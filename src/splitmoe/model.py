from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


@dataclass
class RouterStats:
    aux_loss: torch.Tensor
    z_loss: torch.Tensor
    entropy: torch.Tensor
    expert_fraction: torch.Tensor
    assignments: torch.Tensor | None = None
    shared_norm: torch.Tensor | None = None
    private_norm: torch.Tensor | None = None


class Top1Router(nn.Module):
    def __init__(self, dim: int, n_experts: int, jitter: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(dim, n_experts, bias=False)
        self.n_experts = n_experts
        self.jitter = jitter

    def forward(self, x: torch.Tensor, collect_assignments: bool = False):
        router_x = x
        if self.training and self.jitter > 0:
            router_x = x * torch.empty_like(x).uniform_(1 - self.jitter, 1 + self.jitter)
        logits = self.proj(router_x).float()
        probs = logits.softmax(dim=-1)
        indices = probs.argmax(dim=-1)
        selected = probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        one_hot = F.one_hot(indices, self.n_experts).float()
        fraction = one_hot.mean(dim=tuple(range(one_hot.ndim - 1)))
        mean_prob = probs.mean(dim=tuple(range(probs.ndim - 1)))
        aux = self.n_experts * torch.sum(fraction.detach() * mean_prob)
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        stats = RouterStats(
            aux_loss=aux,
            z_loss=z_loss,
            entropy=entropy.detach(),
            expert_fraction=fraction.detach(),
            assignments=indices.detach() if collect_assignments else None,
        )
        return indices, selected, stats


class RoutedExperts(nn.Module):
    def __init__(self, dim: int, width: int, n_experts: int, dropout: float, weight_mode: str):
        super().__init__()
        self.experts = nn.ModuleList([SwiGLU(dim, width, dropout) for _ in range(n_experts)])
        self.weight_mode = weight_mode

    def forward(self, x: torch.Tensor, indices: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        flat_x = x.reshape(-1, x.size(-1))
        flat_indices = indices.reshape(-1)
        flat_selected = selected.reshape(-1)
        output = None
        for expert_id, expert in enumerate(self.experts):
            positions = torch.where(flat_indices == expert_id)[0]
            if positions.numel() == 0:
                continue
            expert_out = expert(flat_x.index_select(0, positions))
            gate = flat_selected.index_select(0, positions)
            if self.weight_mode == "straight_through":
                gate = gate / gate.detach().clamp_min(1e-6)
            elif self.weight_mode == "none":
                gate = torch.ones_like(gate)
            expert_out = expert_out * gate.unsqueeze(-1).to(expert_out.dtype)
            if output is None:
                # Under AMP, the residual stream can be FP32 while Linear outputs
                # are FP16/BF16. index_copy requires matching source/destination dtypes.
                output = torch.zeros(flat_x.shape, device=flat_x.device, dtype=expert_out.dtype)
            output.index_copy_(0, positions, expert_out)
        if output is None:
            raise RuntimeError("Cannot dispatch an empty token tensor")
        return output.view_as(x)

    @torch.no_grad()
    def pairwise_similarity(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.size(-1))
        outputs = torch.stack([expert(flat).float().flatten() for expert in self.experts])
        outputs = outputs - outputs.mean(dim=1, keepdim=True)
        outputs = F.normalize(outputs, dim=1)
        return outputs @ outputs.T


class StandardMoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.router = Top1Router(cfg.d_model, cfg.n_experts, cfg.router_jitter)
        self.routed = RoutedExperts(
            cfg.d_model, cfg.standard_expert_width, cfg.n_experts, cfg.dropout, cfg.router_weight_mode
        )

    def forward(self, x: torch.Tensor, collect_assignments: bool = False):
        indices, selected, stats = self.router(x, collect_assignments)
        return self.routed(x, indices, selected), stats


class SplitMoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.shared = SwiGLU(cfg.d_model, cfg.shared_width, cfg.dropout)
        self.router = Top1Router(cfg.d_model, cfg.n_experts, cfg.router_jitter)
        self.routed = RoutedExperts(
            cfg.d_model, cfg.private_width, cfg.n_experts, cfg.dropout, cfg.router_weight_mode
        )
        self.output_scale = cfg.split_output_scale

    def forward(self, x: torch.Tensor, collect_assignments: bool = False):
        shared = self.shared(x)
        indices, selected, stats = self.router(x, collect_assignments)
        private = self.routed(x, indices, selected)
        stats.shared_norm = shared.float().norm(dim=-1).mean().detach()
        stats.private_norm = private.float().norm(dim=-1).mean().detach()
        return (shared + private) * self.output_scale, stats


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, seq, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.out(y.transpose(1, 2).contiguous().view(batch, seq, dim))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, use_moe: bool):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        if not use_moe or cfg.moe_type == "dense":
            self.ffn: nn.Module = SwiGLU(cfg.d_model, cfg.dense_ffn_width, cfg.dropout)
        elif cfg.moe_type == "standard":
            self.ffn = StandardMoE(cfg)
        else:
            self.ffn = SplitMoE(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, collect_assignments: bool = False):
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        ffn_input = self.ffn_norm(x)
        if isinstance(self.ffn, (StandardMoE, SplitMoE)):
            y, stats = self.ffn(ffn_input, collect_assignments)
        else:
            y, stats = self.ffn(ffn_input), None
        return x + self.dropout(y), stats


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    lm_loss: torch.Tensor | None
    router_stats: list[RouterStats] = field(default_factory=list)


class DecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=((i + 1) % cfg.moe_every == 0)) for i in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, collect_assignments: bool = False
    ) -> LMOutput:
        _, seq = input_ids.shape
        if seq > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq} exceeds max_seq_len {self.cfg.max_seq_len}")
        positions = torch.arange(seq, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        stats: list[RouterStats] = []
        for block in self.blocks:
            x, block_stats = block(x, collect_assignments)
            if block_stats is not None:
                stats.append(block_stats)
        logits = self.lm_head(self.norm(x))
        lm_loss = None if labels is None else F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        if lm_loss is None:
            loss = None
        else:
            aux = torch.stack([s.aux_loss for s in stats]).mean() if stats else logits.new_zeros(())
            z_loss = torch.stack([s.z_loss for s in stats]).mean() if stats else logits.new_zeros(())
            loss = lm_loss + self.cfg.router_aux_loss_coef * aux + self.cfg.router_z_loss_coef * z_loss
        return LMOutput(logits=logits, loss=loss, lm_loss=lm_loss, router_stats=stats)

    def parameter_summary(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        routed = sum(p.numel() for name, p in self.named_parameters() if ".routed.experts." in name)
        active_routed = sum(
            sum(p.numel() for p in module.experts[0].parameters())
            for module in self.modules()
            if isinstance(module, RoutedExperts)
        )
        shared = sum(p.numel() for name, p in self.named_parameters() if ".ffn.shared." in name)
        router = sum(p.numel() for name, p in self.named_parameters() if ".router." in name)
        activated = total - routed + active_routed
        return {
            "total": total,
            "activated_per_token": activated,
            "routed_experts": routed,
            "shared_ffn": shared,
            "routers": router,
        }
