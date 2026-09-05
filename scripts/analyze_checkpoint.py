"""Run causal MoE ablations and expert-similarity diagnostics on one checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from splitmoe.config import ExperimentConfig
from splitmoe.data import TokenBlockDataset
from splitmoe.model import DecoderLM, SplitMoE, StandardMoE


def balanced_indices(dataset: TokenBlockDataset, blocks_per_domain: int) -> dict[int, list[int]]:
    selected = {}
    domains = np.asarray(dataset.domains)
    for domain_id in range(len(dataset.metadata["domains"])):
        candidates = np.flatnonzero(domains == domain_id)
        if len(candidates) < blocks_per_domain:
            raise ValueError(f"Domain {domain_id} has only {len(candidates)} validation blocks")
        positions = np.linspace(0, len(candidates) - 1, blocks_per_domain, dtype=np.int64)
        selected[domain_id] = candidates[positions].tolist()
    return selected


def wrong_indices(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Assign every token to a deterministic, guaranteed-different expert."""
    positions = torch.arange(indices.numel(), device=indices.device).reshape_as(indices)
    offset = 1 + positions.remainder(n_experts - 1)
    return (indices + offset) % n_experts


def install_interventions(model: DecoderLM) -> list[torch.nn.Module]:
    modules = []
    for module in model.modules():
        if isinstance(module, StandardMoE):
            def standard_forward(self, x, collect_assignments=False):
                indices, selected, stats = self.router(x, collect_assignments)
                if self.analysis_mode == "wrong":
                    indices = wrong_indices(indices, self.router.n_experts)
                return self.routed(x, indices, selected), stats

            module.analysis_mode = "normal"
            module.forward = types.MethodType(standard_forward, module)
            modules.append(module)
        elif isinstance(module, SplitMoE):
            def split_forward(self, x, collect_assignments=False):
                shared = self.shared(x)
                indices, selected, stats = self.router(x, collect_assignments)
                routed_indices = (
                    wrong_indices(indices, self.router.n_experts)
                    if self.analysis_mode == "wrong"
                    else indices
                )
                private = self.routed(x, routed_indices, selected)
                stats.shared_norm = shared.float().norm(dim=-1).mean().detach()
                stats.private_norm = private.float().norm(dim=-1).mean().detach()
                if self.analysis_mode == "shared_only":
                    result = shared
                elif self.analysis_mode == "private_only":
                    result = private
                else:
                    result = shared + private
                return result * self.output_scale, stats

            module.analysis_mode = "normal"
            module.forward = types.MethodType(split_forward, module)
            modules.append(module)
    return modules


def set_mode(modules: list[torch.nn.Module], mode: str) -> None:
    for module in modules:
        module.analysis_mode = mode


@torch.inference_mode()
def evaluate_mode(
    model: DecoderLM,
    modules: list[torch.nn.Module],
    dataset: TokenBlockDataset,
    selected: dict[int, list[int]],
    mode: str,
    batch_size: int,
) -> dict:
    set_mode(modules, mode)
    domain_names = dataset.metadata["domains"]
    loss_sums = torch.zeros(len(domain_names), dtype=torch.float64)
    counts = torch.zeros(len(domain_names), dtype=torch.float64)
    route_counts = None
    ordered = [(domain_id, index) for domain_id, indices in selected.items() for index in indices]
    for start in range(0, len(ordered), batch_size):
        examples = ordered[start : start + batch_size]
        batch = [dataset[index] for _, index in examples]
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        domains = torch.tensor([domain_id for domain_id, _ in examples])
        output = model(inputs, labels, collect_assignments=True)
        per_token = F.cross_entropy(output.logits.transpose(1, 2), labels, reduction="none")
        per_sequence = per_token.mean(dim=1).double()
        loss_sums.index_add_(0, domains, per_sequence)
        counts.add_(torch.bincount(domains, minlength=len(domain_names)).double())
        if output.router_stats:
            if route_counts is None:
                route_counts = torch.zeros(
                    len(output.router_stats), len(domain_names),
                    output.router_stats[0].expert_fraction.numel(), dtype=torch.float64,
                )
            for layer_id, stats in enumerate(output.router_stats):
                assignments = stats.assignments
                token_domains = domains[:, None].expand_as(assignments)
                for domain_id in range(len(domain_names)):
                    routed = assignments[token_domains == domain_id]
                    if routed.numel():
                        route_counts[layer_id, domain_id].add_(
                            torch.bincount(routed, minlength=route_counts.size(-1)).double()
                        )
    losses = loss_sums / counts
    result = {
        "lm_loss": float(loss_sums.sum() / counts.sum()),
        "perplexity": math.exp(float(loss_sums.sum() / counts.sum())),
        "domains": {name: float(losses[i]) for i, name in enumerate(domain_names)},
    }
    if route_counts is not None:
        fractions = route_counts / route_counts.sum(-1, keepdim=True).clamp_min(1)
        result["routing"] = fractions.tolist()
    return result


def cosine_matrix(outputs: list[torch.Tensor]) -> torch.Tensor:
    flattened = torch.stack([output.flatten() for output in outputs]).float()
    flattened -= flattened.mean(dim=1, keepdim=True)
    return F.normalize(flattened, dim=1) @ F.normalize(flattened, dim=1).T


def cka_matrix(outputs: list[torch.Tensor]) -> torch.Tensor:
    kernels = []
    for output in outputs:
        centered = output.float() - output.float().mean(dim=0, keepdim=True)
        kernel = centered @ centered.T
        kernel -= kernel.mean(dim=0, keepdim=True)
        kernel -= kernel.mean(dim=1, keepdim=True)
        kernel += kernel.mean()
        kernels.append(kernel)
    result = torch.empty(len(kernels), len(kernels))
    norms = [kernel.square().sum().sqrt().clamp_min(1e-12) for kernel in kernels]
    for i, left in enumerate(kernels):
        for j, right in enumerate(kernels):
            result[i, j] = (left * right).sum() / (norms[i] * norms[j])
    return result


@torch.inference_mode()
def similarity_analysis(
    model: DecoderLM,
    modules: list[torch.nn.Module],
    dataset: TokenBlockDataset,
    selected: dict[int, list[int]],
    blocks_per_domain: int,
    max_tokens: int,
) -> dict:
    set_mode(modules, "normal")
    module_names = {module: name for name, module in model.named_modules() if module in modules}
    captured: dict[str, dict[str, dict[str, list]]] = defaultdict(dict)
    current_domain = [""]

    def hook(module, inputs, _output):
        flat = inputs[0].reshape(-1, inputs[0].size(-1))
        if flat.size(0) > max_tokens:
            positions = torch.linspace(0, flat.size(0) - 1, max_tokens).long()
            flat = flat.index_select(0, positions)
        outputs = [expert(flat) for expert in module.routed.experts]
        captured[module_names[module]][current_domain[0]] = {
            "centered_cosine": cosine_matrix(outputs).tolist(),
            "linear_cka": cka_matrix(outputs).tolist(),
        }

    hooks = [module.register_forward_hook(hook) for module in modules]
    try:
        for domain_id, domain_name in enumerate(dataset.metadata["domains"]):
            current_domain[0] = domain_name
            indices = selected[domain_id][:blocks_per_domain]
            inputs = torch.stack([dataset[index][0] for index in indices])
            model(inputs)
    finally:
        for hook_handle in hooks:
            hook_handle.remove()
    return dict(captured)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocks-per-domain", type=int, default=16)
    parser.add_argument("--similarity-blocks-per-domain", type=int, default=4)
    parser.add_argument("--similarity-max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    checkpoint_step = int(checkpoint["step"])
    model = DecoderLM(config.model)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    model.eval()
    dataset = TokenBlockDataset(args.validation_data)
    selected = balanced_indices(dataset, args.blocks_per_domain)
    modules = install_interventions(model)
    if config.model.moe_type == "split":
        modes = ["normal", "wrong", "shared_only", "private_only"]
    elif config.model.moe_type == "standard":
        modes = ["normal", "wrong"]
    else:
        modes = ["normal"]
    ablations = {
        mode: evaluate_mode(model, modules, dataset, selected, mode, args.batch_size)
        for mode in modes
    }
    similarities = (
        similarity_analysis(
            model, modules, dataset, selected,
            args.similarity_blocks_per_domain, args.similarity_max_tokens,
        )
        if modules else {}
    )
    result = {
        "architecture": config.model.moe_type,
        "seed": config.train.seed,
        "checkpoint_step": checkpoint_step,
        "evaluation": {
            "blocks_per_domain": args.blocks_per_domain,
            "selection": "Evenly spaced deterministic sample within each validation domain",
            "wrong_expert": "Per-token deterministic assignment to a guaranteed non-selected expert",
            "ablation_scale": "Retains the trained SplitMoE output scale",
        },
        "ablations": ablations,
        "similarities": similarities,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"architecture": result["architecture"], "seed": result["seed"], "losses": {
        mode: values["lm_loss"] for mode, values in ablations.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
