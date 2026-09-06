"""Run causal MoE ablations and expert-similarity diagnostics on one checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import statistics
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


def wrong_indices(indices: torch.Tensor, n_experts: int, offset: int) -> torch.Tensor:
    """Move every selected expert by a fixed, guaranteed-nonzero cyclic offset."""
    if not 1 <= offset < n_experts:
        raise ValueError(f"Wrong-expert offset must be in [1, {n_experts - 1}], got {offset}")
    return (indices + offset) % n_experts


def install_interventions(model: DecoderLM) -> list[torch.nn.Module]:
    modules = []
    for module in model.modules():
        if isinstance(module, StandardMoE):
            def standard_forward(self, x, collect_assignments=False):
                indices, selected, stats = self.router(x, collect_assignments)
                if self.analysis_mode == "wrong":
                    indices = wrong_indices(
                        indices, self.router.n_experts, self.analysis_wrong_offset
                    )
                return self.routed(x, indices, selected), stats

            module.analysis_mode = "normal"
            module.analysis_wrong_offset = 1
            module.forward = types.MethodType(standard_forward, module)
            modules.append(module)
        elif isinstance(module, SplitMoE):
            def split_forward(self, x, collect_assignments=False):
                shared = self.shared(x)
                indices, selected, stats = self.router(x, collect_assignments)
                routed_indices = (
                    wrong_indices(
                        indices, self.router.n_experts, self.analysis_wrong_offset
                    )
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
            module.analysis_wrong_offset = 1
            module.forward = types.MethodType(split_forward, module)
            modules.append(module)
    return modules


def set_intervention(
    modules: list[torch.nn.Module],
    mode: str,
    wrong_offset: int = 1,
    target_module: int | None = None,
) -> None:
    for module_index, module in enumerate(modules):
        module.analysis_mode = mode if target_module is None or module_index == target_module else "normal"
        module.analysis_wrong_offset = wrong_offset


@torch.inference_mode()
def evaluate_mode(
    model: DecoderLM,
    modules: list[torch.nn.Module],
    dataset: TokenBlockDataset,
    selected: dict[int, list[int]],
    mode: str,
    batch_size: int,
    device: torch.device,
    wrong_offset: int = 1,
    target_module: int | None = None,
) -> dict:
    set_intervention(modules, mode, wrong_offset, target_module)
    domain_names = dataset.metadata["domains"]
    loss_sums = torch.zeros(len(domain_names), dtype=torch.float64)
    counts = torch.zeros(len(domain_names), dtype=torch.float64)
    route_counts = None
    ordered = [(domain_id, index) for domain_id, indices in selected.items() for index in indices]
    for start in range(0, len(ordered), batch_size):
        examples = ordered[start : start + batch_size]
        batch = [dataset[index] for _, index in examples]
        inputs = torch.stack([item[0] for item in batch]).to(device)
        labels = torch.stack([item[1] for item in batch]).to(device)
        domains = torch.tensor([domain_id for domain_id, _ in examples])
        output = model(inputs, labels, collect_assignments=True)
        per_token = F.cross_entropy(output.logits.transpose(1, 2), labels, reduction="none")
        per_sequence = per_token.mean(dim=1).double().cpu()
        loss_sums.index_add_(0, domains, per_sequence)
        counts.add_(torch.bincount(domains, minlength=len(domain_names)).double())
        if output.router_stats:
            if route_counts is None:
                route_counts = torch.zeros(
                    len(output.router_stats), len(domain_names),
                    output.router_stats[0].expert_fraction.numel(), dtype=torch.float64,
                )
            for layer_id, stats in enumerate(output.router_stats):
                assignments = stats.assignments.cpu()
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


def average_evaluations(evaluations: list[dict]) -> dict:
    mean_loss = statistics.mean(item["lm_loss"] for item in evaluations)
    result = {
        "lm_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "domains": {
            domain: statistics.mean(item["domains"][domain] for item in evaluations)
            for domain in evaluations[0]["domains"]
        },
    }
    if "routing" in evaluations[0]:
        result["routing"] = np.asarray([item["routing"] for item in evaluations]).mean(axis=0).tolist()
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
    device: torch.device,
) -> dict:
    set_intervention(modules, "normal")
    module_names = {module: name for name, module in model.named_modules() if module in modules}
    captured: dict[str, dict[str, dict[str, list]]] = defaultdict(dict)
    current_domain = [""]

    def hook(module, inputs, _output):
        flat = inputs[0].reshape(-1, inputs[0].size(-1))
        if flat.size(0) > max_tokens:
            positions = torch.linspace(
                0, flat.size(0) - 1, max_tokens, device=flat.device
            ).long()
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
            inputs = torch.stack([dataset[index][0] for index in indices]).to(device)
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
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete device")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    checkpoint_step = int(checkpoint["step"])
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = DecoderLM(config.model)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    model.to(device).eval()
    dataset = TokenBlockDataset(args.validation_data)
    selected = balanced_indices(dataset, args.blocks_per_domain)
    modules = install_interventions(model)
    ablations = {
        "normal": evaluate_mode(
            model, modules, dataset, selected, "normal", args.batch_size, device
        )
    }
    wrong_experts = {}
    layerwise_wrong = {}
    if modules:
        for offset in range(1, config.model.n_experts):
            wrong_experts[f"offset_{offset}"] = evaluate_mode(
                model, modules, dataset, selected, "wrong", args.batch_size,
                device, wrong_offset=offset,
            )
        ablations["wrong"] = average_evaluations(list(wrong_experts.values()))
        layer_numbers = list(range(config.model.moe_every, config.model.n_layers + 1, config.model.moe_every))
        for module_index, layer_number in enumerate(layer_numbers):
            alternatives = {}
            for offset in range(1, config.model.n_experts):
                alternatives[f"offset_{offset}"] = evaluate_mode(
                    model, modules, dataset, selected, "wrong", args.batch_size,
                    device, wrong_offset=offset, target_module=module_index,
                )
            layerwise_wrong[str(layer_number)] = {
                "alternatives": alternatives,
                "mean": average_evaluations(list(alternatives.values())),
            }
    if config.model.moe_type == "split":
        for mode in ("shared_only", "private_only"):
            ablations[mode] = evaluate_mode(
                model, modules, dataset, selected, mode, args.batch_size, device
            )
    similarities = (
        similarity_analysis(
            model, modules, dataset, selected,
            args.similarity_blocks_per_domain, args.similarity_max_tokens, device,
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
            "wrong_expert": "Every token is evaluated against every non-selected expert via cyclic offsets; reported wrong loss is their arithmetic mean",
            "layerwise_wrong_expert": "Only one MoE layer is corrupted at a time; all other layers retain their routed selections",
            "ablation_scale": "Retains the trained SplitMoE output scale",
            "device": str(device),
        },
        "ablations": ablations,
        "wrong_experts": wrong_experts,
        "layerwise_wrong": layerwise_wrong,
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
