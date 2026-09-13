"""Evaluate a model-only SplitMoE checkpoint on pretokenized LAMBADA."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from splitmoe.config import ExperimentConfig
from splitmoe.heldout import HeldoutTokenDataset, length_grouped_batches
from splitmoe.model import DecoderLM


@torch.inference_mode()
def evaluate(
    checkpoint_path: Path,
    data_path: Path,
    batch_size: int,
    device_name: str,
    max_examples: int | None = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    model = DecoderLM(config.model)
    model.load_state_dict(checkpoint["model"])
    step = int(checkpoint["step"])
    del checkpoint

    device = torch.device(device_name)
    model.to(device).eval()
    dataset = HeldoutTokenDataset(data_path)
    precision = config.train.precision
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    use_amp = device.type == "cuda" and precision in {"fp16", "bf16"}

    full_nll = 0.0
    full_tokens = 0
    target_nll = 0.0
    target_tokens = 0
    exact_examples = 0
    examples = 0
    entropy_sums: list[float] = []
    route_counts: list[torch.Tensor] = []
    started = time.perf_counter()

    selected = None if max_examples is None else range(min(max_examples, len(dataset)))
    for documents, target_starts in length_grouped_batches(
        dataset,
        batch_size=batch_size,
        max_seq_len=config.model.max_seq_len,
        indices=selected,
    ):
        documents = documents.to(device)
        inputs, labels = documents[:, :-1], documents[:, 1:]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model(inputs, collect_assignments=True)
        per_token = F.cross_entropy(
            output.logits.float().flatten(0, 1), labels.flatten(), reduction="none"
        ).view_as(labels)
        full_nll += float(per_token.sum())
        full_tokens += labels.numel()

        positions = torch.arange(labels.size(1), device=device).unsqueeze(0)
        target_mask = positions >= (target_starts.to(device).unsqueeze(1) - 1)
        target_nll += float(per_token[target_mask].sum())
        target_tokens += int(target_mask.sum())
        correct = output.logits.argmax(dim=-1).eq(labels) | ~target_mask
        exact_examples += int(correct.all(dim=1).sum())
        examples += labels.size(0)

        if not entropy_sums:
            entropy_sums = [0.0] * len(output.router_stats)
            route_counts = [
                torch.zeros(config.model.n_experts, dtype=torch.float64)
                for _ in output.router_stats
            ]
        token_count = inputs.numel()
        for layer, stats in enumerate(output.router_stats):
            entropy_sums[layer] += float(stats.entropy) * token_count
            assignments = stats.assignments.reshape(-1)
            route_counts[layer] += torch.bincount(
                assignments.cpu(), minlength=config.model.n_experts
            ).double()

    elapsed = time.perf_counter() - started
    result = {
        "checkpoint": checkpoint_path.name,
        "architecture": config.model.moe_type,
        "seed": config.train.seed,
        "checkpoint_step": step,
        "dataset": dataset.metadata,
        "evaluation": {
            "max_context_tokens": config.model.max_seq_len,
            "long_document_policy": "Keep the final max_seq_len+1 tokens",
            "full_lm_loss": full_nll / full_tokens,
            "full_perplexity": math.exp(full_nll / full_tokens),
            "last_word_token_loss": target_nll / target_tokens,
            "last_word_perplexity": math.exp(target_nll / target_tokens),
            "last_word_exact_match": exact_examples / examples,
            "examples": examples,
            "scored_tokens": full_tokens,
            "target_tokens": target_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": full_tokens / elapsed,
            "device": str(device),
            "precision": f"{precision} autocast" if use_amp else "fp32",
            "batching": "Documents grouped by identical token length; no padding or cross-document context",
        },
        "routing": [
            {
                "layer": (index + 1) * config.model.moe_every,
                "entropy": entropy_sums[index] / full_tokens,
                "expert_fraction": (counts / counts.sum()).tolist(),
            }
            for index, counts in enumerate(route_counts)
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = evaluate(
        Path(args.checkpoint), Path(args.data), args.batch_size, args.device, args.max_examples
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
