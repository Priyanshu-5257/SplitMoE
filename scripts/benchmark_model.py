"""Standardized forward-only architecture benchmark with no logging or validation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from splitmoe.config import ExperimentConfig
from splitmoe.model import DecoderLM


@torch.inference_mode()
def benchmark(config_path: Path, batch_size: int, warmup: int, iterations: int, device_name: str) -> dict:
    config = ExperimentConfig.from_json(config_path)
    torch.manual_seed(1729)
    device = torch.device(device_name)
    model = DecoderLM(config.model).to(device).eval()
    tokens = torch.randint(
        config.model.vocab_size,
        (batch_size, config.model.max_seq_len),
        device=device,
    )
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16

    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            model(tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies = []
    for _ in range(iterations):
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            model(tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - started)

    tokens_per_iteration = tokens.numel()
    return {
        "config": config_path.name,
        "architecture": config.model.moe_type,
        "model": config.model.__dict__,
        "parameters": model.parameter_summary(),
        "protocol": {
            "seed": 1729,
            "batch_size": batch_size,
            "sequence_length": config.model.max_seq_len,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "precision": "fp16 autocast" if use_amp else "fp32",
            "weights": "Deterministically initialized; forward runtime is architecture-only",
            "logging": "disabled",
        },
        "systems": {
            "median_latency_seconds": statistics.median(latencies),
            "mean_latency_seconds": statistics.mean(latencies),
            "median_tokens_per_second": tokens_per_iteration / statistics.median(latencies),
            "mean_tokens_per_second": tokens_per_iteration / statistics.mean(latencies),
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = benchmark(
        Path(args.config), args.batch_size, args.warmup, args.iterations, args.device
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["systems"], indent=2))


if __name__ == "__main__":
    main()
