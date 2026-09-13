"""Combine standardized single-T4 benchmark outputs with final validation quality."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = {
    "Top-4 Standard": "top4_standard.json",
    "Top-4 Practical Split": "top4_practical_split.json",
    "Top-4 Equal-active Split": "top4_equal_active_split.json",
    "Top-4 Shared expert": "top4_shared_expert.json",
}


def main() -> None:
    root = Path("results/systems")
    quality = json.loads(Path("results/paper_scaling/summary.json").read_text())["models"]
    rows = []
    raw = {}
    for model, filename in MODELS.items():
        record = json.loads((root / "raw" / filename).read_text())
        raw[model] = record
        rows.append({
            "model": model,
            "validation_lm_loss_mean": quality[model]["validation_lm_loss"]["mean"],
            "total_parameters": record["parameters"]["total"],
            "activated_parameters": record["parameters"]["activated_per_token"],
            "median_tokens_per_second": record["systems"]["median_tokens_per_second"],
            "median_latency_seconds": record["systems"]["median_latency_seconds"],
            "peak_allocated_bytes": record["systems"]["peak_allocated_bytes"],
        })
    summary = {
        "protocol": next(iter(raw.values()))["protocol"],
        "hardware": {
            key: next(iter(raw.values()))["systems"][key]
            for key in ("device", "torch_version", "cuda_version")
        },
        "models": {row["model"]: row for row in rows},
        "caveat": (
            "Deterministically initialized weights isolate architecture runtime but do not reproduce "
            "trained routing distributions. Report alongside, not instead of, measured training throughput."
        ),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, root / "standardized_t4.png")
    print(json.dumps(summary, indent=2))


def plot(rows: list[dict], output: Path) -> None:
    labels = [row["model"].replace("Top-4 ", "") for row in rows]
    throughput = [row["median_tokens_per_second"] / 1000 for row in rows]
    memory = [row["peak_allocated_bytes"] / 2**30 for row in rows]
    colors = ["#5664d2", "#53a77a", "#d39b43", "#e07255"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    axes[0].bar(x, throughput, color=colors)
    axes[0].set(ylabel="Median thousand tokens/s ↑", title="Forward throughput")
    axes[1].bar(x, memory, color=colors)
    axes[1].set(ylabel="Peak allocated GiB ↓", title="Resident model + forward peak")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Standardized single-T4 inference, batch 4 × context 256", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
