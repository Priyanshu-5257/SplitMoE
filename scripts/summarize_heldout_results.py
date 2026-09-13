"""Aggregate paired LAMBADA evaluations and render paper-ready figures."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUPS = {
    "Top-1 Standard": "top1_standard_seed{seed}.json",
    "Top-1 Split-25": "top1_split_seed{seed}.json",
    "Top-4 Equal-active Split": "equal_active_split_seed{seed}.json",
    "Top-4 Shared expert": "shared_expert_seed{seed}.json",
}
SEEDS = (1337, 2027, 3407)
METRICS = ("full_lm_loss", "last_word_token_loss", "last_word_exact_match")
T_CRITICAL_95_DF2 = 4.302652729911275


def mean_ci(values: list[float]) -> dict:
    mean = statistics.mean(values)
    radius = T_CRITICAL_95_DF2 * statistics.stdev(values) / len(values) ** 0.5
    return {"mean": mean, "ci95": [mean - radius, mean + radius], "values": values}


def main() -> None:
    root = Path("results/heldout")
    raw = root / "raw"
    records: dict[str, list[dict]] = {}
    for model, pattern in GROUPS.items():
        records[model] = []
        for seed in SEEDS:
            record = json.loads((raw / pattern.format(seed=seed)).read_text())
            if int(record["seed"]) != seed:
                raise ValueError(f"Seed mismatch for {model}: expected {seed}")
            if int(record["checkpoint_step"]) != 6500:
                raise ValueError(f"Unexpected checkpoint step for {model}, seed {seed}")
            records[model].append(record)

    manifest = records["Top-1 Standard"][0]["dataset"]
    for rows in records.values():
        for row in rows:
            if row["dataset"] != manifest:
                raise ValueError("Held-out dataset manifests differ across runs")
            if row["evaluation"]["device"] != "cpu":
                raise ValueError("Held-out aggregation requires the standardized CPU path")

    summary = {
        "protocol": {
            "seeds": list(SEEDS),
            "uncertainty": "Two-sided 95% Student's t interval over three training seeds",
            "pairing": "Differences are paired by training seed",
            "dataset": manifest,
            "evaluation": {
                "device": "CPU",
                "precision": "FP32",
                "primary_metric": "Full-token autoregressive language-model loss",
                "target_word_diagnostics": (
                    "Loss and exact argmax token-sequence match under teacher forcing"
                ),
            },
        },
        "models": {},
        "paired": {},
    }
    table_rows = []
    for model, rows in records.items():
        summary["models"][model] = {
            metric: mean_ci([float(row["evaluation"][metric]) for row in rows])
            for metric in METRICS
        }
        for seed, row in zip(SEEDS, rows, strict=True):
            table_rows.append({
                "model": model,
                "seed": seed,
                **{metric: row["evaluation"][metric] for metric in METRICS},
                "full_perplexity": row["evaluation"]["full_perplexity"],
                "last_word_perplexity": row["evaluation"]["last_word_perplexity"],
            })

    comparisons = {
        "Top-1 Split-25 minus Standard": ("Top-1 Split-25", "Top-1 Standard"),
        "Top-4 Equal-active Split minus Shared expert": (
            "Top-4 Equal-active Split", "Top-4 Shared expert"
        ),
    }
    for label, (left, right) in comparisons.items():
        summary["paired"][label] = {
            metric: mean_ci([
                float(a["evaluation"][metric]) - float(b["evaluation"][metric])
                for a, b in zip(records[left], records[right], strict=True)
            ])
            for metric in METRICS
        }

    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table_rows)
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_pair(records, "Top-1 Standard", "Top-1 Split-25", root / "top1_lambada.png")
    plot_pair(
        records,
        "Top-4 Equal-active Split",
        "Top-4 Shared expert",
        root / "top4_lambada.png",
    )
    print(json.dumps(summary["paired"], indent=2))


def plot_pair(records: dict, left: str, right: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    colors = {left: "#5664d2", right: "#e07255"}
    for axis, metric, title, ylabel in (
        (axes[0], "full_lm_loss", "Full-token language modeling", "LAMBADA LM loss ↓"),
        (
            axes[1],
            "last_word_exact_match",
            "Teacher-forced final-word tokens",
            "Exact token-sequence match ↑",
        ),
    ):
        for x, model in enumerate((left, right)):
            values = [float(row["evaluation"][metric]) for row in records[model]]
            axis.scatter(np.full(len(values), x), values, s=48, color=colors[model], zorder=3)
            axis.plot([x - 0.18, x + 0.18], [statistics.mean(values)] * 2, color="black", lw=2)
        for seed_index in range(len(SEEDS)):
            axis.plot(
                [0, 1],
                [
                    records[left][seed_index]["evaluation"][metric],
                    records[right][seed_index]["evaluation"][metric],
                ],
                color="#999999",
                alpha=0.55,
                lw=1,
            )
        axis.set_xticks([0, 1], [left.replace("Top-1 ", "").replace("Top-4 ", ""), right.replace("Top-1 ", "").replace("Top-4 ", "")])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Independent held-out evaluation (three paired seeds)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
