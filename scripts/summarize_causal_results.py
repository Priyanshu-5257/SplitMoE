"""Aggregate causal-ablation JSON files produced by analyze_checkpoint.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARCHITECTURES = ["dense", "standard", "split"]
DOMAINS = ["stories", "wiki", "code", "math"]
COLORS = {"dense": "#64748B", "standard": "#2563EB", "split": "#E11D48"}
T_CRITICAL_95_DF4 = 2.776445105


def mean_ci(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half_width = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
    return {"mean": mean, "sample_sd": sd, "ci95": [mean - half_width, mean + half_width]}


def off_diagonal_mean(matrix: list[list[float]]) -> float:
    return statistics.mean(
        matrix[i][j] for i in range(len(matrix)) for j in range(i + 1, len(matrix))
    )


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/causal/raw")
    parser.add_argument("--output-dir", default="results/causal")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {
        architecture: [json.loads(path.read_text()) for path in sorted(input_dir.glob(f"{architecture}_seed_*.json"))]
        for architecture in ARCHITECTURES
    }
    if any(len(items) != 5 for items in records.values()):
        raise RuntimeError({architecture: len(items) for architecture, items in records.items()})

    summary = {
        "method": {
            "seeds": [record["seed"] for record in records["dense"]],
            "blocks_per_domain": records["dense"][0]["evaluation"]["blocks_per_domain"],
            "domains": DOMAINS,
            "confidence_intervals": "Two-sided 95% t intervals across seeds (df=4)",
            "wrong_expert": records["split"][0]["evaluation"]["wrong_expert"],
            "similarity": "Four blocks/domain; 128 evenly spaced tokens/layer; off-diagonal expert-pair mean",
        },
        "losses": {},
        "causal_penalties": {},
        "expert_similarity": {},
    }
    rows = []
    for architecture, items in records.items():
        summary["losses"][architecture] = {}
        for mode in items[0]["ablations"]:
            values = [item["ablations"][mode]["lm_loss"] for item in items]
            summary["losses"][architecture][mode] = mean_ci(values)
            for item, value in zip(items, values, strict=True):
                rows.append({"architecture": architecture, "seed": item["seed"], "mode": mode, "lm_loss": value})

    penalty_specs = {
        "standard_wrong_expert": ("standard", "wrong", "normal"),
        "split_wrong_expert": ("split", "wrong", "normal"),
        "split_remove_private": ("split", "shared_only", "normal"),
        "split_remove_shared": ("split", "private_only", "normal"),
    }
    penalties = {}
    for label, (architecture, altered, reference) in penalty_specs.items():
        values = [
            item["ablations"][altered]["lm_loss"] - item["ablations"][reference]["lm_loss"]
            for item in records[architecture]
        ]
        penalties[label] = values
        summary["causal_penalties"][label] = mean_ci(values)
        summary["causal_penalties"][label]["positive_seeds"] = sum(value > 0 for value in values)
    wrong_vs_absent = [
        item["ablations"]["wrong"]["lm_loss"] - item["ablations"]["shared_only"]["lm_loss"]
        for item in records["split"]
    ]
    summary["causal_penalties"]["split_wrong_expert_minus_no_private"] = mean_ci(wrong_vs_absent)
    summary["causal_penalties"]["split_wrong_expert_minus_no_private"]["positive_seeds"] = sum(
        value > 0 for value in wrong_vs_absent
    )
    summary["causal_penalties"]["wrong_expert_by_domain"] = {}
    for architecture in ("standard", "split"):
        summary["causal_penalties"]["wrong_expert_by_domain"][architecture] = {}
        for domain in DOMAINS:
            values = [
                item["ablations"]["wrong"]["domains"][domain]
                - item["ablations"]["normal"]["domains"][domain]
                for item in records[architecture]
            ]
            summary["causal_penalties"]["wrong_expert_by_domain"][architecture][domain] = mean_ci(values)

    similarity_by_layer = {architecture: {metric: {} for metric in ("centered_cosine", "linear_cka")} for architecture in ("standard", "split")}
    similarity_overall = {architecture: {metric: [] for metric in ("centered_cosine", "linear_cka")} for architecture in ("standard", "split")}
    for architecture in ("standard", "split"):
        layers = list(records[architecture][0]["similarities"])
        for metric in ("centered_cosine", "linear_cka"):
            for layer in layers:
                per_seed = []
                for item in records[architecture]:
                    values = [
                        off_diagonal_mean(item["similarities"][layer][domain][metric])
                        for domain in DOMAINS
                    ]
                    per_seed.append(statistics.mean(values))
                similarity_by_layer[architecture][metric][layer] = per_seed
            similarity_overall[architecture][metric] = [
                statistics.mean(
                    similarity_by_layer[architecture][metric][layer][seed_index]
                    for layer in layers
                )
                for seed_index in range(5)
            ]
            summary["expert_similarity"].setdefault(architecture, {})[metric] = {
                "overall": mean_ci(similarity_overall[architecture][metric]),
                "by_layer": {
                    layer: mean_ci(values)
                    for layer, values in similarity_by_layer[architecture][metric].items()
                },
            }
    for metric in ("centered_cosine", "linear_cka"):
        differences = [
            split - standard
            for split, standard in zip(
                similarity_overall["split"][metric], similarity_overall["standard"][metric], strict=True
            )
        ]
        summary["expert_similarity"][f"split_minus_standard_{metric}"] = mean_ci(differences)

    with (output_dir / "ablation_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["architecture", "seed", "mode", "lm_loss"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    configure_plots()
    plot_penalties(penalties, output_dir)
    plot_domain_penalties(summary, output_dir)
    plot_similarity(similarity_by_layer, output_dir)
    print(json.dumps(summary, indent=2))


def plot_penalties(penalties: dict[str, list[float]], output_dir: Path) -> None:
    labels = ["Standard\nwrong expert", "Split\nwrong expert", "Split\nremove private", "Split\nremove shared"]
    keys = ["standard_wrong_expert", "split_wrong_expert", "split_remove_private", "split_remove_shared"]
    colors = [COLORS["standard"], COLORS["split"], "#F97316", "#8B5CF6"]
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(keys))
    for seed_index in range(5):
        axis.scatter(x, [penalties[key][seed_index] for key in keys], color=colors, alpha=0.55, s=24, zorder=3)
    means = [statistics.mean(penalties[key]) for key in keys]
    cis = [T_CRITICAL_95_DF4 * statistics.stdev(penalties[key]) / math.sqrt(5) for key in keys]
    bars = axis.bar(x, means, color=colors, alpha=0.3, yerr=cis, capsize=5)
    axis.bar_label(bars, labels=[f"+{value:.3f}" for value in means], padding=5)
    axis.set_xticks(x, labels)
    axis.set(ylabel="Increase in balanced LM loss", title="Causal intervention penalties across five seeds")
    fig.tight_layout()
    fig.savefig(output_dir / "causal_penalties.png", bbox_inches="tight")
    plt.close(fig)


def plot_domain_penalties(summary: dict, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(DOMAINS))
    width = 0.34
    for offset, architecture in ((-0.5, "standard"), (0.5, "split")):
        stats = summary["causal_penalties"]["wrong_expert_by_domain"][architecture]
        means = [stats[domain]["mean"] for domain in DOMAINS]
        cis = [(stats[domain]["ci95"][1] - stats[domain]["ci95"][0]) / 2 for domain in DOMAINS]
        axis.bar(x + offset * width, means, width, yerr=cis, capsize=4, color=COLORS[architecture], label=architecture.title())
    axis.set_xticks(x, [domain.title() for domain in DOMAINS])
    axis.set(ylabel="Wrong-expert LM-loss penalty", title="Correct routing matters in every domain")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "wrong_expert_by_domain.png", bbox_inches="tight")
    plt.close(fig)


def plot_similarity(similarity_by_layer: dict, output_dir: Path) -> None:
    metrics = [("centered_cosine", "Centered cosine similarity"), ("linear_cka", "Linear CKA")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    x = np.arange(4)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for architecture in ("standard", "split"):
            layers = list(similarity_by_layer[architecture][metric])
            values = [similarity_by_layer[architecture][metric][layer] for layer in layers]
            means = [statistics.mean(item) for item in values]
            cis = [T_CRITICAL_95_DF4 * statistics.stdev(item) / math.sqrt(5) for item in values]
            axis.errorbar(x, means, yerr=cis, marker="o", capsize=4, linewidth=2, color=COLORS[architecture], label=architecture.title())
        axis.set_xticks(x, ["2", "4", "6", "8"])
        axis.set(xlabel="Transformer layer", ylabel="Mean expert-pair similarity", title=title)
        axis.legend()
    fig.suptitle("Private SplitMoE experts are less redundant", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "expert_similarity.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
