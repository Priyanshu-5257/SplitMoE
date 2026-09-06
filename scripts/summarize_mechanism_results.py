"""Aggregate stricter causal and width-controlled similarity analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = [1337, 2027, 3407, 4517, 5651]
DOMAINS = ["stories", "wiki", "code", "math"]
LAYERS = ["2", "4", "6", "8"]
METRICS = ["centered_cosine", "linear_cka"]
T_CRITICAL_95_DF4 = 2.776445105
COLORS = {
    "Standard-1024": "#1D4ED8",
    "Standard-640": "#3B82F6",
    "Standard-512": "#93C5FD",
    "Split private-512": "#E11D48",
}


def mean_ci(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
    return {"mean": mean, "sample_sd": sd, "ci95": [mean - half, mean + half]}


def paired_summary(values: list[float]) -> dict:
    result = mean_ci(values)
    result["positive_seeds"] = sum(value > 0 for value in values)
    result["seed_count"] = len(values)
    return result


def off_diagonal_mean(matrix: list[list[float]]) -> float:
    return statistics.mean(
        matrix[i][j] for i in range(len(matrix)) for j in range(i + 1, len(matrix))
    )


def load_records(directory: Path, prefix: str) -> list[dict]:
    records = [
        json.loads((directory / f"{prefix}_seed_{seed}.json").read_text()) for seed in SEEDS
    ]
    if [int(record["seed"]) for record in records] != SEEDS:
        raise RuntimeError(f"Seed mismatch for {prefix}")
    return records


def per_seed_similarity(records: list[dict], metric: str) -> dict[str, list[float]]:
    result = {layer: [] for layer in LAYERS}
    for record in records:
        named_layers = list(record["similarities"])
        if len(named_layers) != len(LAYERS):
            raise RuntimeError(f"Expected four similarity layers, found {named_layers}")
        for layer, named_layer in zip(LAYERS, named_layers, strict=True):
            domain_values = [
                off_diagonal_mean(record["similarities"][named_layer][domain][metric])
                for domain in DOMAINS
            ]
            result[layer].append(statistics.mean(domain_values))
    return result


def overall_by_seed(by_layer: dict[str, list[float]]) -> list[float]:
    return [
        statistics.mean(by_layer[layer][seed_index] for layer in LAYERS)
        for seed_index in range(len(SEEDS))
    ]


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
    parser.add_argument("--input-dir", default="results/mechanism/raw")
    parser.add_argument("--original-dir", default="results/causal/raw")
    parser.add_argument("--output-dir", default="results/mechanism")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    standard = load_records(input_dir, "standard640")
    standard_512 = load_records(input_dir, "standard512")
    split = load_records(input_dir, "split50")
    standard_1024 = load_records(Path(args.original_dir), "standard")
    records = {
        "Standard-640": standard,
        "Standard-512": standard_512,
        "Split-50": split,
    }

    summary = {
        "method": {
            "seeds": SEEDS,
            "blocks_per_domain": standard[0]["evaluation"]["blocks_per_domain"],
            "domains": DOMAINS,
            "wrong_expert": standard[0]["evaluation"]["wrong_expert"],
            "layerwise_wrong_expert": standard[0]["evaluation"]["layerwise_wrong_expert"],
            "confidence_intervals": "Two-sided 95% paired t intervals across seeds (df=4)",
        },
        "all_wrong_experts": {},
        "layerwise_wrong_expert": {},
        "expert_similarity": {},
    }
    causal_rows = []
    wrong_penalties: dict[str, dict[str, list[float]]] = {}
    layer_penalties: dict[str, dict[str, list[float]]] = {}
    for model_name, items in records.items():
        normal = [item["ablations"]["normal"]["lm_loss"] for item in items]
        wrong_penalties[model_name] = {}
        for offset in range(1, 4):
            key = f"offset_{offset}"
            values = [
                item["wrong_experts"][key]["lm_loss"] - reference
                for item, reference in zip(items, normal, strict=True)
            ]
            wrong_penalties[model_name][key] = values
            summary["all_wrong_experts"].setdefault(model_name, {})[key] = paired_summary(values)
            for seed, value in zip(SEEDS, values, strict=True):
                causal_rows.append(
                    {"model": model_name, "seed": seed, "intervention": key, "layer": "all", "loss_penalty": value}
                )
        mean_values = [
            item["ablations"]["wrong"]["lm_loss"] - reference
            for item, reference in zip(items, normal, strict=True)
        ]
        wrong_penalties[model_name]["mean"] = mean_values
        summary["all_wrong_experts"][model_name]["mean"] = paired_summary(mean_values)

        layer_penalties[model_name] = {}
        summary["layerwise_wrong_expert"][model_name] = {}
        for layer in LAYERS:
            values = [
                item["layerwise_wrong"][layer]["mean"]["lm_loss"] - reference
                for item, reference in zip(items, normal, strict=True)
            ]
            layer_penalties[model_name][layer] = values
            summary["layerwise_wrong_expert"][model_name][layer] = paired_summary(values)
            for seed, value in zip(SEEDS, values, strict=True):
                causal_rows.append(
                    {"model": model_name, "seed": seed, "intervention": "all_wrong_mean", "layer": layer, "loss_penalty": value}
                )

    split_wrong_minus_absent = [
        item["ablations"]["wrong"]["lm_loss"] - item["ablations"]["shared_only"]["lm_loss"]
        for item in split
    ]
    summary["all_wrong_experts"]["Split-50"]["wrong_minus_no_private"] = paired_summary(
        split_wrong_minus_absent
    )

    similarity_records = {
        "Standard-1024": standard_1024,
        "Standard-640": standard,
        "Standard-512": standard_512,
        "Split private-512": split,
    }
    similarity_values = {}
    for model_name, items in similarity_records.items():
        summary["expert_similarity"][model_name] = {}
        similarity_values[model_name] = {}
        for metric in METRICS:
            by_layer = per_seed_similarity(items, metric)
            overall = overall_by_seed(by_layer)
            similarity_values[model_name][metric] = by_layer
            summary["expert_similarity"][model_name][metric] = {
                "overall": mean_ci(overall),
                "by_layer": {layer: mean_ci(values) for layer, values in by_layer.items()},
            }
    for baseline in ("Standard-512", "Standard-640", "Standard-1024"):
        label = f"Split_private_512_minus_{baseline}"
        summary["expert_similarity"][label] = {}
        for metric in METRICS:
            split_overall = overall_by_seed(similarity_values["Split private-512"][metric])
            standard_overall = overall_by_seed(similarity_values[baseline][metric])
            differences = [
                left - right for left, right in zip(split_overall, standard_overall, strict=True)
            ]
            summary["expert_similarity"][label][metric] = mean_ci(differences)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "causal_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "seed", "intervention", "layer", "loss_penalty"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(causal_rows)

    configure_plots()
    plot_wrong_alternatives(wrong_penalties, output_dir)
    plot_layerwise(layer_penalties, output_dir)
    plot_similarity(similarity_values, output_dir)
    print(json.dumps(summary, indent=2))


def plot_wrong_alternatives(penalties: dict, output_dir: Path) -> None:
    labels = ["Wrong +1", "Wrong +2", "Wrong +3", "Mean"]
    keys = ["offset_1", "offset_2", "offset_3", "mean"]
    x = np.arange(len(keys))
    width = 0.25
    fig, axis = plt.subplots(figsize=(8.3, 4.6))
    for offset, model_name, color in (
        (-1, "Standard-640", COLORS["Standard-640"]),
        (0, "Standard-512", COLORS["Standard-512"]),
        (1, "Split-50", COLORS["Split private-512"]),
    ):
        values = [penalties[model_name][key] for key in keys]
        means = [statistics.mean(item) for item in values]
        cis = [T_CRITICAL_95_DF4 * statistics.stdev(item) / math.sqrt(5) for item in values]
        axis.bar(x + offset * width, means, width, yerr=cis, capsize=4, color=color, label=model_name)
    axis.set_xticks(x, labels)
    axis.set(
        ylabel="Increase in balanced validation LM loss",
        title="Correct routing beats every alternative expert",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "all_wrong_experts.png", bbox_inches="tight")
    plt.close(fig)


def plot_layerwise(penalties: dict, output_dir: Path) -> None:
    x = np.arange(len(LAYERS))
    fig, axis = plt.subplots(figsize=(8.3, 4.6))
    for model_name, color in (
        ("Standard-640", COLORS["Standard-640"]),
        ("Standard-512", COLORS["Standard-512"]),
        ("Split-50", COLORS["Split private-512"]),
    ):
        values = [penalties[model_name][layer] for layer in LAYERS]
        means = [statistics.mean(item) for item in values]
        cis = [T_CRITICAL_95_DF4 * statistics.stdev(item) / math.sqrt(5) for item in values]
        axis.errorbar(x, means, yerr=cis, marker="o", linewidth=2, capsize=4, color=color, label=model_name)
    axis.set_xticks(x, LAYERS)
    axis.set(
        xlabel="Transformer layer corrupted in isolation",
        ylabel="Increase in balanced validation LM loss",
        title="Routing dependence appears at every MoE layer",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "layerwise_wrong_expert.png", bbox_inches="tight")
    plt.close(fig)


def plot_similarity(values: dict, output_dir: Path) -> None:
    titles = {"centered_cosine": "Centered cosine", "linear_cka": "Linear CKA"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    x = np.arange(len(LAYERS))
    for axis, metric in zip(axes, METRICS, strict=True):
        for model_name in (
            "Standard-1024", "Standard-640", "Standard-512", "Split private-512"
        ):
            per_layer = [values[model_name][metric][layer] for layer in LAYERS]
            means = [statistics.mean(item) for item in per_layer]
            cis = [T_CRITICAL_95_DF4 * statistics.stdev(item) / math.sqrt(5) for item in per_layer]
            axis.errorbar(
                x, means, yerr=cis, marker="o", linewidth=2, capsize=4,
                color=COLORS[model_name], label=model_name,
            )
        axis.set_xticks(x, LAYERS)
        axis.set(
            xlabel="Transformer layer",
            ylabel="Mean expert-pair similarity",
            title=titles[metric],
        )
        axis.legend()
    fig.suptitle("Same-width control addresses the expert-width confound", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "width_controlled_similarity.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
