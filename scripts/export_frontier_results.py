"""Export Standard-640 and SplitMoE width-sweep results from W&B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


SEEDS = [1337, 2027, 3407, 4517, 5651]
NEW_RUNS = {
    "Standard-640": ("splitmoe-param-matched", ["8r3e1vy9", "xsgn5dss", "zr0wkues", "bdvg72oh", "8y3vvsny"]),
    "Split-25": ("splitmoe-width-sweep", ["lxak7vdp", "tzs47qq0", "8uj0395o", "79coor5k", "btnk91om"]),
    "Split-75": ("splitmoe-width-sweep", ["vv1xb66z", "rj8sxug7", "s9t3hb20", "1mh80uq8", "s5ko7k8h"]),
}
OLD_NAMES = {"Dense": "Dense", "Standard MoE": "Standard-1024", "SplitMoE": "Split-50"}
DOMAINS = ["stories", "wiki", "code", "math"]
TOTAL_PARAMETERS = {
    "Dense": 46_277_120,
    "Standard-640": 55_722_496,
    "Standard-1024": 65_159_680,
    "Split-25": 60_441_088,
    "Split-50": 55_722_496,
    "Split-75": 51_003_904,
}
ACTIVATED_PARAMETERS = {
    "Dense": 46_277_120,
    "Standard-640": 43_926_016,
    "Standard-1024": 46_285_312,
    "Split-25": 46_285_312,
    "Split-50": 46_285_312,
    "Split-75": 46_285_312,
}
COLORS = {
    "Dense": "#64748B",
    "Standard-640": "#60A5FA",
    "Standard-1024": "#1D4ED8",
    "Split-25": "#BE123C",
    "Split-50": "#E11D48",
    "Split-75": "#FB7185",
}
T_CRITICAL_95_DF4 = 2.776445105


def mean_ci(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
    return {"mean": mean, "sample_sd": sd, "ci95": [mean - half, mean + half]}


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
    parser.add_argument("--entity", default="hbpkillerx")
    parser.add_argument("--old-results", default="results/five_seed/final_metrics.csv")
    parser.add_argument("--output-dir", default="results/frontier")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    records: dict[str, list[dict]] = {name: [] for name in TOTAL_PARAMETERS}
    with Path(args.old_results).open() as file:
        for row in csv.DictReader(file):
            name = OLD_NAMES[row["model"]]
            records[name].append(
                {
                    "model": name,
                    "seed": int(row["seed"]),
                    "run_id": row["run_id"],
                    "project": "splitmoe-seeds",
                    "validation_lm_loss": float(row["validation_lm_loss"]),
                    "validation_perplexity": float(row["validation_perplexity"]),
                    "median_tokens_per_second": float(row["median_tokens_per_second"]),
                    "runtime_seconds": float(row["runtime_seconds"]),
                    **{f"{domain}_lm_loss": float(row[f"{domain}_lm_loss"]) for domain in DOMAINS},
                }
            )
    for name, (project, run_ids) in NEW_RUNS.items():
        for seed, run_id in zip(SEEDS, run_ids, strict=True):
            run = api.run(f"{args.entity}/{project}/{run_id}")
            if run.state != "finished":
                raise RuntimeError(f"Run {run_id} is not finished: {run.state}")
            model_config = run.config["model"]
            if name == "Standard-640" and model_config["standard_expert_width"] != 640:
                raise RuntimeError(f"Unexpected Standard width in {run_id}")
            speed = [
                row["train/tokens_per_second"]
                for row in run.scan_history(keys=["_step", "train/tokens_per_second"], page_size=1000)
                if row.get("train/tokens_per_second") is not None and row["_step"] >= 100
            ]
            summary = run.summary
            records[name].append(
                {
                    "model": name,
                    "seed": seed,
                    "run_id": run_id,
                    "project": project,
                    "validation_lm_loss": float(summary["validation/lm_loss"]),
                    "validation_perplexity": float(summary["validation/perplexity"]),
                    "median_tokens_per_second": statistics.median(speed),
                    "runtime_seconds": float(summary["_runtime"]),
                    **{
                        f"{domain}_lm_loss": float(summary[f"validation/domain/{domain}/lm_loss"])
                        for domain in DOMAINS
                    },
                }
            )
    for values in records.values():
        values.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in values] != SEEDS:
            raise RuntimeError("Every model must contain the same five seeds")

    summary = {"models": {}, "paired_comparisons": {}, "provenance": {}}
    for name, values in records.items():
        summary["models"][name] = {
            "total_parameters": TOTAL_PARAMETERS[name],
            "activated_parameters_per_token": ACTIVATED_PARAMETERS[name],
            "validation_lm_loss": mean_ci([row["validation_lm_loss"] for row in values]),
            "validation_perplexity": mean_ci([row["validation_perplexity"] for row in values]),
            "median_tokens_per_second_mean": statistics.mean(row["median_tokens_per_second"] for row in values),
            "runtime_minutes_mean": statistics.mean(row["runtime_seconds"] for row in values) / 60,
            "domains": {
                domain: mean_ci([row[f"{domain}_lm_loss"] for row in values]) for domain in DOMAINS
            },
        }
    comparisons = [
        ("Split-25", "Standard-1024"),
        ("Split-25", "Split-50"),
        ("Split-75", "Split-50"),
        ("Split-75", "Dense"),
        ("Split-50", "Standard-640"),
        ("Standard-640", "Dense"),
    ]
    for left, right in comparisons:
        differences = [
            a["validation_lm_loss"] - b["validation_lm_loss"]
            for a, b in zip(records[left], records[right], strict=True)
        ]
        result = mean_ci(differences)
        result["left_wins"] = sum(value < 0 for value in differences)
        result["seed_count"] = len(differences)
        summary["paired_comparisons"][f"{left}_minus_{right}"] = result
    summary["provenance"] = {
        "seeds": SEEDS,
        "projects": {
            "original": f"https://wandb.ai/{args.entity}/splitmoe-seeds",
            "parameter_matched": f"https://wandb.ai/{args.entity}/splitmoe-param-matched",
            "width_sweep": f"https://wandb.ai/{args.entity}/splitmoe-width-sweep",
        },
        "confidence_intervals": "Two-sided 95% t intervals across five paired seeds (df=4)",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = list(next(iter(records.values()))[0])
    with (output_dir / "final_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in TOTAL_PARAMETERS:
            writer.writerows(records[name])

    configure_plots()
    plot_frontier(summary, output_dir)
    plot_width_sweep(records, output_dir)
    plot_paired_controls(records, output_dir)
    plot_domains(summary, output_dir)
    print(json.dumps(summary, indent=2))


def plot_frontier(summary: dict, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5))
    offsets = {
        "Dense": (6, 8), "Split-75": (6, -17), "Standard-640": (8, 8),
        "Split-50": (8, -18), "Split-25": (-72, -19), "Standard-1024": (-110, 9),
    }
    for name, stats in summary["models"].items():
        mean = stats["validation_lm_loss"]["mean"]
        ci = stats["validation_lm_loss"]["ci95"]
        marker = "s" if name.startswith("Standard") else ("D" if name == "Dense" else "o")
        axis.errorbar(
            TOTAL_PARAMETERS[name] / 1e6, mean,
            yerr=[[mean - ci[0]], [ci[1] - mean]], fmt=marker,
            markersize=8, capsize=4, color=COLORS[name], zorder=3,
        )
        axis.annotate(name, (TOTAL_PARAMETERS[name] / 1e6, mean), xytext=offsets[name], textcoords="offset points")
    axis.set(
        title="Validation quality versus stored parameters",
        xlabel="Total parameters (millions)", ylabel="Balanced validation LM loss (lower is better)",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "quality_vs_parameters.png", bbox_inches="tight")
    plt.close(fig)


def plot_width_sweep(records: dict[str, list[dict]], output_dir: Path) -> None:
    percentages = [0, 25, 50, 75, 100]
    names = ["Standard-1024", "Split-25", "Split-50", "Split-75", "Dense"]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for seed_index, seed in enumerate(SEEDS):
        values = [records[name][seed_index]["validation_lm_loss"] for name in names]
        axis.plot(percentages, values, color="#94A3B8", alpha=0.55, linewidth=1)
    means = [statistics.mean(row["validation_lm_loss"] for row in records[name]) for name in names]
    cis = [
        T_CRITICAL_95_DF4 * statistics.stdev(row["validation_lm_loss"] for row in records[name]) / math.sqrt(5)
        for name in names
    ]
    axis.errorbar(percentages, means, yerr=cis, color="#E11D48", marker="o", linewidth=2.2, capsize=5)
    for x, y, name in zip(percentages, means, names, strict=True):
        axis.annotate(f"{name}\n{TOTAL_PARAMETERS[name] / 1e6:.2f}M", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    axis.set(
        title="Active-width allocation sweep across five seeds",
        xlabel="Fraction of active FFN width assigned to the shared path (%)",
        ylabel="Balanced validation LM loss",
        xticks=percentages,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "width_sweep.png", bbox_inches="tight")
    plt.close(fig)


def plot_paired_controls(records: dict[str, list[dict]], output_dir: Path) -> None:
    comparisons = [
        ("Standard-640", "Split-50", "Equal stored parameters"),
        ("Standard-1024", "Split-25", "Equal activated parameters"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for axis, (left, right, title) in zip(axes, comparisons, strict=True):
        for seed_index in range(5):
            values = [records[left][seed_index]["validation_lm_loss"], records[right][seed_index]["validation_lm_loss"]]
            axis.plot([0, 1], values, color="#94A3B8", alpha=0.7)
            axis.scatter([0, 1], values, color=[COLORS[left], COLORS[right]], s=28, zorder=3)
        axis.set_xticks([0, 1], [left, right])
        axis.set(title=title, ylabel="Final validation LM loss")
    fig.suptitle("Paired architectural controls", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "paired_controls.png", bbox_inches="tight")
    plt.close(fig)


def plot_domains(summary: dict, output_dir: Path) -> None:
    names = ["Standard-1024", "Split-25", "Split-50", "Split-75"]
    fig, axis = plt.subplots(figsize=(9.5, 4.7))
    x = np.arange(len(DOMAINS))
    width = 0.19
    for model_index, name in enumerate(names):
        stats = summary["models"][name]["domains"]
        means = [stats[domain]["mean"] for domain in DOMAINS]
        cis = [(stats[domain]["ci95"][1] - stats[domain]["ci95"][0]) / 2 for domain in DOMAINS]
        axis.bar(x + (model_index - 1.5) * width, means, width, yerr=cis, capsize=3, color=COLORS[name], label=name)
    axis.set_xticks(x, [domain.title() for domain in DOMAINS])
    axis.set(title="Width-sweep performance by domain", ylabel="Final validation LM loss")
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "domain_loss.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
