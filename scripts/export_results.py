"""Export the initial W&B experiment into reproducible tables and plots."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


RUNS = {
    "Dense": "3ewbsmej",
    "Standard MoE": "qi97vvu9",
    "SplitMoE": "pp7z3b7x",
}
COLORS = {
    "Dense": "#6B7280",
    "Standard MoE": "#2563EB",
    "SplitMoE": "#DC2626",
}


def history(run, keys: list[str]) -> list[dict]:
    return [
        row for row in run.scan_history(keys=["_step", *keys], page_size=1000)
        if all(row.get(key) is not None for key in keys)
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
            "grid.alpha": 0.22,
            "legend.frameon": False,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbpkillerx")
    parser.add_argument("--project", default="splitmoe")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    runs = {name: api.run(f"{args.entity}/{args.project}/{run_id}") for name, run_id in RUNS.items()}
    validation: dict[str, list[dict]] = {}
    train_speed: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for name, run in runs.items():
        validation[name] = history(run, ["validation/lm_loss", "validation/perplexity"])
        train_speed[name] = history(run, ["train/tokens_per_second"])
        final = validation[name][-1]
        best = min(validation[name], key=lambda row: row["validation/lm_loss"])
        total = int(run.summary["parameters/total"])
        routed = int(run.summary.get("parameters/routed_experts", 0))
        n_experts = int(run.config["model"]["n_experts"])
        activated = total - routed + (routed // n_experts if routed else 0)
        speeds = [
            row["train/tokens_per_second"] for row in train_speed[name]
            if row["_step"] >= 100
        ]
        summary[name] = {
            "run_id": run.id,
            "url": run.url,
            "total_parameters": total,
            "activated_parameters_per_token": activated,
            "best_step": int(best["_step"]),
            "best_validation_lm_loss": best["validation/lm_loss"],
            "final_validation_lm_loss": final["validation/lm_loss"],
            "final_validation_perplexity": final["validation/perplexity"],
            "median_tokens_per_second": statistics.median(speeds),
            "runtime_seconds": run.summary.get("_runtime"),
        }

    steps = sorted(set.intersection(*[{row["_step"] for row in rows} for rows in validation.values()]))
    by_name = {name: {row["_step"]: row for row in rows} for name, rows in validation.items()}
    with (output_dir / "validation_history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", *[f"{name}_lm_loss" for name in RUNS]])
        for step in steps:
            writer.writerow([step, *[by_name[name][step]["validation/lm_loss"] for name in RUNS]])

    split_router = runs["SplitMoE"]
    ratio_keys = [f"router/layer{layer}/shared_private_ratio" for layer in range(4)]
    ratio_history = history(split_router, ratio_keys)
    with (output_dir / "split_norm_history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", *ratio_keys])
        for row in ratio_history:
            writer.writerow([row["_step"], *[row[key] for key in ratio_keys]])

    paired = {}
    for left, right in (("Dense", "Standard MoE"), ("Dense", "SplitMoE"), ("Standard MoE", "SplitMoE")):
        differences = [by_name[right][step]["validation/lm_loss"] - by_name[left][step]["validation/lm_loss"] for step in steps]
        paired[f"{right}_minus_{left}"] = {
            "mean_lm_loss_delta": statistics.mean(differences),
            "final_lm_loss_delta": differences[-1],
            "right_better_evaluations": sum(value < 0 for value in differences),
            "evaluation_count": len(differences),
        }
    payload = {
        "generated_from": f"https://wandb.ai/{args.entity}/{args.project}",
        "validation_scope": "First 50 DDP validation batches; due source-ordered storage this slice contains stories only.",
        "runs": summary,
        "paired_comparisons": paired,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    configure_plots()
    plot_validation(validation, output_dir)
    plot_efficiency(summary, output_dir)
    plot_norm_ratios(ratio_history, ratio_keys, output_dir)
    print(json.dumps(payload, indent=2))


def plot_validation(validation: dict[str, list[dict]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for name, rows in validation.items():
        x = [row["_step"] for row in rows]
        y = [row["validation/lm_loss"] for row in rows]
        axes[0].plot(x, y, color=COLORS[name], linewidth=2, label=name)
        late = [(step, loss) for step, loss in zip(x, y, strict=True) if step >= 2500]
        axes[1].plot(*zip(*late, strict=True), color=COLORS[name], linewidth=2, label=name)
    axes[0].set(title="Full training", xlabel="Optimizer step", ylabel="Validation LM loss")
    axes[1].set(title="Late-training detail", xlabel="Optimizer step", ylabel="Validation LM loss")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Validation convergence (story slice; lower is better)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_convergence.png", bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(summary: dict[str, dict], output_dir: Path) -> None:
    names = list(RUNS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    total = [summary[name]["total_parameters"] / 1e6 for name in names]
    active = [summary[name]["activated_parameters_per_token"] / 1e6 for name in names]
    positions = list(range(len(names)))
    axes[0].bar(positions, total, color=[COLORS[name] for name in names], alpha=0.35, label="Total")
    axes[0].bar(positions, active, color=[COLORS[name] for name in names], label="Activated/token")
    axes[0].set_xticks(positions, names, rotation=10)
    axes[0].set(ylabel="Parameters (millions)", title="Stored vs activated parameters")
    axes[0].legend()
    speeds = [summary[name]["median_tokens_per_second"] / 1000 for name in names]
    bars = axes[1].bar(positions, speeds, color=[COLORS[name] for name in names])
    axes[1].set_xticks(positions, names, rotation=10)
    axes[1].set(ylabel="Median training throughput (k tokens/s)", title="Observed 2×T4 throughput")
    axes[1].bar_label(bars, fmt="%.1f")
    fig.suptitle("Parameter and systems efficiency", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "efficiency.png", bbox_inches="tight")
    plt.close(fig)


def plot_norm_ratios(rows: list[dict], keys: list[str], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.2))
    for layer, key in enumerate(keys):
        axis.plot([row["_step"] for row in rows], [row[key] for row in rows], linewidth=1.6, label=f"MoE layer {2 * (layer + 1)}")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    axis.set(
        title="SplitMoE shared/private activation norm ratio",
        xlabel="Optimizer step",
        ylabel="mean ||shared|| / mean ||private||",
    )
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "split_norm_ratio.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
