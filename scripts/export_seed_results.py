"""Export the five-seed W&B experiment into reproducible tables and plots."""

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


RUNS = {
    "Dense": ["ot3kdxvy", "qb1x67lt", "fa9fjzkx", "a8xg4vpp", "qskugdtn"],
    "Standard MoE": ["d0wfjnn9", "ebj2hihj", "8ir8s7uv", "57byzli2", "ci3egspk"],
    "SplitMoE": ["95197l1u", "ikla4b81", "88fnjv13", "m2a3t8ny", "ejeh4t10"],
}
SEEDS = [1337, 2027, 3407, 4517, 5651]
DOMAINS = ["stories", "wiki", "code", "math"]
COLORS = {"Dense": "#64748B", "Standard MoE": "#2563EB", "SplitMoE": "#E11D48"}
T_CRITICAL_95_DF4 = 2.776445105


def history(run, keys: list[str]) -> list[dict]:
    return [
        row
        for row in run.scan_history(keys=["_step", *keys], page_size=1000)
        if all(row.get(key) is not None for key in keys)
    ]


def mean_ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    ci = T_CRITICAL_95_DF4 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, ci


def paired_summary(right: list[float], left: list[float]) -> dict:
    differences = [r - l for r, l in zip(right, left, strict=True)]
    mean, ci = mean_ci(differences)
    return {
        "mean_difference": mean,
        "sample_sd": statistics.stdev(differences),
        "ci95": [mean - ci, mean + ci],
        "right_wins": sum(value < 0 for value in differences),
        "seed_count": len(differences),
    }


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
    parser.add_argument("--project", default="splitmoe-seeds")
    parser.add_argument("--output-dir", default="results/five_seed")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    validation_keys = ["validation/lm_loss", "validation/perplexity"] + [
        f"validation/domain/{domain}/lm_loss" for domain in DOMAINS
    ]
    fetched: dict[str, list[dict]] = {name: [] for name in RUNS}
    for name, ids in RUNS.items():
        for seed, run_id in zip(SEEDS, ids, strict=True):
            run = api.run(f"{args.entity}/{args.project}/{run_id}")
            validation = history(run, validation_keys)
            speed = history(run, ["train/tokens_per_second"])
            if not validation:
                raise RuntimeError(f"Run {run_id} has no complete validation records")
            fetched[name].append(
                {
                    "seed": seed,
                    "id": run_id,
                    "url": run.url,
                    "run": run,
                    "validation": validation,
                    "speed": speed,
                }
            )

    final_rows = []
    summary: dict[str, dict] = {}
    final_losses: dict[str, list[float]] = {}
    for name, records in fetched.items():
        losses = []
        perplexities = []
        speeds = []
        runtimes = []
        for record in records:
            final = record["validation"][-1]
            speed_values = [
                row["train/tokens_per_second"]
                for row in record["speed"]
                if row["_step"] >= 100
            ]
            median_speed = statistics.median(speed_values)
            runtime = float(record["run"].summary["_runtime"])
            row = {
                "model": name,
                "seed": record["seed"],
                "run_id": record["id"],
                "validation_lm_loss": final["validation/lm_loss"],
                "validation_perplexity": final["validation/perplexity"],
                "median_tokens_per_second": median_speed,
                "runtime_seconds": runtime,
            }
            for domain in DOMAINS:
                row[f"{domain}_lm_loss"] = final[f"validation/domain/{domain}/lm_loss"]
            final_rows.append(row)
            losses.append(row["validation_lm_loss"])
            perplexities.append(row["validation_perplexity"])
            speeds.append(median_speed)
            runtimes.append(runtime)
        final_losses[name] = losses
        loss_mean, loss_ci = mean_ci(losses)
        ppl_mean, ppl_ci = mean_ci(perplexities)
        summary[name] = {
            "validation_lm_loss": {
                "mean": loss_mean,
                "sample_sd": statistics.stdev(losses),
                "ci95": [loss_mean - loss_ci, loss_mean + loss_ci],
            },
            "validation_perplexity": {
                "mean": ppl_mean,
                "sample_sd": statistics.stdev(perplexities),
                "ci95": [ppl_mean - ppl_ci, ppl_mean + ppl_ci],
            },
            "median_tokens_per_second_mean": statistics.mean(speeds),
            "runtime_minutes_mean": statistics.mean(runtimes) / 60,
            "domains": {},
        }
        for domain in DOMAINS:
            values = [row[f"{domain}_lm_loss"] for row in final_rows if row["model"] == name]
            domain_mean, domain_ci = mean_ci(values)
            summary[name]["domains"][domain] = {
                "mean": domain_mean,
                "sample_sd": statistics.stdev(values),
                "ci95": [domain_mean - domain_ci, domain_mean + domain_ci],
            }

    summary["paired_comparisons"] = {
        "Standard MoE minus Dense": paired_summary(final_losses["Standard MoE"], final_losses["Dense"]),
        "SplitMoE minus Dense": paired_summary(final_losses["SplitMoE"], final_losses["Dense"]),
        "SplitMoE minus Standard MoE": paired_summary(
            final_losses["SplitMoE"], final_losses["Standard MoE"]
        ),
    }
    summary["provenance"] = {
        "project": f"https://wandb.ai/{args.entity}/{args.project}",
        "seeds": SEEDS,
        "validation": "Fixed domain-balanced sample, identical across architectures and seeds.",
        "confidence_intervals": "Two-sided 95% t intervals across five seeds (df=4).",
    }

    fields = list(final_rows[0])
    with (output_dir / "final_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(final_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    steps = sorted(
        set.intersection(
            *[
                {row["_step"] for record in records for row in record["validation"]}
                for records in fetched.values()
            ]
        )
    )
    with (output_dir / "validation_history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "seed", "step", "lm_loss", "perplexity"])
        for name, records in fetched.items():
            for record in records:
                by_step = {row["_step"]: row for row in record["validation"]}
                for step in steps:
                    row = by_step[step]
                    writer.writerow(
                        [name, record["seed"], step, row["validation/lm_loss"], row["validation/perplexity"]]
                    )

    split_ratio_keys = [f"router/layer{layer}/shared_private_ratio" for layer in range(4)]
    ratio_records = []
    for record in fetched["SplitMoE"]:
        rows = history(record["run"], split_ratio_keys)
        ratio_records.append({"seed": record["seed"], "rows": rows})
    with (output_dir / "split_norm_history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "step", *split_ratio_keys])
        for record in ratio_records:
            for row in record["rows"]:
                writer.writerow([record["seed"], row["_step"], *[row[key] for key in split_ratio_keys]])

    configure_plots()
    plot_convergence(fetched, output_dir)
    plot_final_losses(final_losses, output_dir)
    plot_domains(final_rows, output_dir)
    plot_efficiency(summary, output_dir)
    plot_norm_ratios(ratio_records, split_ratio_keys, output_dir)
    print(json.dumps(summary, indent=2))


def plot_convergence(fetched: dict[str, list[dict]], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 4.5))
    for name, records in fetched.items():
        common_steps = sorted(set.intersection(*[{r["_step"] for r in x["validation"]} for x in records]))
        curves = []
        for record in records:
            by_step = {row["_step"]: row["validation/lm_loss"] for row in record["validation"]}
            curves.append([by_step[step] for step in common_steps])
        values = np.asarray(curves)
        mean = values.mean(axis=0)
        ci = T_CRITICAL_95_DF4 * values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        axis.plot(common_steps, mean, color=COLORS[name], linewidth=2, label=name)
        axis.fill_between(common_steps, mean - ci, mean + ci, color=COLORS[name], alpha=0.14)
    axis.set(title="Five-seed validation convergence", xlabel="Optimizer step", ylabel="Balanced validation LM loss")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_convergence.png", bbox_inches="tight")
    plt.close(fig)


def plot_final_losses(final_losses: dict[str, list[float]], output_dir: Path) -> None:
    names = list(RUNS)
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for seed_index, seed in enumerate(SEEDS):
        values = [final_losses[name][seed_index] for name in names]
        axis.plot(range(len(names)), values, color="#94A3B8", alpha=0.6, linewidth=1)
        axis.scatter(range(len(names)), values, color=[COLORS[name] for name in names], s=28, zorder=3)
    means = [statistics.mean(final_losses[name]) for name in names]
    cis = [mean_ci(final_losses[name])[1] for name in names]
    axis.errorbar(range(len(names)), means, yerr=cis, fmt="none", ecolor="black", capsize=5, linewidth=1.8, zorder=4)
    axis.scatter(range(len(names)), means, marker="D", color="black", s=38, label="Mean ± 95% CI", zorder=5)
    axis.set_xticks(range(len(names)), names)
    axis.set(ylabel="Final balanced validation LM loss", title="Paired results across five seeds")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "final_loss_by_seed.png", bbox_inches="tight")
    plt.close(fig)


def plot_domains(final_rows: list[dict], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(DOMAINS))
    width = 0.24
    for model_index, name in enumerate(RUNS):
        means, cis = [], []
        rows = [row for row in final_rows if row["model"] == name]
        for domain in DOMAINS:
            values = [row[f"{domain}_lm_loss"] for row in rows]
            mean, ci = mean_ci(values)
            means.append(mean)
            cis.append(ci)
        axis.bar(x + (model_index - 1) * width, means, width, yerr=cis, capsize=3, color=COLORS[name], label=name)
    axis.set_xticks(x, [domain.title() for domain in DOMAINS])
    axis.set(ylabel="Final validation LM loss", title="Performance by validation domain (mean ± 95% CI)")
    axis.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "domain_loss.png", bbox_inches="tight")
    plt.close(fig)


def plot_efficiency(summary: dict, output_dir: Path) -> None:
    names = list(RUNS)
    total = [46.277120, 65.159680, 55.722496]
    active = [46.277120, 46.285312, 46.285312]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(x, total, color=[COLORS[name] for name in names], alpha=0.35, label="Stored")
    axes[0].bar(x, active, color=[COLORS[name] for name in names], label="Activated/token")
    axes[0].set_xticks(x, names, rotation=8)
    axes[0].set(title="Parameters", ylabel="Millions")
    axes[0].legend()
    speeds = [summary[name]["median_tokens_per_second_mean"] / 1000 for name in names]
    bars = axes[1].bar(x, speeds, color=[COLORS[name] for name in names])
    axes[1].set_xticks(x, names, rotation=8)
    axes[1].set(title="Measured training throughput", ylabel="Mean of per-run medians (k tokens/s)")
    axes[1].bar_label(bars, fmt="%.1f")
    fig.tight_layout()
    fig.savefig(output_dir / "efficiency.png", bbox_inches="tight")
    plt.close(fig)


def plot_norm_ratios(ratio_records: list[dict], keys: list[str], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 4.5))
    for layer, key in enumerate(keys):
        common_steps = sorted(set.intersection(*[{row["_step"] for row in record["rows"]} for record in ratio_records]))
        curves = []
        for record in ratio_records:
            by_step = {row["_step"]: row[key] for row in record["rows"]}
            curves.append([by_step[step] for step in common_steps])
        values = np.asarray(curves)
        mean = values.mean(axis=0)
        ci = T_CRITICAL_95_DF4 * values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        color = plt.colormaps["viridis"](0.1 + 0.25 * layer)
        axis.plot(common_steps, mean, color=color, linewidth=1.8, label=f"Transformer layer {2 * (layer + 1)}")
        axis.fill_between(common_steps, mean - ci, mean + ci, color=color, alpha=0.12)
    axis.axhline(1, color="black", linestyle="--", linewidth=1, alpha=0.65)
    axis.set(
        title="SplitMoE shared/private activation balance",
        xlabel="Optimizer step",
        ylabel="Mean ||shared|| / mean ||private||",
    )
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "split_norm_ratio.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
