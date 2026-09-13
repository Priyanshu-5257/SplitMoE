"""Export the completed all-MoE scaling experiments from public W&B runs."""

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


SEEDS = [1337, 2027, 3407]
DOMAINS = ["stories", "wiki", "code", "math"]
T_CRITICAL_95_DF2 = 4.302652729
RUNS = {
    "Top-1 Standard": ("splitmoe-paper-allmoe-8e-top1", ["zzxrqkze", "kftluv5e", "octg2jnh"]),
    "Top-1 Split-25": ("splitmoe-paper-allmoe-8e-top1", ["l91x6pv6", "e5rik0fi", "dtztt4d2"]),
    "Top-4 Standard": ("splitmoe-paper-allmoe-16e-top4-kaggle", ["9g6yc85g", "pkgb4b6u", "3qn2rqsi"]),
    "Top-4 Practical Split": ("splitmoe-paper-allmoe-16e-top4-kaggle", ["vwhdjqw9", "jgualbqu", "kme7pr0u"]),
    "Top-4 Equal-active Split": ("splitmoe-paper-allmoe-16e-top4-equal-active", ["9w9acspt", "ndqjysdn", "56ojhfjt"]),
    "Top-4 Shared expert": ("splitmoe-paper-allmoe-shared-expert-baseline", ["k7mayb57", "owlki4d5", "lxdf76bl"]),
}
PARAMETERS = {
    "Top-1 Standard": (134_390_272, 46_309_888),
    "Top-1 Split-25": (112_370_176, 46_309_888),
    "Top-4 Standard": (235_086_336, 84_091_392),
    "Top-4 Practical Split": (187_900_416, 74_654_208),
    "Top-4 Equal-active Split": (225_649_152, 84_091_392),
    "Top-4 Shared expert": (235_082_240, 84_087_296),
}
EXPERTS = {
    "Top-1 Standard": 8,
    "Top-1 Split-25": 8,
    "Top-4 Standard": 16,
    "Top-4 Practical Split": 16,
    "Top-4 Equal-active Split": 16,
    "Top-4 Shared expert": 15,
}
COLORS = {
    "Top-1 Standard": "#2563EB",
    "Top-1 Split-25": "#E11D48",
    "Top-4 Standard": "#2563EB",
    "Top-4 Practical Split": "#F97316",
    "Top-4 Equal-active Split": "#E11D48",
    "Top-4 Shared expert": "#059669",
}


def mean_ci(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL_95_DF2 * sd / math.sqrt(len(values))
    return {"mean": mean, "sample_sd": sd, "ci95": [mean - half, mean + half]}


def paired(records: dict[str, list[dict]], left: str, right: str) -> dict:
    differences = [
        a["validation_lm_loss"] - b["validation_lm_loss"]
        for a, b in zip(records[left], records[right], strict=True)
    ]
    result = mean_ci(differences)
    result.update({"differences": differences, "left_wins": sum(x < 0 for x in differences)})
    return result


def configure_plots() -> None:
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 180, "font.size": 10,
        "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.2, "legend.frameon": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbpkillerx")
    parser.add_argument("--output-dir", default="results/paper_scaling")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    records: dict[str, list[dict]] = {name: [] for name in RUNS}
    histories: dict[tuple[str, int], dict[int, float]] = {}
    router_rows = []

    for name, (project, run_ids) in RUNS.items():
        for seed, run_id in zip(SEEDS, run_ids, strict=True):
            run = api.run(f"{args.entity}/{project}/{run_id}")
            if run.state != "finished" or int(run.summary.get("_step", -1)) != 6500:
                raise RuntimeError(f"Run {run_id} is not a finished step-6500 run")
            total, activated = PARAMETERS[name]
            if int(run.summary["parameters/total"]) != total:
                raise RuntimeError(f"Unexpected parameter count for {run_id}")
            if int(run.summary["parameters/activated_per_token"]) != activated:
                raise RuntimeError(f"Unexpected activated count for {run_id}")
            history = list(run.scan_history(
                keys=["_step", "validation/lm_loss", "train/tokens_per_second"], page_size=1000
            ))
            validation = {
                int(row["_step"]): float(row["validation/lm_loss"])
                for row in history if row.get("validation/lm_loss") is not None
            }
            speed = [
                float(row["train/tokens_per_second"]) for row in history
                if row.get("train/tokens_per_second") is not None and row.get("_step", 0) >= 100
            ]
            if 6500 not in validation or not speed:
                raise RuntimeError(f"Incomplete history for {run_id}")
            histories[(name, seed)] = validation
            summary = run.summary
            row = {
                "regime": "Top-1" if name.startswith("Top-1") else "Top-4",
                "model": name,
                "seed": seed,
                "project": project,
                "run_id": run_id,
                "url": run.url,
                "total_parameters": total,
                "activated_parameters_per_token": activated,
                "validation_lm_loss": float(summary["validation/lm_loss"]),
                "validation_perplexity": float(summary["validation/perplexity"]),
                "median_tokens_per_second": statistics.median(speed),
                "runtime_seconds": float(summary["_runtime"]),
                "peak_reserved_bytes": int(summary["system/peak_vram_reserved_bytes"]),
                **{
                    f"{domain}_lm_loss": float(summary[f"validation/domain/{domain}/lm_loss"])
                    for domain in DOMAINS
                },
            }
            records[name].append(row)
            for layer in range(8):
                fractions = [
                    summary.get(f"router/layer{layer}/expert{expert}_fraction")
                    for expert in range(EXPERTS[name])
                ]
                fractions = [float(value) for value in fractions if value is not None]
                router_rows.append({
                    "model": name, "seed": seed, "layer": layer + 1,
                    "entropy": float(summary[f"router/layer{layer}/entropy"]),
                    "minimum_expert_fraction": min(fractions),
                    "maximum_expert_fraction": max(fractions),
                    "shared_private_ratio": summary.get(f"router/layer{layer}/shared_private_ratio"),
                })

    summary = {"models": {}, "paired_comparisons": {}, "provenance": {}}
    for name, rows in records.items():
        summary["models"][name] = {
            "total_parameters": PARAMETERS[name][0],
            "activated_parameters_per_token": PARAMETERS[name][1],
            "validation_lm_loss": mean_ci([row["validation_lm_loss"] for row in rows]),
            "validation_perplexity": mean_ci([row["validation_perplexity"] for row in rows]),
            "median_tokens_per_second_mean": statistics.mean(row["median_tokens_per_second"] for row in rows),
            "runtime_minutes_mean": statistics.mean(row["runtime_seconds"] for row in rows) / 60,
            "peak_reserved_gib_mean": statistics.mean(row["peak_reserved_bytes"] for row in rows) / 2**30,
            "domains": {
                domain: mean_ci([row[f"{domain}_lm_loss"] for row in rows]) for domain in DOMAINS
            },
        }
    comparisons = [
        ("Top-1 Split-25", "Top-1 Standard"),
        ("Top-4 Practical Split", "Top-4 Standard"),
        ("Top-4 Equal-active Split", "Top-4 Standard"),
        ("Top-4 Shared expert", "Top-4 Standard"),
        ("Top-4 Shared expert", "Top-4 Equal-active Split"),
        ("Top-4 Shared expert", "Top-4 Practical Split"),
        ("Top-4 Equal-active Split", "Top-4 Practical Split"),
    ]
    for left, right in comparisons:
        summary["paired_comparisons"][f"{left}_minus_{right}"] = paired(records, left, right)
    summary["provenance"] = {
        "seeds": SEEDS,
        "steps": 6500,
        "validation": "Fixed domain-balanced validation sample, identical across paired runs.",
        "confidence_intervals": "Two-sided 95% Student t intervals over three paired seeds (df=2).",
        "projects": sorted({f"https://wandb.ai/{args.entity}/{project}" for project, _ in RUNS.values()}),
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    final_rows = [row for name in RUNS for row in records[name]]
    with (output_dir / "final_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(final_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(final_rows)
    with (output_dir / "validation_history.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["regime", "model", "seed", "step", "validation_lm_loss"])
        for (name, seed), values in histories.items():
            for step, loss in sorted(values.items()):
                writer.writerow(["Top-1" if name.startswith("Top-1") else "Top-4", name, seed, step, loss])
    with (output_dir / "router_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(router_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(router_rows)

    configure_plots()
    plot_paired(records, ["Top-1 Standard", "Top-1 Split-25"], output_dir / "top1_paired_loss.png", "8-expert Top-1 paired results")
    top4 = ["Top-4 Standard", "Top-4 Practical Split", "Top-4 Equal-active Split", "Top-4 Shared expert"]
    plot_paired(records, top4, output_dir / "top4_paired_loss.png", "16-expert Top-4 paired results")
    plot_convergence(histories, top4, output_dir)
    plot_frontier(summary, top4, output_dir)
    plot_domains(summary, top4, output_dir)
    plot_systems(summary, top4, output_dir)
    plot_ratios(router_rows, output_dir)
    print(json.dumps(summary, indent=2))


def plot_paired(records, names, path, title):
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    for index, seed in enumerate(SEEDS):
        values = [records[name][index]["validation_lm_loss"] for name in names]
        axis.plot(range(len(names)), values, color="#94A3B8", alpha=0.65, linewidth=1)
        axis.scatter(range(len(names)), values, color=[COLORS[name] for name in names], s=30, zorder=3)
    means = [statistics.mean(row["validation_lm_loss"] for row in records[name]) for name in names]
    cis = [mean_ci([row["validation_lm_loss"] for row in records[name]])["ci95"] for name in names]
    errors = [[mean - ci[0] for mean, ci in zip(means, cis)], [ci[1] - mean for mean, ci in zip(means, cis)]]
    axis.errorbar(range(len(names)), means, yerr=errors, fmt="D", color="black", capsize=5, zorder=4)
    axis.set_xticks(range(len(names)), [name.replace("Top-4 ", "").replace("Top-1 ", "") for name in names])
    axis.set(title=title, ylabel="Final balanced validation LM loss")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(histories, names, output_dir):
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    for name in names:
        steps = sorted(set.intersection(*[set(histories[(name, seed)]) for seed in SEEDS]))
        values = np.asarray([[histories[(name, seed)][step] for step in steps] for seed in SEEDS])
        mean = values.mean(axis=0)
        ci = T_CRITICAL_95_DF2 * values.std(axis=0, ddof=1) / math.sqrt(3)
        axis.plot(steps, mean, color=COLORS[name], linewidth=2, label=name.replace("Top-4 ", ""))
        axis.fill_between(steps, mean - ci, mean + ci, color=COLORS[name], alpha=0.12)
    axis.set(title="16-expert Top-4 validation convergence", xlabel="Optimizer step", ylabel="Balanced validation LM loss")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "top4_convergence.png", bbox_inches="tight")
    plt.close(fig)


def plot_frontier(summary, names, output_dir):
    fig, axis = plt.subplots(figsize=(8.8, 5))
    offsets = [(8, 8), (-110, 8), (-120, -20), (8, -18)]
    for name, offset in zip(names, offsets, strict=True):
        stats = summary["models"][name]
        loss = stats["validation_lm_loss"]
        x = stats["total_parameters"] / 1e6
        axis.errorbar(x, loss["mean"], yerr=[[loss["mean"] - loss["ci95"][0]], [loss["ci95"][1] - loss["mean"]]], fmt="o", color=COLORS[name], capsize=5)
        axis.annotate(name.replace("Top-4 ", ""), (x, loss["mean"]), xytext=offset, textcoords="offset points")
    axis.set(title="Top-4 quality versus stored parameters", xlabel="Total parameters (millions)", ylabel="Final validation LM loss")
    fig.tight_layout()
    fig.savefig(output_dir / "top4_parameter_frontier.png", bbox_inches="tight")
    plt.close(fig)


def plot_domains(summary, names, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    labels = [name.replace("Top-4 ", "") for name in names]
    for axis, domain in zip(axes.flat, DOMAINS, strict=True):
        means = [summary["models"][name]["domains"][domain]["mean"] for name in names]
        intervals = [summary["models"][name]["domains"][domain]["ci95"] for name in names]
        errors = [
            [mean - interval[0] for mean, interval in zip(means, intervals, strict=True)],
            [interval[1] - mean for mean, interval in zip(means, intervals, strict=True)],
        ]
        axis.errorbar(
            range(len(names)), means, yerr=errors, fmt="none", ecolor="#475569",
            capsize=4, linewidth=1.2, zorder=2,
        )
        axis.scatter(range(len(names)), means, color=[COLORS[name] for name in names], s=42, zorder=3)
        axis.set_xticks(range(len(names)), labels, rotation=18)
        axis.set(title=domain.title(), ylabel="LM loss")
    fig.suptitle("Top-4 validation loss by domain", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "top4_domain_loss.png", bbox_inches="tight")
    plt.close(fig)


def plot_systems(summary, names, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = [name.replace("Top-4 ", "") for name in names]
    axes[0].bar(labels, [summary["models"][name]["median_tokens_per_second_mean"] for name in names], color=[COLORS[name] for name in names])
    axes[0].set(title="Training throughput", ylabel="Median tokens/second")
    axes[1].bar(labels, [summary["models"][name]["peak_reserved_gib_mean"] for name in names], color=[COLORS[name] for name in names])
    axes[1].set(title="Peak reserved VRAM", ylabel="GiB per GPU")
    for axis in axes: axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "top4_systems.png", bbox_inches="tight")
    plt.close(fig)


def plot_ratios(router_rows, output_dir):
    names = ["Top-4 Practical Split", "Top-4 Equal-active Split", "Top-4 Shared expert"]
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    for name in names:
        means = [statistics.mean(float(row["shared_private_ratio"]) for row in router_rows if row["model"] == name and row["layer"] == layer) for layer in range(1, 9)]
        axis.plot(range(1, 9), means, marker="o", linewidth=2, color=COLORS[name], label=name.replace("Top-4 ", ""))
    axis.axhline(1, color="#64748B", linestyle="--", linewidth=1)
    axis.set(title="Shared/private activation norm by layer", xlabel="Transformer layer", ylabel="Shared norm / private norm", xticks=range(1, 9))
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "top4_shared_private_ratio.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
