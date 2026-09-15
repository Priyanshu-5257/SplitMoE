"""Export and summarize the preregistered output-scale reviewer controls."""

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
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


SEEDS = [1337, 2027, 3407]
DOMAINS = ["stories", "wiki", "code", "math"]
T_CRITICAL_95_DF2 = 4.302652729
RUNS = {
    "Standard-1024 scale-1": (
        "offline",
        ["a1", "a2", "a3"],
        1.0,
        134_390_272,
        46_309_888,
    ),
    "Standard-1024 scale-0.7071": (
        "splitmoe-paper-review-scale",
        ["lstu6dji", "10mrog7t", "gbr2y0v4"],
        2**-0.5,
        134_390_272,
        46_309_888,
    ),
    "Split-25 scale-1": (
        "splitmoe-paper-review-scale",
        ["jxqwolwq", "hk4e6oad", "cof2ale0"],
        1.0,
        112_370_176,
        46_309_888,
    ),
    "Split-25 scale-0.7071": (
        "offline",
        ["a4", "a5", "a6"],
        2**-0.5,
        112_370_176,
        46_309_888,
    ),
    "Standard-800 scale-1": (
        "splitmoe-paper-review-storage",
        ["vme4rov2", "nvar9f18", "qj02qhlh"],
        1.0,
        112_370_176,
        43_557_376,
    ),
}
KAGGLE_URLS = {
    "a1": "https://www.kaggle.com/code/zenitsu1agatsuma/review-std1-b128-s1337",
    "a2": "https://www.kaggle.com/code/hbpkillerx/review-std1-b128-s2027",
    "a3": "https://www.kaggle.com/code/aivenger1st/review-std1-b128-s3407",
    "a4": "https://www.kaggle.com/code/priyanshu3rdvivek/review-split07071-b128-s1337",
    "a5": "https://www.kaggle.com/code/copy1cat/review-split07071-b128-s2027",
    "a6": "https://www.kaggle.com/code/prinshu5257/review-split07071-b128-s3407",
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
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "legend.frameon": False,
    })


def item_key(item) -> str:
    return item.key or "/".join(item.nested_key)


def item_value(item):
    return json.loads(item.value_json)


def read_offline_run(account_dir: Path) -> tuple[str, list[dict], dict]:
    paths = list(account_dir.rglob("run-*.wandb"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one offline W&B file under {account_dir}, found {len(paths)}")
    path = paths[0]
    datastore = DataStore()
    datastore.open_for_scan(str(path))
    history: list[dict] = []
    summary: dict = {}
    while (data := datastore.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        record_type = record.WhichOneof("record_type")
        if record_type == "history":
            history.append({item_key(item): item_value(item) for item in record.history.item})
        elif record_type == "summary":
            summary.update({item_key(item): item_value(item) for item in record.summary.update})
            for item in record.summary.remove:
                summary.pop(item_key(item), None)
    return path.stem.removeprefix("run-"), history, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbpkillerx")
    parser.add_argument("--output-dir", default="results/reviewer_controls")
    parser.add_argument("--offline-root", default="/tmp/splitmoe-review-complete")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    records: dict[str, list[dict]] = {name: [] for name in RUNS}
    trajectories: list[dict] = []

    for name, (project, run_ids, scale, total, activated) in RUNS.items():
        for seed, source_id in zip(SEEDS, run_ids, strict=True):
            if project == "offline":
                run_id, history, summary = read_offline_run(Path(args.offline_root) / source_id)
                run_url = KAGGLE_URLS[source_id]
                source = "Kaggle offline W&B artifact"
            else:
                run = api.run(f"{args.entity}/{project}/{source_id}")
                if run.state != "finished":
                    raise RuntimeError(f"Run {source_id} is not finished")
                run_id = source_id
                history = list(run.scan_history(page_size=1000))
                summary = run.summary
                run_url = run.url
                source = "W&B API"
            if int(summary.get("_step", -1)) != 6500:
                raise RuntimeError(f"Run {run_id} is not a finished step-6500 run")
            if int(summary["parameters/total"]) != total:
                raise RuntimeError(f"Unexpected total parameter count for {run_id}")
            if int(summary["parameters/activated_per_token"]) != activated:
                raise RuntimeError(f"Unexpected activated parameter count for {run_id}")
            validation = {
                int(row["_step"]): float(row["validation/lm_loss"])
                for row in history if row.get("validation/lm_loss") is not None
            }
            speed = [
                float(row["train/tokens_per_second"])
                for row in history
                if row.get("train/tokens_per_second") is not None and row.get("_step", 0) >= 100
            ]
            if 6500 not in validation or not speed:
                raise RuntimeError(f"Incomplete history for {run_id}")
            for step, loss in validation.items():
                trajectories.append({"model": name, "seed": seed, "step": step, "lm_loss": loss})
            records[name].append({
                "model": name,
                "seed": seed,
                "project": project if project != "offline" else "splitmoe-paper-review-scale",
                "run_id": run_id,
                "url": run_url,
                "source": source,
                "output_scale": scale,
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
            })

    comparisons = [
        ("Split-25 scale-1", "Standard-1024 scale-1"),
        ("Split-25 scale-0.7071", "Standard-1024 scale-0.7071"),
        ("Split-25 scale-1", "Split-25 scale-0.7071"),
        ("Standard-1024 scale-1", "Standard-1024 scale-0.7071"),
        ("Split-25 scale-1", "Standard-800 scale-1"),
        ("Split-25 scale-1", "Standard-1024 scale-0.7071"),
    ]
    result = {
        "models": {},
        "paired_comparisons": {},
        "provenance": {
            "seeds": SEEDS,
            "steps": 6500,
            "world_size": 2,
            "effective_batch_sequences": 128,
            "training_token_positions": 212_992_000,
            "primary_outcome": "Final domain-balanced validation LM loss.",
            "confidence_intervals": "Two-sided 95% Student t intervals over three paired seeds (df=2).",
        },
    }
    for name, rows in records.items():
        result["models"][name] = {
            "output_scale": RUNS[name][2],
            "total_parameters": RUNS[name][3],
            "activated_parameters_per_token": RUNS[name][4],
            "validation_lm_loss": mean_ci([row["validation_lm_loss"] for row in rows]),
            "validation_perplexity": mean_ci([row["validation_perplexity"] for row in rows]),
            "median_tokens_per_second_mean": statistics.mean(row["median_tokens_per_second"] for row in rows),
            "runtime_minutes_mean": statistics.mean(row["runtime_seconds"] for row in rows) / 60,
            "peak_reserved_gib_mean": statistics.mean(row["peak_reserved_bytes"] for row in rows) / 2**30,
            "domains": {
                domain: mean_ci([row[f"{domain}_lm_loss"] for row in rows]) for domain in DOMAINS
            },
        }
    for left, right in comparisons:
        result["paired_comparisons"][f"{left}_minus_{right}"] = paired(records, left, right)

    interaction = []
    for seed_index in range(len(SEEDS)):
        split_1 = records["Split-25 scale-1"][seed_index]["validation_lm_loss"]
        standard_1 = records["Standard-1024 scale-1"][seed_index]["validation_lm_loss"]
        split_scaled = records["Split-25 scale-0.7071"][seed_index]["validation_lm_loss"]
        standard_scaled = records["Standard-1024 scale-0.7071"][seed_index]["validation_lm_loss"]
        interaction.append((split_1 - standard_1) - (split_scaled - standard_scaled))
    result["architecture_by_scale_interaction"] = {
        **mean_ci(interaction),
        "differences": interaction,
        "definition": "(Split-Standard at scale 1) - (Split-Standard at scale 1/sqrt(2)).",
    }

    rows = [row for name in RUNS for row in records[name]]
    with (output_dir / "final_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "validation_trajectories.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "seed", "step", "lm_loss"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(trajectories, key=lambda row: (row["model"], row["seed"], row["step"])))
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    configure_plots()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.array([2**-0.5, 1.0])
    for architecture, names, color in (
        ("Standard-1024", ["Standard-1024 scale-0.7071", "Standard-1024 scale-1"], "#2563EB"),
        ("Split-25", ["Split-25 scale-0.7071", "Split-25 scale-1"], "#E11D48"),
    ):
        means = np.array([result["models"][name]["validation_lm_loss"]["mean"] for name in names])
        cis = [result["models"][name]["validation_lm_loss"]["ci95"] for name in names]
        errors = np.array([[mean - ci[0], ci[1] - mean] for mean, ci in zip(means, cis)]).T
        ax.errorbar(x, means, yerr=errors, marker="o", capsize=4, linewidth=2.2, color=color, label=architecture)
    ax.set_xticks(x, [r"$1/\sqrt{2}$", "1"])
    ax.set_xlabel("FFN output scale")
    ax.set_ylabel("Validation LM loss (lower is better)")
    ax.set_title("Batch-matched output-scale factorial")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "output_scale_factorial.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    labels = ["Standard-800\nscale 1", "Split-25\nscale 1"]
    names = ["Standard-800 scale-1", "Split-25 scale-1"]
    for seed_index, seed in enumerate(SEEDS):
        values = [records[name][seed_index]["validation_lm_loss"] for name in names]
        ax.plot([0, 1], values, marker="o", alpha=0.65, color="#64748B", label="paired seeds" if seed_index == 0 else None)
    means = [result["models"][name]["validation_lm_loss"]["mean"] for name in names]
    ax.scatter([0, 1], means, s=90, color=["#2563EB", "#E11D48"], zorder=3, label="mean")
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Validation LM loss (lower is better)")
    ax.set_title("Equal storage: 112.37M total parameters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "equal_storage_paired.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
