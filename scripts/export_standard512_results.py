"""Export the five-seed Standard-512 mechanism-control metrics from W&B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import wandb


SEEDS = [1337, 2027, 3407, 4517, 5651]
RUN_IDS = ["cexu0thw", "0h6uifsu", "vrvd3n66", "knrjjnyn", "ybao9jl3"]
DOMAINS = ["stories", "wiki", "code", "math"]
T_CRITICAL_95_DF4 = 2.776445105


def mean_ci(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL_95_DF4 * sd / math.sqrt(len(values))
    return {"mean": mean, "sample_sd": sd, "ci95": [mean - half, mean + half]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbpkillerx")
    parser.add_argument("--project", default="splitmoe-mechanism-control")
    parser.add_argument("--baseline-metrics", default="results/frontier/final_metrics.csv")
    parser.add_argument("--output-dir", default="results/mechanism")
    args = parser.parse_args()
    api = wandb.Api()
    rows = []
    for seed, run_id in zip(SEEDS, RUN_IDS, strict=True):
        run = api.run(f"{args.entity}/{args.project}/{run_id}")
        if run.state != "finished":
            raise RuntimeError(f"Run {run_id} is not finished: {run.state}")
        if run.config["model"]["standard_expert_width"] != 512:
            raise RuntimeError(f"Run {run_id} is not Standard-512")
        speed = [
            row["train/tokens_per_second"]
            for row in run.scan_history(
                keys=["_step", "train/tokens_per_second"], page_size=1000
            )
            if row.get("train/tokens_per_second") is not None and row["_step"] >= 100
        ]
        summary = run.summary
        rows.append(
            {
                "seed": seed,
                "run_id": run_id,
                "validation_lm_loss": float(summary["validation/lm_loss"]),
                "validation_perplexity": float(summary["validation/perplexity"]),
                "median_tokens_per_second": statistics.median(speed),
                "runtime_seconds": float(summary["_runtime"]),
                **{
                    f"{domain}_lm_loss": float(
                        summary[f"validation/domain/{domain}/lm_loss"]
                    )
                    for domain in DOMAINS
                },
            }
        )

    baselines: dict[str, list[dict]] = {}
    with Path(args.baseline_metrics).open() as file:
        for row in csv.DictReader(file):
            baselines.setdefault(row["model"], []).append(row)
    for values in baselines.values():
        values.sort(key=lambda row: int(row["seed"]))

    summary = {
        "model": "Standard-512",
        "seeds": SEEDS,
        "run_ids": RUN_IDS,
        "total_parameters": 52_576_768,
        "activated_parameters_per_token": 43_139_584,
        "validation_lm_loss": mean_ci([row["validation_lm_loss"] for row in rows]),
        "validation_perplexity": mean_ci(
            [row["validation_perplexity"] for row in rows]
        ),
        "median_tokens_per_second": mean_ci(
            [row["median_tokens_per_second"] for row in rows]
        ),
        "runtime_minutes": mean_ci([row["runtime_seconds"] / 60 for row in rows]),
        "domains": {
            domain: mean_ci([row[f"{domain}_lm_loss"] for row in rows])
            for domain in DOMAINS
        },
        "paired_comparisons": {},
        "wandb_project": f"https://wandb.ai/{args.entity}/{args.project}",
    }
    for baseline in ("Dense", "Standard-640", "Split-50", "Standard-1024"):
        differences = [
            row["validation_lm_loss"] - float(reference["validation_lm_loss"])
            for row, reference in zip(rows, baselines[baseline], strict=True)
        ]
        comparison = mean_ci(differences)
        comparison["standard512_wins"] = sum(value < 0 for value in differences)
        summary["paired_comparisons"][f"Standard-512_minus_{baseline}"] = comparison

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "standard512_training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (output_dir / "standard512_training_metrics.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
