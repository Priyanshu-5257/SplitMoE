"""Budget-aware Modal runner for the all-MoE paper experiments.

Examples:
    modal run scripts/modal_train.py --action prepare
    modal run scripts/modal_train.py --action benchmark --gpu T4 --steps 100
    modal run scripts/modal_train.py --action train --config paper_allmoe_split25.json --seed 1337 --gpu L4
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = "splitmoe-paper"
REMOTE_ROOT = Path("/workspace")
VOLUME_ROOT = Path("/vol")
DATA_ROOT = VOLUME_ROOT / "data"
CHECKPOINT_ROOT = VOLUME_ROOT / "checkpoints"
EXPECTED_TRAIN_BLOCKS = 396_537
EXPECTED_VALIDATION_BLOCKS = 3_463
GPU_HOURLY_USD = {
    "T4": 0.5904,
    "L4": 0.7992,
    "A10": 1.1016,
    "L40S": 1.9512,
    "A100-40GB": 2.0988,
    "A100-80GB": 2.4984,
    "H100": 3.9492,
}
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
TRAIN_CPU_CORES = 2
TRAIN_MEMORY_MIB = 4096

local_root = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "numpy==2.2.6",
        "transformers==4.55.2",
        "datasets==4.0.0",
        "wandb==0.21.1",
    )
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src"), "PYTHONUNBUFFERED": "1"})
    .add_local_dir(local_root / "src", str(REMOTE_ROOT / "src"))
    .add_local_dir(local_root / "configs", str(REMOTE_ROOT / "configs"))
    .add_local_file(
        local_root / "data_sources.example.json", str(REMOTE_ROOT / "data_sources.json")
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("splitmoe-paper", create_if_missing=True)
huggingface_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


def _dataset_is_ready() -> bool:
    expected = {
        "train": EXPECTED_TRAIN_BLOCKS,
        "validation": EXPECTED_VALIDATION_BLOCKS,
    }
    for split, expected_blocks in expected.items():
        metadata_path = DATA_ROOT / split / "metadata.json"
        if not metadata_path.exists():
            return False
        metadata = json.loads(metadata_path.read_text())
        if int(metadata.get("num_blocks", -1)) != expected_blocks:
            return False
        if int(metadata.get("block_size", -1)) != 256:
            return False
        for filename in ("tokens.bin", "domains.bin"):
            if not (DATA_ROOT / split / filename).exists():
                return False
    return True


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    volumes={VOLUME_ROOT: volume},
    secrets=[huggingface_secret],
    timeout=6 * 60 * 60,
)
def prepare_data(force: bool = False) -> dict:
    """Create the exact pretokenized corpus once and retain it on the Volume."""
    if _dataset_is_ready() and not force:
        return {"status": "already-ready", "path": str(DATA_ROOT)}
    if DATA_ROOT.exists() and force:
        import shutil

        shutil.rmtree(DATA_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HOME"] = str(VOLUME_ROOT / "huggingface-cache")
    command = [
        "python",
        "-m",
        "splitmoe.prepare_data",
        "--sources",
        str(REMOTE_ROOT / "data_sources.json"),
        "--output-dir",
        str(DATA_ROOT),
        "--tokenizer",
        "HuggingFaceTB/SmolLM2-135M",
        "--block-size",
        "256",
        "--validation-fraction",
        "0.01",
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, env=env, check=True)
    if not _dataset_is_ready():
        raise RuntimeError("Pretokenization completed but the dataset manifest is unexpected")
    volume.commit()
    return {
        "status": "prepared",
        "path": str(DATA_ROOT),
        "train_blocks": EXPECTED_TRAIN_BLOCKS,
        "validation_blocks": EXPECTED_VALIDATION_BLOCKS,
    }


def _remote_config(
    config_name: str,
    seed: int,
    *,
    steps: int | None,
    benchmark: bool,
    micro_batch_size: int,
) -> tuple[Path, dict, Path]:
    if Path(config_name).name != config_name or not config_name.endswith(".json"):
        raise ValueError("config must be a JSON filename from the bundled configs directory")
    source = REMOTE_ROOT / "configs" / config_name
    if not source.exists():
        raise FileNotFoundError(source)
    raw = json.loads(source.read_text())
    if "experiments" in raw:
        raise ValueError("Modal runs one variant and seed per invocation, not suite configs")

    train = raw["train"]
    base_name = train["wandb_run_name"]
    run_name = f"{base_name}-seed-{seed}"
    output = CHECKPOINT_ROOT / run_name
    train.update(
        {
            "train_data": str(DATA_ROOT / "train"),
            "validation_data": str(DATA_ROOT / "validation"),
            "output_dir": str(output),
            "seed": seed,
            "seeds": [seed],
            "wandb_run_name": run_name,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": 64 // micro_batch_size,
        }
    )
    if 64 % micro_batch_size:
        raise ValueError("micro_batch_size must divide the fixed effective batch size of 64")
    if steps is not None:
        train["max_steps"] = int(steps)

    if benchmark:
        output = Path("/tmp") / run_name
        train.update(
            {
                "output_dir": str(output),
                "wandb_mode": "disabled",
                "eval_interval": int(train["max_steps"]) + 1,
                "save_interval": int(train["max_steps"]) + 1,
                "save_model_only_final": True,
            }
        )
    else:
        final_path = output / "final.pt"
        latest_path = output / "latest.pt"
        if final_path.exists():
            return source, raw, output
        train["resume"] = str(latest_path) if latest_path.exists() else None

    generated = Path("/tmp") / f"{run_name}.json"
    generated.write_text(json.dumps(raw, indent=2) + "\n")
    return generated, raw, output


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16_384,
    volumes={VOLUME_ROOT: volume},
    timeout=12 * 60 * 60,
    single_use_containers=True,
)
def train_remote(
    config_name: str,
    seed: int = 1337,
    steps: int | None = None,
    benchmark: bool = False,
    micro_batch_size: int = 4,
    gpu_name: str = "T4",
) -> dict:
    if not _dataset_is_ready():
        raise RuntimeError("Dataset is not prepared; run with --action prepare first")
    config_path, raw, output = _remote_config(
        config_name,
        seed,
        steps=steps,
        benchmark=benchmark,
        micro_batch_size=micro_batch_size,
    )
    final_path = output / "final.pt"
    if not benchmark and final_path.exists():
        return {"status": "already-complete", "output": str(output)}

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "WANDB_DIR": "/tmp/wandb",
            "WANDB_CACHE_DIR": "/tmp/wandb-cache",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command = ["python", "-m", "splitmoe.train", "--config", str(config_path)]
    started = time.perf_counter()
    throughputs: list[float] = []
    pattern = re.compile(r"tokens/s=([0-9]+(?:\.[0-9]+)?)")
    checkpoint_pattern = re.compile(r"^checkpoint ([0-9]+):")
    process = subprocess.Popen(
        command,
        cwd=REMOTE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        match = pattern.search(line)
        if match:
            throughputs.append(float(match.group(1)))
        if not benchmark and checkpoint_pattern.match(line):
            # Persist each periodic latest.pt while the container is still alive.
            # This preserves a usable resume point if the cost cap terminates the job.
            volume.commit()
    return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    if not benchmark:
        volume.commit()

    gpu_hourly_rate = GPU_HOURLY_USD[gpu_name]
    total_hourly_rate = (
        gpu_hourly_rate
        + CPU_USD_PER_CORE_SECOND * TRAIN_CPU_CORES * 3600
        + MEMORY_USD_PER_GIB_SECOND * (TRAIN_MEMORY_MIB / 1024) * 3600
    )
    result = {
        "status": "complete",
        "config": config_name,
        "seed": seed,
        "steps": raw["train"]["max_steps"],
        "gpu": gpu_name,
        "elapsed_seconds": elapsed,
        "estimated_gpu_cost_usd": elapsed * gpu_hourly_rate / 3600,
        "estimated_compute_cost_usd": elapsed * total_hourly_rate / 3600,
        "median_tokens_per_second": statistics.median(throughputs) if throughputs else None,
        "mean_tokens_per_second": statistics.mean(throughputs) if throughputs else None,
        "output": str(output),
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(image=image, cpu=1, memory=1024, volumes={VOLUME_ROOT: volume}, timeout=300)
def volume_status() -> dict:
    result = {"data_ready": _dataset_is_ready(), "runs": {}}
    if CHECKPOINT_ROOT.exists():
        for directory in sorted(path for path in CHECKPOINT_ROOT.iterdir() if path.is_dir()):
            result["runs"][directory.name] = {
                "latest": (directory / "latest.pt").exists(),
                "final": (directory / "final.pt").exists(),
            }
    return result


def _timeout_from_cost(gpu: str, max_cost: float, *, include_train_resources: bool) -> int:
    if gpu not in GPU_HOURLY_USD:
        raise ValueError(f"Unsupported priced GPU {gpu!r}; choose from {sorted(GPU_HOURLY_USD)}")
    if max_cost <= 0:
        raise ValueError("max_cost must be positive")
    hourly_rate = GPU_HOURLY_USD[gpu]
    if include_train_resources:
        hourly_rate += CPU_USD_PER_CORE_SECOND * TRAIN_CPU_CORES * 3600
        hourly_rate += MEMORY_USD_PER_GIB_SECOND * (TRAIN_MEMORY_MIB / 1024) * 3600
    return max(60, min(24 * 60 * 60, math.floor(max_cost / hourly_rate * 3600)))


@app.local_entrypoint()
def main(
    action: str = "status",
    config: str = "paper_allmoe_standard_1024.json",
    seed: int = 1337,
    gpu: str = "T4",
    steps: int = 100,
    micro_batch_size: int = 4,
    max_cost: float = 0.25,
    force: bool = False,
):
    """Prepare data, inspect state, benchmark a GPU, or run one budget-capped training job."""
    if action == "prepare":
        print(json.dumps(prepare_data.remote(force=force), indent=2))
        return
    if action == "status":
        print(json.dumps(volume_status.remote(), indent=2))
        return
    if action not in {"benchmark", "train"}:
        raise ValueError("action must be prepare, status, benchmark, or train")

    timeout = _timeout_from_cost(
        gpu, max_cost, include_train_resources=action == "train"
    )
    requested_steps = steps if action == "benchmark" else None
    print(
        f"Launching {action} on {gpu} with a ${max_cost:.2f} "
        f"{'total-compute' if action == 'train' else 'GPU-time'} cap "
        f"({timeout / 60:.1f} minutes)"
    )
    options = {"gpu": gpu, "timeout": timeout}
    if action == "train":
        options.update(
            {
                "cpu": TRAIN_CPU_CORES,
                "memory": TRAIN_MEMORY_MIB,
                "secrets": [wandb_secret],
            }
        )
    result = train_remote.with_options(**options).remote(
        config_name=config,
        seed=seed,
        steps=requested_steps,
        benchmark=action == "benchmark",
        micro_batch_size=micro_batch_size,
        gpu_name=gpu,
    )
    print(json.dumps(result, indent=2))
