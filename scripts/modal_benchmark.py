"""Run standardized architecture benchmarks on one Modal T4."""

from __future__ import annotations

import json
from pathlib import Path

import modal


REMOTE_ROOT = Path("/workspace")
CONFIGS = {
    "top4_standard": "kaggle_allmoe_16e_top4_standard_seed1337.json",
    "top4_practical_split": "kaggle_allmoe_16e_top4_split25_seed1337.json",
    "top4_equal_active_split": "kaggle_allmoe_16e_top4_equal_active_split_seed1337.json",
    "top4_shared_expert": "kaggle_allmoe_shared1_routed15_top3_seed1337.json",
}
local_root = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "numpy==2.2.6")
    .env({"PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'scripts'}"})
    .add_local_dir(local_root / "src", str(REMOTE_ROOT / "src"))
    .add_local_dir(local_root / "configs", str(REMOTE_ROOT / "configs"))
    .add_local_file(
        local_root / "scripts" / "benchmark_model.py",
        str(REMOTE_ROOT / "scripts" / "benchmark_model.py"),
    )
)
app = modal.App("splitmoe-systems-benchmark")


@app.function(image=image, gpu="T4", cpu=2, memory=8192, timeout=30 * 60, single_use_containers=True)
def run_benchmark(key: str) -> dict:
    from benchmark_model import benchmark

    if key not in CONFIGS:
        raise ValueError(f"Unknown benchmark key: {key}")
    return benchmark(
        REMOTE_ROOT / "configs" / CONFIGS[key],
        batch_size=4,
        warmup=10,
        iterations=50,
        device_name="cuda",
    )


@app.local_entrypoint()
def main(key: str, output: str):
    if key not in CONFIGS:
        raise ValueError(f"--key must be one of {sorted(CONFIGS)}")
    result = run_benchmark.remote(key)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["systems"], indent=2))
