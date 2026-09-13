"""Evaluate checkpoints already stored on a Modal volume against pinned LAMBADA data."""

from __future__ import annotations

import json
from pathlib import Path

import modal


REMOTE_ROOT = Path("/workspace")
VOLUME_ROOT = Path("/vol")
RESULT_ROOT = VOLUME_ROOT / "heldout_results"
local_root = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "numpy==2.2.6")
    .env({"PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'scripts'}"})
    .add_local_dir(local_root / "src", str(REMOTE_ROOT / "src"))
    .add_local_file(
        local_root / "scripts" / "evaluate_heldout.py",
        str(REMOTE_ROOT / "scripts" / "evaluate_heldout.py"),
    )
    .add_local_dir(
        local_root / "data" / "heldout" / "lambada_openai_en",
        str(REMOTE_ROOT / "data" / "heldout" / "lambada_openai_en"),
    )
)

app = modal.App("splitmoe-volume-heldout")
volume = modal.Volume.from_name("splitmoe-paper", create_if_missing=False)


@app.function(
    image=image,
    gpu="T4",
    cpu=2,
    memory=8192,
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 60,
    single_use_containers=True,
)
def evaluate_volume_checkpoint(checkpoint_run: str, result_key: str) -> dict:
    from evaluate_heldout import evaluate

    if Path(checkpoint_run).name != checkpoint_run or Path(result_key).name != result_key:
        raise ValueError("checkpoint_run and result_key must be plain names")
    volume.reload()
    checkpoint = VOLUME_ROOT / "checkpoints" / checkpoint_run / "final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result = evaluate(
        checkpoint,
        REMOTE_ROOT / "data" / "heldout" / "lambada_openai_en",
        batch_size=16,
        device_name="cuda",
        max_examples=None,
    )
    result.update({"result_key": result_key, "modal_checkpoint_run": checkpoint_run})
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESULT_ROOT / f"{result_key}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return {"status": "complete", "result": str(output), "metrics": result["evaluation"]}


@app.function(image=image, cpu=1, memory=1024, volumes={VOLUME_ROOT: volume}, timeout=300)
def read_result(result_key: str) -> dict | None:
    volume.reload()
    path = RESULT_ROOT / f"{result_key}.json"
    return json.loads(path.read_text()) if path.exists() else None


@app.local_entrypoint()
def main(
    action: str = "launch",
    checkpoint_run: str = "",
    result_key: str = "",
    output_dir: str = "results/heldout/raw",
):
    if not result_key:
        raise ValueError("--result-key is required")
    if action in {"launch", "run"}:
        if not checkpoint_run:
            raise ValueError("--checkpoint-run is required for launch/run")
        if action == "launch":
            call = evaluate_volume_checkpoint.spawn(checkpoint_run, result_key)
            print(json.dumps({"status": "launched", "call_id": call.object_id, "result_key": result_key}))
        else:
            print(json.dumps(evaluate_volume_checkpoint.remote(checkpoint_run, result_key), indent=2))
        return
    if action == "download":
        result = read_result.remote(result_key)
        if result is None:
            print(json.dumps({"status": "pending", "result_key": result_key}))
            return
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / f"{result_key}.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": "downloaded", "output": str(output)}))
        return
    raise ValueError("action must be launch, run, or download")
