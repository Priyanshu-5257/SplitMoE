"""End-to-end CPU/CUDA smoke run without downloads or W&B login."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from splitmoe.data import write_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true", help="Run with two local DDP workers")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 (requires CUDA)")
    parser.add_argument("--multi-seed", action="store_true", help="Exercise sequential seed orchestration")
    parser.add_argument("--suite", action="store_true", help="Exercise sequential experiment orchestration")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="splitmoe-smoke-") as temporary:
        root = Path(temporary)
        write_synthetic_dataset(root / "train", vocab_size=128, block_size=16, num_blocks=32)
        write_synthetic_dataset(root / "validation", vocab_size=128, block_size=16, num_blocks=8, seed=1)
        config = {
            "model": {
                "vocab_size": 128, "max_seq_len": 16, "n_layers": 2, "d_model": 32,
                "n_heads": 4, "dense_ffn_width": 64, "moe_every": 1, "moe_type": "split",
                "n_experts": 4, "shared_width": 32, "private_width": 32,
            },
            "train": {
                "train_data": str(root / "train"), "validation_data": str(root / "validation"),
                "output_dir": str(root / "checkpoints"), "micro_batch_size": 2,
                "seeds": [11, 22] if args.multi_seed else [1337],
                "gradient_accumulation_steps": 1, "max_steps": 2, "eval_interval": 2,
                "eval_batches": 1, "log_interval": 1, "save_interval": 2,
                "warmup_steps": 1, "precision": "fp16" if args.fp16 else "fp32", "num_workers": 0,
                "wandb_mode": "disabled",
            },
        }
        output_directories = [root / "checkpoints"]
        config_path = root / "smoke.json"
        if args.suite:
            references = []
            output_directories = []
            for name, shared, private in (("split-25", 16, 48), ("split-75", 48, 16)):
                variant = copy.deepcopy(config)
                variant["model"]["shared_width"] = shared
                variant["model"]["private_width"] = private
                output = root / "checkpoints" / name
                variant["train"]["output_dir"] = str(output)
                variant["train"]["wandb_run_name"] = name
                variant_path = root / f"{name}.json"
                variant_path.write_text(json.dumps(variant))
                references.append(variant_path.name)
                output_directories.append(output)
            config_path.write_text(json.dumps({"experiments": references}))
        else:
            config_path.write_text(json.dumps(config))
        command = [sys.executable]
        if args.ddp:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        command += ["-m", "splitmoe.train", "--config", str(config_path)]
        environment = os.environ.copy()
        if args.ddp:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(command, check=True, env=environment)
        for output in output_directories:
            if args.multi_seed:
                assert (output / "seed-11" / "final.pt").exists()
                assert (output / "seed-22" / "final.pt").exists()
            else:
                assert (output / "final.pt").exists()
    qualifiers = " ".join(label for enabled, label in ((args.ddp, "DDP"), (args.suite, "suite")) if enabled)
    print(f"SplitMoE {qualifiers} end-to-end smoke test passed")


if __name__ == "__main__":
    main()
