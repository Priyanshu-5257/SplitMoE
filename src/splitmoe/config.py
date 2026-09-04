from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    max_seq_len: int = 256
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    dense_ffn_width: int = 1024
    moe_every: int = 2
    moe_type: str = "split"  # dense, standard, split
    n_experts: int = 4
    standard_expert_width: int = 1024
    shared_width: int = 512
    private_width: int = 512
    dropout: float = 0.0
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_jitter: float = 0.0
    router_weight_mode: str = "straight_through"  # probability, straight_through, none
    split_output_scale: float = 0.7071067811865476
    tie_embeddings: bool = True

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.moe_type not in {"dense", "standard", "split"}:
            raise ValueError(f"Unknown moe_type: {self.moe_type}")
        if self.router_weight_mode not in {"probability", "straight_through", "none"}:
            raise ValueError(f"Unknown router_weight_mode: {self.router_weight_mode}")
        if min(self.n_layers, self.d_model, self.n_heads, self.max_seq_len) <= 0:
            raise ValueError("Model dimensions must be positive")


@dataclass
class TrainConfig:
    train_data: str = "data/train"
    validation_data: str = "data/validation"
    output_dir: str = "checkpoints/split"
    seed: int = 1337
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 10000
    eval_interval: int = 250
    eval_batches: int = 50
    log_interval: int = 10
    save_interval: int = 500
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "fp16"  # fp32, fp16, bf16
    num_workers: int = 2
    compile: bool = False
    resume: str | None = None
    wandb_project: str = "splitmoe"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"  # online, offline, disabled


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        cfg = cls(
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )
        cfg.model.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
