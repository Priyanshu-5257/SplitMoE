from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .config import ExperimentConfig
from .data import TokenBlockDataset
from .distributed import cleanup, initialize, reduce_mean
from .model import DecoderLM


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def make_loader(dataset, batch_size, workers, distributed, rank, world_size, train):
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=train, seed=1337, drop_last=train
        )
    else:
        sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=train,
    )
    return loader, sampler


def infinite_batches(loader, sampler):
    epoch = 0
    while True:
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def learning_rate(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def unwrap_model(model):
    result = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(result, "_orig_mod", result)


@torch.no_grad()
def evaluate(model, loader, device, cfg, domain_names, distributed):
    model.eval()
    losses: list[torch.Tensor] = []
    lm_losses: list[torch.Tensor] = []
    route_counts = None
    for batch_index, (inputs, labels, domains) in enumerate(loader):
        if batch_index >= cfg.eval_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)
        with autocast_context(device, cfg.precision):
            output = model(inputs, labels, collect_assignments=True)
        losses.append(output.loss.detach().float())
        lm_losses.append(output.lm_loss.detach().float())
        if output.router_stats:
            if route_counts is None:
                route_counts = torch.zeros(
                    len(output.router_stats), len(domain_names), output.router_stats[0].expert_fraction.numel(),
                    device=device, dtype=torch.float64,
                )
            for layer_id, stats in enumerate(output.router_stats):
                assignments = stats.assignments
                token_domains = domains[:, None].expand_as(assignments)
                for domain_id in range(len(domain_names)):
                    selected = assignments[token_domains == domain_id]
                    if selected.numel():
                        route_counts[layer_id, domain_id].add_(
                            torch.bincount(selected, minlength=route_counts.size(-1)).double()
                        )
    if not losses:
        raise RuntimeError("Validation loader yielded no batches")
    metrics = {
        "validation/loss": reduce_mean(torch.stack(losses).mean()).item(),
        "validation/lm_loss": reduce_mean(torch.stack(lm_losses).mean()).item(),
    }
    metrics["validation/perplexity"] = math.exp(min(20.0, metrics["validation/lm_loss"]))
    if route_counts is not None:
        if distributed:
            dist.all_reduce(route_counts, op=dist.ReduceOp.SUM)
        fractions = route_counts / route_counts.sum(-1, keepdim=True).clamp_min(1)
        for layer_id in range(fractions.size(0)):
            for domain_id, domain in enumerate(domain_names):
                for expert_id in range(fractions.size(-1)):
                    metrics[f"routing/layer{layer_id}/{domain}/expert{expert_id}"] = fractions[
                        layer_id, domain_id, expert_id
                    ].item()
    model.train()
    return metrics


def router_metrics(output) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for layer_id, stats in enumerate(output.router_stats):
        entropy = reduce_mean(stats.entropy)
        fractions = reduce_mean(stats.expert_fraction)
        metrics[f"router/layer{layer_id}/entropy"] = entropy.item()
        for expert_id, value in enumerate(fractions):
            metrics[f"router/layer{layer_id}/expert{expert_id}_fraction"] = value.item()
        if stats.shared_norm is not None and stats.private_norm is not None:
            shared_norm = reduce_mean(stats.shared_norm)
            private_norm = reduce_mean(stats.private_norm)
            metrics[f"router/layer{layer_id}/shared_norm"] = shared_norm.item()
            metrics[f"router/layer{layer_id}/private_norm"] = private_norm.item()
            metrics[f"router/layer{layer_id}/shared_private_ratio"] = (
                shared_norm / private_norm.clamp_min(1e-8)
            ).item()
    return metrics


def save_checkpoint(path: Path, model, optimizer, scaler, step, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": config.to_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dense, standard-MoE, or SplitMoE decoder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-steps", type=int, default=None, help="Override max steps for a quick validation")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.smoke_steps is not None:
        config.train.max_steps = args.smoke_steps
        config.train.eval_interval = max(1, args.smoke_steps)
        config.train.save_interval = max(1, args.smoke_steps)
        config.train.wandb_mode = "disabled"

    distributed, rank, local_rank, world_size, device = initialize()
    seed_everything(config.train.seed, rank)
    train_data = TokenBlockDataset(config.train.train_data)
    validation_data = TokenBlockDataset(config.train.validation_data)
    if train_data.block_size != config.model.max_seq_len:
        raise ValueError("Pretokenized block size must equal model.max_seq_len")
    data_vocab = train_data.metadata.get("vocab_size")
    if data_vocab is not None and int(data_vocab) != config.model.vocab_size:
        raise ValueError(f"Data vocab_size={data_vocab}, model vocab_size={config.model.vocab_size}")
    train_loader, train_sampler = make_loader(
        train_data, config.train.micro_batch_size, config.train.num_workers,
        distributed, rank, world_size, True,
    )
    validation_loader, _ = make_loader(
        validation_data, config.train.micro_batch_size, config.train.num_workers,
        distributed, rank, world_size, False,
    )
    batches = infinite_batches(train_loader, train_sampler)

    model = DecoderLM(config.model).to(device)
    summary = model.parameter_summary()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate,
        betas=(config.train.beta1, config.train.beta2), weight_decay=config.train.weight_decay,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.train.precision == "fp16")
    start_step = 0
    if config.train.resume:
        checkpoint = torch.load(config.train.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
    if config.train.compile:
        model = torch.compile(model)
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True
        )

    run = None
    if rank == 0 and config.train.wandb_mode != "disabled":
        import wandb
        run = wandb.init(
            project=config.train.wandb_project,
            name=config.train.wandb_run_name,
            mode=config.train.wandb_mode,
            config=config.to_dict(),
        )
        run.summary.update({f"parameters/{key}": value for key, value in summary.items()})
    if rank == 0:
        effective_batch = config.train.micro_batch_size * config.train.gradient_accumulation_steps * world_size
        print(json.dumps({"device": str(device), "world_size": world_size, "effective_batch": effective_batch, **summary}))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_time = time.perf_counter()
    completed_step = start_step
    try:
        for step in range(start_step, config.train.max_steps):
            lr = learning_rate(step, config.train)
            for group in optimizer.param_groups:
                group["lr"] = lr
            last_output = None
            for micro_step in range(config.train.gradient_accumulation_steps):
                inputs, labels, _ = next(batches)
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                sync = micro_step == config.train.gradient_accumulation_steps - 1
                sync_context = contextlib.nullcontext() if sync or not distributed else model.no_sync()
                with sync_context, autocast_context(device, config.train.precision):
                    output = model(inputs, labels)
                    loss = output.loss / config.train.gradient_accumulation_steps
                scaler.scale(loss).backward()
                last_output = output
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            completed_step = step + 1
            if completed_step % config.train.log_interval == 0:
                elapsed = time.perf_counter() - last_time
                tokens = (
                    config.train.micro_batch_size * config.train.gradient_accumulation_steps
                    * world_size * config.model.max_seq_len * config.train.log_interval
                )
                metrics = {
                    "train/loss": reduce_mean(last_output.loss.detach().float()).item(),
                    "train/lm_loss": reduce_mean(last_output.lm_loss.detach().float()).item(),
                    "train/learning_rate": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/tokens_per_second": tokens / elapsed,
                }
                metrics.update(router_metrics(last_output))
                if rank == 0:
                    print(f"step {completed_step}: loss={metrics['train/loss']:.4f}, tokens/s={metrics['train/tokens_per_second']:.0f}")
                    if run is not None:
                        run.log(metrics, step=completed_step)
                last_time = time.perf_counter()

            if completed_step % config.train.eval_interval == 0:
                metrics = evaluate(
                    model, validation_loader, device, config.train,
                    validation_data.metadata.get("domains", ["unknown"]), distributed,
                )
                if rank == 0:
                    print(f"validation {completed_step}: loss={metrics['validation/loss']:.4f}")
                    if run is not None:
                        run.log(metrics, step=completed_step)

            if rank == 0 and completed_step % config.train.save_interval == 0:
                save_checkpoint(Path(config.train.output_dir) / "latest.pt", model, optimizer, scaler, completed_step, config)
    finally:
        if rank == 0:
            if completed_step >= config.train.max_steps:
                save_checkpoint(
                    Path(config.train.output_dir) / "final.pt",
                    model, optimizer, scaler, completed_step, config,
                )
            if run is not None:
                run.finish()
        cleanup()


if __name__ == "__main__":
    main()
