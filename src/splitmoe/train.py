from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from .config import ExperimentConfig
from .data import DomainBalancedSampler, TokenBlockDataset
from .distributed import cleanup, initialize, reduce_mean
from .model import DecoderLM


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def make_loader(
    dataset, batch_size, workers, distributed, rank, world_size, train, *, seed, eval_batches=None
):
    if not train:
        if eval_batches is None:
            raise ValueError("eval_batches is required for the balanced validation sampler")
        sampler = DomainBalancedSampler(
            dataset,
            samples_per_replica=eval_batches * batch_size,
            num_replicas=world_size,
            rank=rank,
        )
    elif distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True
        )
    else:
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
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
    domain_loss_sum = torch.zeros(len(domain_names), device=device, dtype=torch.float64)
    domain_sequence_count = torch.zeros(len(domain_names), device=device, dtype=torch.float64)
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
        per_token_loss = F.cross_entropy(
            output.logits.detach().float().transpose(1, 2), labels, reduction="none"
        )
        per_sequence_loss = per_token_loss.mean(dim=1).double()
        domain_loss_sum.index_add_(0, domains, per_sequence_loss)
        domain_sequence_count.add_(torch.bincount(domains, minlength=len(domain_names)).double())
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
    if distributed:
        dist.all_reduce(domain_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(domain_sequence_count, op=dist.ReduceOp.SUM)
    domain_losses = domain_loss_sum / domain_sequence_count.clamp_min(1)
    for domain_id, domain in enumerate(domain_names):
        value = domain_losses[domain_id].item()
        metrics[f"validation/domain/{domain}/lm_loss"] = value
        metrics[f"validation/domain/{domain}/perplexity"] = math.exp(min(20.0, value))
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


def train_one(config, distributed, rank, local_rank, world_size, device, train_data, validation_data) -> None:
    seed_everything(config.train.seed, rank)
    train_loader, train_sampler = make_loader(
        train_data, config.train.micro_batch_size, config.train.num_workers,
        distributed, rank, world_size, True, seed=config.train.seed,
    )
    validation_loader, _ = make_loader(
        validation_data, config.train.micro_batch_size, config.train.num_workers,
        distributed, rank, world_size, False, seed=config.train.seed,
        eval_batches=config.train.eval_batches,
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
            group=config.train.wandb_run_name.rsplit("-seed-", 1)[0],
            job_type="train",
            mode=config.train.wandb_mode,
            config=config.to_dict(),
        )
        run.summary.update({f"parameters/{key}": value for key, value in summary.items()})
    if rank == 0:
        effective_batch = config.train.micro_batch_size * config.train.gradient_accumulation_steps * world_size
        print(json.dumps({
            "seed": config.train.seed, "device": str(device), "world_size": world_size,
            "effective_batch": effective_batch, **summary,
        }))

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
                # The final checkpoint supersedes the same run's periodic resume file.
                (Path(config.train.output_dir) / "latest.pt").unlink(missing_ok=True)
            if run is not None:
                run.finish()

    del model, optimizer, scaler, train_loader, validation_loader, batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dense, standard-MoE, or SplitMoE decoder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-steps", type=int, default=None, help="Override max steps for a quick validation")
    args = parser.parse_args()
    base_config = ExperimentConfig.from_json(args.config)
    if args.smoke_steps is not None:
        base_config.train.max_steps = args.smoke_steps
        base_config.train.eval_interval = max(1, args.smoke_steps)
        base_config.train.save_interval = max(1, args.smoke_steps)
        base_config.train.wandb_mode = "disabled"
        base_config.train.seeds = [base_config.train.seed]
    seeds = base_config.train.seeds
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("train.seeds must be a non-empty list of unique integers")
    if base_config.train.resume and len(seeds) != 1:
        raise ValueError("Resume supports one seed at a time; set train.seeds to the resumed seed")

    distributed, rank, local_rank, world_size, device = initialize()
    train_data = TokenBlockDataset(base_config.train.train_data)
    validation_data = TokenBlockDataset(base_config.train.validation_data)
    if train_data.block_size != base_config.model.max_seq_len:
        raise ValueError("Pretokenized block size must equal model.max_seq_len")
    data_vocab = train_data.metadata.get("vocab_size")
    if data_vocab is not None and int(data_vocab) != base_config.model.vocab_size:
        raise ValueError(f"Data vocab_size={data_vocab}, model vocab_size={base_config.model.vocab_size}")

    original_output = Path(base_config.train.output_dir)
    original_name = base_config.train.wandb_run_name or base_config.model.moe_type
    try:
        for run_number, seed in enumerate(seeds, start=1):
            config = copy.deepcopy(base_config)
            config.train.seed = int(seed)
            if len(seeds) > 1:
                config.train.output_dir = str(original_output / f"seed-{seed}")
                config.train.wandb_run_name = f"{original_name}-seed-{seed}"
            if rank == 0:
                print(f"Starting seed {seed} ({run_number}/{len(seeds)}): {config.train.wandb_run_name}")
            train_one(
                config, distributed, rank, local_rank, world_size, device,
                train_data, validation_data,
            )
            if distributed:
                dist.barrier()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
