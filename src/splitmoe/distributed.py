from __future__ import annotations

import os

import torch
import torch.distributed as dist


def initialize():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return distributed, rank, local_rank, world_size, device


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return value
    result = value.detach().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result / dist.get_world_size()


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
