"""Distributed helpers.

Training and evaluation are launched with ``torchrun``; running the scripts
directly falls back to a single process on one device.
"""

import os
import random

import numpy as np
import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialise the process group if torchrun provided one.

    Returns:
        ``(rank, world_size, device)``.
    """
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, torch.device("cuda", local_rank)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def world_size() -> int:
    """Number of processes in the group (1 when not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def all_reduce_mean(values: torch.Tensor) -> torch.Tensor:
    """Average a tensor across processes (a no-op when running on one)."""
    if not is_distributed():
        return values
    values = values.clone()
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values / dist.get_world_size()


def all_reduce_sum(values: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across processes (a no-op when running on one)."""
    if not is_distributed():
        return values
    values = values.clone()
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def global_std(values: torch.Tensor) -> torch.Tensor:
    """Standard deviation over every element on every rank.

    ``Tensor.std()`` on a sharded batch measures only the local shard, so an
    augmentation scaled by it would be a slightly different size at every world
    size (and different again on each rank). Reducing the moments instead makes
    the scale a property of the global batch, which is what it is on one
    process. A no-op on one process, so single-GPU runs are unchanged.
    """
    if not is_distributed():
        return values.std()
    moments = torch.stack(
        [
            values.sum(),
            values.pow(2).sum(),
            torch.tensor(float(values.numel()), dtype=values.dtype, device=values.device),
        ]
    )
    total, squares, count = all_reduce_sum(moments).unbind()
    variance = (squares - total * total / count) / (count - 1)
    return variance.clamp(min=0).sqrt()


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed python, numpy and torch. Each rank gets its own stream."""
    seed = seed + rank
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The underlying module, whether or not it is wrapped in DDP."""
    return getattr(model, "module", model)
