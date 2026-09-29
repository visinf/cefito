"""Checkpoint saving and loading."""

from pathlib import Path

import torch

from .distributed import unwrap


def save_checkpoint(path, model, ema_model, optimizer, epoch: int, step: int, config) -> None:
    """Write a checkpoint. Only rank 0 should call this."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model": unwrap(model).state_dict(),
            "ema": unwrap(ema_model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
        },
        path,
    )


def load_checkpoint(path, model, ema_model=None, optimizer=None, map_location="cpu"):
    """Restore weights (and optionally the optimiser) from a checkpoint.

    Returns the ``(epoch, step)`` the checkpoint was written at.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    unwrap(model).load_state_dict(checkpoint["model"])
    if ema_model is not None and "ema" in checkpoint:
        unwrap(ema_model).load_state_dict(checkpoint["ema"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def load_weights(path, model, prefer_ema: bool = True, map_location="cpu") -> None:
    """Load just the weights, preferring the EMA copy when the checkpoint has one."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    key = "ema" if prefer_ema and "ema" in checkpoint else "model"
    unwrap(model).load_state_dict(checkpoint[key])
