"""Procedure planning metrics.

All three are reported in percent, higher is better.
"""

import torch

EPS = 1e-6


def success_rate(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fraction of plans that match the ground-truth sequence exactly.

    Args:
        predictions: ``[B, T]`` predicted action ids.
        targets: ``[B, T]`` ground-truth action ids.
    """
    return predictions.eq(targets).all(dim=1).float().mean() * 100.0


def mean_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fraction of individual action steps predicted correctly."""
    return predictions.eq(targets).float().mean() * 100.0


def mean_iou(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Element-wise mean intersection over union, averaged over the batch.

    This is the element-wise mIoU of the reference implementation: a bitwise
    ``and`` / ``or`` taken position by position over the integer action ids
    themselves, then reduced per sample. It is kept as is so that numbers stay
    comparable with prior work; it is not a set-overlap IoU.
    """
    intersection = torch.bitwise_and(predictions, targets).sum(dim=1)
    union = torch.bitwise_or(predictions, targets).sum(dim=1)
    return ((intersection + EPS) / (union + EPS) * 100.0).mean()


class AverageMeter:
    """Running weighted average of a scalar."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += float(value) * weight
        self.count += weight

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0
