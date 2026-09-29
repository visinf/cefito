"""Planning evaluation."""

import torch

from ..planning.search import ActionSpace
from ..utils.distributed import all_reduce_sum, unwrap
from ..utils.metrics import mean_accuracy, mean_iou, success_rate


@torch.no_grad()
def evaluate(model, loader, chunk_size: int = 50_000, progress_every: int = 0, logger=None):
    """Plan for every sample in ``loader`` and report SR, mAcc and mIoU.

    Metrics are accumulated as (sum, count) pairs and reduced across processes,
    so sharding the loader over ranks gives the same answer as a single process.

    Args:
        model: a :class:`~cefito.models.cefito.CEFITO` (optionally DDP-wrapped).
        loader: yields ``(states, actions, task, gt_task)``.
        chunk_size: candidate rows scored per predictor call.

    Returns:
        ``{"success_rate": ..., "mean_accuracy": ..., "mean_iou": ...,
        "task_accuracy": ...}`` in percent. ``task_accuracy`` reports how often
        the task used for planning matched the ground-truth task.
    """
    module = unwrap(model)
    module.eval()
    device = next(module.parameters()).device
    # A(c) is the same for every batch, so build the device tensors once.
    action_space = ActionSpace(module.task_to_actions)

    totals = torch.zeros(4, device=device)
    count = torch.zeros(1, device=device)

    for index, (states, actions, task, gt_task) in enumerate(loader):
        states = states.to(device, non_blocking=True).float()
        actions = actions.to(device, non_blocking=True)
        task = task.to(device, non_blocking=True)
        gt_task = gt_task.to(device, non_blocking=True)

        plans, _ = module.plan(
            states[:, 0],
            states[:, -1],
            task,
            chunk_size=chunk_size,
            top_k=1,
        )
        predictions = plans[:, 0]

        batch = actions.size(0)
        totals += batch * torch.stack(
            [
                success_rate(predictions, actions),
                mean_accuracy(predictions, actions),
                mean_iou(predictions, actions),
                task.eq(gt_task).float().mean() * 100.0,
            ]
        )
        count += batch

        if progress_every and logger and (index + 1) % progress_every == 0:
            running = (totals / count).tolist()
            logger.info(
                f"  [{index + 1}/{len(loader)}] SR {running[0]:.2f} "
                f"mAcc {running[1]:.2f} mIoU {running[2]:.2f}"
            )

    totals = all_reduce_sum(totals)
    count = all_reduce_sum(count)
    averages = (totals / count).tolist()
    return {
        "success_rate": averages[0],
        "mean_accuracy": averages[1],
        "mean_iou": averages[2],
        "task_accuracy": averages[3],
    }
