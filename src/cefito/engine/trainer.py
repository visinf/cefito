"""Contrastive training loop for the predictor."""

import copy

import torch

from ..utils.distributed import all_reduce_mean, global_std, unwrap
from ..utils.metrics import AverageMeter

#: Loss components tracked per epoch, in the order they are reduced across
#: ranks. The two auxiliary halves are tracked separately because their ratio
#: to `contrastive` is what says how much of the gradient budget is actually
#: reaching the energy field.
LOSS_NAMES = (
    "total",
    "contrastive",
    "auxiliary",
)


def _update_meters(meters, losses, batch: int) -> None:
    for name in LOSS_NAMES:
        meters[name].update(getattr(losses, name).item(), batch)


class EMA:
    """Exponential moving average of the model weights.

    The EMA copy is the one used for planning: it is what gets evaluated and
    what the released checkpoints contain.
    """

    def __init__(self, model, decay: float, start_step: int, update_every: int):
        self.model = copy.deepcopy(unwrap(model)).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.start_step = start_step
        self.update_every = update_every

    @torch.no_grad()
    def update(self, model, step: int) -> None:
        if step % self.update_every:
            return
        source = unwrap(model)
        if step < self.start_step:
            self.model.load_state_dict(source.state_dict())
            return
        for ema_param, param in zip(self.model.parameters(), source.parameters()):
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.model.buffers(), source.buffers()):
            ema_buffer.copy_(buffer)


def train_one_epoch(
    model, loader, optimizer, ema, step: int, device, logger=None, log_every=10,
    observation_noise: float = 0.0,
):
    """One pass over the training set.

    Returns the ``(average losses, next step)``.
    """
    model.train()
    meters = {name: AverageMeter() for name in LOSS_NAMES}

    for index, (states, actions, task, _) in enumerate(loader):
        states = states.to(device, non_blocking=True).float()
        actions = actions.to(device, non_blocking=True)
        task = task.to(device, non_blocking=True)

        initial, goal = states[:, 0], states[:, -1]
        if observation_noise:
            # Augment only the two OBSERVED states -- the intermediate ones are
            # never read. Scaled by each tensor's own std so the setting is
            # scale-free across datasets and feature extractors. Guarded so that
            # at 0.0 no RNG is consumed.
            #
            # The std is reduced across ranks: a local `.std()` would measure
            # only this rank's shard (16 rows at world size 8 against 128 on one
            # process), making the augmentation a slightly different size at
            # every world size and on every rank.
            initial = initial + observation_noise * global_std(initial) * torch.randn_like(initial)
            goal = goal + observation_noise * global_std(goal) * torch.randn_like(goal)
        losses = model(initial, goal, actions, task)

        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        optimizer.step()

        step += 1
        ema.update(model, step)

        batch = actions.size(0)
        _update_meters(meters, losses, batch)

        if logger and (index + 1) % log_every == 0:
            logger.info(
                f"  [{index + 1}/{len(loader)}] loss {meters['total'].average:.4f} "
                f"(contrastive {meters['contrastive'].average:.4f}, "
                f"aux {meters['auxiliary'].average:.4f})"
            )

    averages = all_reduce_mean(
        torch.tensor([meters[name].average for name in meters], device=device)
    )
    return dict(zip(meters, averages.tolist())), step


@torch.no_grad()
def validation_loss(model, loader, device):
    """Average loss on the validation split (cheap; no planning)."""
    model.eval()
    meters = {name: AverageMeter() for name in LOSS_NAMES}

    for states, actions, _, gt_task in loader:
        states = states.to(device, non_blocking=True).float()
        actions = actions.to(device, non_blocking=True)
        gt_task = gt_task.to(device, non_blocking=True)

        losses = unwrap(model).compute_losses(states[:, 0], states[:, -1], actions, gt_task)
        _update_meters(meters, losses, actions.size(0))

    averages = all_reduce_mean(
        torch.tensor([meters[name].average for name in meters], device=device)
    )
    return dict(zip(meters, averages.tolist()))
