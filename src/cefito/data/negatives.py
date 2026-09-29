"""Mixed hard/easy negative action sequences (Sec. 3.2).

For every ground-truth sequence we draw ``N`` distinct wrong sequences:

* **hard** negatives share the ground-truth task and differ only in the ordering
  or composition of its actions -- they teach the low-level structure of a plan;
* **easy** negatives are drawn from a different task entirely -- they teach the
  high-level separation between tasks.

``hard_negative_ratio`` (r in the paper) is the probability that a candidate is
drawn from the ground-truth task.
"""

import random

import torch


def sample_negative_sequences(
    actions: torch.Tensor,
    task_ids: torch.Tensor,
    task_to_actions: dict[int, tuple[int, ...]],
    num_negatives: int,
    hard_negative_ratio: float,
    generator: random.Random | None = None,
    permutation_negative_ratio: float = 0.0,
) -> torch.Tensor:
    """Sample negatives for a batch of ground-truth sequences.

    Args:
        actions: ``[B, T]`` ground-truth action ids.
        task_ids: ``[B]`` ground-truth task ids.
        task_to_actions: A(c) for every task.
        num_negatives: N, the number of negatives per ground-truth sequence.
        hard_negative_ratio: r, the probability of drawing from the same task.
        permutation_negative_ratio: probability of drawing a *permutation* of
            the ground truth instead -- same action multiset, wrong order. The
            i.i.d. draws below reach one only by chance (~T!/|A(c)|^T), so this
            is the only way to reliably train the field to penalise ordering.

    Returns:
        ``[B, N, T]`` negative action sequences on the device of ``actions``.
        Within one sample the N sequences are distinct and none equals the
        ground truth.
    """
    rng = generator or random
    horizon = actions.size(1)
    all_tasks = list(task_to_actions)

    # Guard against asking for more distinct negatives than the sampling pool
    # can offer, which would otherwise spin forever.
    if hard_negative_ratio >= 1.0:
        smallest = min(len(task_to_actions[t]) for t in task_ids.tolist())
        if smallest**horizon - 1 < num_negatives:
            raise ValueError(
                f"cannot draw {num_negatives} distinct hard negatives: the smallest "
                f"task in this batch has only {smallest}**{horizon} - 1 alternative "
                f"sequences. Lower loss.num_negatives or loss.hard_negative_ratio."
            )

    # Without permutation negatives the plain loop below is used as is. Every
    # draw comes from the shared `random` stream, so an extra draw -- even on a
    # branch that is never taken -- would change every later negative. Hence a
    # separate branch rather than a guard threaded through one loop body.
    order_aware = permutation_negative_ratio > 0.0

    negatives = []
    for sequence, task in zip(actions.tolist(), task_ids.tolist()):
        target = tuple(sequence)
        same_task_actions = task_to_actions[task]
        other_tasks = [t for t in all_tasks if t != task]
        seen: dict[tuple[int, ...], None] = {}

        if not order_aware:
            while len(seen) < num_negatives:
                if other_tasks and rng.random() >= hard_negative_ratio:
                    pool = task_to_actions[rng.choice(other_tasks)]  # easy negative
                else:
                    pool = same_task_actions  # hard negative
                candidate = tuple(rng.choice(pool) for _ in range(horizon))
                if candidate != target:
                    seen[candidate] = None
        else:
            # A permutation of an all-identical-action target is the target
            # itself, so it can never be drawn as a negative; skip that branch
            # for such sequences rather than spin on a draw that can't succeed.
            can_permute = len(set(target)) >= 2
            # The permutation branch draws from a small, bounded pool (at most
            # T! sequences). If a task is small enough that `num_negatives`
            # exceeds what that pool can offer, and the ratio is high enough to
            # keep landing in it, the loop below would spin forever. `stall`
            # counts draws that fail to add a new candidate; once it's clearly not the ordinary rejection-sampling
            # noise the unbounded general pool already tolerates, permanently
            # fall back to that general pool (proven to terminate -- it is the
            # same code this whole study has run without incident).
            stall, exhausted = 0, False
            STALL_LIMIT = 200
            while len(seen) < num_negatives:
                before = len(seen)
                u = rng.random()
                if not exhausted and permutation_negative_ratio and can_permute and u < permutation_negative_ratio:
                    shuffled = list(target)
                    rng.shuffle(shuffled)
                    candidate = tuple(shuffled)
                elif other_tasks and rng.random() >= hard_negative_ratio:
                    pool = task_to_actions[rng.choice(other_tasks)]
                    candidate = tuple(rng.choice(pool) for _ in range(horizon))
                else:
                    candidate = tuple(rng.choice(same_task_actions) for _ in range(horizon))
                if candidate != target:
                    seen[candidate] = None
                if len(seen) == before:
                    stall += 1
                    if stall > STALL_LIMIT:
                        exhausted = True
                else:
                    stall = 0
        negatives.append(list(seen))

    return torch.tensor(negatives, dtype=torch.long, device=actions.device)


def adaptive_margins(
    positives: torch.Tensor,
    negatives: torch.Tensor,
    min_margin: float,
    max_margin: float,
    margin_order_weight: float = 0.0,
) -> torch.Tensor:
    """Per-negative triplet margin tau_i (Eq. 3).

    The margin grows with the fraction of the negative's actions that do not
    occur in the ground truth, so a re-ordering of the correct actions is asked
    to stay closer to the goal than a sequence of unrelated actions. Order and
    repetition are ignored: the comparison is between the *sets* A+ and A-.

    Set membership is resolved by comparing the T actions of a sequence against
    each other and against the ground truth, which costs O(B N T^2) -- rather
    than one-hot encoding over the whole action vocabulary, which would cost
    O(B N T C) for a C that reaches 778 on COIN.

    Args:
        positives: ``[B, T]`` ground-truth action ids.
        negatives: ``[B, N, T]`` negative action ids.
        margin_order_weight: blends the set-difference severity above with a
            positional (Hamming) one, so a *permutation* of the ground truth --
            same set, `unseen = 0` under set comparison alone -- still earns a
            real margin. At ``0.0`` this is exactly the original set-only
            severity; positional distance is `>=` set distance always, so this
            only ever strengthens a margin, never weakens one.

    Returns:
        ``[B, N]`` margins in ``[min_margin, max_margin]``.
    """
    # Keep only a sequence's first occurrence of each action, so repeats in a
    # negative do not inflate either count.
    same = negatives.unsqueeze(-1) == negatives.unsqueeze(-2)  # [B, N, T, T]
    earlier = torch.ones_like(same).tril(diagonal=-1)
    is_first = ~(same & earlier).any(dim=-1)  # [B, N, T]

    in_positive = (negatives.unsqueeze(-1) == positives[:, None, None, :]).any(dim=-1)

    distinct = is_first.sum(-1).clamp(min=1)
    unseen = (is_first & ~in_positive).sum(-1)
    severity = unseen / distinct

    if margin_order_weight:
        positional = (negatives != positives[:, None, :]).float().mean(dim=-1)
        severity = (1.0 - margin_order_weight) * severity + margin_order_weight * positional

    return min_margin + (max_margin - min_margin) * severity
