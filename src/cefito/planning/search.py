"""Task-constrained inference-time optimisation (Sec. 3.3, Eq. 4).

Planning is a discrete search: the predicted task c^ restricts the action space
to A(c^), every sequence in A(c^)^T is scored by the energy field, and the
sequence with the lowest energy is the plan.

The search is exhaustive, so it costs |A(c^)|^T predictor forward passes per
sample -- the exponential growth in T discussed in the paper's limitations.
Three things keep that tractable in practice:

* **task grouping** -- samples in a batch that share a task share their
  candidate set, so they are scored together instead of one sample at a time;
* **streaming candidates** -- the candidate sequences of a chunk are decoded
  from their indices on the fly, so the full |A(c)|^T table is never
  materialised;
* **running top-k** -- only the best k plans seen so far are kept, so memory
  does not grow with the number of candidates.
"""

from typing import Callable

import torch


class ActionSpace:
    """A(c) per task, as device tensors, built once and reused across batches."""

    def __init__(self, task_to_actions: dict[int, tuple[int, ...]]):
        self.task_to_actions = task_to_actions
        self._cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def get(self, task_id: int, device: torch.device) -> torch.Tensor:
        key = (task_id, device)
        if key not in self._cache:
            self._cache[key] = torch.tensor(
                self.task_to_actions[task_id], dtype=torch.long, device=device
            )
        return self._cache[key]


def decode_candidates(
    indices: torch.Tensor, actions: torch.Tensor, horizon: int
) -> torch.Tensor:
    """Turn flat candidate indices into action sequences.

    Candidate ``i`` is the base-|A(c)| representation of ``i`` with the first
    action as the most significant digit, i.e. the same enumeration order as
    ``itertools.product(A(c), repeat=T)``.

    Args:
        indices: ``[M]`` candidate indices in ``[0, |A(c)| ** T)``.
        actions: ``[|A(c)|]`` the action ids of this task.
        horizon: planning horizon T.

    Returns:
        ``[M, T]`` action sequences.
    """
    base = actions.numel()
    digits = []
    for _ in range(horizon):
        digits.append(indices % base)
        indices = indices // base
    return actions[torch.stack(digits[::-1], dim=1)]


def task_constrained_search(
    energy_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    goal_state: torch.Tensor,
    task_ids: torch.Tensor,
    task_to_actions: dict[int, tuple[int, ...]],
    horizon: int,
    chunk_size: int = 50_000,
    top_k: int = 1,
    action_space: ActionSpace | None = None,
):
    """Plan by minimising the energy over the task-restricted action space.

    Args:
        energy_fn: scores ``(initial_state, actions, goal_state)`` row-wise and
            returns one energy per row.
        initial_state: ``[B, E]`` normalised initial observations x_s.
        goal_state: ``[B, E]`` normalised goal observations x_g.
        task_ids: ``[B]`` task ids c^ restricting the search.
        horizon: planning horizon T.
        chunk_size: (sample, candidate) pairs scored per ``energy_fn`` call.
        top_k: number of lowest-energy plans to return per sample.
        action_space: reused A(c) cache; created on the fly when omitted.

    Returns:
        ``plans`` ``[B, top_k, T]`` ordered by increasing energy, and
        ``energies`` ``[B, top_k]``.
    """
    action_space = action_space or ActionSpace(task_to_actions)
    device = initial_state.device

    plans = torch.zeros(initial_state.size(0), top_k, horizon, dtype=torch.long, device=device)
    energies = torch.zeros(initial_state.size(0), top_k, device=device)

    for task_id in task_ids.unique().tolist():
        members = (task_ids == task_id).nonzero(as_tuple=True)[0]
        group_size = members.numel()
        actions = action_space.get(task_id, device)
        num_candidates = actions.numel() ** horizon
        k = min(top_k, num_candidates)

        member_initial = initial_state[members]
        member_goal = goal_state[members]
        candidates_per_chunk = max(1, chunk_size // group_size)

        best_energies, best_indices = [], []
        for start in range(0, num_candidates, candidates_per_chunk):
            stop = min(start + candidates_per_chunk, num_candidates)
            indices = torch.arange(start, stop, device=device)
            chunk = decode_candidates(indices, actions, horizon)  # [m, T]

            scores = energy_fn(
                member_initial.repeat_interleave(chunk.size(0), dim=0),
                chunk.repeat(group_size, 1),
                member_goal.repeat_interleave(chunk.size(0), dim=0),
            ).view(group_size, chunk.size(0))

            top_energy, top_index = scores.topk(min(k, chunk.size(0)), dim=1, largest=False)
            best_energies.append(top_energy)
            best_indices.append(indices[top_index])

        # Merge the per-chunk winners into the global top-k.
        merged_energies = torch.cat(best_energies, dim=1)
        merged_indices = torch.cat(best_indices, dim=1)
        top_energy, position = merged_energies.topk(k, dim=1, largest=False)
        winners = merged_indices.gather(1, position)  # [m, k] candidate indices

        plans[members, :k] = decode_candidates(winners.reshape(-1), actions, horizon).view(
            group_size, k, horizon
        )
        energies[members, :k] = top_energy
        if k < top_k:  # fewer candidates than requested: repeat the worst plan
            plans[members, k:] = plans[members, k - 1: k]
            energies[members, k:] = energies[members, k - 1: k]

    return plans, energies
