"""CEFITO: contrastive training of the energy field, and planning on top of it.

Training (Sec. 3.2) shapes P_theta into an energy field: the predicted goal of
the correct action sequence is pulled towards the observed goal latent, the
predicted goals of N wrong sequences are pushed away by an adaptive margin, and
an auxiliary head reconstructs the actions from the predicted intermediate
states.

Inference (Sec. 3.3) turns planning into a search: score every action sequence
the predicted task admits and keep the one whose predicted goal sits closest to
the observed goal.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..data.negatives import adaptive_margins, sample_negative_sequences
from ..planning.search import task_constrained_search
from ..utils.distributed import all_reduce_sum, is_distributed, world_size
from .predictor import Predictor


@dataclass
class Losses:
    total: torch.Tensor
    contrastive: torch.Tensor
    auxiliary: torch.Tensor


class CEFITO(nn.Module):
    """Predictor plus the contrastive objective and the planning search.

    Args:
        task_to_actions: A(c) for every task, used both to sample hard negatives
            during training and to constrain the search at inference.
    """

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        horizon: int,
        action_embeddings: torch.Tensor,
        task_to_actions: dict[int, tuple[int, ...]],
        hidden_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        causal_attention: bool = True,
        num_negatives: int = 50,
        hard_negative_ratio: float = 0.8,
        min_margin: float = 0.01,
        max_margin: float = 0.1,
        aux_weight: float = 1.0,
        permutation_negative_ratio: float = 0.0,
        margin_order_weight: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.num_actions = num_actions
        self.task_to_actions = task_to_actions
        self.num_negatives = num_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.aux_weight = aux_weight
        self.permutation_negative_ratio = permutation_negative_ratio
        self.margin_order_weight = margin_order_weight

        self.predictor = Predictor(
            observation_dim=observation_dim,
            num_actions=num_actions,
            action_embeddings=action_embeddings,
            horizon=horizon,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            causal_attention=causal_attention,
        )
        # Placeholders for the unobserved intermediate/goal states and, in the
        # auxiliary pass, for the actions to be reconstructed.
        self.state_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.action_mask_token = nn.Parameter(torch.zeros(hidden_dim))

    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize(observation: torch.Tensor) -> torch.Tensor:
        """Layer-normalise a visual feature over its channel dimension."""
        return F.layer_norm(observation, (observation.size(-1),))

    def energy(
        self, initial_state: torch.Tensor, actions: torch.Tensor, goal_state: torch.Tensor
    ) -> torch.Tensor:
        """Energy of candidate sequences: the squared distance between the goal
        the sequence predicts and the goal actually observed (Eq. 1).

        Args:
            initial_state: ``[B, E]`` normalised initial observation.
            actions: ``[B, T]`` candidate action ids.
            goal_state: ``[B, E]`` normalised goal observation.

        Returns:
            ``[B]`` energies. Lower is better.
        """
        predicted_goal, _ = self.predictor.predict_states(
            initial_state, self.predictor.embed_actions(actions), self.state_mask_token
        )
        return (predicted_goal - goal_state).pow(2).mean(dim=-1)

    # ------------------------------------------------------------------ #

    def forward(self, *args, **kwargs) -> Losses:
        """Training step. Routed through ``forward`` so DDP can sync gradients."""
        return self.compute_losses(*args, **kwargs)

    def compute_losses(
        self,
        initial_state: torch.Tensor,
        goal_state: torch.Tensor,
        actions: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> Losses:
        """Contrastive + auxiliary loss for one batch (Eq. 2, Eq. 3, Supp. A).

        Args:
            initial_state: ``[B, E]`` raw initial observations.
            goal_state: ``[B, E]`` raw goal observations.
            actions: ``[B, T]`` ground-truth action sequences.
            task_ids: ``[B]`` ground-truth task ids.
        """
        initial_state = self.normalize(initial_state)
        goal_state = self.normalize(goal_state)
        batch, horizon = actions.shape

        negatives = sample_negative_sequences(
            actions,
            task_ids,
            self.task_to_actions,
            self.num_negatives,
            self.hard_negative_ratio,
            permutation_negative_ratio=self.permutation_negative_ratio,
        )  # [B, N, T]
        num_negatives = negatives.size(1)

        # Score the ground truth and all of its negatives in a single pass: they
        # share the token layout, so they only differ in the action tokens.
        candidates = torch.cat([actions.unsqueeze(1), negatives], dim=1)  # [B, 1 + N, T]
        flat_candidates = candidates.reshape(-1, horizon)
        flat_initial = initial_state.repeat_interleave(1 + num_negatives, dim=0)

        predicted_goal, state_latents = self.predictor.predict_states(
            flat_initial,
            self.predictor.embed_actions(flat_candidates),
            self.state_mask_token,
        )
        distances = (
            (predicted_goal - goal_state.repeat_interleave(1 + num_negatives, dim=0))
            .pow(2)
            .mean(dim=-1)
            .view(batch, 1 + num_negatives)
        )

        positive_distance, negative_distances = distances[:, :1], distances[:, 1:]

        margins = adaptive_margins(actions, negatives, self.min_margin, self.max_margin,
                                   margin_order_weight=self.margin_order_weight)
        triplet = F.relu(positive_distance - negative_distances + margins)
        # Average over the violating triplets only, so satisfied negatives do not
        # dilute the gradient signal.
        #
        # The denominator is a global count, not a per-rank one. This is a ratio
        # of two batch-dependent quantities, so averaging each rank's ratio -- as
        # DDP does to gradients -- is not the same estimator as the ratio of the
        # sums, and the two diverge exactly where it matters: near convergence
        # only a handful of triplets are still active, most ranks hold none, and
        # `clamp(min=1)` then makes every empty rank divide by 1 instead of
        # contributing to a shared denominator. The auxiliary term, a plain
        # mean, is unaffected, so a per-rank count would silently reweight the
        # two objectives.
        #
        # Reducing the count across ranks and scaling the local sum by the world
        # size undoes DDP's 1/R gradient averaging, so the gradient assembled
        # from R processes equals the single-process one for any R. The count
        # carries no gradient, so a plain (non-differentiable) all-reduce is
        # correct for it.
        active = triplet > 0
        total, count = triplet.sum(), active.sum()
        if is_distributed():
            count = all_reduce_sum(count)
            total = total * world_size()
        contrastive = total / count.clamp(min=1)

        # Auxiliary action reconstruction, on the ground truth and negatives alike.
        logits = self.predictor.reconstruct_actions(
            flat_initial, state_latents, self.action_mask_token
        )
        per_sequence = F.cross_entropy(
            logits.reshape(-1, self.num_actions), flat_candidates.reshape(-1), reduction="none"
        ).view(batch, 1 + num_negatives, horizon).mean(dim=(0, 2))

        # The ground truth and the negatives are weighted equally: each candidate
        # reconstructs *its own* action ids, so the negative branch asks the model
        # to keep a wrong sequence's predicted goal latent decodable back into
        # that wrong sequence. `state_latents` is not detached, so this flows
        # straight back through the predictor.
        auxiliary = (per_sequence[0] + per_sequence[1:].mean()) / 2.0

        return Losses(
            total=contrastive + self.aux_weight * auxiliary,
            contrastive=contrastive,
            auxiliary=auxiliary,
        )

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def plan(
        self,
        initial_state: torch.Tensor,
        goal_state: torch.Tensor,
        task_ids: torch.Tensor,
        chunk_size: int = 50_000,
        top_k: int = 1,
    ):
        """Plan action sequences by task-constrained energy minimisation (Eq. 4).

        Args:
            initial_state: ``[B, E]`` raw initial observations.
            goal_state: ``[B, E]`` raw goal observations.
            task_ids: ``[B]`` task ids restricting the search to A(c).
            chunk_size: candidate rows scored per predictor call.
            top_k: how many of the lowest-energy plans to return.

        Returns:
            ``plans`` ``[B, top_k, T]`` ranked by increasing energy, and
            ``energies`` ``[B, top_k]``.
        """
        return task_constrained_search(
            energy_fn=self.energy,
            initial_state=self.normalize(initial_state),
            goal_state=self.normalize(goal_state),
            task_ids=task_ids,
            task_to_actions=self.task_to_actions,
            horizon=self.horizon,
            chunk_size=chunk_size,
            top_k=top_k,
        )
