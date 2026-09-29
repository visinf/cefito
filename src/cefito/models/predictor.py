"""The predictor P_theta (Sec. 3.2).

P_theta maps an initial observation and a candidate action sequence to the goal
latent that sequence would reach. It is a small causal transformer over the
interleaved token sequence

    [ x_s, a_1, s_1, a_2, s_2, ..., a_T, x_g ]

where the ``s_t`` are the (unobserved) intermediate states and ``x_g`` the goal.
Both are supplied as learnable mask tokens and read out of the transformer
output: the goal token gives x~_g, whose distance to the encoded goal defines
the energy of the candidate; the intermediate tokens feed the auxiliary action
reconstruction pass (Supp. A).
"""

import torch
from torch import nn

from .transformer import TransformerBlock, sinusoidal_positional_embedding


class Predictor(nn.Module):
    """Action-conditioned state predictor.

    Args:
        observation_dim: dimension E of the frozen visual features.
        num_actions: size of the action vocabulary (for the auxiliary head).
        action_embeddings: ``[num_actions, embedding_dim]`` frozen CLIP text
            embeddings of the action names; the embedding dimension is taken
            from this tensor. Kept as a non-persistent buffer so it follows the
            module across devices but stays out of checkpoints.
        horizon: planning horizon T.
    """

    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        action_embeddings: torch.Tensor,
        horizon: int,
        hidden_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        causal_attention: bool = True,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_tokens = 2 * horizon + 1

        self.state_embed = nn.Linear(observation_dim, hidden_dim)
        self.action_embed = nn.Linear(action_embeddings.size(1), hidden_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_dim, num_heads, mlp_ratio, causal=causal_attention)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.state_head = nn.Linear(hidden_dim, observation_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)

        self.register_buffer(
            "pos_embed", sinusoidal_positional_embedding(self.num_tokens, hidden_dim)
        )
        self.register_buffer("action_embeddings", action_embeddings, persistent=False)

    # ------------------------------------------------------------------ #

    def embed_actions(self, action_ids: torch.Tensor) -> torch.Tensor:
        """``[B, T]`` action ids -> ``[B, T, hidden_dim]`` action tokens."""
        return self.action_embed(self.action_embeddings[action_ids])

    def _apply_blocks(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def predict_states(
        self,
        initial_state: torch.Tensor,
        action_tokens: torch.Tensor,
        state_mask_token: torch.Tensor,
    ):
        """Roll a candidate action sequence out from the initial state.

        Args:
            initial_state: ``[B, E]`` encoded initial observation x_s.
            action_tokens: ``[B, T, hidden_dim]`` embedded action sequence.
            state_mask_token: ``[hidden_dim]`` learnable placeholder for the
                unobserved intermediate and goal states.

        Returns:
            ``predicted_goal`` ``[B, E]`` -- x~_g, and ``latents`` ``[B, T,
            hidden_dim]`` -- the transformer outputs at the T masked state
            positions (the last one being the goal position).
        """
        batch = initial_state.size(0)
        mask = state_mask_token.expand(batch, 1, -1)

        tokens = [self.state_embed(initial_state).unsqueeze(1)]
        for step in range(self.horizon):
            tokens.append(action_tokens[:, step:step + 1])
            tokens.append(mask)
        tokens = self._apply_blocks(torch.cat(tokens, dim=1))

        latents = tokens[:, 2::2]  # the T masked state positions
        predicted_goal = self.state_head(self.norm(latents[:, -1]))
        return predicted_goal, latents

    def reconstruct_actions(
        self,
        initial_state: torch.Tensor,
        state_latents: torch.Tensor,
        action_mask_token: torch.Tensor,
    ) -> torch.Tensor:
        """Read the action sequence back out of the predicted state latents.

        The second, weight-shared pass of the auxiliary objective (Supp. A):
        states are given, actions are masked, and the model has to name the
        actions that connect them.

        Note that the last of ``state_latents`` is the *predicted* goal latent
        from the first pass, not the encoded ground-truth goal. The supplement
        describes this pass as receiving x_g; the released models were trained
        by feeding the prediction, so that is what is implemented here.

        Args:
            initial_state: ``[B, E]`` encoded initial observation.
            state_latents: ``[B, T, hidden_dim]`` latents from
                :meth:`predict_states`.
            action_mask_token: ``[hidden_dim]`` learnable action placeholder.

        Returns:
            ``[B, T, num_actions]`` action logits.
        """
        batch = initial_state.size(0)
        mask = action_mask_token.expand(batch, 1, -1)

        tokens = [self.state_embed(initial_state).unsqueeze(1)]
        for step in range(self.horizon):
            tokens.append(mask)
            tokens.append(state_latents[:, step:step + 1])
        tokens = self._apply_blocks(torch.cat(tokens, dim=1))

        return self.action_head(tokens[:, 1::2])  # the T masked action positions
