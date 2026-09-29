"""Models: the predictor P_theta, the CEFITO wrapper, and the task classifier."""

from ..data.action_space import load_action_embeddings, load_task_to_actions
from .cefito import CEFITO, Losses
from .predictor import Predictor
from .task_classifier import TaskClassifier

__all__ = ["CEFITO", "Losses", "Predictor", "TaskClassifier", "build_model"]


def build_model(config) -> CEFITO:
    """Instantiate CEFITO from an experiment config."""
    return CEFITO(
        observation_dim=config.model.observation_dim,
        num_actions=config.num_actions,
        horizon=config.horizon,
        action_embeddings=load_action_embeddings(
            config.paths.action_embeddings, config.num_actions
        ),
        task_to_actions=load_task_to_actions(config.paths.taxonomy),
        hidden_dim=config.model.hidden_dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        causal_attention=config.model.causal_attention,
        num_negatives=config.loss.num_negatives,
        hard_negative_ratio=config.loss.hard_negative_ratio,
        min_margin=config.loss.min_margin,
        max_margin=config.loss.max_margin,
        aux_weight=config.loss.aux_weight,
        permutation_negative_ratio=config.loss.permutation_negative_ratio,
        margin_order_weight=config.loss.margin_order_weight,
    )
