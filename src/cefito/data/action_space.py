"""The task-conditioned action space A(c), the action names, and the frozen
action embeddings.

A(c) and the action names both come from the dataset's taxonomy file::

    {"<task id>_<task name>": {"<action id>": "<action name>", ...}, ...}

All three are static properties of a dataset, so they are loaded once and
cached rather than re-read from disk on every call.
"""

import json
import os
from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=None)
def load_taxonomy(path: str) -> dict[int, dict[int, str]]:
    """``{task id: {action id: action name}}``, in the order of the file.

    The order matters: negatives are drawn by indexing into A(c), so it is part
    of what makes a training run reproducible.
    """
    with open(path) as handle:
        raw = json.load(handle)
    taxonomy = {
        int(task.split("_", 1)[0]): {int(action): name for action, name in actions.items()}
        for task, actions in raw.items()
    }
    empty = [task for task, actions in taxonomy.items() if not actions]
    if empty:
        raise ValueError(f"tasks {empty} have an empty action set in {path}")
    return taxonomy


def load_task_to_actions(path: str) -> dict[int, tuple[int, ...]]:
    """Map every task id to the action ids that make up A(c) for that task."""
    return {task: tuple(actions) for task, actions in load_taxonomy(path).items()}


def load_action_names(path: str) -> dict[int, str]:
    """``{action id: action name}`` over every task."""
    return {
        action: name
        for actions in load_taxonomy(path).values()
        for action, name in actions.items()
    }


@lru_cache(maxsize=None)
def load_action_embeddings(path: str, num_actions: int) -> torch.Tensor:
    """``[num_actions, embedding_dim]`` frozen CLIP text embeddings, indexed by
    action id. Produced by ``scripts/extract_action_embeddings.py``."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist. Generate the action embeddings with "
            f"`python scripts/extract_action_embeddings.py` (see the README)."
        )
    table = np.load(path, allow_pickle=True).item()
    missing = [i for i in range(num_actions) if i not in table]
    if missing:
        raise ValueError(
            f"{path} is missing embeddings for action ids {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    return torch.as_tensor(np.stack([table[i] for i in range(num_actions)]), dtype=torch.float32)
