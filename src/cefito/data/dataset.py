"""Procedure planning dataset.

One sample is a video segment covering ``horizon`` consecutive annotated action
steps. It yields

* ``states``   -- ``[horizon + 1, observation_dim]`` S3D features of the states
  between the actions; ``states[0]`` is the initial observation x_s and
  ``states[-1]`` the goal observation x_g,
* ``actions``  -- ``[horizon]`` ground-truth action ids a_1:T,
* ``task``     -- the high-level task id used to constrain the search at
  inference (see ``task_source``),
* ``gt_task``  -- the ground-truth task id, always. Used for training and for
  scoring the task classifier.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import REPO_ROOT

#: Number of consecutive S3D features averaged into one state observation.
FRAMES_PER_STATE = 3


class ProcedurePlanningDataset(Dataset):
    """CrossTask / COIN / NIV procedure planning splits.

    Args:
        annotation_file: json listing the videos of this split.
        horizon: planning horizon T.
        task_source: which task label to expose as ``task``. ``"ground_truth"``
            for training and for the oracle experiment; ``"predicted"`` reads the
            task-classifier prediction (``event_class_top1``) stored in the
            annotation file, which is what the main tables use.
        feature_root: rebases the feature paths stored in the annotation file
            onto this directory. Leave as ``None`` to use them as they are;
            relative ones are then resolved against the repository root.
        max_videos: keep only the first N videos. Smoke tests only.
    """

    def __init__(
        self,
        annotation_file: str,
        horizon: int = 3,
        task_source: str = "ground_truth",
        feature_root: str | None = None,
        max_videos: int | None = None,
    ):
        if task_source not in ("ground_truth", "predicted"):
            raise ValueError(f"unknown task_source '{task_source}'")
        self.annotation_file = annotation_file
        self.horizon = horizon
        self.task_source = task_source
        self.feature_root = feature_root
        self.samples: list[dict] = []
        self._load(max_videos)

    # ------------------------------------------------------------------ #

    def _feature_path(self, path: str) -> str:
        if self.feature_root is not None:
            return os.path.join(self.feature_root, os.path.basename(path))
        return path if os.path.isabs(path) else str(REPO_ROOT / path)

    def _load(self, max_videos: int | None) -> None:
        with open(self.annotation_file) as handle:
            videos = json.load(handle)
        if max_videos is not None:
            videos = videos[:max_videos]

        for record_index, video in enumerate(videos):
            meta = video["id"]
            video_id = meta.get("vid") or os.path.splitext(os.path.basename(meta["feature"]))[0]
            segments = meta["legal_range"]
            gt_task = int(meta["task_id"])

            features = np.load(self._feature_path(meta["feature"]), allow_pickle=True)
            features = features["frames_features"]

            states, actions = [], []
            for start, end, action_id in segments:
                window = self._state_window(features, start, end)
                if window is None:
                    # Segment lies outside the extracted features; repeat the
                    # previous step so the sequence keeps its length.
                    if not states:
                        continue
                    window, action_id = states[-1], actions[-1]
                states.append(window)
                actions.append(int(action_id))

            if not states:
                continue

            # Looked up only for records that are kept: a record with no usable
            # segment never reaches the task classifier, so it has no prediction.
            if self.task_source == "predicted":
                if "event_class_top1" not in meta:
                    raise KeyError(
                        f"{self.annotation_file} has no 'event_class_top1' for video "
                        f"{video_id}. Generate it with scripts/predict_task_labels.py, "
                        f"or evaluate with eval.oracle_task=true."
                    )
                task = int(meta["event_class_top1"])
            else:
                task = gt_task

            # Short videos are padded by repeating their last step, matching the
            # reference protocol.
            while len(states) < self.horizon:
                states.append(states[-1])
                actions.append(actions[-1])
            if len(states) != self.horizon:
                raise ValueError(
                    f"video {video_id} in {self.annotation_file} has {len(states)} "
                    f"annotated steps but the horizon is {self.horizon}; use the "
                    f"annotation file cut to this horizon."
                )

            # Reduce to the [T + 1, observation_dim] state sequence once here
            # rather than on every __getitem__; it is also the smaller of the
            # two representations to keep in memory.
            stacked = np.stack(states)  # [T, 2, frames, feat]
            stacked = stacked.reshape(stacked.shape[0], stacked.shape[1], -1)
            self.samples.append(
                {
                    "states": self._states_between_actions(stacked).astype(np.float32),
                    "actions": np.asarray(actions, dtype=np.int64),
                    "task": task,
                    "gt_task": gt_task,
                    "video_id": video_id,
                    # Position in the annotation file. A video contributes one
                    # record per window, so the video id alone is not a key.
                    "record_index": record_index,
                }
            )

    def _state_window(self, features: np.ndarray, start: int, end: int):
        """Features of the observations bracketing one action step.

        Returns ``[2, FRAMES_PER_STATE, feature_dim]`` -- a short window centred
        on the start frame and one centred on the end frame -- or ``None`` if the
        step falls outside the extracted features.
        """
        if start < 0 or end >= features.shape[0]:
            return None
        half = FRAMES_PER_STATE // 2
        windows = []
        for centre in (start, end):
            window = features[max(0, centre - half): min(centre + half + 1, features.shape[0])]
            while window.shape[0] < FRAMES_PER_STATE:
                window = np.concatenate([window, window[-1:]], axis=0)
            windows.append(window)
        return np.stack(windows)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _states_between_actions(states: np.ndarray) -> np.ndarray:
        """``[T, 2, D]`` per-action (before, after) features -> ``[T + 1, D]`` states.

        Consecutive actions share the state between them, so the observation
        after action t and before action t+1 are averaged into a single state.
        """
        before = np.concatenate([states[:, 0:1], states[-1:, -1:]], axis=0)
        after = np.concatenate([states[0:1, 0:1], states[:, -1:]], axis=0)
        return np.concatenate([before, after], axis=1).mean(axis=1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return (
            torch.as_tensor(sample["states"], dtype=torch.float32),
            torch.as_tensor(sample["actions"], dtype=torch.long),
            torch.as_tensor(sample["task"], dtype=torch.long),
            torch.as_tensor(sample["gt_task"], dtype=torch.long),
        )


def build_datasets(config, max_videos: int | None = None):
    """Training and validation splits for an experiment config."""
    task_source = "ground_truth" if config.eval.oracle_task else "predicted"
    train = ProcedurePlanningDataset(
        config.paths.train_json,
        horizon=config.horizon,
        task_source="ground_truth",
        feature_root=config.paths.feature_root,
        max_videos=max_videos,
    )
    val = ProcedurePlanningDataset(
        config.paths.val_json,
        horizon=config.horizon,
        task_source=task_source,
        feature_root=config.paths.feature_root,
        max_videos=max_videos,
    )
    return train, val
