"""Experiment configuration.

A run is described by two YAML files:

* ``configs/paths.yaml``  -- where each dataset's files live. Every path may use
  ``{root}`` and ``{horizon}`` placeholders; relative paths are resolved against
  the repository root. Override the file with ``--paths`` or the
  ``CEFITO_PATHS`` environment variable.
* ``configs/<experiment>.yaml`` -- dataset name, planning horizon, model size,
  loss and optimisation hyper-parameters.

Nothing else in the codebase reads a hard-coded path.
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_FILE = REPO_ROOT / "configs" / "paths.yaml"


@dataclass
class DatasetPaths:
    """Resolved on-disk locations for one dataset."""

    root: str
    train_json: str
    val_json: str
    #: ``{"<task id>_<task name>": {"<action id>": "<action name>"}}``; defines
    #: A(c) for every task and names every action.
    taxonomy: str
    #: ``{action id: 512-d vector}``, written by
    #: ``scripts/extract_action_embeddings.py``.
    action_embeddings: str
    #: Rebases the per-video feature paths stored inside the json files onto
    #: this directory. ``None`` uses them as stored.
    feature_root: str | None = None


@dataclass
class ModelConfig:
    observation_dim: int = 1536
    hidden_dim: int = 384
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    causal_attention: bool = True


@dataclass
class LossConfig:
    #: N -- negative action sequences sampled per ground-truth sequence.
    num_negatives: int = 50
    #: r -- fraction of negatives drawn from the same task (hard negatives).
    hard_negative_ratio: float = 0.8
    #: tau_min / tau_max -- bounds of the adaptive triplet margin.
    min_margin: float = 0.01
    max_margin: float = 0.1
    #: lambda -- weight of the auxiliary action-reconstruction loss.
    aux_weight: float = 1.0
    #: Probability of drawing a *permutation* of the ground truth as a
    #: negative -- same actions, wrong order. The i.i.d. sampler only reaches
    #: one by chance (~T!/|A(c)|^T per draw).
    permutation_negative_ratio: float = 0.0
    #: Blends the margin's set-difference severity with a positional (Hamming)
    #: one, so a permutation negative earns a real margin instead of the
    #: weakest one. ``0.0`` is set-only; positional distance is never smaller
    #: than set distance, so raising this only strengthens a margin.
    margin_order_weight: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 5.0e-4
    #: Gaussian augmentation on the two observed states during training, as a
    #: multiple of the observation's own standard deviation. Scale-free, so the
    #: same value means the same thing on every dataset.
    observation_noise: float = 0.0
    weight_decay: float = 1.0e-3
    ema_decay: float = 0.995
    step_start_ema: int = 400
    update_ema_every: int = 10
    #: Run planning metrics on the validation set every N epochs.
    eval_every: int = 3
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 23


@dataclass
class EvalConfig:
    batch_size: int = 32
    #: Candidate rows scored per predictor call. Trades memory for speed.
    chunk_size: int = 50000
    #: Plan with the ground-truth task instead of the classifier prediction
    #: (the oracle experiment, Tab. 6).
    oracle_task: bool = False


@dataclass
class TaskClassifierConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    num_workers: int = 0
    seed: int = 23


@dataclass
class Config:
    dataset: str = "crosstask"
    horizon: int = 3
    num_actions: int = 133
    num_tasks: int = 18
    output_dir: str = "runs"
    run_name: str = "cefito"

    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    task_classifier: TaskClassifierConfig = field(default_factory=TaskClassifierConfig)

    #: Filled in by :func:`load_config` from ``paths.yaml``.
    paths: DatasetPaths = field(default_factory=lambda: DatasetPaths("", "", "", "", ""))

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_dict(cls, data: dict[str, Any]):
    """Build a (possibly nested) dataclass, rejecting unknown keys."""
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        field_type = known[key].type
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _from_dict(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _absolute(path: str) -> str:
    """Expand ``~`` and variables; anchor a relative path at the repository root."""
    path = os.path.expandvars(os.path.expanduser(path))
    return path if os.path.isabs(path) else str(REPO_ROOT / path)


def _resolve_paths(entry: dict[str, Any], horizon: int) -> DatasetPaths:
    root = _absolute(entry["root"])
    resolved = {"root": root}
    for key, value in entry.items():
        if key == "root":
            continue
        if isinstance(value, str):
            value = _absolute(value.format(root=root, horizon=horizon))
        resolved[key] = value
    return _from_dict(DatasetPaths, resolved)


def load_dataset_paths(
    dataset: str, horizon: int, paths_file: str | Path | None = None
) -> DatasetPaths:
    """Resolve one dataset's entry of ``paths.yaml`` for a planning horizon."""
    paths_file = paths_file or os.environ.get("CEFITO_PATHS") or DEFAULT_PATHS_FILE
    with open(paths_file) as handle:
        all_paths = yaml.safe_load(handle) or {}
    if dataset not in all_paths:
        raise KeyError(
            f"dataset '{dataset}' is not described in {paths_file}. "
            f"Known datasets: {sorted(all_paths)}"
        )
    return _resolve_paths(all_paths[dataset], horizon)


def _apply_overrides(config: Config, overrides: list[str]) -> None:
    """Apply ``a.b=value`` command line overrides in place."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override '{override}' is not of the form key=value")
        key, raw = override.split("=", 1)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise KeyError(f"unknown config key '{key}'")
        current = getattr(target, leaf)
        # YAML parses ints, floats, bools, null and strings the same way the
        # config files do, so overrides and files agree on types.
        value = yaml.safe_load(raw)
        if isinstance(current, bool):
            if not isinstance(value, bool):
                raise TypeError(f"'{key}' expects a boolean, got {raw!r}")
        elif isinstance(current, (int, float)) and isinstance(value, str):
            # PyYAML follows YAML 1.1, which only recognises a float in
            # scientific notation when it has both a decimal point and a signed
            # exponent -- so "1e-4" parses as the *string* "1e-4". Coerce here
            # rather than letting it reach the optimiser as a string.
            caster = float if isinstance(current, float) else int
            try:
                value = caster(value)
            except ValueError:
                raise TypeError(
                    f"'{key}' expects {caster.__name__}, got {raw!r}"
                ) from None
        setattr(target, leaf, value)


def load_config(
    experiment: str | Path,
    paths_file: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load an experiment config and resolve its dataset paths."""
    with open(experiment) as handle:
        raw = yaml.safe_load(handle) or {}
    config = _from_dict(Config, raw)
    if overrides:
        _apply_overrides(config, overrides)
    config.paths = load_dataset_paths(config.dataset, config.horizon, paths_file)
    return config


def add_config_arguments(parser) -> None:
    """Register the config flags shared by every entry point."""
    parser.add_argument("config", type=str, help="path to an experiment yaml in configs/")
    parser.add_argument(
        "--paths",
        type=str,
        default=None,
        help="path to paths.yaml (default: configs/paths.yaml or $CEFITO_PATHS)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config entry, e.g. --set train.lr=0.0002 (repeatable)",
    )
