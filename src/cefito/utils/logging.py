"""Console (and optional Weights & Biases) logging."""

import logging
import sys
from pathlib import Path


def setup_logging(run_dir=None, rank: int = 0) -> logging.Logger:
    """Configure a logger that only speaks on rank 0."""
    logger = logging.getLogger("cefito")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    # Progress goes to stderr so that stdout stays a clean, parseable channel
    # for the metrics json the scripts print at the end.
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if run_dir is not None and rank == 0:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class WandbLogger:
    """Thin optional wrapper; a no-op unless ``--wandb`` was passed on rank 0."""

    def __init__(self, enabled: bool, project: str | None, name: str, config: dict):
        self.run = None
        if not enabled:
            return
        import wandb  # imported lazily so wandb stays an optional dependency

        self.run = wandb.init(project=project or "cefito", name=name, config=config)

    def log(self, metrics: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
