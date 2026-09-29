"""Train the task classifier that constrains the search at inference (Tab. 5).

    python scripts/train_task_classifier.py configs/crosstask_t3.yaml

Writes ``runs/<run_name>/task_classifier.pth``. Feed it to
``scripts/predict_task_labels.py`` to stamp the predictions into an annotation
file, which is what the planner reads at inference time.
"""

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cefito.config import add_config_arguments, load_config
from cefito.data import ProcedurePlanningDataset
from cefito.models import TaskClassifier
from cefito.utils import AverageMeter, seed_everything, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--output", type=str, default=None, help="where to write the checkpoint")
    parser.add_argument("--max_videos", type=int, default=None, help="cap the split (smoke tests)")
    return parser.parse_args()


def observations(states: torch.Tensor) -> torch.Tensor:
    """``[B, T + 1, E]`` states -> the ``[B, 2, E]`` (initial, goal) pair."""
    return torch.stack([states[:, 0], states[:, -1]], dim=1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Top-1 and top-3 task accuracy on a split."""
    model.eval()
    top1, top3 = AverageMeter(), AverageMeter()
    for states, _, _, gt_task in loader:
        states = states.to(device).float()
        gt_task = gt_task.to(device)
        logits = model(observations(states))
        batch = gt_task.size(0)
        top1.update((logits.argmax(-1) == gt_task).float().mean().item() * 100, batch)
        ranked = logits.topk(min(3, logits.size(-1)), dim=-1).indices
        top3.update((ranked == gt_task.unsqueeze(1)).any(1).float().mean().item() * 100, batch)
    return top1.average, top3.average


def main():
    args = parse_args()
    config = load_config(args.config, args.paths, args.overrides)
    seed_everything(config.task_classifier.seed)
    logger = setup_logging(config.run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {
        name: ProcedurePlanningDataset(
            path,
            horizon=config.horizon,
            task_source="ground_truth",
            feature_root=config.paths.feature_root,
            max_videos=args.max_videos,
        )
        for name, path in (("train", config.paths.train_json), ("val", config.paths.val_json))
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.task_classifier.batch_size,
            shuffle=name == "train",
            num_workers=config.task_classifier.num_workers,
            drop_last=name == "train",
        )
        for name, dataset in splits.items()
    }
    logger.info(f"train {len(splits['train'])} | val {len(splits['val'])} samples")

    model = TaskClassifier(config.model.observation_dim, config.num_tasks).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.task_classifier.lr,
        weight_decay=config.task_classifier.weight_decay,
    )

    output = args.output or config.run_dir / "task_classifier.pth"
    best = -1.0
    for epoch in range(config.task_classifier.epochs):
        model.train()
        loss_meter = AverageMeter()
        for states, _, _, gt_task in loaders["train"]:
            states = states.to(device).float()
            gt_task = gt_task.to(device)
            loss = F.cross_entropy(model(observations(states)), gt_task)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), gt_task.size(0))

        top1, top3 = evaluate(model, loaders["val"], device)
        logger.info(
            f"epoch {epoch + 1}/{config.task_classifier.epochs}: "
            f"loss {loss_meter.average:.4f} | val top-1 {top1:.2f} | top-3 {top3:.2f}"
        )
        if top1 > best:
            best = top1
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "config": config.to_dict()},
                output,
            )

    logger.info(f"best val top-1 accuracy {best:.2f}; checkpoint at {output}")
    print(json.dumps({"task_accuracy_top1": best, "checkpoint": str(output)}, indent=2))


if __name__ == "__main__":
    main()
