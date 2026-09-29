"""Evaluate a trained CEFITO checkpoint on the validation split.

    python scripts/evaluate.py configs/crosstask_t3.yaml \
        --checkpoint runs/crosstask_t3/best.pth

Add ``--dump_predictions out.json`` to write the top-k plans of every sample.
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from cefito.config import add_config_arguments, load_config
from cefito.data import EvalShardSampler, ProcedurePlanningDataset, load_action_names
from cefito.engine import evaluate
from cefito.models import build_model
from cefito.utils import (
    cleanup_distributed,
    is_distributed,
    load_weights,
    seed_everything,
    setup_distributed,
    setup_logging,
    unwrap,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="path to a .pth checkpoint")
    parser.add_argument(
        "--weights",
        choices=("ema", "raw"),
        default="ema",
        help="which copy of the weights to evaluate (default: the EMA copy)",
    )
    parser.add_argument(
        "--dump_predictions", type=str, default=None, help="write top-k plans to this json"
    )
    parser.add_argument("--top_k", type=int, default=5, help="plans to keep when dumping")
    parser.add_argument("--max_videos", type=int, default=None, help="cap the split (smoke tests)")
    return parser.parse_args()


@torch.no_grad()
def dump_predictions(model, dataset, config, path, top_k, device):
    """Write the top-k plans and their energies for every validation sample."""
    module = unwrap(model)
    names = load_action_names(config.paths.taxonomy)
    loader = DataLoader(dataset, batch_size=config.eval.batch_size, shuffle=False)

    records, offset = [], 0
    for states, actions, task, gt_task in loader:
        states = states.to(device).float()
        plans, energies = module.plan(
            states[:, 0],
            states[:, -1],
            task.to(device),
            chunk_size=config.eval.chunk_size,
            top_k=top_k,
        )
        for i in range(actions.size(0)):
            sample = dataset.samples[offset + i]
            records.append(
                {
                    "video_id": sample["video_id"],
                    "task": int(task[i]),
                    "ground_truth_task": int(gt_task[i]),
                    "ground_truth": [int(a) for a in actions[i]],
                    "ground_truth_names": [names.get(int(a)) for a in actions[i]] if names else None,
                    "plans": [
                        {
                            "actions": [int(a) for a in plan],
                            "names": [names.get(int(a)) for a in plan] if names else None,
                            "energy": round(float(energy), 6),
                        }
                        for plan, energy in zip(plans[i], energies[i])
                    ],
                }
            )
        offset += actions.size(0)

    with open(path, "w") as handle:
        json.dump(records, handle, indent=2)
    return len(records)


def main():
    args = parse_args()
    config = load_config(args.config, args.paths, args.overrides)

    rank, world_size, device = setup_distributed()
    seed_everything(config.train.seed, rank)
    logger = setup_logging(None, rank)

    if args.dump_predictions and world_size > 1:
        raise SystemExit(
            "--dump_predictions writes one file over the whole split; run it in a "
            "single process (without torchrun)."
        )

    task_source = "ground_truth" if config.eval.oracle_task else "predicted"
    dataset = ProcedurePlanningDataset(
        config.paths.val_json,
        horizon=config.horizon,
        task_source=task_source,
        feature_root=config.paths.feature_root,
        max_videos=args.max_videos,
    )
    logger.info(
        f"{config.dataset} T={config.horizon}: {len(dataset)} val samples, "
        f"planning with the {task_source.replace('_', ' ')} task"
    )

    model = build_model(config).to(device)
    load_weights(args.checkpoint, model, prefer_ema=args.weights == "ema", map_location=device)
    logger.info(f"loaded {args.weights} weights from {args.checkpoint}")

    sampler = (
        EvalShardSampler(len(dataset), rank, world_size) if is_distributed() else None
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, config.eval.batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=config.train.num_workers,
    )

    metrics = evaluate(
        model, loader, config.eval.chunk_size, progress_every=20, logger=logger
    )
    logger.info(
        f"SR {metrics['success_rate']:.2f} | mAcc {metrics['mean_accuracy']:.2f} | "
        f"mIoU {metrics['mean_iou']:.2f} | task acc {metrics['task_accuracy']:.2f}"
    )
    if rank == 0:
        print(json.dumps(metrics, indent=2))

    if args.dump_predictions and rank == 0:
        count = dump_predictions(
            model, dataset, config, args.dump_predictions, args.top_k, device
        )
        logger.info(f"wrote {count} predictions to {args.dump_predictions}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
