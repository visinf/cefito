"""Train the CEFITO predictor.

    python scripts/train.py configs/crosstask_t3.yaml

Every ``train.eval_every`` epochs the EMA weights are evaluated with the full
planning search; the best checkpoint by success rate is kept as ``best.pth``.
Training resumes automatically from ``last.pth`` if it exists.
"""

import argparse
import json
import platform

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cefito.config import add_config_arguments, load_config
from cefito.data import EvalShardSampler, build_datasets
from cefito.engine import EMA, evaluate, train_one_epoch, validation_loss
from cefito.models import build_model
from cefito.utils import (
    WandbLogger,
    cleanup_distributed,
    is_distributed,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    setup_logging,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument(
        "--max_videos", type=int, default=None, help="cap the split size (smoke tests)"
    )
    return parser.parse_args()


def build_loader(
    dataset,
    batch_size,
    world_size,
    num_workers,
    pin_memory,
    shuffle,
    metric_shard=False,
):
    """A loader, sharded over ranks when running distributed.

    ``metric_shard`` picks the sampler that does not pad the split, for the
    loader whose results are turned into a metric. See
    :class:`~cefito.data.sampler.EvalShardSampler`.
    """
    if not is_distributed():
        sampler = None
    elif metric_shard:
        sampler = EvalShardSampler(len(dataset), torch.distributed.get_rank(), world_size)
    else:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    return (
        DataLoader(
            dataset,
            batch_size=max(1, batch_size // world_size),
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        sampler,
    )


def main():
    args = parse_args()
    config = load_config(args.config, args.paths, args.overrides)

    rank, world_size, device = setup_distributed()
    seed_everything(config.train.seed, rank)
    logger = setup_logging(config.run_dir, rank)
    # Recorded because results depend on the GPU model even at a fixed seed.
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    logger.info(
        f"host: {platform.node()} | device: {device_name} | world size: {world_size}"
    )
    logger.info(f"config:\n{json.dumps(config.to_dict(), indent=2)}")

    train_set, val_set = build_datasets(config, max_videos=args.max_videos)
    logger.info(f"train samples: {len(train_set)} | val samples: {len(val_set)}")

    loader_args = (world_size, config.train.num_workers, config.train.pin_memory)
    train_loader, train_sampler = build_loader(
        train_set,
        config.train.batch_size,
        *loader_args,
        shuffle=True,
    )
    val_loss_loader, _ = build_loader(
        val_set, config.train.batch_size, *loader_args, shuffle=False
    )
    val_plan_loader, _ = build_loader(
        val_set, config.eval.batch_size, *loader_args, shuffle=False, metric_shard=True
    )

    model = build_model(config).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"predictor parameters: {parameters / 1e6:.2f}M")

    ema = EMA(
        model,
        config.train.ema_decay,
        config.train.step_start_ema,
        config.train.update_ema_every,
    )
    ema.model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    start_epoch, step, best_metrics = 0, 0, {"success_rate": -1.0, "mean_accuracy": -1.0}
    last_path = config.run_dir / "last.pth"
    best_metrics_path = config.run_dir / "best_metrics.json"
    if last_path.exists():
        start_epoch, step = load_checkpoint(
            last_path, model, ema.model, optimizer, map_location=device
        )
        # Carry the best score across the resume, so a worse epoch cannot
        # overwrite a better checkpoint saved before the interruption.
        if best_metrics_path.exists():
            best_metrics = json.loads(best_metrics_path.read_text())
        logger.info(
            f"resumed from {last_path} at epoch {start_epoch} "
            f"(best SR so far {best_metrics['success_rate']:.2f})"
        )

    if is_distributed():
        model = DistributedDataParallel(model, device_ids=[device.index])

    wandb = WandbLogger(
        args.wandb and rank == 0, args.wandb_project, config.run_name, config.to_dict()
    )

    for epoch in range(start_epoch, config.train.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        logger.info(f"epoch {epoch + 1}/{config.train.epochs}")

        train_losses, step = train_one_epoch(
            model, train_loader, optimizer, ema, step, device, logger,
            observation_noise=config.train.observation_noise,
        )
        val_losses = validation_loss(model, val_loss_loader, device)
        logger.info(
            f"epoch {epoch + 1}: train loss {train_losses['total']:.4f} | "
            f"val loss {val_losses['total']:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        metrics = {
            **{f"train/{k}_loss": v for k, v in train_losses.items()},
            **{f"val/{k}_loss": v for k, v in val_losses.items()},
        }

        if (epoch + 1) % config.train.eval_every == 0:
            planning = evaluate(ema.model, val_plan_loader, config.eval.chunk_size)
            logger.info(
                f"epoch {epoch + 1}: SR {planning['success_rate']:.2f} | "
                f"mAcc {planning['mean_accuracy']:.2f} | mIoU {planning['mean_iou']:.2f}"
            )
            metrics.update({f"val/{k}": v for k, v in planning.items()})

            improved = planning["success_rate"] > best_metrics["success_rate"] or (
                planning["success_rate"] == best_metrics["success_rate"]
                and planning["mean_accuracy"] > best_metrics["mean_accuracy"]
            )
            if improved and rank == 0:
                best_metrics = planning
                save_checkpoint(
                    config.run_dir / "best.pth", model, ema.model, optimizer, epoch + 1, step, config
                )
                best_metrics_path.write_text(
                    json.dumps({"epoch": epoch + 1, **planning}, indent=2)
                )
                logger.info(f"new best: SR {planning['success_rate']:.2f}")

        if rank == 0:
            save_checkpoint(last_path, model, ema.model, optimizer, epoch + 1, step, config)
        wandb.log(metrics, step=epoch + 1)

    logger.info(f"best: {json.dumps(best_metrics, indent=2)}")
    wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
