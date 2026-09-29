"""Stamp task-classifier predictions into an annotation file.

    python scripts/predict_task_labels.py configs/coin_t3.yaml \
        --checkpoint runs/taskcls_coin/task_classifier.pth \
        --source dataset/COIN/coin_test_30_3.json \
        --output dataset/COIN/coin_test_30_3_with_predictions.json

The planner reads the resulting ``event_class_top1`` field to constrain its
search, so this is the step that connects the task classifier to inference.
``dataset/`` already ships the stamped test files; run this (or
``run/task_labels.sh``) only to regenerate them.
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from cefito.config import add_config_arguments, load_config
from cefito.data import ProcedurePlanningDataset
from cefito.models import TaskClassifier
from cefito.utils import setup_logging

from train_task_classifier import observations  # noqa: E402  (sibling script)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="annotation file to stamp (default: the config's val_json)",
    )
    parser.add_argument("--output", type=str, required=True, help="where to write the new json")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config, args.paths, args.overrides)
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source = args.source or config.paths.val_json
    dataset = ProcedurePlanningDataset(
        source,
        horizon=config.horizon,
        task_source="ground_truth",
        feature_root=config.paths.feature_root,
    )

    model = TaskClassifier(config.model.observation_dim, config.num_tasks).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])

    predictions, correct = [], 0
    for states, _, _, gt_task in DataLoader(dataset, batch_size=256, shuffle=False):
        logits = model(observations(states.to(device).float()))
        predicted = logits.argmax(-1).cpu()
        correct += int((predicted == gt_task).sum())
        predictions.extend(predicted.tolist())

    # Match predictions back by record position, not video id: a video has one
    # record per window, each with its own (initial, goal) pair and so its own
    # prediction. The dataset can skip a record with no usable segment, which
    # is why the position is carried rather than assumed.
    by_record = {
        sample["record_index"]: prediction
        for sample, prediction in zip(dataset.samples, predictions)
    }
    with open(source) as handle:
        records = json.load(handle)
    stamped = 0
    for index, record in enumerate(records):
        if index in by_record:
            record["id"]["event_class_top1"] = int(by_record[index])
            stamped += 1

    with open(args.output, "w") as handle:
        json.dump(records, handle)
    logger.info(
        f"task accuracy {100 * correct / len(predictions):.2f}% over {len(predictions)} samples; "
        f"stamped {stamped}/{len(records)} records into {args.output}"
    )


if __name__ == "__main__":
    main()
