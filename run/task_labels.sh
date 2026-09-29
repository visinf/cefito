#!/usr/bin/env bash
# Regenerate the task-classifier predictions shipped in dataset/.
#
# One classifier per dataset, trained on the T=3 training split with the recipe
# in configs/<dataset>_t3.yaml, labels both the T=3 and T=4 test splits. Test
# videos are disjoint from every training video. The stamped copies are written
# next to the originals as *_with_predictions.json, which is what
# configs/paths.yaml points val_json at.
set -euo pipefail
cd "$(dirname "$0")/.."

for dataset in crosstask coin; do
    checkpoint=runs/taskcls_${dataset}/task_classifier.pth
    python scripts/train_task_classifier.py "configs/${dataset}_t3.yaml" \
        --set run_name="taskcls_${dataset}"
    for horizon in 3 4; do
        case $dataset in
            crosstask) test=dataset/CrossTask/test_list_${horizon}_133 ;;
            coin)      test=dataset/COIN/coin_test_30_${horizon} ;;
        esac
        python scripts/predict_task_labels.py "configs/${dataset}_t${horizon}.yaml" \
            --checkpoint "$checkpoint" \
            --source "${test}.json" \
            --output "${test}_with_predictions.json"
    done
done
