<div align="center">
<h1>Contrastive Energy Fields for Inference-Time Procedure Planning in Instructional Videos</h1>


[**Mohamed Afham**](https://mohamedafham.github.io/)<sup>1,3</sup>    [**Christoph
Reich**](https://christophreich1996.github.io/)<sup>1,2,3,4</sup>    [**Oliver
Hahn**](https://olvrhhn.github.io)<sup>1</sup>    [**Daniel
Cremers**](https://cvg.cit.tum.de/members/cremers)<sup>2,3,4</sup>    [**Stefan
Roth**](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp)<sup>1,3,5</sup>


<sup>1</sup>TU Darmstadt  <sup>2</sup>TU Munich  <sup>3</sup>ELIZA  <sup>4</sup>MCML  <sup>5</sup>hessian.AI
<h3>GCPR 2026 · Best Paper Honorable Mention</h3>

[Project page](https://visinf.github.io/cefito) · [arXiv](https://arxiv.org/abs/2608.16457)

</div>

---

Procedure planning estimates the sequence of actions that turns an observed
initial state into a given goal state. CEFITO splits that into two problems:

1. **Learn an action-conditioned energy field.** A predictor `P_θ` maps an
   initial observation and a candidate action sequence to the goal latent that
   sequence would reach. A margin-based triplet loss pulls the prediction of the
   correct sequence towards the observed goal and pushes wrong sequences away,
   with the margin adapted to how much each negative overlaps with the ground
   truth. No intermediate-state supervision is needed.
2. **Plan by minimising that energy at inference.** A lightweight task
   classifier predicts the high-level task, which restricts the search to the
   actions that task admits. Every sequence in that restricted space is scored,
   and the lowest-energy one is the plan.

## Install

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .
```

Only PyTorch, NumPy and PyYAML are required. Optional extras:
`".[logging]"` adds Weights & Biases for `--wandb`, and `".[embeddings]"` adds
the CLIP text encoder used to regenerate the action embeddings.

## Data

| Dataset | Videos | Tasks | Actions | S3D features |
| --- | --- | --- | --- | --- |
| [CrossTask](https://github.com/DmZhukov/CrossTask) | 2 750 | 18 | 133 | [download](https://drive.google.com/file/d/146VQTl9Pw4VmmlTUt3RVmZtDW8Dcw4uL/view) |
| [COIN](https://coin-dataset.github.io/) | 11 827 | 180 | 778 | [download](https://drive.google.com/file/d/1aQAUQP8jpmZW3iqzVoSeDGJfJOT3-8sv/view) |
| [NIV](https://www.di.ens.fr/willow/research/instructionvideos/) | 150 | 5 | 48 | [download](https://drive.google.com/file/d/1nmsYW0hRGWChCfuTnsvn8lfW2YWmFTfA/view) |

`dataset/` ships the annotations, the task taxonomies and the task-classifier
predictions. Two things are not included:

**1. S3D features** (pre-extracted on HowTo100M). Unzip each archive inside its
dataset folder:

```bash
pip install gdown
gdown 146VQTl9Pw4VmmlTUt3RVmZtDW8Dcw4uL -O crosstask.zip && unzip crosstask.zip -d dataset/CrossTask
gdown 1aQAUQP8jpmZW3iqzVoSeDGJfJOT3-8sv -O coin.zip      && unzip coin.zip      -d dataset/COIN
gdown 1nmsYW0hRGWChCfuTnsvn8lfW2YWmFTfA -O niv.zip       && unzip niv.zip       -d dataset/NIV
```

**2. Action embeddings.** Each action is embedded by its name with the frozen
CLIP ViT-B/32 text encoder:

```bash
uv pip install -e ".[embeddings]"
python scripts/extract_action_embeddings.py
```

This leaves

```
dataset/
  CrossTask/
    crosstask_features/processed_data/*.npy     S3D features            (downloaded)
    action_embeddings_133.npy                   action embeddings       (generated)
    train_list_{3,4}_133.json                   training windows
    test_list_{3,4}_133.json                    test windows
    test_list_{3,4}_133_with_predictions.json   test windows + predicted task
    crosstask_taxonomy_133.json                 task -> {action id: name}
  COIN/
    full_npy/*.npy, action_embeddings.npy
    coin_{train_70,test_30}_{3,4}.json, coin_test_30_{3,4}_with_predictions.json
    coin_taxonomy.json
  NIV/
    processed_data/*.npy, action_embeddings.npy
    {train70,test30}_{3,4}.json
    niv_taxonomy.json
```

If the features already live elsewhere, set `feature_root` for that dataset in
`configs/paths.yaml` instead. To keep machine-specific paths out of the
repository, use a copy of that file via `--paths my_paths.yaml` or
`export CEFITO_PATHS=my_paths.yaml`.

The planner restricts its search with the task classifier's prediction, stored
as `event_class_top1` in the `*_with_predictions.json` files. To regenerate
them:

```bash
bash run/task_labels.sh
```

## Usage

Each experiment is one YAML file in `configs/`; any entry can be overridden with
`--set key=value`.

```bash
# Train
python scripts/train.py configs/crosstask_t3.yaml

# Evaluate
python scripts/evaluate.py configs/crosstask_t3.yaml --checkpoint runs/crosstask_t3/best.pth

# Dump the top-5 plans and their energies
python scripts/evaluate.py configs/crosstask_t3.yaml \
    --checkpoint runs/crosstask_t3/best.pth --dump_predictions plans.json
```

Training resumes from `runs/<name>/last.pth` and writes the best planning
metrics to `runs/<name>/best_metrics.json`.

To check the pipeline end to end on a few videos:

```bash
python scripts/train.py configs/crosstask_t3.yaml --max_videos 24 \
    --set train.epochs=1 --set train.eval_every=1 --set run_name=_smoke
```

The exhaustive search grows as `|A(ĉ)|^T`; lower `eval.chunk_size` if
evaluation runs out of GPU memory.

## Citation

```bibtex
@inproceedings{afham2026cefito,
  title     = {Contrastive Energy Fields for Inference-Time Procedure Planning
               in Instructional Videos},
  author    = {Afham, Mohamed and Reich, Christoph and Hahn, Oliver and
               Cremers, Daniel and Roth, Stefan},
  booktitle = {Proceedings of the 48th German Conference on Pattern Recognition (GCPR)},
  year      = {2026}
}
```

## Acknowledgements

We thank the authors of [PDPP](https://github.com/mcg-nju/pdpp) and [ViterbiPlanNet](https://github.com/Gigi-G/ViterbiPlanNet) for open-sourcing their code. 
