"""Encode every action name with the frozen CLIP text encoder.

    pip install -e ".[embeddings]"
    python scripts/extract_action_embeddings.py              # all datasets
    python scripts/extract_action_embeddings.py coin niv     # a subset

Reads each dataset's action names from its taxonomy file and writes
``{action id: 512-d float32 vector}`` to the ``action_embeddings`` path given in
``configs/paths.yaml``. That is the file the predictor embeds actions with.

The vector is the output of CLIP ViT-B/32's text transformer at the
end-of-text token, after the final layer norm and *before* the text projection,
computed from the bare action name with no prompt template. This is the
representation the released models were trained with.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from cefito.config import DEFAULT_PATHS_FILE, load_dataset_paths
from cefito.data import load_action_names

#: OpenAI's CLIP uses QuickGELU; open_clip only applies it under this name.
MODEL, PRETRAINED = "ViT-B-32-quickgelu", "openai"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="*", help="datasets to encode (default: every one in paths.yaml)"
    )
    parser.add_argument("--paths", type=str, default=None, help="path to paths.yaml")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def encode(model, tokenizer, names: list[str], device, batch_size: int) -> np.ndarray:
    """``[len(names), 512]`` end-of-text features before the text projection."""
    out = []
    for start in range(0, len(names), batch_size):
        tokens = tokenizer(names[start:start + batch_size]).to(device)
        x = model.token_embedding(tokens) + model.positional_embedding
        x = model.ln_final(model.transformer(x, attn_mask=model.attn_mask))
        # The end-of-text token has the largest id in CLIP's vocabulary.
        out.append(x[torch.arange(x.size(0)), tokens.argmax(dim=-1)].float().cpu())
    return torch.cat(out).numpy()


def main():
    args = parse_args()
    import open_clip  # optional dependency: pip install -e ".[embeddings]"

    paths_file = args.paths or DEFAULT_PATHS_FILE
    datasets = args.datasets or list(yaml.safe_load(open(paths_file)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL)

    for dataset in datasets:
        paths = load_dataset_paths(dataset, horizon=0, paths_file=paths_file)
        names = load_action_names(paths.taxonomy)
        ids = sorted(names)
        vectors = encode(model, tokenizer, [names[i] for i in ids], device, args.batch_size)

        output = Path(paths.action_embeddings)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, {i: v for i, v in zip(ids, vectors)}, allow_pickle=True)
        print(f"{dataset}: {len(ids)} actions -> {output}")


if __name__ == "__main__":
    main()
