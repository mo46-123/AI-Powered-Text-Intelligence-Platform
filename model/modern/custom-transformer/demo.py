import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MODEL_DIR = REPO_ROOT / "models"

CHECKPOINT_FILE = MODEL_DIR / "transformer.pth"
CONFIG_FILE = MODEL_DIR / "transformer_config.json"
TOKENIZER_FILE = MODEL_DIR / "custom_tokenizer.json"

LABEL_NAMES = {0: "poor", 1: "good", 2: "excellent"}

MODEL_KEYS = {"embedding_dim", "nhead", "num_layers", "dropout", "max_len"}

sys.path.append(str(HERE))

from transformer import TransformerCls


def resolve_device(name):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint=CHECKPOINT_FILE, config_file=CONFIG_FILE,
               tokenizer_file=TOKENIZER_FILE, device=None):
    for path, hint in (
        (tokenizer_file, "build it with tokenizer.ipynb"),
        (config_file, "copy it from the training notebook"),
        (checkpoint, "train it with train.ipynb, or pass --checkpoint"),
    ):
        if not path.exists():
            raise SystemExit(f"missing {path}\n  -> {hint}")

    with open(config_file) as f:
        config = json.load(f)

    unknown = set(config) - MODEL_KEYS
    if unknown:
        raise SystemExit(f"{config_file} has unknown keys {sorted(unknown)}")

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    model = TransformerCls(tokenizer, **config)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise SystemExit(f"{checkpoint} does not fit {config_file}:\n{exc}") from None

    return model.to(resolve_device(device)).eval()


def predict(model, texts, batch_size=64):
    """Return one integer label per text."""
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(text) for text in texts]
    if not texts:
        return np.empty(0, dtype=np.int64)

    labels = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            logits = model(texts[start:start + batch_size])
            labels.append(logits.argmax(dim=-1).cpu())

    return torch.cat(labels).numpy()


def build_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Demo the custom transformer sentiment classifier.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-c",
        "--comment",
        help="customer comment"
    )
    source.add_argument(
        "-cf",
        "--comment-file",
        help="csv file containing a bunch of customer comments"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_FILE,
        help="model weights (default: %(default)s)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE,
        help="model hyperparameters (default: %(default)s)"
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=TOKENIZER_FILE,
        help="trained tokenizer (default: %(default)s)"
    )
    parser.add_argument(
        "--device",
        help="cpu or cuda (default: cuda when available)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="comments per forward pass when reading a file (default: %(default)s)"
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = build_args(argv)

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    model = load_model(args.checkpoint, args.config, args.tokenizer, args.device)

    if args.comment_file:
        df = pd.read_csv(args.comment_file)
        if "input_text" not in df.columns:
            raise SystemExit(
                f"{args.comment_file} has no 'input_text' column "
                f"(found {list(df.columns)})"
            )

        df["prediction"] = predict(model, df["input_text"], args.batch_size)
        df["prediction_label"] = df["prediction"].map(LABEL_NAMES)

        print(df[["input_text", "prediction_label"]].head(3).to_string(index=False))
        print()
        print("Label counts:")
        print(df["prediction_label"].value_counts().to_string())
    else:
        label = predict(model, [args.comment], args.batch_size)[0]
        print(f"Prediction: {LABEL_NAMES[int(label)]}")


if __name__ == "__main__":
    main()
