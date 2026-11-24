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

LABEL_NAMES = {0: "poor", 1: "good", 2: "excellent"}

sys.path.append(str(HERE))

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def resolve_device(name):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir=MODEL_DIR, device=None):
    model_dir = MODEL_DIR / "fine_tuned"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    return tokenizer, model

@torch.inference_mode()
def predict_batch(texts, model, tokenizer, device, batch_size=64, max_length=256):
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[i:i + batch_size],
            truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        ).to(device)
        logits = model(**batch).logits
        all_probs.append(logits.softmax(dim=-1).cpu())
    return torch.cat(all_probs, dim=0)   


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

    tokenizer, model = load_model(args.device)

    if args.comment_file:
        df = pd.read_csv(args.comment_file)
        if "input_text" not in df.columns:
            raise SystemExit(
                f"{args.comment_file} has no 'input_text' column "
                f"(found {list(df.columns)})"
            )
        texts = df["input_text"].astype(str).tolist()
        df["prediction"] = predict_batch(texts, model, tokenizer, args.device, args.batch_size)
        df["prediction_label"] = df["prediction"].map(LABEL_NAMES)

        print(df[["input_text", "prediction_label"]].head(3).to_string(index=False))
        print()
        print("Label counts:")
        print(df["prediction_label"].value_counts().to_string())
    else:
        label = predict_batch([args.comment], model, tokenizer, args.device, args.batch_size)[0]
        print(f"Prediction: {LABEL_NAMES[int(label)]}")


if __name__ == "__main__":
    main()
