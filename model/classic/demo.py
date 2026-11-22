from pathlib import Path
import argparse
import pandas as pd
from shared import evaluate

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = REPO_ROOT / "models" / "naive_bayes.joblib"
VECTORIZER_FILE = REPO_ROOT / "models" / "naive_bayes_vectorizer.joblib"

LABEL_NAMES = {0: "poor", 1: "good", 2: "excellent"}


def build_args():
    parser = argparse.ArgumentParser()

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

    return parser.parse_args()


if __name__ == '__main__':
    args = build_args()

    if args.comment_file:
        df = pd.read_csv(args.comment_file)
        df['prediction'] = evaluate(
            df['input_text'],
            model_file=MODEL_FILE,
            vectorizer_file=VECTORIZER_FILE,
        )
        df['prediction_label'] = df['prediction'].map(LABEL_NAMES)

        print(df[['input_text', 'prediction_label']].head(3).to_string(index=False))
    else:
        label = evaluate(
            [args.comment],
            model_file=MODEL_FILE,
            vectorizer_file=VECTORIZER_FILE,
        )[0]
        print(f"Prediction: {LABEL_NAMES[int(label)]}")
