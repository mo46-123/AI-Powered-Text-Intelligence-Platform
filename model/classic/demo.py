from pathlib import Path
import argparse
import pandas as pd
from shared import evaluate

REPO_ROOT = Path(__file__).resolve().parents[2]
MDOEL_DIR = REPO_ROOT / "models"

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

    source.add_argument(
        "-m",
        "--model",
        choices=['naive_bayes', 'logistic_regression'],
        help="choose which model to demo"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = build_args()

    model_file = MDOEL_DIR / f"{args.model}.joblib"
    vectorizer_file = MDOEL_DIR / f"{args.model}_vectorizer.joblib"

    if args.comment_file:
        df = pd.read_csv(args.comment_file)
        df['prediction'] = evaluate(
            df['input_text'],
            model_file=model_file,
            vectorizer_file=vectorizer_file,
        )
        df['prediction_label'] = df['prediction'].map(LABEL_NAMES)

        print(df[['input_text', 'prediction_label']].head(3).to_string(index=False))
    else:
        label = evaluate(
            [args.comment],
            model_file=model_file,
            vectorizer_file=vectorizer_file,
        )[0]
        print(f"Prediction: {LABEL_NAMES[int(label)]}")
