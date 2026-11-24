import sys
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from model.classic.shared import evaluate, save_cls, test
from model.utils import report_metrices, load_data

def train_cls(x, y, C):
    cls = LogisticRegression(
        C=C,
        class_weight="balanced",  
        solver="lbfgs",
        max_iter=2000,
    )
    cls.fit(x, y)
    return cls


def full_training(n_trial=5, c_min=0.25, c_max=4.0):

    train = load_data("train")
    val = load_data("val")

    tfidf = TfidfVectorizer(min_df=3)
    x_train = tfidf.fit_transform(train["input_text"])
    y_train = train["Sentiment"]
    print(f"features: {x_train.shape[0]} docs x {x_train.shape[1]} terms", flush=True)

    best_macro = -np.inf
    best_metrics = None
    best_model = None
    best_c = None

    for c in np.geomspace(c_min, c_max, n_trial):
    
        c = round(float(c), 4)
        cls = train_cls(x_train, y_train, c)
        
        print(f"metrics for using c = {c}")

        y_pred = evaluate(val["input_text"], cls, vectorizer=tfidf, dense=False)
        metrics = report_metrices(val["Sentiment"], y_pred)

        if metrics["f1"]["macro"] > best_macro:
            best_macro = metrics["f1"]["macro"]
            best_metrics = metrics
            best_model = cls
            best_c = c

    best_config = {
        "C": best_c,
        **best_metrics,
    }
    return best_config, best_model, tfidf


if __name__ == '__main__':
    # train
    best_config, best_model, best_vectorizer = full_training(5, 0.25, 4.0)

    print("\nmetrics on test dataset")
    metrices = test(best_model, best_vectorizer, dense=False)

    # save
    save_cls("logistic_regression", best_config, metrices, best_model, best_vectorizer)
