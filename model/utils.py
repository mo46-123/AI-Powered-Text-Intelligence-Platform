import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]

def report_metrices(y_true, y_pred):

    # f1 
    f1 = {
        'per_class': [float(v) for v in f1_score(y_true, y_pred, average=None, zero_division=0)],
        'macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    # precision
    precision = {
        'per_class': [float(v) for v in precision_score(y_true, y_pred, average=None, zero_division=0)],
        'macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro': float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
        'weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    #recall
    recall = {
        'per_class': [float(v) for v in recall_score(y_true, y_pred, average=None, zero_division=0)],
        'macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro': float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
        'weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'labels': labels.tolist(),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    for name in ('f1', 'precision', 'recall'):
        values = metrics[name]
        print(f"{name.capitalize():<10} per-class {values['per_class']}  "
              f"macro {values['macro']:.3f}  micro {values['micro']:.3f}  "
              f"weighted {values['weighted']:.3f}")
    print(f"{'Accuracy':<10} {metrics['accuracy']:.3f}")

    print(f"Confusion Matrix  labels {metrics['labels']}")
    print(np.array(metrics['confusion_matrix']))

    return metrics

def load_data(name):
    return pd.read_csv(REPO_ROOT / 'data' / f"{name}.csv")


def save_classic_cls(best_config, metrices, best_model, best_vectorizer):
    # save 
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "naive_bayes_config.json", 'w') as f:
        json.dump(best_config, f, indent=4)
    
    with open(models_dir / 'test_result.json', 'w') as f:
        json.dump(metrices, f, indent=4)

    joblib.dump(best_model, models_dir / 'naive_bayes.joblib')
    joblib.dump(best_vectorizer, models_dir / 'naive_bayes_vectorizer.joblib')
