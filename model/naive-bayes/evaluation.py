from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import joblib
from pathlib import Path
import numpy as np

def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x

def evaluate(x, model=None, model_file=None, vectorizer=None,
             vectorizer_file=None, batch=True):
    #load trained model
    if model_file:
        cls = joblib.load(model_file)
    elif model is not None:
        cls = model

    if vectorizer_file:
        tfidf = joblib.load(vectorizer_file)
    elif vectorizer is not None:
        tfidf = vectorizer

    # get prediction
    x = to_dense(tfidf.transform(x))
    if not batch: x = np.expand_dims(x, axis=0)
    return cls.predict(x)

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

    
