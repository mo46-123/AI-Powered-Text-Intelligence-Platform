import joblib
import json
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from model.utils import load_data, report_metrices

# evaulation methods
def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x

def evaluate(x, model=None, model_file=None, vectorizer=None,
             vectorizer_file=None, batch=True, dense=True):
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
    x = tfidf.transform(x)
    if dense: x = to_dense(x)
    if not batch: x = np.expand_dims(x, axis=0)
    return cls.predict(x)

# tf-idf selection 
def get_vocab(k, x_full, tfidf_full):
    # get idx of vocabs with heighest average tfidf
    term_scores = np.asarray(x_full.mean(axis=0)).ravel()
    top_idx = np.argsort(term_scores)[-k:]

    # restore idx to vocab
    idx_to_term = {i: t for t, i in tfidf_full.vocabulary_.items()}
    return sorted(idx_to_term[i] for i in top_idx)

# test model
def test(cls, vectorizer=None, vectorizer_file=None, dense=True):
    test_df = load_data('test')

    y_pred = evaluate(test_df['input_text'], cls,
                      vectorizer=vectorizer, vectorizer_file=vectorizer_file,
                      dense=dense)
    return report_metrices(test_df['Sentiment'], y_pred)

# model saving
def save_cls(name, best_config, metrices, best_model, best_vectorizer):
    # save 
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / f"{name}_config.json", 'w') as f:
        json.dump(best_config, f, indent=4)
    
    with open(models_dir / f'{name}_test_result.json', 'w') as f:
        json.dump(metrices, f, indent=4)

    joblib.dump(best_model, models_dir / f'{name}.joblib')
    joblib.dump(best_vectorizer, models_dir / f'{name}_vectorizer.joblib')
