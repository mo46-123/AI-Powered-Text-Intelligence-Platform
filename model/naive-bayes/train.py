from sklearn.naive_bayes import GaussianNB, ComplementNB 
import joblib
import os
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
from evaluation import evaluate, report_metrices, to_dense
import json

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_data(name):
    return pd.read_csv(REPO_ROOT / 'data' / f"{name}.csv")

def train_cls(x, y):
    cls = GaussianNB()
    cls.fit(to_dense(x), y)
    return cls

def get_vocab(k, x_full, tfidf_full):
    # get idx of vocabs with heighest average tfidf
    term_scores = np.asarray(x_full.mean(axis=0)).ravel()
    top_idx = np.argsort(term_scores)[-k:]

    # restore idx to vocab
    idx_to_term = {i: t for t, i in tfidf_full.vocabulary_.items()}
    return sorted(idx_to_term[i] for i in top_idx)


def full_training(n_trial = 10, k_min=500, k_max=1500):
    tfidf_full = TfidfVectorizer(min_df=3)

    train = load_data('train')
    val = load_data('val')
    x_full = tfidf_full.fit_transform(train['input_text'])

    config = {}
    best_macro = -np.inf
    best_model = None
    best_vectorizer = None
    best_k = None

    step = max(1, (k_max - k_min) // n_trial)

    for k in range(k_min, k_max+1, step):
        
        print(f"metrics for using the top {k} vocabs")

        # init tfidf 
        top_terms = get_vocab(k, x_full, tfidf_full)
        tfidf = TfidfVectorizer(vocabulary=top_terms)
        
        # train model
        x = tfidf.fit_transform(train['input_text'])
        y = train['Sentiment']
        cls = train_cls(x, y)

        # validate result  
        y_pred = evaluate(val['input_text'], cls, vectorizer=tfidf)
        metrics = report_metrices(val['Sentiment'], y_pred)
        
        # save to config
        config[str(k)] = {
            'k': k, 
            **metrics,
            'vocab': top_terms
        }

        if metrics['f1']['macro'] > best_macro:
            best_macro = metrics['f1']['macro']
            best_model = cls
            best_vectorizer = tfidf
            best_k = k

    return config[str(best_k)], best_model, best_vectorizer

def test(cls, vectorizer=None, vectorizer_file=None):
    test_df = load_data('test')

    y_pred = evaluate(test_df['input_text'], cls,
                      vectorizer=vectorizer, vectorizer_file=vectorizer_file)
    return report_metrices(test_df['Sentiment'], y_pred)


if __name__ == '__main__':
    # train 
    best_config, best_model, best_vectorizer = full_training(10, 800, 2400)
    
    # test
    print("metrics on test dataset")
    metrices = test(best_model, best_vectorizer)

    # save 
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "naive_bayes_config.json", 'w') as f:
        json.dump(best_config, f, indent=4)
    
    with open(models_dir / 'test_result.json', 'w') as f:
        json.dump(metrices, f, indent=4)

    joblib.dump(best_model, models_dir / 'naive_bayes.joblib')
    joblib.dump(best_vectorizer, models_dir / 'naive_bayes_vectorizer.joblib')
   
