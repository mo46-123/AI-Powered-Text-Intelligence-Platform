import sys
import numpy as np
from pathlib import Path
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from model.classic.shared import evaluate, to_dense, save_cls, get_vocab, test
from model.utils import report_metrices, load_data

def train_cls(x, y):
    cls = GaussianNB()
    cls.fit(to_dense(x), y)
    return cls

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


if __name__ == '__main__':
    # train 
    best_config, best_model, best_vectorizer = full_training(10, 800, 2400)
    
    # test
    print("metrics on test dataset")
    metrices = test(best_model, best_vectorizer)

    # save 
    save_cls("naive_bayes", best_config, metrices, best_model, best_vectorizer)
