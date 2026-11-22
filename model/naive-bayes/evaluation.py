from sklearn.naive_bayes import GaussianNB 
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
    
