import os
import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --- load the three splits ---
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

# --- define the two representations ---
bow = CountVectorizer(min_df=5, max_features=20000)
tfidf = TfidfVectorizer(min_df=5, max_features=20000)

# --- fit on TRAIN ONLY, then transform train ---
X_bow_train = bow.fit_transform(train["input_text"])
X_tfidf_train = tfidf.fit_transform(train["input_text"])

# --- transform val and test (no refitting) ---
X_bow_val = bow.transform(val["input_text"])
X_bow_test = bow.transform(test["input_text"])
X_tfidf_val = tfidf.transform(val["input_text"])
X_tfidf_test = tfidf.transform(test["input_text"])

print("BoW vocab size:", len(bow.vocabulary_))
print("TF-IDF vocab size:", len(tfidf.vocabulary_))
print("BoW   train/val/test:", X_bow_train.shape, X_bow_val.shape, X_bow_test.shape)
print("TFIDF train/val/test:", X_tfidf_train.shape, X_tfidf_val.shape, X_tfidf_test.shape)

# --- save fitted vectorizers (the app and teammates load these) ---
os.makedirs("models", exist_ok=True)
joblib.dump(bow, "models/bow_vectorizer.joblib")
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")

# --- save the feature matrices (sparse .npz, same row order as the CSVs) ---
os.makedirs("data/features", exist_ok=True)
matrices = {
    "bow_train": X_bow_train, "bow_val": X_bow_val, "bow_test": X_bow_test,
    "tfidf_train": X_tfidf_train, "tfidf_val": X_tfidf_val, "tfidf_test": X_tfidf_test,
}
for name, mat in matrices.items():
    sp.save_npz(f"data/features/{name}.npz", mat)

print("Saved vectorizers to models/ and matrices to data/features/")