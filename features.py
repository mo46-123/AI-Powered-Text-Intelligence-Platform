import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train = pd.read_csv("data/train.csv")

bow = CountVectorizer(min_df=5, max_features=20000)
tfidf = TfidfVectorizer(min_df=5, max_features=20000)

# fit on TRAIN ONLY
X_bow = bow.fit_transform(train["input_text"])
X_tfidf = tfidf.fit_transform(train["input_text"])

print("BoW vocab size:", len(bow.vocabulary_), "| matrix:", X_bow.shape)
print("TF-IDF vocab size:", len(tfidf.vocabulary_), "| matrix:", X_tfidf.shape)