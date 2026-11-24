from pathlib import Path
import joblib
import pandas as pd
import streamlit as app

models_dir = Path(__file__).parent/"models"
label_names = {0: "poor", 1: "good", 2: "excellent"}

app.set_page_config(page_title="Review Sentiment")
app.title("AI-Powered Text Intelligence Platform")
app.write("Sentiment analysis of Amazon food reviews.")

@app.cache_resource
def load_models():
    models = {}
    models["Naive Bayes"] = (
        joblib.load(models_dir / "naive_bayes.joblib"),
        joblib.load(models_dir / "naive_bayes_vectorizer.joblib"),
    )
    return models

models = load_models()

def predict(model, vectorizer, text):
    x = vectorizer.transform([text]).toarray()
    label = int(model.predict(x)[0])
    probs = model.predict_proba(x)[0]
    return label, probs

review = app.text_area("Paste a food review", height=150)
if app.button("Predict"):
    if not review.strip():
        app.warning("No input")
    else:
        columns = app.columns(len(models))
        for column, (name, (model, vectorizer)) in zip(columns, models.items()):
            label, probs = predict(model, vectorizer, review)
            with column:
                app.subheader(name)
                chances = {label_names[int(c)]: p for c, p in zip(model.classes_, probs)}
                app.bar_chart(pd.Series(chances))