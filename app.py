from pathlib import Path
import joblib
import streamlit as app

models_dir = Path(__file__).parent/"models"
label_names = {0: "poor", 1: "good", 2: "excellent"}

app.set_page_config(page_title="Review Sentiment")
app.title("AI-Powered Text Intelligence Platform")
app.write("Sentiment analysis of Amazon food reviews.")

@app.cache_resource
def load_naive_bayes():
    model = joblib.load(models_dir/"naive_bayes.joblib")
    vectorizer = joblib.load(models_dir/"naive_bayes_vectorizer.joblib")
    return model, vectorizer
model, vectorizer = load_naive_bayes()

review = app.text_area("Paste a food review", height=150)
if app.button("Predict"):
    if not review.strip():
        app.warning("No input")
    else:
        x = vectorizer.transform([review]).toarray()
        label = int(model.predict(x)[0]) 
        app.success(f"Prediction: {label_names[label]}")