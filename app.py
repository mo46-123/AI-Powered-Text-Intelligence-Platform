from pathlib import Path
import altair as alt
import joblib
import pandas as pd
import streamlit as app

models_dir = Path(__file__).parent/"models"
label_names = {0: "poor", 1: "good", 2: "excellent"}
label_order = ["poor", "good", "excellent"]

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
    models["Logistic Regression"] = (
        joblib.load(models_dir / "logistic_regression.joblib"),
        joblib.load(models_dir / "logistic_regression_vectorizer.joblib"),
    )
    return models


models = load_models()


def predict(model, vectorizer, text):
    x = vectorizer.transform([text]).toarray()
    label = int(model.predict(x)[0])
    probs = model.predict_proba(x)[0]
    return label, probs


def probability_chart(chances):
    data = pd.DataFrame({
        "label": label_order,
        "probability": [chances[l] for l in label_order],
    })
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("label", sort=label_order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("probability", scale=alt.Scale(domain=[0, 1]), title=""),
        tooltip=["label", alt.Tooltip("probability", format=".0%")],
    )


with app.form("review_form"):
    review = app.text_area("Paste a food review", height=150)
    submitted = app.form_submit_button("Predict")

if submitted:
    if not review.strip():
        app.warning("No input")
    else:
        columns = app.columns(len(models))
        for column, (name, (model, vectorizer)) in zip(columns, models.items()):
            label, probs = predict(model, vectorizer, review)
            with column:
                app.subheader(name)
                app.metric("Prediction", label_names[label])
                chances = {label_names[int(c)]: p for c, p in zip(model.classes_, probs)}
                app.altair_chart(probability_chart(chances))