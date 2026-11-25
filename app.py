import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as app
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

root = Path(__file__).parent
models_dir = root / "models"
sys.path.append(str(root / "model" / "modern" / "custom-transformer"))
from transformer import TransformerCls 

label_names = {0: "poor", 1: "good", 2: "excellent"}
label_order = ["poor", "good", "excellent"]

app.set_page_config(page_title="Review Sentiment", layout="wide")
app.title("AI-Powered Text Intelligence Platform")
app.write("Sentiment analysis of Amazon food reviews.")

def load_classic(file):
    model = joblib.load(models_dir / f"{file}.joblib")
    vectorizer = joblib.load(models_dir / f"{file}_vectorizer.joblib")
    return ("classic", model, vectorizer)

def load_custom_transformer():
    config = json.loads((models_dir / "transformer_config.json").read_text())
    tokenizer = Tokenizer.from_file(str(models_dir / "custom_tokenizer.json"))
    model = TransformerCls(tokenizer, **config)
    weights = torch.load(models_dir / "transformer.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    return ("transformer", model, None)

def load_bert():
    folder = models_dir / "fine_tuned"
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder)
    model.eval()
    return ("bert", model, tokenizer)

@app.cache_resource
def load_models():
    models = {}
    models["Naive Bayes"] = load_classic("naive_bayes")
    models["Logistic Regression"] = load_classic("logistic_regression")
    if (models_dir / "transformer.pth").exists():
        models["Custom Transformer"] = load_custom_transformer()
    if (models_dir / "fine_tuned").exists():
        models["Fine-tuned BERT"] = load_bert()
    return models

models = load_models()

def predict(kind, model, helper, text):
    if kind == "classic":
        x = helper.transform([text]).toarray()
        probs = dict(zip(model.classes_, model.predict_proba(x)[0]))
        return [float(probs[c]) for c in (0, 1, 2)]
    with torch.no_grad():
        if kind == "transformer":
            scores = model([text])
        else:  # bert
            tokens = helper(text, truncation=True, max_length=256, return_tensors="pt")
            scores = model(**tokens).logits
    return scores.softmax(dim=-1)[0].tolist()

def probability_chart(chances):
    data = pd.DataFrame({
        "label": label_order,
        "probability": [chances[l] for l in label_order],
    })
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("label", sort=label_order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("probability", scale=alt.Scale(domain=[0, 1]), title="Probability"),
        tooltip=["label", alt.Tooltip("probability", format=".0%")],
    )

with app.form("review_form"):
    review = app.text_area("Paste a food review", height=150)
    submitted = app.form_submit_button("Predict")

if submitted:
    if not review.strip():
        app.warning("No input")
    else:
        groups = {
            "Classic models": {n: m for n, m in models.items() if m[0] == "classic"},
            "Modern models": {n: m for n, m in models.items() if m[0] != "classic"},
        }
        for group_title, group in groups.items():
            if not group:
                continue
            app.header(group_title)
            columns = app.columns(len(group))
            for column, (name, (kind, model, helper)) in zip(columns, group.items()):
                probs = predict(kind, model, helper, review)
                label = probs.index(max(probs))
                with column:
                    app.subheader(name)
                    app.metric("Prediction", label_names[label])
                    chances = {label_names[i]: p for i, p in enumerate(probs)}
                    app.altair_chart(probability_chart(chances))