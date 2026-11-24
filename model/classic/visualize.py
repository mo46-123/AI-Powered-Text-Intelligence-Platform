from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

SAMPLE_SIZE = 5000 # only part of the data, the full data can't run due to memory limit 
RANDOM_STATE = 46

PCA_COMPONENTS = 50  # intermediate dimensionality handed to t-SNE
PERPLEXITY = 30  # rough number of nearest neighbours each point attends to

LABEL_NAMES = {0: "poor", 1: "good", 2: "excellent"}
COLORS = {0: "#d62728", 1: "#7f7f7f", 2: "#2ca02c"}


def load_sample():
    data = pd.read_csv(REPO_ROOT / "data" / "clean_reviews.csv")
    if len(data) > SAMPLE_SIZE:
        data = data.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    return data.reset_index(drop=True)


def embed(x):

    x_scaled = StandardScaler().fit_transform(x)

    # PCA
    n_components = min(PCA_COMPONENTS, x_scaled.shape[0], x_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    x_pca = pca.fit_transform(x_scaled)
    variance = float(pca.explained_variance_ratio_.sum())

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    )
    coords = tsne.fit_transform(x_pca)
    return coords, n_components, variance


def plot(coords, labels, n_components, variance, n_features, output_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.65,
            color=COLORS.get(label, None),
            label=LABEL_NAMES.get(label, str(label)),
        )

    ax.set_title(f"PCA + t-SNE projection of {len(labels):,} reviews")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(title="Sentiment", markerscale=2)
    fig.text(
        0.5,
        0.01,
        f"PCA: {n_features:,} -> {n_components} dims ({variance:.1%} variance)  |  "
        f"t-SNE: {n_components} -> 2 dims (perplexity={PERPLEXITY}, "
        f"random_state={RANDOM_STATE})",
        ha="center",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    data = load_sample()

    tfidf = joblib.load(MODELS_DIR / "naive_bayes_vectorizer.joblib")
    x = tfidf.transform(data["input_text"]).toarray()

    coords, n_components, variance = embed(x)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "tsne.png"
    plot(
        coords,
        data["Sentiment"].to_numpy(),
        n_components,
        variance,
        x.shape[1],
        output_path,
    )

    print(f"Saved plot to {output_path}")
