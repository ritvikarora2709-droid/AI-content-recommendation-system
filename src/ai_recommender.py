"""
AI-powered recommendation system using NLP embeddings
Uses a global dataset (TMDB + Bollywood IMDB)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "content_global.csv"

# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH)

# Safety: fill missing fields
for col in ["title", "description", "genres", "extra", "source", "text"]:
    if col in df.columns:
        df[col] = df[col].fillna("")

# =========================
# Load embedding model (cached by Streamlit later)
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# Generate embeddings (once)
# =========================
content_embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True
)

# =========================
# Explanation helper
# =========================
def explain_recommendation(query, row):
    reasons = []

    if row["genres"]:
        reasons.append("genre match")
    if row["source"] == "Bollywood_IMDB":
        reasons.append("Indian cinema")
    if row["source"] == "TMDB":
        reasons.append("popular global cinema")

    if not reasons:
        return f"Semantically similar to your query: '{query}'"

    return f"Recommended due to {', '.join(reasons)}"

# =========================
# Main recommend function
# =========================
def recommend(query, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, content_embeddings)[0]

    top_indices = similarities.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["score"] = similarities[top_indices]

    results["explanation"] = results.apply(
        lambda row: explain_recommendation(query, row),
        axis=1
    )

    return results[[
        "title",
        "description",
        "genres",
        "source",
        "score",
        "explanation"
    ]]

# =========================
# Local test
# =========================
#if __name__ == "__main__":
 #   q = "an epic fantasy adventure"
  #  print(recommend(q))
