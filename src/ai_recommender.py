"""
AI-powered recommendation system using NLP embeddings
This is the main system used in the final application
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Resolve project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "content_enriched.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Combine text fields
df["text"] = df["title"] + " " + df["description"]

# Load pre-trained AI embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for all content
content_embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True
)

def explain_recommendation(query, description):
    return f"Recommended because it aligns with themes like {description.replace(' ', ', ')}"

def recommend(query, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, content_embeddings)[0]

    top_indices = similarities.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["score"] = similarities[top_indices]

    results["explanation"] = results["description"].apply(
        lambda x: explain_recommendation(query, x)
    )

    return results

if __name__ == "__main__":
    query = "space travel with emotional story"
    print("User Query:", query)
    print("\nAI Recommendations:")
    print(recommend(query))