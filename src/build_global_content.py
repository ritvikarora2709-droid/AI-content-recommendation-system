import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# =========================
# Load datasets
# =========================
tmdb = pd.read_csv(DATA_DIR / "tmdb_movies.csv")
bollywood = pd.read_csv(DATA_DIR / "IMDB-Movie-Dataset(2023-1951).csv")

# =========================
# TMDB CLEANING
# =========================
tmdb_clean = pd.DataFrame()

tmdb_clean["title"] = tmdb["title"]
tmdb_clean["description"] = tmdb["overview"].fillna("")
tmdb_clean["genres"] = ""  # TMDB dataset has no genres
tmdb_clean["extra"] = (
    "Language: " + tmdb["original_language"].fillna("") +
    " Popularity: " + tmdb["popularity"].astype(str)
)
tmdb_clean["source"] = "TMDB"

tmdb_clean.dropna(subset=["title"], inplace=True)

# =========================
# BOLLYWOOD / IMDB CLEANING
# =========================
bollywood_clean = pd.DataFrame()

bollywood_clean["title"] = bollywood["movie_name"]
bollywood_clean["description"] = bollywood["overview"].fillna("")
bollywood_clean["genres"] = bollywood["genre"].fillna("")
bollywood_clean["extra"] = (
    "Director: " + bollywood["director"].fillna("") +
    " Cast: " + bollywood["cast"].fillna("")
)
bollywood_clean["source"] = "Bollywood_IMDB"

bollywood_clean.dropna(subset=["title"], inplace=True)

# =========================
# COMBINE DATASETS
# =========================
combined = pd.concat([tmdb_clean, bollywood_clean], ignore_index=True)

# Unified text field for embeddings
combined["text"] = (
    combined["title"].astype(str) + " " +
    combined["genres"].astype(str) + " " +
    combined["description"].astype(str) + " " +
    combined["extra"].astype(str)
)

# Save final dataset
OUTPUT_PATH = DATA_DIR / "content_global.csv"
combined.to_csv(OUTPUT_PATH, index=False)

print("✅ Global dataset created successfully")
print("📊 Total movies:", len(combined))
print("📁 Saved to:", OUTPUT_PATH)
