"""
topic_embeddings.py

Embedding-based multi-label topic assignment for Trump posts.

Idea:
- Represent each post and each topic description as a sentence embedding.
- Compute cosine similarity between post embedding and each topic embedding.
- Convert similarities into:
  - per-topic similarity scores
  - per-topic binary flags (similarity >= threshold)
  - top-1 topic + score
  - number of topics per post

Model:
- sentence-transformers/all-MiniLM-L6-v2

Input:
- data_processed/dataset_es_classic.parquet   (must contain: post_id, created_at, text)

Output:
- data_processed/features_topics_embed.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# -----------------------
# Config
# -----------------------

ID_COL = "post_id"
DATE_COL = "created_at"
TEXT_COL = "text"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Topic definitions (you can cite these in your paper as the topic ontology)
TOPICS: Dict[str, str] = {
    "stock_market": (
        "Tweets discussing stock prices, equity markets (e.g., the S&P 500), "
        "market rallies or crashes, investor sentiment, or investment booms and busts."
    ),
    "economy_macro": (
        "Tweets related to the overall economy, including economic growth or recession, "
        "unemployment and jobs, inflation, and monetary policy such as actions by the Federal Reserve."
    ),
    "immigration": (
        "Tweets discussing immigration policy, border security, the border wall, illegal immigration, "
        "or related enforcement and migration issues."
    ),
    "government_domestic": (
        "Tweets referring to the U.S. federal or state government, politicians, political institutions, "
        "Congress, courts, or domestic political decision-making."
    ),
    "foreign_policy": (
        "Tweets discussing foreign countries or international relations, including trade policy and tariffs, "
        "diplomacy, conflicts, or relations with countries such as China, Russia, or Venezuela."
    ),
    "media": (
        "Tweets discussing news media and journalists, including references to outlets such as CNN or Fox News, "
        "criticism of the press, or claims about fake or biased news coverage."
    ),
    "entertainment_celebrity": (
        "Tweets discussing entertainment and show business, including television, movies, books, "
        "celebrities, or cultural events and personalities."
    ),
}

# Similarity threshold for multi-label assignment
SIM_THRESHOLD = 0.45

# Embedding batch size (adjust if you run out of RAM)
BATCH_SIZE = 256


# -----------------------
# Helpers
# -----------------------

def _require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _normalize_text(s: pd.Series) -> pd.Series:
    # Minimal normalization; avoid heavy cleaning here
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 256) -> np.ndarray:
    """
    Return normalized embeddings for cosine similarity via dot product.
    normalize_embeddings=True makes cosine_similarity(u,v) == u @ v.
    """
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def compute_topic_features(
    df_posts: pd.DataFrame,
    topics: Dict[str, str],
    model_name: str = MODEL_NAME,
    sim_threshold: float = SIM_THRESHOLD,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Compute per-topic similarity scores and binary flags.

    Returns a DataFrame keyed by (post_id, created_at) with topic features.
    """
    _require_columns(df_posts, [ID_COL, DATE_COL, TEXT_COL], "df_posts")

    df = df_posts[[ID_COL, DATE_COL, TEXT_COL]].copy()
    df[TEXT_COL] = _normalize_text(df[TEXT_COL])

    # Drop blank text (should already be filtered upstream, but keep safe)
    df = df[df[TEXT_COL].ne("")].reset_index(drop=True)

    model = SentenceTransformer(model_name)

    # Topic embeddings
    topic_keys = list(topics.keys())
    topic_descs = [topics[k] for k in topic_keys]
    topic_emb = embed_texts(model, topic_descs, batch_size=min(batch_size, 64))  # small list
    # shape: (K, D)

    # Post embeddings
    post_texts = df[TEXT_COL].tolist()
    post_emb = embed_texts(model, post_texts, batch_size=batch_size)
    # shape: (N, D)

    # Cosine similarity (because both normalized): sim = dot product
    sims = post_emb @ topic_emb.T  # (N, K)

    # Build output
    out = df[[ID_COL, DATE_COL]].copy()

    # Per-topic score + flag
    for j, k in enumerate(topic_keys):
        score_col = f"topic_{k}_score"
        flag_col = f"topic_{k}_flag"
        out[score_col] = sims[:, j].astype(np.float32)
        out[flag_col] = (out[score_col] >= sim_threshold).astype("int8")

    score_cols = [f"topic_{k}_score" for k in topic_keys]
    flag_cols = [f"topic_{k}_flag" for k in topic_keys]

    # Top-1 topic and score
    top_idx = np.argmax(sims, axis=1)
    out["top_topic"] = [topic_keys[i] for i in top_idx]
    out["top_score"] = sims[np.arange(len(df)), top_idx].astype(np.float32)

    # Number of topics assigned (multi-label)
    out["n_topics"] = out[flag_cols].sum(axis=1).astype("int16")

    # Optional: also keep max score (same as top_score), but explicit name can help
    out["max_topic_similarity"] = out["top_score"]

    return out


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_es_classic.parquet"
    out_path = PROJECT_ROOT / "data_processed" / "features_topics_embed.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Expected {data_path}")

    df = pd.read_parquet(data_path)
    _require_columns(df, [ID_COL, DATE_COL, TEXT_COL], "dataset_model")

    feats = compute_topic_features(
        df_posts=df,
        topics=TOPICS,
        model_name=MODEL_NAME,
        sim_threshold=SIM_THRESHOLD,
        batch_size=BATCH_SIZE,
    )

    feats.to_parquet(out_path, index=False)

    # Minimal sanity prints
    print("Saved:", out_path)
    print("Rows:", len(feats))
    print("Columns:", len(feats.columns))

    # Coverage diagnostics
    flag_cols = [c for c in feats.columns if c.endswith("_flag")]
    coverage = feats[flag_cols].mean().sort_values(ascending=False)
    print("\n=== Topic coverage (share of posts flagged) ===")
    print(coverage.to_string())

    print("\n=== n_topics distribution ===")
    print(feats["n_topics"].value_counts().sort_index().head(15).to_string())
