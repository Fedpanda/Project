"""
vader_sentiment.py

Compute VADER sentiment scores for each post text.

Input:
- data_processed/events_es.parquet

Output:
- data_processed/features_vader.parquet

# ===========================
# YOUR COMMENTS HERE
# - Explain why lexicon-based sentiment is a strong baseline for political text
# - Mention limitations (sarcasm, domain mismatch)
# ===========================
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def _ensure_vader():
    """
    Ensure NLTK VADER resources exist. Downloads if missing.
    """
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def compute_vader(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Compute VADER sentiment scores.

    Required columns:
    - post_id
    - created_at
    - text
    """
    required = {"post_id", "created_at", text_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input df missing required columns: {sorted(missing)}")

    _ensure_vader()

    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()

    def score(text: str):
        if text is None or (isinstance(text, float) and pd.isna(text)):
            text = ""
        s = sia.polarity_scores(str(text))
        return s["neg"], s["neu"], s["pos"], s["compound"]

    scores = df[text_col].map(score)

    out = pd.DataFrame(
        {
            "post_id": df["post_id"].astype(str),
            "created_at": df["created_at"],
            "vader_neg": scores.map(lambda x: x[0]).astype("float32"),
            "vader_neu": scores.map(lambda x: x[1]).astype("float32"),
            "vader_pos": scores.map(lambda x: x[2]).astype("float32"),
            "vader_compound": scores.map(lambda x: x[3]).astype("float32"),
        }
    )

    return out


def save_vader(features: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)


if __name__ == "__main__":
    """
    PyCharm-friendly runner.
    Reads events_es.parquet and writes features_vader.parquet.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    IN_PATH = PROJECT_ROOT / "data_processed" / "events_es.parquet"
    OUT_PATH = PROJECT_ROOT / "data_processed" / "features_vader.parquet"

    events_es = pd.read_parquet(IN_PATH)

    feats = compute_vader(events_es, text_col="text")
    save_vader(feats, OUT_PATH)

    print("Saved:", OUT_PATH)
    print("Rows:", len(feats))
    print("Columns:", list(feats.columns))
    print(feats.head())
