"""
text_stats.py

Baseline, interpretable text features extracted from post text.

Input:
- data_processed/events_es.parquet  (we use ES events as the canonical row set)

Output:
- data_processed/features_text_stats.parquet

Notes:
- We keep this deliberately simple and fast.
- These are "baseline" features before sentiment/embeddings.

# ===========================
# YOUR COMMENTS HERE
# - Explain why simple text stats are useful as baselines
# - Mention interpretability and speed
# ===========================
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#\w+")


def _safe_text(x) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def compute_text_stats(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Compute baseline text features.

    Required columns in df:
    - post_id
    - created_at
    - text
    Optionally used (if present):
    - is_retweet
    - platform
    """
    required = {"post_id", "created_at", text_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input df missing required columns: {sorted(missing)}")

    s = df[text_col].map(_safe_text)

    # Basic counts
    n_chars = s.str.len()

    # Word count: split on whitespace after stripping
    n_words = s.str.strip().str.split().map(len)

    # Punctuation emphasis
    n_exclam = s.str.count("!")
    n_question = s.str.count(r"\?")

    # Uppercase share: ratio of uppercase letters to all alphabetic letters
    def upper_share(text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        upper = sum(1 for c in letters if c.isupper())
        return upper / len(letters)

    share_upper = s.map(upper_share)

    # Simple flags
    has_url = s.map(lambda t: bool(_URL_RE.search(t))).astype("int8")
    has_hashtag = s.map(lambda t: bool(_HASHTAG_RE.search(t))).astype("int8")

    out = pd.DataFrame(
        {
            "post_id": df["post_id"].astype(str),
            "created_at": df["created_at"],
            "n_chars": n_chars.astype("int32"),
            "n_words": n_words.astype("int32"),
            "n_exclam": n_exclam.astype("int16"),
            "n_question": n_question.astype("int16"),
            "share_upper": share_upper.astype("float32"),
            "has_url": has_url,
            "has_hashtag": has_hashtag,
        }
    )

    # Carry through helpful identifiers if present (not strictly "features", but useful keys)
    if "platform" in df.columns:
        out["platform"] = df["platform"].astype(str)
    if "is_retweet" in df.columns:
        out["is_retweet"] = df["is_retweet"].astype(bool)

    return out


def save_text_stats(features: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)


if __name__ == "__main__":
    """
    PyCharm-friendly runner.
    Reads events_es.parquet and writes features_text_stats.parquet.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    IN_PATH = PROJECT_ROOT / "data_processed" / "events_es.parquet"
    OUT_PATH = PROJECT_ROOT / "data_processed" / "features_text_stats.parquet"

    events_es = pd.read_parquet(IN_PATH)

    feats = compute_text_stats(events_es, text_col="text")
    save_text_stats(feats, OUT_PATH)

    print("Saved:", OUT_PATH)
    print("Rows:", len(feats))
    print("Columns:", list(feats.columns))
    print(feats.head())