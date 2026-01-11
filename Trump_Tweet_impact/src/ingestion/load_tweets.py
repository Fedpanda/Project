"""
load_tweets.py

Ingestion module for Trump social-media posts.
Loads and normalizes Twitter (CSV) and Truth Social (JSON) data
into a single DataFrame.

Schema (canonical output):
- post_id: str
- created_at: timezone-aware datetime (UTC)
- text: str
- platform: str ('twitter' or 'truth_social')
- is_retweet: bool

Important:
- We DROP posts with blank/whitespace-only text at ingestion.
  Rationale: images/videos are not available in the dataset and blank text is unusable
  for EDA and text features. Keeping them creates noise and confusion downstream.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List
import html
import pandas as pd


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """
    Remove HTML tags from text. Keep punctuation/emojis/casing as-is.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = html.unescape(text)   # converts &amp; → &
    return text


def _drop_blank_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop posts with no usable text (NaN, empty string, whitespace-only).
    """
    if "text" not in df.columns:
        return df
    s = df["text"].fillna("").astype(str)
    keep = s.str.strip().ne("")
    return df.loc[keep].copy()


def _load_twitter_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load Trump Archive Twitter CSV (2009–2021) and normalize schema.
    """
    df = pd.read_csv(
        csv_path,
        dtype={"id": str},  # critical: avoid float precision loss
    )

    # Rename and select relevant columns
    df = df.rename(
        columns={
            "id": "post_id",
            "date": "created_at",
            "isRetweet": "is_retweet",
            "text": "text",
        }
    )

    df = df[["post_id", "created_at", "text", "is_retweet"]]

    # Parse timestamps (UTC, tz-aware)
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
        utc=True,
    )

    # Normalize boolean (Trump Archive sometimes uses 't'/'f')
    df["is_retweet"] = df["is_retweet"].astype(str).str.lower().eq("t")

    # Clean HTML just in case (harmless if none)
    df["text"] = df["text"].apply(_strip_html)

    df["platform"] = "twitter"
    return df


def _load_truth_social_json(json_path: Path) -> pd.DataFrame:
    """
    Load one Truth Social JSON batch and normalize schema.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)

    df = df.rename(
        columns={
            "id": "post_id",
            "date": "created_at",
            "isRetweet": "is_retweet",
            "text": "text",
        }
    )

    df = df[["post_id", "created_at", "text", "is_retweet"]]

    # Convert epoch milliseconds to UTC datetime
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        unit="ms",
        errors="coerce",
        utc=True,
    )

    # Clean HTML from text
    df["text"] = df["text"].apply(_strip_html)

    # Ensure boolean
    # (Truth Social export may already be boolean; this is safe)
    df["is_retweet"] = df["is_retweet"].astype(bool)

    df["platform"] = "truth_social"
    return df


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_all_posts(raw_data_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate all Trump social-media posts.

    Parameters
    ----------
    raw_data_dir : Path
        Root directory containing raw Trump Archive data.

    Returns
    -------
    pd.DataFrame
        Canonical DataFrame with all posts, sorted by timestamp, with blank-text dropped.
    """
    raw_data_dir = Path(raw_data_dir)

    # ---- Twitter CSV ----
    twitter_csv = raw_data_dir / "trump_archive" / "twitter_2009_2021.csv"
    twitter_df = _load_twitter_csv(twitter_csv)

    # ---- Truth Social JSON batches ----
    truth_dir = raw_data_dir / "trump_archive" / "truth_social"
    json_files: List[Path] = sorted(truth_dir.glob("*.json"))

    truth_dfs = [_load_truth_social_json(jf) for jf in json_files]
    truth_df = pd.concat(truth_dfs, ignore_index=True) if truth_dfs else pd.DataFrame(
        columns=["post_id", "created_at", "text", "is_retweet", "platform"]
    )

    # ---- Combine ----
    all_posts = pd.concat([twitter_df, truth_df], ignore_index=True)

    # Drop rows with invalid timestamps
    all_posts = all_posts.dropna(subset=["created_at"])

    # Drop blank/whitespace-only text posts (image/video-only in our dataset)
    before = len(all_posts)
    all_posts = _drop_blank_text(all_posts)
    after = len(all_posts)

    # Sort chronologically
    all_posts = all_posts.sort_values("created_at").reset_index(drop=True)

    # Optional sanity prints (comment out once verified)
    print(f"[load_all_posts] Blank-text removed: {before - after} (kept {after})")
    print("[load_all_posts] platform counts:")
    print(all_posts["platform"].value_counts(dropna=False))

    return all_posts


# ---------------------------------------------------------------------
# Optional manual test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    root = Path("../../data_raw")
    df = load_all_posts(root)
    print(df.head())
    print(df.tail())
    print("Empty text check (should be False only):")
    print(df["text"].fillna("").astype(str).str.strip().eq("").value_counts())
