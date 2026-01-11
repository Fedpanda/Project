"""
build_dataset_es_classic.py

Merge event-window outcomes with text features into one modeling dataset.

Design choice:
- events_es.parquet defines the sample (tweets with valid forward ES outcomes).
- Features are left-merged onto this base to avoid changing the sample when adding features.

Inputs:
- data_processed/events_es.parquet
- data_processed/features_text_stats.parquet
- data_processed/features_vader.parquet
- data_processed/features_topics_embed.parquet

Output:
- data_processed/dataset_es_classic.parquet
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


KEYS = ["post_id", "created_at"]


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    events_es = pd.read_parquet(root / "data_processed" / "events_es.parquet")
    feats_stats = pd.read_parquet(root / "data_processed" / "features_text_stats.parquet")
    feats_vader = pd.read_parquet(root / "data_processed" / "features_vader.parquet")
    feats_topics = pd.read_parquet(root / "data_processed" / "features_topics_embed.parquet")
    return events_es, feats_stats, feats_vader, feats_topics


def assign_regime(created_at: pd.Series) -> pd.Series:
    """
    Assign political regime based on Trump’s status.

    Regimes:
    - pre:    before campaign announcement (2009-01-01 – 2015-06-15)
    - power:  campaign + presidency (2015-06-16 – 2021-01-20)
    - post:   after presidency (2021-01-21 – )
    """
    dt = created_at.dt.tz_convert("UTC")

    pre_end = pd.Timestamp("2015-06-15", tz="UTC")
    power_end = pd.Timestamp("2021-01-20", tz="UTC")

    return pd.Series(
        pd.cut(
            dt,
            bins=[
                pd.Timestamp.min.tz_localize("UTC"),
                pre_end,
                power_end,
                pd.Timestamp.max.tz_localize("UTC"),
            ],
            labels=["pre", "power", "post"],
            right=True,
        ),
        index=created_at.index,
        name="regime",
    )


def _require_keys(df: pd.DataFrame, name: str) -> None:
    missing = set(KEYS) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing key columns: {sorted(missing)}")


def _coalesce_columns(df: pd.DataFrame, base: str) -> pd.DataFrame:
    """
    Ensure df has exactly one column named `base`.

    If `base` already exists, keep it.
    Otherwise, build it from `base_x`, `base_y` (or any base_* columns),
    taking first non-null across candidates.

    Then drop all duplicate-suffixed columns for that base.
    """
    candidates = []
    if base in df.columns:
        candidates.append(base)

    for c in [f"{base}_x", f"{base}_y"]:
        if c in df.columns and c not in candidates:
            candidates.append(c)

    for c in df.columns:
        if c.startswith(base + "_") and c not in candidates:
            candidates.append(c)

    if not candidates:
        return df

    if base not in df.columns:
        s = None
        for c in candidates:
            s = df[c] if s is None else s.combine_first(df[c])
        df[base] = s

    drop_cols = [
        c for c in df.columns
        if c != base and (c == f"{base}_x" or c == f"{base}_y" or c.startswith(base + "_"))
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def build_dataset(
    events_es: pd.DataFrame,
    feats_stats: pd.DataFrame,
    feats_vader: pd.DataFrame,
    feats_topics: pd.DataFrame,
) -> pd.DataFrame:
    # Key checks
    _require_keys(events_es, "events_es")
    _require_keys(feats_stats, "feats_stats")
    _require_keys(feats_vader, "feats_vader")
    _require_keys(feats_topics, "feats_topics")

    # Base sample = tweets with valid ES event windows
    df = events_es.copy()

    # Left-merge features (do not change sample)
    df = df.merge(feats_stats, on=KEYS, how="left", validate="one_to_one")
    df = df.merge(feats_vader, on=KEYS, how="left", validate="one_to_one")
    df = df.merge(feats_topics, on=KEYS, how="left", validate="one_to_one")

    # Coalesce platform / is_retweet (if duplicates were created by merges)
    df = _coalesce_columns(df, "platform")
    df = _coalesce_columns(df, "is_retweet")

    # Assign political regime
    df["regime"] = assign_regime(df["created_at"])

    # Targets (main horizons)
    df["es_up_30m"] = (df["es_ret_30m"] > 0).astype("int8")
    df["es_up_60m"] = (df["es_ret_60m"] > 0).astype("int8")

    # Sanity checks for magnitude targets
    if "es_absret_30m" not in df.columns or "es_absret_60m" not in df.columns:
        raise ValueError("Expected es_absret_30m and es_absret_60m in events_es / merged dataset.")

    # Final cleanup: ensure no accidental platform_* or is_retweet_* remain
    extra_platform = [c for c in df.columns if c.startswith("platform_")]
    extra_isrt = [c for c in df.columns if c.startswith("is_retweet_")]
    df = df.drop(columns=extra_platform + extra_isrt, errors="ignore")

    return df


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    out_path = PROJECT_ROOT / "data_processed" / "dataset_es_classic.parquet"

    events_es, feats_stats, feats_vader, feats_topics = load_inputs(PROJECT_ROOT)
    df = build_dataset(events_es, feats_stats, feats_vader, feats_topics)
    save_dataset(df, out_path)

    print("Saved:", out_path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # Guarantee no duplicate merge columns remain
    dup_cols = [
        c for c in df.columns
        if c in ("platform_x", "platform_y", "is_retweet_x", "is_retweet_y")
        or c.startswith("platform_") or c.startswith("is_retweet_")
    ]
    if dup_cols:
        raise RuntimeError(f"Duplicate columns still present (should never happen): {dup_cols}")

    # Coverage check for topic fields
    if "top_topic" in df.columns:
        print("\nTop topic value counts (including NaN):")
        print(df["top_topic"].value_counts(dropna=False).head(10))

    print("\nClass balance (es_up_30m):")
    print(df["es_up_30m"].value_counts(normalize=True))
    print("\nClass balance (es_up_60m):")
    print(df["es_up_60m"].value_counts(normalize=True))

    print("\nRegime counts:")
    print(df["regime"].value_counts(dropna=False))
    print("\nRegime date ranges:")
    print(df.groupby("regime")["created_at"].agg(["min", "max"]))



