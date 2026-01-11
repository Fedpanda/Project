"""
build_dataset_vx_llm.py

Merge VX event-window outcomes with text stats + controls + LLM features into one modeling dataset.

Inputs:
- data_processed/events_vx.parquet
- data_processed/features_text_stats.parquet
- data_processed/features_vader.parquet        (loaded only to preserve compatibility; not required for LLM models)
- data_processed/features_topics_embed.parquet (optional; loaded only to preserve compatibility)
- data_processed/features_llm_all.parquet      (LLM features by post_id)

Output:
- data_processed/dataset_vx_llm.parquet

Notes:
- VX outcomes can be missing (NaN) for some horizons. Keep NaNs.
  Model scripts must dropna() on the specific target they use.
- We merge LLM features on post_id (not created_at) because your LLM file contains post_id only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


KEYS = ["post_id", "created_at"]

LLM_COLS = [
    "post_id",
    "market_relevance",
    "market_topic",
    "uncertainty_shock",
    "policy_surprise",
    "novelty",
    "tone_valence",
    "tail_risk_severity",
]


def assign_regime(created_at: pd.Series) -> pd.Series:
    dt = created_at.dt.tz_convert("UTC")
    pre_end = pd.Timestamp("2015-06-15", tz="UTC")
    power_end = pd.Timestamp("2021-01-20", tz="UTC")

    return pd.cut(
        dt,
        bins=[
            pd.Timestamp.min.tz_localize("UTC"),
            pre_end,
            power_end,
            pd.Timestamp.max.tz_localize("UTC"),
        ],
        labels=["pre", "power", "post"],
        right=True,
    )


def _coalesce_columns(df: pd.DataFrame, base: str) -> pd.DataFrame:
    candidates: List[str] = []
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
    return df.drop(columns=drop_cols, errors="ignore")


def load_inputs(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    root = Path(root)

    events_vx = pd.read_parquet(root / "data_processed" / "events_vx.parquet")
    feats_stats = pd.read_parquet(root / "data_processed" / "features_text_stats.parquet")

    # Keep these to preserve compatibility / allow later comparisons, even if not used in LLM-only models.
    feats_vader = pd.read_parquet(root / "data_processed" / "features_vader.parquet")
    topics_path = root / "data_processed" / "features_topics_embed.parquet"
    feats_topics = pd.read_parquet(topics_path) if topics_path.exists() else None

    llm_path = root / "data_processed" / "features_llm_all.parquet"
    llm = pd.read_parquet(llm_path)

    return events_vx, feats_stats, feats_vader, feats_topics, llm


def build_dataset_vx_llm(
    events_vx: pd.DataFrame,
    feats_stats: pd.DataFrame,
    feats_vader: pd.DataFrame,
    feats_topics: Optional[pd.DataFrame],
    llm: pd.DataFrame,
) -> pd.DataFrame:
    # Key checks
    for dfx, name in [
        (events_vx, "events_vx"),
        (feats_stats, "features_text_stats"),
        (feats_vader, "features_vader"),
    ]:
        missing = set(KEYS) - set(dfx.columns)
        if missing:
            raise ValueError(f"{name} missing key columns: {sorted(missing)}")

    # Base merge: keep the exact VX sample definition you had before (inner on KEYS)
    df = events_vx.merge(feats_stats, on=KEYS, how="inner", validate="one_to_one")

    # Optional (for comparison later): bring VADER / topics through, but you can ignore them in models
    df = df.merge(feats_vader, on=KEYS, how="inner", validate="one_to_one")
    if feats_topics is not None:
        missing = set(KEYS) - set(feats_topics.columns)
        if missing:
            raise ValueError(f"features_topics_embed missing key columns: {sorted(missing)}")
        df = df.merge(feats_topics, on=KEYS, how="left", validate="one_to_one")

    # Coalesce duplicates
    df = _coalesce_columns(df, "platform")
    df = _coalesce_columns(df, "is_retweet")
    df = _coalesce_columns(df, "text")

    # Ensure types
    df["post_id"] = df["post_id"].astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"]).copy()

    # LLM: keep only expected cols, merge on post_id (left merge; do not change sample)
    if "post_id" not in llm.columns:
        raise ValueError("features_llm_all.parquet must contain post_id")

    llm = llm.copy()
    llm["post_id"] = llm["post_id"].astype(str)

    # enforce uniqueness
    if llm["post_id"].duplicated().any():
        example = llm.loc[llm["post_id"].duplicated(), "post_id"].iloc[0]
        raise ValueError(f"features_llm_all has duplicate post_id (example): {example}")

    keep = [c for c in LLM_COLS if c in llm.columns]
    missing_llm = set(LLM_COLS) - set(keep)
    if missing_llm:
        raise ValueError(f"features_llm_all missing expected columns: {sorted(missing_llm)}")

    llm = llm[keep]

    df = df.merge(llm, on="post_id", how="left", validate="many_to_one")

    # Regime
    df["regime"] = assign_regime(df["created_at"])

    # Sanity check: ensure VX outcome columns exist (at least the 30/60 you care about)
    needed = ["vx_ret_30m", "vx_ret_60m", "vx_absret_30m", "vx_absret_60m"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing VX outcome columns in merged dataset: {miss}")

    return df


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    out_path = PROJECT_ROOT / "data_processed" / "dataset_vx_llm.parquet"

    events_vx, feats_stats, feats_vader, feats_topics, llm = load_inputs(PROJECT_ROOT)
    df = build_dataset_vx_llm(events_vx, feats_stats, feats_vader, feats_topics, llm)
    save_dataset(df, out_path)

    print("Saved:", out_path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nVX outcome availability (share observed):")
    for c in ["vx_ret_30m", "vx_ret_60m", "vx_absret_30m", "vx_absret_60m"]:
        print(f"{c}: {(df[c].notna().mean() * 100):.2f}%")

    print("\nLLM coverage (share observed):")
    for c in ["market_relevance", "market_topic", "uncertainty_shock", "novelty"]:
        if c in df.columns:
            print(f"{c}: {(df[c].notna().mean() * 100):.2f}%")

    print("\nRegime counts:")
    print(df["regime"].value_counts(dropna=False))
