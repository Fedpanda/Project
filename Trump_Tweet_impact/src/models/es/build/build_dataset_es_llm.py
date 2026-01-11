"""
build_dataset_es_llm.py

Build modeling dataset with LLM features merged onto the baseline dataset.

Design choice:
- dataset_es_classic.parquet defines the base sample (tweets with valid ES outcomes).
- LLM features are LEFT-merged on post_id to avoid changing the sample.
- We do NOT overwrite dataset_es_classic.parquet; we write dataset_es_llm.parquet.

Inputs:
- data_processed/dataset_es_classic.parquet
- data_processed/features_llm_all.parquet   (must contain: post_id + LLM feature columns)

Output:
- data_processed/dataset_es_llm.parquet
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


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


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    base = pd.read_parquet(root / "data_processed" / "dataset_es_classic.parquet")
    llm = pd.read_parquet(root / "data_processed" / "features_llm_all.parquet")
    return base, llm


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def build_dataset_llm(base: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    _require_cols(base, ["post_id"], "dataset_model")
    _require_cols(llm, ["post_id"], "features_llm_all")

    # Keep only expected LLM cols (ignore anything else)
    keep = [c for c in LLM_COLS if c in llm.columns]
    _require_cols(llm, keep, "features_llm_all")

    llm = llm[keep].copy()

    # Ensure join key type
    base = base.copy()
    base["post_id"] = base["post_id"].astype(str)
    llm["post_id"] = llm["post_id"].astype(str)

    # Enforce unique key in LLM features
    if llm["post_id"].duplicated().any():
        dup = llm.loc[llm["post_id"].duplicated(), "post_id"].iloc[0]
        raise ValueError(f"features_llm_all has duplicate post_id (example): {dup}")

    # LEFT MERGE: keep base sample unchanged
    out = base.merge(llm, on="post_id", how="left", validate="one_to_one")

    # Optional: strong typing (so downstream models don't choke)
    int_cols = [
        "market_relevance",
        "uncertainty_shock",
        "policy_surprise",
        "novelty",
        "tone_valence",
        "tail_risk_severity",
    ]
    for c in int_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    if "market_topic" in out.columns:
        out["market_topic"] = out["market_topic"].astype("string")

    return out


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    out_path = PROJECT_ROOT / "data_processed" / "dataset_es_llm.parquet"

    base, llm = load_inputs(PROJECT_ROOT)
    df = build_dataset_llm(base, llm)
    save_dataset(df, out_path)

    print("Saved:", out_path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # Coverage diagnostics
    print("\nLLM feature coverage (non-null rate):")
    for c in ["market_relevance", "market_topic", "novelty"]:
        if c in df.columns:
            print(f"  {c}: {(df[c].notna().mean() * 100):.2f}%")

    if "market_topic" in df.columns:
        print("\nmarket_topic value counts (including NaN):")
        print(df["market_topic"].value_counts(dropna=False).head(15))
