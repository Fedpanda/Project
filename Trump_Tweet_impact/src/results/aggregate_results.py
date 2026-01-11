"""
aggregate_results.py

Aggregate per-script model results into master tables for reporting.

Update vs previous
------------------
Adds support for XGBoost runs (model="xgb") by unifying metrics from:
- logit_acc/logit_auc
- rf_acc/rf_auc
- xgb_acc/xgb_auc

Everything else stays the same.

Expected inputs
---------------
Each model script should save a tidy results parquet into:
    <PROJECT_ROOT>/results/runs/*.parquet

Each file should contain (at minimum) these columns:
- market: "es" or "vx"
- feature_track: "classic" or "llm"
- model: "logit" or "rf" or "xgb"
- analysis: "baseline" or "ablation"
- regime: "pre"|"power"|"post" (or NaN for pooled)
- target: e.g. "es_up_30m", "vx_spike_60m"
- spec (for baseline) OR feature_set (for ablation)
- n_train, n_test
- baseline_acc
- plus model metrics in one of these pairs:
    - logit_acc, logit_auc
    - rf_acc, rf_auc
    - xgb_acc, xgb_auc

Outputs
-------
Writes to:
    <PROJECT_ROOT>/results/tables/all_results.parquet
    <PROJECT_ROOT>/results/tables/baselines.parquet
    <PROJECT_ROOT>/results/tables/ablations.parquet
    <PROJECT_ROOT>/results/tables/best_by_auc.parquet
    <PROJECT_ROOT>/results/tables/ablation_top_singles_by_auc.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


# -----------------------
# Paths
# -----------------------

def project_root() -> Path:
    # .../src/results/aggregate_results.py -> root = parents[2]
    return Path(__file__).resolve().parents[2]


def _discover_run_files(runs_dir: Path) -> List[Path]:
    files = sorted(runs_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet results found in: {runs_dir}")
    return files


# -----------------------
# Normalization helpers
# -----------------------

def _ensure_cols(df: pd.DataFrame, cols: List[str], fname: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{fname} missing required columns: {missing}")


def _coerce_str(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].astype("string")


def _coerce_numeric(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _unify_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unified columns:
      - acc
      - auc
    from model-specific metric columns.
    Keeps original columns intact too.
    """
    df = df.copy()

    # Accuracy
    if "acc" not in df.columns:
        for c in ["logit_acc", "rf_acc", "xgb_acc"]:
            if c in df.columns:
                df["acc"] = df[c]
                break

    # AUC
    if "auc" not in df.columns:
        for c in ["logit_auc", "rf_auc", "xgb_auc"]:
            if c in df.columns:
                df["auc"] = df[c]
                break

    _coerce_numeric(df, "acc")
    _coerce_numeric(df, "auc")
    return df


def _add_spec_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provide a single column 'spec_or_feature_set' that is always present:
      - baseline: spec
      - ablation: feature_set
    """
    df = df.copy()
    if "spec" in df.columns:
        df["spec_or_feature_set"] = df["spec"]
    elif "feature_set" in df.columns:
        df["spec_or_feature_set"] = df["feature_set"]
    else:
        df["spec_or_feature_set"] = pd.NA
    _coerce_str(df, "spec_or_feature_set")
    return df


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # minimal common requirements
    _ensure_cols(
        df,
        ["market", "feature_track", "model", "analysis", "regime", "target", "n_train", "n_test", "baseline_acc"],
        fname=path.name,
    )

    # baseline vs ablation identifier presence
    analysis_lower = df["analysis"].astype(str).str.lower()
    if analysis_lower.eq("baseline").any():
        if "spec" not in df.columns and "spec_or_feature_set" not in df.columns:
            raise ValueError(f"{path.name}: analysis=baseline but no 'spec' column found.")
    if analysis_lower.eq("ablation").any():
        if "feature_set" not in df.columns and "spec_or_feature_set" not in df.columns:
            raise ValueError(f"{path.name}: analysis=ablation but no 'feature_set' column found.")

    # types: strings
    for c in ["market", "feature_track", "model", "analysis", "regime", "target"]:
        _coerce_str(df, c)

    # types: numerics
    numeric_cols = [
        "n_train", "n_test", "baseline_acc",
        "logit_acc", "logit_auc",
        "rf_acc", "rf_auc",
        "xgb_acc", "xgb_auc",
        "acc", "auc",
    ]
    for c in numeric_cols:
        _coerce_numeric(df, c)

    # attach source
    df["source_file"] = path.name
    _coerce_str(df, "source_file")

    # unify metrics and spec name
    df = _unify_metric_columns(df)
    df = _add_spec_col(df)

    return df


# -----------------------
# Aggregations
# -----------------------

def _best_by_auc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each (market, feature_track, model, analysis, regime, target),
    pick the row with max AUC (ties broken by acc, then n_test).
    """
    key_cols = ["market", "feature_track", "model", "analysis", "regime", "target"]
    df2 = df.dropna(subset=["auc"]).copy()

    # stable sort for tie-breakers
    df2 = df2.sort_values(
        key_cols + ["auc", "acc", "n_test"],
        ascending=[True] * len(key_cols) + [False, False, False],
    )

    best = df2.groupby(key_cols, as_index=False).head(1).copy()
    return best


def _ablation_top_singles(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    For ablation runs only: keep SINGLE:* feature sets, return top K by AUC
    per (market, feature_track, model, regime, target).
    """
    ab = df[df["analysis"].str.lower() == "ablation"].copy()
    if ab.empty:
        return ab

    name_col = "feature_set" if "feature_set" in ab.columns else "spec_or_feature_set"
    ab = ab[ab[name_col].astype(str).str.startswith("SINGLE:")].copy()
    if ab.empty:
        return ab

    key = ["market", "feature_track", "model", "regime", "target"]
    ab = ab.dropna(subset=["auc"]).sort_values(key + ["auc"], ascending=[True] * len(key) + [False])
    return ab.groupby(key, as_index=False).head(top_k).copy()


# -----------------------
# Main
# -----------------------

def main() -> None:
    root = project_root()
    runs_dir = root / "results" / "runs"
    out_dir = root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _discover_run_files(runs_dir)

    dfs = []
    for p in files:
        dfs.append(_read_one(p))

    all_results = pd.concat(dfs, ignore_index=True)

    # split views
    baselines = all_results[all_results["analysis"].str.lower() == "baseline"].copy()
    ablations = all_results[all_results["analysis"].str.lower() == "ablation"].copy()

    best_by_auc = _best_by_auc(all_results)
    ablation_top_singles = _ablation_top_singles(all_results, top_k=10)

    # write outputs
    all_results_path = out_dir / "all_results.parquet"
    baselines_path = out_dir / "baselines.parquet"
    ablations_path = out_dir / "ablations.parquet"
    best_path = out_dir / "best_by_auc.parquet"
    singles_path = out_dir / "ablation_top_singles_by_auc.parquet"

    all_results.to_parquet(all_results_path, index=False)
    baselines.to_parquet(baselines_path, index=False)
    ablations.to_parquet(ablations_path, index=False)
    best_by_auc.to_parquet(best_path, index=False)
    ablation_top_singles.to_parquet(singles_path, index=False)

    # quick console summary
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 140)

    print(f"[ok] Loaded run files: {len(files)} from {runs_dir}")
    print(f"[ok] All rows: {len(all_results)}")
    print(f"[ok] Baseline rows: {len(baselines)} | Ablation rows: {len(ablations)}")
    print(f"[ok] Wrote: {all_results_path}")
    print(f"[ok] Wrote: {baselines_path}")
    print(f"[ok] Wrote: {ablations_path}")
    print(f"[ok] Wrote: {best_path}")
    print(f"[ok] Wrote: {singles_path}")

    print("\n=== Best-by-AUC (one per market/track/model/analysis/regime/target) ===")
    show_cols = [
        "market", "feature_track", "model", "analysis", "regime", "target",
        "spec_or_feature_set", "n_test", "baseline_acc", "acc", "auc", "source_file",
    ]
    keep = [c for c in show_cols if c in best_by_auc.columns]
    print(
        best_by_auc.sort_values(["market", "feature_track", "model", "analysis", "regime", "target"])[keep]
        .to_string(index=False)
    )

    if not ablation_top_singles.empty:
        print("\n=== Ablation: top SINGLE features by AUC (top 10 per group) ===")
        keep2 = [c for c in show_cols if c in ablation_top_singles.columns]
        print(
            ablation_top_singles.sort_values(
                ["market", "feature_track", "model", "regime", "target", "auc"],
                ascending=[True, True, True, True, True, False],
            )[keep2]
            .head(80)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
