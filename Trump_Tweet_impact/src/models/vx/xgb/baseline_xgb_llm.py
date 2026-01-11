"""
baseline_xgb_vx_llm.py

XGBoost benchmarks for VX volatility spikes after posts, using LLM-derived features.

Input:
- data_processed/dataset_vx_llm.parquet

Targets (constructed WITHOUT leakage and SPEC-INVARIANT across feature specs)
----------------------------------------------------------------------------
For horizon h in {30m, 60m}:
- vx_spike_h = 1{ vx_absret_h > Q_train(q) }, else 0

IMPORTANT:
- The spike threshold Q_train(q) is computed using ONLY the TRAIN split AND using ONLY vx_absret_h
  (i.e., NOT after dropping rows due to missing features). This keeps the target definition
  comparable across specs (stats_only vs llm_only vs ...).

Validation:
- within each regime, sort by created_at
- train on first 70%, test on last 30% (time-respecting split)

Feature specs:
1) stats_only:
   - text stats + controls (platform, is_retweet)
2) llm_only:
   - LLM numeric/binary + controls + market_topic
3) stats_plus_llm:
   - text stats + LLM numeric/binary + controls
4) stats_plus_llm_plus_topic:
   - stats_plus_llm + market_topic

Outputs:
- spike threshold (train quantile)
- test spike rate
- baseline accuracy (majority class on test)
- XGB accuracy and ROC-AUC

Notes:
- AUC is usually more informative than accuracy here because spikes are rare.
- XGBoost is flexible; keep hyperparameters conservative to reduce overfitting.
- This file saves results to results/runs for downstream aggregation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except Exception as e:  # pragma: no cover
    raise ImportError(
        "xgboost is not installed. Install it with:\n"
        "  pip install xgboost\n"
        f"Original import error: {e}"
    )


# -----------------------
# Config
# -----------------------

HORIZONS = [30, 60]
SPIKE_Q = 0.95
TRAIN_FRAC = 0.70

TEXT_STATS = [
    "n_chars",
    "n_words",
    "n_exclam",
    "n_question",
    "share_upper",
    "has_url",
    "has_hashtag",
]

LLM_NUMBOOL = [
    "market_relevance",
    "uncertainty_shock",
    "policy_surprise",
    "novelty",
    "tone_valence",
    "tail_risk_severity",
]

# Controls
BOOL_CONTROLS = ["is_retweet"]
CAT_CONTROLS = ["platform"]

# LLM topic categorical
LLM_TOPIC_CAT = ["market_topic"]

SPECS: Dict[str, Dict[str, List[str]]] = {
    "stats_only": {
        "num": TEXT_STATS + BOOL_CONTROLS,
        "cat": CAT_CONTROLS,
    },
    "llm_only": {
        "num": LLM_NUMBOOL + BOOL_CONTROLS,
        "cat": CAT_CONTROLS + LLM_TOPIC_CAT,
    },
    "stats_plus_llm": {
        "num": TEXT_STATS + LLM_NUMBOOL + BOOL_CONTROLS,
        "cat": CAT_CONTROLS,
    },
    "stats_plus_llm_plus_topic": {
        "num": TEXT_STATS + LLM_NUMBOOL + BOOL_CONTROLS,
        "cat": CAT_CONTROLS + LLM_TOPIC_CAT,
    },
}


# -----------------------
# Helpers
# -----------------------

def time_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("created_at").reset_index(drop=True)
    n = len(df)
    cut = int(np.floor(train_frac * n))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    p1 = float(np.mean(y_true))
    return float(max(p1, 1.0 - p1))


def _auc_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _compute_threshold_train_only(train: pd.DataFrame, abs_col: str, q: float) -> float:
    """
    Compute spike threshold using ONLY train absret values (spec-invariant).
    """
    train_abs = train[abs_col].dropna()
    if len(train_abs) == 0:
        return float("nan")
    return float(train_abs.quantile(q))


def make_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    """
    XGB pipeline:
    - numeric passthrough
    - categorical one-hot
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )

    # Conservative defaults for rare-event / weak-signal setting
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(steps=[("pre", pre), ("xgb", xgb)])


def eval_horizon_spec(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    spec_name: str,
) -> Dict[str, float]:
    """
    Evaluate XGB for one horizon + spec.

    Key point: threshold is computed on TRAIN absret only (not after feature dropna),
    keeping spike definition identical across specs within the same regime/horizon.
    """
    abs_col = f"vx_absret_{horizon}m"
    if abs_col not in train.columns or abs_col not in test.columns:
        raise ValueError(f"Missing required VX column: {abs_col}")

    spec = SPECS[spec_name]
    num_cols = spec["num"]
    cat_cols = spec["cat"]

    # 1) Spec-invariant threshold (train-only)
    thr = _compute_threshold_train_only(train, abs_col, SPIKE_Q)
    if not np.isfinite(thr):
        return {
            "n_train": 0,
            "n_test": 0,
            "thr_train_q95": float("nan"),
            "test_spike_rate": float("nan"),
            "baseline_acc": float("nan"),
            "xgb_acc": float("nan"),
            "xgb_auc": float("nan"),
        }

    # 2) Build labels for rows where absret observed
    y_train_s = (train[abs_col] > thr).where(train[abs_col].notna())
    y_test_s = (test[abs_col] > thr).where(test[abs_col].notna())

    # 3) Apply spec feature availability + align y by index
    cols_needed = [abs_col] + num_cols + cat_cols

    trainX_all = train[cols_needed].copy()
    testX_all = test[cols_needed].copy()

    train_mask = trainX_all.notna().all(axis=1) & y_train_s.notna()
    test_mask = testX_all.notna().all(axis=1) & y_test_s.notna()

    train2 = train.loc[train_mask].copy()
    test2 = test.loc[test_mask].copy()

    if len(train2) < 300 or len(test2) < 300:
        return {
            "n_train": int(len(train2)),
            "n_test": int(len(test2)),
            "thr_train_q95": float(thr),
            "test_spike_rate": float("nan"),
            "baseline_acc": float("nan"),
            "xgb_acc": float("nan"),
            "xgb_auc": float("nan"),
        }

    y_train = y_train_s.loc[train2.index].astype(int).to_numpy()
    y_test = y_test_s.loc[test2.index].astype(int).to_numpy()

    X_train = train2[num_cols + cat_cols]
    X_test = test2[num_cols + cat_cols]

    pipe = make_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    y_hat = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    return {
        "n_train": int(len(train2)),
        "n_test": int(len(test2)),
        "thr_train_q95": float(thr),
        "test_spike_rate": float(np.mean(y_test)),
        "baseline_acc": float(majority_baseline_accuracy(y_test)),
        "xgb_acc": float(accuracy_score(y_test, y_hat)),
        "xgb_auc": float(_auc_safe(y_test, y_prob)),
    }


def run_by_regime(
    df: pd.DataFrame,
    regimes: List[str] = ["pre", "power", "post"],
    specs: List[str] = ["stats_only", "llm_only", "stats_plus_llm", "stats_plus_llm_plus_topic"],
) -> pd.DataFrame:
    rows = []
    for reg in regimes:
        sub = df.loc[df["regime"] == reg].copy()
        if len(sub) < 500:
            continue

        train, test = time_split(sub, train_frac=TRAIN_FRAC)

        for h in HORIZONS:
            for spec_name in specs:
                m = eval_horizon_spec(train, test, horizon=h, spec_name=spec_name)
                rows.append({"regime": reg, "target": f"vx_spike_{h}m", "spec": spec_name, **m})

    return pd.DataFrame(rows)


def _check_required_columns(df: pd.DataFrame) -> None:
    required = {"created_at", "regime"} | {f"vx_absret_{h}m" for h in HORIZONS}
    for spec in SPECS.values():
        required |= set(spec["num"])
        required |= set(spec["cat"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset_vx_llm missing columns needed for baseline_xgb_vx_llm: {sorted(missing)}")


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_vx_llm.parquet"

    df = pd.read_parquet(data_path)
    _check_required_columns(df)

    results = run_by_regime(df)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 140)

    print("\n=== XGBoost (VX spike targets, LLM features) ===")
    print(results.sort_values(["regime", "target", "spec"]).to_string(index=False))

    print("\n=== Pooled (all regimes combined) ===")
    train_all, test_all = time_split(df, train_frac=TRAIN_FRAC)
    for h in HORIZONS:
        for spec_name in ["stats_only", "llm_only", "stats_plus_llm", "stats_plus_llm_plus_topic"]:
            m = eval_horizon_spec(train_all, test_all, horizon=h, spec_name=spec_name)
            print(
                f"vx_spike_{h}m | {spec_name}: "
                f"n_train={m['n_train']}, n_test={m['n_test']}, "
                f"thr_train_q95={m['thr_train_q95']:.6g}, test_spike_rate={m['test_spike_rate']:.3f}, "
                f"baseline_acc={m['baseline_acc']:.4f}, xgb_acc={m['xgb_acc']:.4f}, xgb_auc={m['xgb_auc']:.4f}"
            )

    # Save as parquet under results/runs for aggregation
    out_dir = PROJECT_ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = results.copy()
    out["market"] = "vx"
    out["feature_track"] = "llm"
    out["model"] = "xgb"
    out["analysis"] = "baseline"

    out_path = out_dir / "baseline_xgb_vx_llm.parquet"
    out.to_parquet(out_path, index=False)
    print("Saved results:", out_path)
