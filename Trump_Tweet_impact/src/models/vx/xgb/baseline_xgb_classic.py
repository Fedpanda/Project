"""
baseline_xgb_vx_classic.py

XGBoost benchmarks for VX volatility spikes after posts, using "classic" features
(text stats + VADER + optional topic embeddings features).

Input:
- data_processed/dataset_vx_classic.parquet

Targets (constructed WITHOUT leakage and SPEC-INVARIANT across feature specs)
----------------------------------------------------------------------------
For horizon h in {30m, 60m}:
- vx_spike_h = 1{ vx_absret_h > Q_train(q) }, else 0

IMPORTANT:
- The spike threshold Q_train(q) is computed using ONLY the TRAIN split AND using ONLY vx_absret_h
  (i.e., NOT after dropping rows due to missing features). This keeps the target definition
  comparable across specs (stats_only vs stats_plus_vader vs ...).

Validation:
- within each regime, sort by created_at
- train on first 70%, test on last 30% (time-respecting split)

Feature specs:
1) stats_only:
   - text stats + controls (platform, is_retweet)
2) stats_plus_vader:
   - stats_only + VADER
3) stats_plus_vader_plus_topics:
   - stats_plus_vader + topic features (only included if topic columns exist)

Outputs:
- spike threshold (train quantile)
- test spike rate
- baseline accuracy (majority class on test)
- XGB accuracy and ROC-AUC

Notes:
- AUC is usually more informative than accuracy here because spikes are rare.
- XGBoost is a flexible non-linear learner; keep hyperparameters conservative to reduce overfitting.
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

# Classic sentiment block
VADER = ["vader_pos"]

# Controls
BOOL_CONTROLS = ["is_retweet"]
CAT_CONTROLS = ["platform"]


def detect_topic_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Topic columns are optional and depend on your features_topics_embed output.

    We support:
    - top_topic (categorical)
    - numeric topic summaries: top_score, max_topic_similarity, n_topics
    - optional binary flags: topic_*_flag
    """
    topic_num: List[str] = []
    topic_cat: List[str] = []

    for c in ["top_score", "max_topic_similarity", "n_topics"]:
        if c in df.columns:
            topic_num.append(c)

    if "top_topic" in df.columns:
        topic_cat.append("top_topic")

    topic_num.extend([c for c in df.columns if c.startswith("topic_") and c.endswith("_flag")])

    topic_num = sorted(set(topic_num))
    return topic_num, topic_cat


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
    train_abs = train[abs_col].dropna()
    if len(train_abs) == 0:
        return float("nan")
    return float(train_abs.quantile(q))


def make_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    """
    XGB pipeline:
    - numeric passthrough
    - categorical one-hot
    - XGBClassifier
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )

    # Conservative defaults (you can tune later if needed)
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
    num_cols: List[str],
    cat_cols: List[str],
) -> Dict[str, float]:
    abs_col = f"vx_absret_{horizon}m"
    if abs_col not in train.columns or abs_col not in test.columns:
        raise ValueError(f"Missing required VX column: {abs_col}")

    # 1) Spec-invariant threshold (uses only absret and train split)
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

    # 2) Labels defined wherever absret observed
    y_train_s = (train[abs_col] > thr).where(train[abs_col].notna())
    y_test_s = (test[abs_col] > thr).where(test[abs_col].notna())

    # 3) Enforce feature availability and align y (index-safe)
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


def run_by_regime(df: pd.DataFrame, regimes: List[str] = ["pre", "power", "post"]) -> pd.DataFrame:
    rows = []

    topic_num, topic_cat = detect_topic_columns(df)

    specs: List[Tuple[str, List[str], List[str]]] = [
        ("stats_only", TEXT_STATS + BOOL_CONTROLS, CAT_CONTROLS),
        ("stats_plus_vader", TEXT_STATS + VADER + BOOL_CONTROLS, CAT_CONTROLS),
    ]
    if topic_num or topic_cat:
        specs.append(
            ("stats_plus_vader_plus_topics", TEXT_STATS + VADER + BOOL_CONTROLS + topic_num, CAT_CONTROLS + topic_cat)
        )

    for reg in regimes:
        sub = df.loc[df["regime"] == reg].copy()
        if len(sub) < 500:
            continue

        train, test = time_split(sub, train_frac=TRAIN_FRAC)

        for h in HORIZONS:
            for spec_name, num_cols, cat_cols in specs:
                m = eval_horizon_spec(train, test, horizon=h, spec_name=spec_name, num_cols=num_cols, cat_cols=cat_cols)
                rows.append({"regime": reg, "target": f"vx_spike_{h}m", "spec": spec_name, **m})

    return pd.DataFrame(rows)


def _check_required_columns(df: pd.DataFrame) -> None:
    required = {"created_at", "regime"} | {f"vx_absret_{h}m" for h in HORIZONS}
    required |= set(TEXT_STATS) | set(BOOL_CONTROLS) | set(CAT_CONTROLS) | set(VADER)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset_vx_classic missing required columns for baseline_xgb_vx_classic: {sorted(missing)}")


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_vx_classic.parquet"

    df = pd.read_parquet(data_path)
    _check_required_columns(df)

    results = run_by_regime(df)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 140)

    print("\n=== XGBoost (VX spike targets, classic features) ===")
    print(results.sort_values(["regime", "target", "spec"]).to_string(index=False))

    print("\n=== Pooled (all regimes combined) ===")
    train_all, test_all = time_split(df, train_frac=TRAIN_FRAC)

    topic_num, topic_cat = detect_topic_columns(df)
    pooled_specs: List[Tuple[str, List[str], List[str]]] = [
        ("stats_only", TEXT_STATS + BOOL_CONTROLS, CAT_CONTROLS),
        ("stats_plus_vader", TEXT_STATS + VADER + BOOL_CONTROLS, CAT_CONTROLS),
    ]
    if topic_num or topic_cat:
        pooled_specs.append(
            ("stats_plus_vader_plus_topics", TEXT_STATS + VADER + BOOL_CONTROLS + topic_num, CAT_CONTROLS + topic_cat)
        )

    for h in HORIZONS:
        for spec_name, num_cols, cat_cols in pooled_specs:
            m = eval_horizon_spec(train_all, test_all, horizon=h, spec_name=spec_name, num_cols=num_cols, cat_cols=cat_cols)
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
    out["feature_track"] = "classic"
    out["model"] = "xgb"
    out["analysis"] = "baseline"

    out_path = out_dir / "baseline_xgb_vx_classic.parquet"
    out.to_parquet(out_path, index=False)
    print("Saved results:", out_path)
