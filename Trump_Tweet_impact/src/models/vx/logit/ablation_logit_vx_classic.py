"""
ablation_logit_vx_classic.py

Ablation-style diagnostics for predicting VX volatility spikes after posts
using CLASSIC (non-LLM) features: text stats, VADER, and (optionally) embedding-topic features.

Input:
- data_processed/dataset_vx_classic.parquet

Target (no leakage; consistent across specs)
-------------------------------------------
For each regime and horizon h in {30m, 60m}:

1) Split by time: train = first 70%, test = last 30% (within regime).
2) Compute threshold thr = Q_train(SPIKE_Q) using ONLY train vx_absret_{h}m
   (dropping NaNs in vx_absret only; NOT conditioning on feature availability).
3) Define labels:
   y = 1{ vx_absret_{h}m > thr } else 0

This guarantees the spike definition is:
- time-respecting (train-only threshold)
- identical across all feature specs (no spec-dependent threshold)

Design:
- Regime-specific evaluation (pre / power / post)
- Metric: ROC-AUC (primary), accuracy printed but not emphasized
- Model: LogisticRegression (L2)
- Many comparisons => treat small differences cautiously (multiple testing).

Feature sets:
- Grouped blocks
- Single-feature models (numeric + categorical)

Topic handling (classic)
------------------------
If topic columns exist, we include both:
- top_topic (categorical)
- numeric topic summaries (top_score, max_topic_similarity, n_topics)

If they don't exist, topic sets are omitted automatically.

Notes:
- VX outcomes can be NaN; we drop NaNs only where needed (absret for threshold/label,
  and then features for each spec).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------
# Config
# -----------------------

HORIZONS = [30, 60]
SPIKE_Q = 0.95
REGIMES = ["pre", "power", "post"]
TRAIN_FRAC = 0.70

# Baseline text stats
TEXT_STATS = [
    "n_chars",
    "n_words",
    "n_exclam",
    "n_question",
    "share_upper",
    "has_url",
    "has_hashtag",
]

# VADER block (keep full set for ablation/diagnostics)
VADER = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]

# Controls
BOOL_CONTROLS = ["is_retweet"]
CAT_CONTROLS = ["platform"]

# Topics (optional)
TOPIC_CAT = ["top_topic"]  # categorical if present
TOPIC_NUM = ["top_score", "max_topic_similarity", "n_topics"]  # numeric summaries if present


# -----------------------
# Utilities
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


def make_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    transformers = []
    if num_features:
        transformers.append(("num", StandardScaler(), num_features))
    if cat_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def compute_spike_threshold(train: pd.DataFrame, abs_col: str, q: float) -> float:
    """
    Compute spike threshold using ONLY train absret values (dropna on absret only).
    Done once per (regime, horizon) and reused across all feature specs.
    """
    s = train[abs_col].dropna()
    if len(s) < 300:
        return float("nan")
    return float(s.quantile(q))


def make_labels(df: pd.DataFrame, abs_col: str, thr: float) -> pd.Series:
    """
    Create spike labels for rows with non-missing absret.
    Returns Series aligned to df.index, with NaN where absret is NaN.
    """
    y = pd.Series(index=df.index, dtype="float64")
    mask = df[abs_col].notna()
    y.loc[mask] = (df.loc[mask, abs_col] > thr).astype(int).astype(float)
    return y


# -----------------------
# Feature sets
# -----------------------

def build_feature_sets(df: pd.DataFrame) -> List[Tuple[str, List[str], List[str]]]:
    """
    Returns list of tuples: (set_name, numeric_features, categorical_features)
    Includes grouped sets + single-feature sets.
    Topic-related sets are included only if required columns exist.
    """
    sets: List[Tuple[str, List[str], List[str]]] = []

    has_topics_cat = all(c in df.columns for c in TOPIC_CAT)
    has_topics_num = all(c in df.columns for c in TOPIC_NUM)

    # --- Grouped blocks (core) ---
    sets.append(("CONTROLS_ONLY", BOOL_CONTROLS, CAT_CONTROLS))
    sets.append(("TEXT_STATS_ONLY", TEXT_STATS, []))
    sets.append(("VADER_ONLY", VADER, []))
    sets.append(("TEXT+VADER", TEXT_STATS + VADER, []))

    sets.append(("TEXT+CONTROLS", TEXT_STATS + BOOL_CONTROLS, CAT_CONTROLS))
    sets.append(("VADER+CONTROLS", VADER + BOOL_CONTROLS, CAT_CONTROLS))
    sets.append(("TEXT+VADER+CONTROLS", TEXT_STATS + VADER + BOOL_CONTROLS, CAT_CONTROLS))

    # --- Topic groups (optional) ---
    if has_topics_cat:
        sets.append(("TOPIC_ONLY_CAT", [], TOPIC_CAT))
        sets.append(("TEXT+TOPIC_CAT", TEXT_STATS, TOPIC_CAT))
        sets.append(("VADER+TOPIC_CAT", VADER, TOPIC_CAT))
        sets.append(("TEXT+VADER+TOPIC_CAT", TEXT_STATS + VADER, TOPIC_CAT))
        sets.append(("TEXT+VADER+TOPIC_CAT+CONTROLS", TEXT_STATS + VADER + BOOL_CONTROLS, CAT_CONTROLS + TOPIC_CAT))

    if has_topics_num:
        sets.append(("TOPIC_ONLY_NUM", TOPIC_NUM, []))
        sets.append(("TEXT+TOPIC_NUM", TEXT_STATS + TOPIC_NUM, []))
        sets.append(("VADER+TOPIC_NUM", VADER + TOPIC_NUM, []))
        sets.append(("TEXT+VADER+TOPIC_NUM", TEXT_STATS + VADER + TOPIC_NUM, []))
        sets.append(("TEXT+VADER+TOPIC_NUM+CONTROLS", TEXT_STATS + VADER + TOPIC_NUM + BOOL_CONTROLS, CAT_CONTROLS))

    # --- Single-feature models ---
    single_numeric = TEXT_STATS + VADER + BOOL_CONTROLS
    if has_topics_num:
        single_numeric = single_numeric + TOPIC_NUM

    for f in single_numeric:
        if f in df.columns:
            sets.append((f"SINGLE:{f}", [f], []))

    single_cat = CAT_CONTROLS.copy()
    if has_topics_cat:
        single_cat = single_cat + TOPIC_CAT

    for f in single_cat:
        if f in df.columns:
            sets.append((f"SINGLE:{f}", [], [f]))

    return sets


# -----------------------
# Evaluation
# -----------------------

def eval_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    abs_col: str,
    thr: float,
    num_features: List[str],
    cat_features: List[str],
    set_name: str,
) -> Dict[str, float | str | int]:
    """
    Evaluate one feature set given a fixed spike threshold (computed from train absret only).
    Drops NaNs in: (absret, selected features).
    """
    if not np.isfinite(thr):
        return {
            "feature_set": set_name,
            "n_train": 0,
            "n_test": 0,
            "thr_train_q": float("nan"),
            "test_spike_rate": float("nan"),
            "baseline_acc": float("nan"),
            "logit_acc": float("nan"),
            "logit_auc": float("nan"),
        }

    # Build labels from fixed threshold
    y_train_s = make_labels(train, abs_col, thr)
    y_test_s = make_labels(test, abs_col, thr)

    cols_needed = [abs_col] + num_features + cat_features
    trainX = train[cols_needed].copy()
    testX = test[cols_needed].copy()

    train_mask = trainX.notna().all(axis=1) & y_train_s.notna()
    test_mask = testX.notna().all(axis=1) & y_test_s.notna()

    train2 = train.loc[train_mask].copy()
    test2 = test.loc[test_mask].copy()

    n_train = int(len(train2))
    n_test = int(len(test2))

    if n_train < 300 or n_test < 300:
        return {
            "feature_set": set_name,
            "n_train": n_train,
            "n_test": n_test,
            "thr_train_q": float(thr),
            "test_spike_rate": float("nan"),
            "baseline_acc": float("nan"),
            "logit_acc": float("nan"),
            "logit_auc": float("nan"),
        }

    y_train = y_train_s.loc[train2.index].astype(int).to_numpy()
    y_test = y_test_s.loc[test2.index].astype(int).to_numpy()

    X_train = train2[num_features + cat_features]
    X_test = test2[num_features + cat_features]

    pipe = make_pipeline(num_features, cat_features)
    pipe.fit(X_train, y_train)

    y_hat = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    return {
        "feature_set": set_name,
        "n_train": n_train,
        "n_test": n_test,
        "thr_train_q": float(thr),
        "test_spike_rate": float(np.mean(y_test)),
        "baseline_acc": float(majority_baseline_accuracy(y_test)),
        "logit_acc": float(accuracy_score(y_test, y_hat)),
        "logit_auc": float(_auc_safe(y_test, y_prob)),
    }


def run(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sets = build_feature_sets(df)

    for reg in REGIMES:
        sub = df.loc[df["regime"] == reg].copy()
        if len(sub) < 500:
            continue

        train, test = time_split(sub, train_frac=TRAIN_FRAC)

        for h in HORIZONS:
            abs_col = f"vx_absret_{h}m"
            thr = compute_spike_threshold(train, abs_col, q=SPIKE_Q)

            for set_name, num_feats, cat_feats in sets:
                res = eval_one(
                    train=train,
                    test=test,
                    abs_col=abs_col,
                    thr=thr,
                    num_features=num_feats,
                    cat_features=cat_feats,
                    set_name=set_name,
                )
                rows.append(
                    {
                        "regime": reg,
                        "target": f"vx_spike_{h}m",
                        "feature_set": set_name,
                        **res,
                    }
                )

    return pd.DataFrame(rows)


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_vx_classic.parquet"

    df = pd.read_parquet(data_path)

    # Minimal sanity checks (core columns only; topics are optional)
    required_core = {
        "created_at",
        "regime",
        "vx_absret_30m",
        "vx_absret_60m",
        *TEXT_STATS,
        *VADER,
        *BOOL_CONTROLS,
        *CAT_CONTROLS,
    }
    missing = set(required_core) - set(df.columns)
    if missing:
        raise ValueError(f"dataset_vx_classic missing required columns for ablation_logit_vx_classic: {sorted(missing)}")

    results = run(df)

    # Sort: grouped first, then singles
    def order_key(name: str) -> int:
        # Put “full-ish” classic sets early if present
        if name.startswith("TEXT+VADER+TOPIC_CAT+CONTROLS"):
            return 0
        if name.startswith("TEXT+VADER+TOPIC_NUM+CONTROLS"):
            return 1
        if name == "TEXT+VADER+CONTROLS":
            return 2
        if name == "TEXT+VADER":
            return 3
        if name == "TEXT+CONTROLS":
            return 4
        if name == "VADER+CONTROLS":
            return 5
        if name.startswith("TEXT+VADER+TOPIC_"):
            return 6
        if name.startswith("TEXT+TOPIC_"):
            return 7
        if name.startswith("VADER+TOPIC_"):
            return 8
        if name.startswith("TOPIC_ONLY"):
            return 9
        if name == "TEXT_STATS_ONLY":
            return 10
        if name == "VADER_ONLY":
            return 11
        if name == "CONTROLS_ONLY":
            return 12
        if name.startswith("SINGLE:platform"):
            return 20
        if name.startswith("SINGLE:top_topic"):
            return 21
        if name.startswith("SINGLE:"):
            return 30
        return 99

    results["__ord"] = results["feature_set"].map(order_key)
    results = results.sort_values(["regime", "target", "__ord", "feature_set"]).drop(columns="__ord")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 140)

    print("\n=== Ablation results (Logistic Regression) — VX spikes with CLASSIC features ===")
    print(results.to_string(index=False))


    #save as a .parquet(0 under runs in results, so we can aggregate all results and analyze them
    out_dir = PROJECT_ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # add metadata columns
    results = results.copy()
    results["market"] = "vx"          #  vx    es
    results["feature_track"] = "classic"  # llm    classic
    results["model"] = "logit"        # logit     rf
    results["analysis"] = "ablation"  # baseline   ablation

    out_path = out_dir / "ablation_logit_vx_classic.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)