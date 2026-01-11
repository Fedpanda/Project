"""
ablation_rf_vx_llm.py

Ablation-style Random Forest diagnostics for predicting VX volatility spikes after posts,
using LLM-derived features (and optional text stats / controls).

Input:
- data_processed/dataset_vx_llm.parquet

Target (NO leakage, spec-invariant)
-----------------------------------
For horizon h in {30m, 60m}:
- vx_spike_h = 1{ vx_absret_h > Q_train(q) } else 0
where Q_train(q) is computed on the TRAIN split within each regime
using ONLY vx_absret_h (NOT after dropping rows due to missing features).

Design:
- Regime-specific evaluation (pre / power / post)
- Within each regime: time-respecting split (first 70% train, last 30% test)
- Metric: ROC-AUC (primary), accuracy printed but not emphasized
- Model: RandomForestClassifier (conservative hyperparams)
- Categorical: one-hot (platform, market_topic)

Feature sets:
- Grouped blocks
- Single-feature models (numeric + categorical)

Notes:
- Many comparisons => treat small differences cautiously (multiple testing).
- Spikes are rare => accuracy often equals majority baseline; AUC is more informative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------
# Config
# -----------------------

HORIZONS = [30, 60]
SPIKE_Q = 0.95
REGIMES = ["pre", "power", "post"]
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

BOOL_CONTROLS = ["is_retweet"]
CAT_CONTROLS = ["platform"]
LLM_TOPIC_CAT = ["market_topic"]


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
        transformers.append(("num", "passthrough", num_features))
    if cat_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=8,
        min_samples_leaf=60,
        min_samples_split=120,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight=None,
    )

    return Pipeline(steps=[("pre", pre), ("rf", rf)])


def _compute_threshold_train_only(train: pd.DataFrame, abs_col: str, q: float) -> float:
    train_abs = train[abs_col].dropna()
    if len(train_abs) == 0:
        return float("nan")
    return float(train_abs.quantile(q))


# -----------------------
# Feature sets
# -----------------------

def build_feature_sets(df: pd.DataFrame) -> List[Tuple[str, List[str], List[str]]]:
    """
    Returns list of tuples: (set_name, numeric_features, categorical_features)
    Includes grouped sets + single-feature sets.
    """
    sets: List[Tuple[str, List[str], List[str]]] = []

    # Grouped blocks
    sets.append(("CONTROLS_ONLY", BOOL_CONTROLS, CAT_CONTROLS))
    sets.append(("TEXT_STATS_ONLY", TEXT_STATS, []))
    sets.append(("LLM_ONLY_NUM", LLM_NUMBOOL, []))
    sets.append(("LLM_ONLY_TOPIC", [], LLM_TOPIC_CAT))
    sets.append(("LLM_ONLY_NUM+TOPIC", LLM_NUMBOOL, LLM_TOPIC_CAT))

    sets.append(("TEXT+CONTROLS", TEXT_STATS + BOOL_CONTROLS, CAT_CONTROLS))
    sets.append(("LLM_NUM+CONTROLS", LLM_NUMBOOL + BOOL_CONTROLS, CAT_CONTROLS))
    sets.append(("LLM_NUM+TOPIC+CONTROLS", LLM_NUMBOOL + BOOL_CONTROLS, CAT_CONTROLS + LLM_TOPIC_CAT))

    sets.append(("TEXT+LLM_NUM+CONTROLS", TEXT_STATS + LLM_NUMBOOL + BOOL_CONTROLS, CAT_CONTROLS))
    sets.append(("TEXT+LLM_NUM+TOPIC+CONTROLS", TEXT_STATS + LLM_NUMBOOL + BOOL_CONTROLS, CAT_CONTROLS + LLM_TOPIC_CAT))

    # Single-feature models
    for f in (TEXT_STATS + LLM_NUMBOOL + BOOL_CONTROLS):
        if f in df.columns:
            sets.append((f"SINGLE:{f}", [f], []))

    for f in (CAT_CONTROLS + LLM_TOPIC_CAT):
        if f in df.columns:
            sets.append((f"SINGLE:{f}", [], [f]))

    return sets


# -----------------------
# Evaluation
# -----------------------

def eval_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    num_features: List[str],
    cat_features: List[str],
    set_name: str,
) -> Dict[str, float | str | int]:
    abs_col = f"vx_absret_{horizon}m"
    if abs_col not in train.columns or abs_col not in test.columns:
        raise ValueError(f"Missing required VX column: {abs_col}")

    # Spec-invariant threshold
    thr = _compute_threshold_train_only(train, abs_col, SPIKE_Q)
    if not np.isfinite(thr):
        return {
            "feature_set": set_name,
            "n_train": 0,
            "n_test": 0,
            "thr_train_q95": float("nan"),
            "test_spike_rate": float("nan"),
            "baseline_acc": float("nan"),
            "rf_acc": float("nan"),
            "rf_auc": float("nan"),
        }

    # Labels defined where absret observed
    y_train_s = (train[abs_col] > thr).where(train[abs_col].notna())
    y_test_s = (test[abs_col] > thr).where(test[abs_col].notna())

    cols_needed = [abs_col] + num_features + cat_features
    trainX = train[cols_needed].copy()
    testX = test[cols_needed].copy()

    train_mask = trainX.notna().all(axis=1) & y_train_s.notna()
    test_mask = testX.notna().all(axis=1) & y_test_s.notna()

    train2 = train.loc[train_mask].copy()
    test2 = test.loc[test_mask].copy()

    if len(train2) < 300 or len(test2) < 300:
        return {
            "feature_set": set_name,
            "n_train": int(len(train2)),
            "n_test": int(len(test2)),
            "thr_train_q95": float(thr),
            "test_spike_rate": float("nan"),
            "baseline_acc": float("nan"),
            "rf_acc": float("nan"),
            "rf_auc": float("nan"),
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
        "n_train": int(len(train2)),
        "n_test": int(len(test2)),
        "thr_train_q95": float(thr),
        "test_spike_rate": float(np.mean(y_test)),
        "baseline_acc": float(majority_baseline_accuracy(y_test)),
        "rf_acc": float(accuracy_score(y_test, y_hat)),
        "rf_auc": float(_auc_safe(y_test, y_prob)),
    }


def run(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sets = build_feature_sets(df)

    for reg in REGIMES:
        sub = df.loc[df["regime"] == reg].copy()
        if len(sub) < 500:
            continue

        train, test = time_split(sub, train_frac=TRAIN_FRAC)

        for h in HORIZONS:
            for set_name, num_feats, cat_feats in sets:
                res = eval_one(train, test, h, num_feats, cat_feats, set_name)
                rows.append({"regime": reg, "target": f"vx_spike_{h}m", **res})

    return pd.DataFrame(rows)


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_vx_llm.parquet"

    df = pd.read_parquet(data_path)

    needed = (
        {"created_at", "regime", "vx_absret_30m", "vx_absret_60m"}
        | set(TEXT_STATS)
        | set(LLM_NUMBOOL)
        | set(BOOL_CONTROLS)
        | set(CAT_CONTROLS)
        | set(LLM_TOPIC_CAT)
    )
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"dataset_vx_llm missing columns needed for ablation_rf_vx_llm: {sorted(missing)}")

    results = run(df)

    def order_key(name: str) -> int:
        if name == "TEXT+LLM_NUM+TOPIC+CONTROLS":
            return 0
        if name == "TEXT+LLM_NUM+CONTROLS":
            return 1
        if name == "LLM_NUM+TOPIC+CONTROLS":
            return 2
        if name == "LLM_NUM+CONTROLS":
            return 3
        if name == "TEXT+CONTROLS":
            return 4
        if name.startswith("LLM_ONLY"):
            return 5
        if name.startswith("TEXT_STATS_ONLY"):
            return 6
        if name.startswith("CONTROLS_ONLY"):
            return 7
        if name.startswith("SINGLE:platform"):
            return 20
        if name.startswith("SINGLE:market_topic"):
            return 21
        if name.startswith("SINGLE:"):
            return 30
        return 99

    results["__ord"] = results["feature_set"].map(order_key)
    results = results.sort_values(["regime", "target", "__ord", "feature_set"]).drop(columns="__ord")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_columns", 140)

    print("\n=== Ablation results (Random Forest) — VX spikes with LLM features ===")
    print(results.to_string(index=False))


    #save as a .parquet() under runs in results, so we can aggregate all results and analyze them
    out_dir = PROJECT_ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # add metadata columns
    results = results.copy()
    results["market"] = "vx"          #  vx    es
    results["feature_track"] = "llm"  # llm    classic
    results["model"] = "rf"        # logit     rf
    results["analysis"] = "ablation"  # baseline   ablation

    out_path = out_dir / "ablation_rf_vx_llm.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)