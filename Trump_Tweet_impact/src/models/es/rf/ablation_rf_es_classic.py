"""
ablation_rf_es_classic.py

Ablation-style Random Forest diagnostics.

Purpose
-------
Identify which individual features or feature blocks contain predictive information
when using a non-linear model (Random Forest).

We run:
- grouped feature sets (text stats only, vader only, topics only, combinations)
- single-feature models (one feature at a time)

Input
-----
- data_processed/dataset_es_classic.parquet

Output
------
- printed results table

Design
------
- Regime-specific evaluation (pre / power / post)
- Within each regime: time-respecting split (first 70% train, last 30% test)
- Targets: es_up_30m, es_up_60m
- Model: RandomForestClassifier (conservative hyperparams)
- Categorical: one-hot (platform, top_topic when present)
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


TARGETS = ["es_up_30m", "es_up_60m"]

# Blocks
TEXT_STATS = [
    "n_chars",
    "n_words",
    "n_exclam",
    "n_question",
    "share_upper",
    "has_url",
    "has_hashtag",
]

# Keep the full VADER block here for diagnostic purposes
VADER = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]

# Topic representation
TOPIC_CAT = ["top_topic"]  # categorical
TOPIC_NUM = ["top_score", "max_topic_similarity", "n_topics"]  # numeric summaries (if present)

# Controls
BOOLS = ["is_retweet"]
CATS = ["platform"]

# Singles
SINGLE_NUMERIC = TEXT_STATS + VADER + BOOLS + TOPIC_NUM
SINGLE_CATEGORICAL = ["platform", "top_topic"]


# -----------------------
# Utilities
# -----------------------

def time_split(df: pd.DataFrame, train_frac: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("created_at").reset_index(drop=True)
    n = len(df)
    cut = int(np.floor(train_frac * n))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    p1 = float(np.mean(y_true))
    return float(max(p1, 1.0 - p1))


def make_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    """
    RF pipeline:
    - Numeric passthrough (RF doesn't need scaling)
    - Categorical one-hot
    """
    transformers = []
    if num_features:
        transformers.append(("num", "passthrough", num_features))
    if cat_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=50,
        min_samples_split=100,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(steps=[("pre", pre), ("rf", rf)])


def _auc_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def eval_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    num_features: List[str],
    cat_features: List[str],
    set_name: str,
) -> Dict[str, float | str | int]:
    cols_needed = ["created_at", target] + num_features + cat_features
    train2 = train[cols_needed].dropna().copy()
    test2 = test[cols_needed].dropna().copy()

    X_train = train2[num_features + cat_features]
    y_train = train2[target].astype(int).values

    X_test = test2[num_features + cat_features]
    y_test = test2[target].astype(int).values

    base_acc = majority_baseline_accuracy(y_test)

    if len(train2) < 200 or len(test2) < 200:
        return {
            "feature_set": set_name,
            "n_train": int(len(train2)),
            "n_test": int(len(test2)),
            "baseline_acc": float(base_acc),
            "rf_acc": float("nan"),
            "rf_auc": float("nan"),
        }

    pipe = make_pipeline(num_features, cat_features)
    pipe.fit(X_train, y_train)

    y_hat = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    return {
        "feature_set": set_name,
        "n_train": int(len(train2)),
        "n_test": int(len(test2)),
        "baseline_acc": float(base_acc),
        "rf_acc": float(accuracy_score(y_test, y_hat)),
        "rf_auc": _auc_safe(y_test, y_prob),
    }


# -----------------------
# Feature set builder
# -----------------------

def build_feature_sets(df: pd.DataFrame) -> List[Tuple[str, List[str], List[str]]]:
    """
    Returns list of tuples: (set_name, numeric_features, categorical_features)
    Topic-related sets are included only if required columns exist.
    """
    sets: List[Tuple[str, List[str], List[str]]] = []

    has_topics_cat = all(c in df.columns for c in TOPIC_CAT)
    has_topics_num = all(c in df.columns for c in TOPIC_NUM)

    # Grouped sets
    sets.append(("TEXT_STATS_ONLY", TEXT_STATS, []))
    sets.append(("VADER_ONLY", VADER, []))
    sets.append(("TEXT+VADER", TEXT_STATS + VADER, []))
    sets.append(("TEXT+VADER+RETWEET", TEXT_STATS + VADER + BOOLS, []))
    sets.append(("ALL_BASELINE", TEXT_STATS + VADER + BOOLS, CATS))

    if has_topics_cat:
        sets.append(("TOPIC_ONLY_CAT", [], TOPIC_CAT))
        sets.append(("TEXT+TOPIC_CAT", TEXT_STATS, TOPIC_CAT))
        sets.append(("TEXT+VADER+TOPIC_CAT", TEXT_STATS + VADER, TOPIC_CAT))
        sets.append(("ALL_BASELINE+TOPIC_CAT", TEXT_STATS + VADER + BOOLS, CATS + TOPIC_CAT))

    if has_topics_num:
        sets.append(("TOPIC_ONLY_NUM", TOPIC_NUM, []))
        sets.append(("TEXT+TOPIC_NUM", TEXT_STATS + TOPIC_NUM, []))
        sets.append(("TEXT+VADER+TOPIC_NUM", TEXT_STATS + VADER + TOPIC_NUM, []))

    # Singles
    for f in SINGLE_NUMERIC:
        if f in df.columns:
            sets.append((f"SINGLE:{f}", [f], []))

    for f in SINGLE_CATEGORICAL:
        if f in df.columns:
            sets.append((f"SINGLE:{f}", [], [f]))

    return sets


def run(df: pd.DataFrame, regimes: List[str] = ["pre", "power", "post"]) -> pd.DataFrame:
    rows = []

    sets = build_feature_sets(df)

    for reg in regimes:
        sub = df.loc[df["regime"] == reg].copy()
        if len(sub) < 500:
            continue

        train, test = time_split(sub, train_frac=0.70)

        for target in TARGETS:
            for set_name, num_feats, cat_feats in sets:
                res = eval_one(train, test, target, num_feats, cat_feats, set_name)
                rows.append({"regime": reg, "target": target, **res})

    return pd.DataFrame(rows)


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_es_classic.parquet"

    df = pd.read_parquet(data_path)

    # Minimal sanity checks (universally expected columns)
    needed = {"created_at", "regime"} | set(TARGETS) | set(TEXT_STATS) | set(VADER) | set(BOOLS) | set(CATS)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"dataset_model missing columns needed for ablation_rf: {sorted(missing)}")

    results = run(df)

    # Sort for readability: grouped sets first, then singles
    def order_key(name: str) -> int:
        if name == "ALL_BASELINE+TOPIC_CAT":
            return 0
        if name == "ALL_BASELINE":
            return 1
        if name.startswith("TEXT+VADER+TOPIC"):
            return 2
        if name.startswith("TEXT+VADER"):
            return 3
        if name.startswith("TEXT+TOPIC"):
            return 4
        if name.startswith("TOPIC_ONLY"):
            return 5
        if name.startswith("TEXT_STATS_ONLY"):
            return 6
        if name.startswith("VADER_ONLY"):
            return 7
        if name.startswith("SINGLE:platform"):
            return 20
        if name.startswith("SINGLE:top_topic"):
            return 21
        if name.startswith("SINGLE:"):
            return 30
        return 99

    results["__ord"] = results["feature_set"].map(order_key)
    results = results.sort_values(["regime", "target", "__ord", "feature_set"]).drop(columns="__ord")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_columns", 80)

    print("\n=== Ablation results (Random Forest) ===")
    print(results.to_string(index=False))



    #save as a .parquet(0 under runs in results, so we can aggregate all results and analyze them
    out_dir = PROJECT_ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # add metadata columns
    results = results.copy()
    results["market"] = "es"          #  vx    es
    results["feature_track"] = "classic"  # llm    classic
    results["model"] = "rf"        # logit     rf
    results["analysis"] = "ablation"  # baseline   ablation

    out_path = out_dir / "ablation_rf_es_classic.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)