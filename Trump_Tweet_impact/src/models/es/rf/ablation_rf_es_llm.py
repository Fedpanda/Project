"""
ablation_rf_es_llm.py

Ablation-style Random Forest diagnostics (LLM feature set).

Purpose
-------
Identify which individual features or feature blocks contain predictive information
when using a non-linear model (Random Forest) with LLM-derived annotations.

We run:
- grouped feature sets (text stats only, LLM only, topic only, combinations, controls)
- single-feature models (one feature at a time)

Input
-----
- data_processed/dataset_es_llm.parquet

Output
------
- printed results table

Design
------
- Regime-specific evaluation (pre / power / post)
- Within each regime: time-respecting split (first 70% train, last 30% test)
- Targets: es_up_30m, es_up_60m
- Model: RandomForestClassifier (conservative hyperparams)
- Categorical: one-hot (platform, market_topic when present)
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

# Blocks: classic mechanical text features
TEXT_STATS = [
    "n_chars",
    "n_words",
    "n_exclam",
    "n_question",
    "share_upper",
    "has_url",
    "has_hashtag",
]

# LLM numeric signals
LLM_NUM = [
    "market_relevance",
    "uncertainty_shock",
    "policy_surprise",
    "novelty",
    "tone_valence",
    "tail_risk_severity",
]

# LLM topic (categorical enum)
LLM_TOPIC_CAT = ["market_topic"]

# Controls
BOOLS = ["is_retweet"]
CATS = ["platform"]

# Singles
SINGLE_NUMERIC = TEXT_STATS + LLM_NUM + BOOLS
SINGLE_CATEGORICAL = ["platform", "market_topic"]


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
    LLM topic sets included only if market_topic exists.
    """
    sets: List[Tuple[str, List[str], List[str]]] = []

    has_llm_topic = all(c in df.columns for c in LLM_TOPIC_CAT)

    # Grouped sets (core)
    sets.append(("TEXT_STATS_ONLY", TEXT_STATS, []))
    sets.append(("LLM_ONLY_NUM", LLM_NUM, []))
    sets.append(("LLM_NUM+CONTROLS", LLM_NUM + BOOLS, CATS))
    sets.append(("TEXT+CONTROLS", TEXT_STATS + BOOLS, CATS))
    sets.append(("TEXT+LLM_NUM", TEXT_STATS + LLM_NUM, []))
    sets.append(("TEXT+LLM_NUM+CONTROLS", TEXT_STATS + LLM_NUM + BOOLS, CATS))

    # Topic blocks (if present)
    if has_llm_topic:
        sets.append(("LLM_ONLY_TOPIC", [], LLM_TOPIC_CAT))
        sets.append(("LLM_ONLY_NUM+TOPIC", LLM_NUM, LLM_TOPIC_CAT))
        sets.append(("LLM_NUM+TOPIC+CONTROLS", LLM_NUM + BOOLS, CATS + LLM_TOPIC_CAT))
        sets.append(("TEXT+LLM_NUM+TOPIC+CONTROLS", TEXT_STATS + LLM_NUM + BOOLS, CATS + LLM_TOPIC_CAT))

    # Controls only (useful benchmark)
    sets.append(("CONTROLS_ONLY", BOOLS, CATS))

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
    data_path = PROJECT_ROOT / "data_processed" / "dataset_es_llm.parquet"

    df = pd.read_parquet(data_path)

    # Minimal sanity checks (universally expected columns)
    needed = {"created_at", "regime"} | set(TARGETS) | set(TEXT_STATS) | set(LLM_NUM) | set(BOOLS) | set(CATS)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"dataset_es_llm missing columns needed for ablation_rf_es_llm: {sorted(missing)}")

    results = run(df)

    # Sort for readability: grouped sets first, then singles
    def order_key(name: str) -> int:
        # grouped sets
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
        if name == "LLM_ONLY_NUM+TOPIC":
            return 5
        if name == "LLM_ONLY_NUM":
            return 6
        if name == "LLM_ONLY_TOPIC":
            return 7
        if name == "TEXT_STATS_ONLY":
            return 8
        if name == "CONTROLS_ONLY":
            return 9

        # singles
        if name.startswith("SINGLE:platform"):
            return 20
        if name.startswith("SINGLE:market_topic"):
            return 21
        if name.startswith("SINGLE:"):
            return 30
        return 99

    results["__ord"] = results["feature_set"].map(order_key)
    results = results.sort_values(["regime", "target", "__ord", "feature_set"]).drop(columns="__ord")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_columns", 80)

    print("\n=== Ablation results (Random Forest, ES, LLM features) ===")
    print(results.to_string(index=False))


    #save as a .parquet(0 under runs in results, so we can aggregate all results and analyze them
    out_dir = PROJECT_ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # add metadata columns
    results = results.copy()
    results["market"] = "es"          #  vx    es
    results["feature_track"] = "llm"  # llm    classic
    results["model"] = "rf"        # logit     rf
    results["analysis"] = "ablation"  # baseline   ablation

    out_path = out_dir / "ablation_rf_es_llm.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)