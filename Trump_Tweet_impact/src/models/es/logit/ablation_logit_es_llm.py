"""
ablation_logit_es_llm.py

Ablation-style logistic regression diagnostics for LLM-derived features.

Purpose
-------
Identify which individual LLM features or feature blocks contain predictive information
for ES direction after posts, and whether any incremental value exists beyond text stats.

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
- Model: LogisticRegression (L2) with standardization for numeric features
- Categoricals: one-hot encoding (platform, market_topic when used)

Strictness
----------
- No imputation: for each feature set, drop rows with NA in required columns.
- This avoids introducing arbitrary values for missing LLM outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = ["es_up_30m", "es_up_60m"]

# Text stats (attention / intensity proxies)
TEXT_STATS = [
    "n_chars",
    "n_words",
    "n_exclam",
    "n_question",
    "share_upper",
    "has_url",
    "has_hashtag",
]

# Controls
BOOLS = ["is_retweet"]
CATS = ["platform"]

# LLM numeric/binary
LLM_NUM = [
    "market_relevance",
    "uncertainty_shock",
    "policy_surprise",
    "novelty",
    "tone_valence",
    "tail_risk_severity",
]

# LLM topic categorical
LLM_TOPIC_CAT = ["market_topic"]


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

    base_acc = majority_baseline_accuracy(test2[target].astype(int).values) if len(test2) else float("nan")

    if len(train2) < 200 or len(test2) < 200:
        return {
            "feature_set": set_name,
            "n_train": int(len(train2)),
            "n_test": int(len(test2)),
            "baseline_acc": float(base_acc) if not np.isnan(base_acc) else float("nan"),
            "logit_acc": float("nan"),
            "logit_auc": float("nan"),
        }

    X_train = train2[num_features + cat_features]
    y_train = train2[target].astype(int).values

    X_test = test2[num_features + cat_features]
    y_test = test2[target].astype(int).values

    pipe = make_pipeline(num_features, cat_features)
    pipe.fit(X_train, y_train)

    y_hat = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    return {
        "feature_set": set_name,
        "n_train": int(len(train2)),
        "n_test": int(len(test2)),
        "baseline_acc": float(base_acc),
        "logit_acc": float(accuracy_score(y_test, y_hat)),
        "logit_auc": _auc_safe(y_test, y_prob),
    }


# -----------------------
# Feature set builder
# -----------------------

def build_feature_sets(df: pd.DataFrame) -> List[Tuple[str, List[str], List[str]]]:
    """
    Returns list of tuples: (set_name, numeric_features, categorical_features)
    Only include sets whose columns exist.
    """
    sets: List[Tuple[str, List[str], List[str]]] = []

    has_topic = all(c in df.columns for c in LLM_TOPIC_CAT)

    # --- Grouped sets ---
    sets.append(("TEXT_STATS_ONLY", [c for c in TEXT_STATS if c in df.columns], []))
    sets.append(("LLM_ONLY_NUM", [c for c in LLM_NUM if c in df.columns], []))
    if has_topic:
        sets.append(("LLM_ONLY_TOPIC", [], LLM_TOPIC_CAT))
        sets.append(("LLM_ONLY_NUM+TOPIC", [c for c in LLM_NUM if c in df.columns], LLM_TOPIC_CAT))

    # Controls
    sets.append(("CONTROLS_ONLY", [c for c in BOOLS if c in df.columns], [c for c in CATS if c in df.columns]))

    # Combined
    sets.append(("TEXT+CONTROLS", [c for c in TEXT_STATS + BOOLS if c in df.columns], [c for c in CATS if c in df.columns]))
    sets.append(("LLM_NUM+CONTROLS", [c for c in LLM_NUM + BOOLS if c in df.columns], [c for c in CATS if c in df.columns]))
    sets.append(("TEXT+LLM_NUM+CONTROLS", [c for c in TEXT_STATS + LLM_NUM + BOOLS if c in df.columns], [c for c in CATS if c in df.columns]))

    if has_topic:
        sets.append(("TEXT+LLM_NUM+TOPIC+CONTROLS", [c for c in TEXT_STATS + LLM_NUM + BOOLS if c in df.columns], [c for c in CATS + LLM_TOPIC_CAT if c in df.columns]))

    # --- Single-feature runs (LLM + text + controls) ---
    # Numeric singles
    for f in (TEXT_STATS + BOOLS + LLM_NUM):
        if f in df.columns:
            sets.append((f"SINGLE:{f}", [f], []))

    # Categorical singles
    for f in (CATS + (LLM_TOPIC_CAT if has_topic else [])):
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

    # Minimal sanity checks
    needed = {"created_at", "regime"} | set(TARGETS)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"dataset_model_llm missing required columns: {sorted(missing)}")

    results = run(df)

    # Sort for readability: grouped sets first, then singles
    def order_key(name: str) -> int:
        if name.startswith("TEXT+LLM_NUM+TOPIC+CONTROLS"):
            return 0
        if name.startswith("TEXT+LLM_NUM+CONTROLS"):
            return 1
        if name.startswith("LLM_NUM+CONTROLS"):
            return 2
        if name.startswith("TEXT+CONTROLS"):
            return 3
        if name.startswith("LLM_ONLY_NUM+TOPIC"):
            return 4
        if name.startswith("LLM_ONLY_NUM"):
            return 5
        if name.startswith("LLM_ONLY_TOPIC"):
            return 6
        if name.startswith("TEXT_STATS_ONLY"):
            return 7
        if name.startswith("CONTROLS_ONLY"):
            return 8
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

    print("\n=== Ablation results (Logistic Regression, LLM features) ===")
    print(results.to_string(index=False))

    #save as a .parquet(0 under runs in results, so we can aggregate all results and analyze them
    out_dir = PROJECT_ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # add metadata columns
    results = results.copy()
    results["market"] = "es"          #  vx    es
    results["feature_track"] = "llm"  # llm    classic
    results["model"] = "logit"        # logit     rf
    results["analysis"] = "ablation"  # baseline   ablation

    out_path = out_dir / "ablation_logit_es_llm.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)