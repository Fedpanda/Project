"""
baseline_logit_es_classic.py

Logistic regression benchmarks (with ablation-style feature sets) for ES direction after posts.

Input:
- data_processed/dataset_es_classic.parquet

We estimate models separately by political regime:
- pre
- power
- post

Targets:
- es_up_30m
- es_up_60m

Validation:
- within each regime, sort by created_at
- train on first 70%, test on last 30% (time-respecting split)

Feature sets (three specifications):
1) stats_only:
   - simple text statistics + controls
2) stats_plus_vader:
   - text statistics + VADER sentiment + controls
3) stats_plus_vader_plus_topics:
   - text statistics + VADER sentiment + topic feature(s) + controls

Topic representation:
- We use `top_topic` as a categorical variable (one-hot encoded).
  This avoids threshold tuning and sparsity from topic flags.

Outputs (printed):
- majority-class baseline accuracy
- logistic regression accuracy and ROC-AUC on test set
- results table by (regime, target, spec)

Notes for paper:
- The time split prevents look-ahead and mimics a forecasting setting.
- Regime-specific models are economically motivated (communication role differs by regime).
- Comparing specs is an ablation-style evaluation: incremental value of feature blocks.
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


# -----------------------
# Config
# -----------------------

TARGETS = ["es_up_30m", "es_up_60m"]

# Text-stat features (attention / intensity proxies)
TEXT_STATS = [
    "n_chars",
    "n_words",
    "n_exclam",
    "n_question",
    "share_upper",
    "has_url",
    "has_hashtag",
]

# VADER sentiment:
# Based on ablation diagnostics, we retain only vader_pos.
# Other components (neg, neu, compound) are either inverted or wash out signal
# due to strong correlation and aggregation effects.
VADER = [
#    "vader_neg",
#    "vader_neu",
    "vader_pos",
#    "vader_compound",
]

# Controls
CAT_CONTROLS = ["platform"]        # one-hot
BOOL_CONTROLS = ["is_retweet"]     # numeric 0/1

# Topic representation (categorical)
TOPIC_CAT = ["top_topic"]          # one-hot

# Three specs (ablation-style)
SPECS: Dict[str, Dict[str, List[str]]] = {
    "stats_only": {
        "num": TEXT_STATS + BOOL_CONTROLS,
        "cat": CAT_CONTROLS,
    },
    "stats_plus_vader": {
        "num": TEXT_STATS + VADER + BOOL_CONTROLS,
        "cat": CAT_CONTROLS,
    },
    "stats_plus_vader_plus_topics": {
        "num": TEXT_STATS + VADER + BOOL_CONTROLS,
        "cat": CAT_CONTROLS + TOPIC_CAT,
    },
}


# -----------------------
# Helpers
# -----------------------

def time_split(df: pd.DataFrame, train_frac: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-respecting split: sort by created_at then split by fraction.
    """
    df = df.sort_values("created_at").reset_index(drop=True)
    n = len(df)
    cut = int(np.floor(train_frac * n))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    """
    Accuracy of predicting the majority class in y_true.
    """
    p1 = float(np.mean(y_true))
    return max(p1, 1.0 - p1)


def make_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    """
    Create preprocessing + logistic regression pipeline for a given feature spec.
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        n_jobs=None,  # lbfgs ignores n_jobs; keep explicit
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def eval_target_spec(train: pd.DataFrame, test: pd.DataFrame, target: str, spec_name: str) -> Dict[str, float]:
    """
    Fit logistic regression on train and evaluate on test for a single target under one feature spec.
    """
    spec = SPECS[spec_name]
    num_cols = spec["num"]
    cat_cols = spec["cat"]

    cols_needed = ["created_at", target] + num_cols + cat_cols

    train2 = train[cols_needed].dropna().copy()
    test2 = test[cols_needed].dropna().copy()

    X_train = train2[num_cols + cat_cols]
    y_train = train2[target].astype(int).values

    X_test = test2[num_cols + cat_cols]
    y_test = test2[target].astype(int).values

    pipe = make_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    y_hat = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_hat)
    auc = roc_auc_score(y_test, y_prob)
    base = majority_baseline_accuracy(y_test)

    return {
        "n_train": int(len(train2)),
        "n_test": int(len(test2)),
        "baseline_acc": float(base),
        "logit_acc": float(acc),
        "logit_auc": float(auc),
    }


def run_by_regime(
    df: pd.DataFrame,
    regimes: List[str] = ["pre", "power", "post"],
    specs: List[str] = ["stats_only", "stats_plus_vader", "stats_plus_vader_plus_topics"],
) -> pd.DataFrame:
    """
    Run evaluation for each regime, target, and feature spec.
    """
    rows = []
    for reg in regimes:
        sub = df.loc[df["regime"] == reg].copy()
        if len(sub) < 500:
            continue

        train, test = time_split(sub, train_frac=0.70)

        for target in TARGETS:
            for spec_name in specs:
                metrics = eval_target_spec(train, test, target, spec_name)
                rows.append({"regime": reg, "target": target, "spec": spec_name, **metrics})

    return pd.DataFrame(rows)


def _check_required_columns(df: pd.DataFrame) -> None:
    """
    Sanity-check that the dataset has what we need for the chosen specs.
    """
    required = set(["created_at", "regime"] + TARGETS)

    # Add all columns referenced by specs
    for spec in SPECS.values():
        required |= set(spec["num"])
        required |= set(spec["cat"])

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset_model missing columns needed for baseline_models: {sorted(missing)}")


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_es_classic.parquet"

    df = pd.read_parquet(data_path)

    _check_required_columns(df)

    results = run_by_regime(df)

    # Pretty print
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 60)

    print("\n=== Logistic Regression results by regime/target/spec ===")
    print(results.sort_values(["regime", "target", "spec"]).to_string(index=False))

    # Optional pooled results (all regimes combined)
    print("\n=== Pooled (all regimes combined) ===")
    train_all, test_all = time_split(df, train_frac=0.70)
    for target in TARGETS:
        for spec_name in ["stats_only", "stats_plus_vader", "stats_plus_vader_plus_topics"]:
            m = eval_target_spec(train_all, test_all, target, spec_name)
            print(
                f"{target} | {spec_name}: "
                f"n_train={m['n_train']}, n_test={m['n_test']}, "
                f"baseline_acc={m['baseline_acc']:.4f}, logit_acc={m['logit_acc']:.4f}, logit_auc={m['logit_auc']:.4f}"
            )


    #save as a .parquet(0 under runs in results, so we can aggregate all results and analyze them
    out_dir = PROJECT_ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # add metadata columns
    results = results.copy()
    results["market"] = "es"          #  vx    es
    results["feature_track"] = "classic"  # llm    classic
    results["model"] = "logit"        # logit     rf
    results["analysis"] = "baseline"  # baseline   ablation

    out_path = out_dir / "baseline_logit_es_classic.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)