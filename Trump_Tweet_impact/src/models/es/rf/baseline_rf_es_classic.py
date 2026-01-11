"""
baseline_rf_es_classic.py

Random Forest benchmarks for ES direction after posts.

Goal
----
Test whether simple non-linearities / interactions improve predictive performance
relative to the logistic regression baselines, under the same temporal validation.

Input:
- data_processed/dataset_es_classic.parquet

Regimes:
- pre, power, post (estimated separately)

Targets:
- es_up_30m
- es_up_60m

Validation:
- within each regime, sort by created_at
- train on first 70%, test on last 30% (time-respecting split)

Feature sets (two RF specs)
--------------------------
A) rf_baseline:
   - text stats + vader_pos + controls (platform, is_retweet)

B) rf_baseline_plus_topics:
   - same as A) plus top_topic (one-hot)

Notes
-----
- RF is used as a robustness check for non-linear effects; it is not the primary model.
- We keep hyperparameters conservative (shallow depth, large min_samples_leaf)
  to reduce overfitting in a weak-signal market setting.
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

TARGETS = ["es_up_30m", "es_up_60m"]

TEXT_STATS = [
    "n_chars",
    "n_words",
    "n_exclam",
    "n_question",
    "share_upper",
    "has_url",
    "has_hashtag",
]

# Per ablation diagnostics, keep only vader_pos
VADER = ["vader_pos"]

BOOL_CONTROLS = ["is_retweet"]
CAT_CONTROLS = ["platform"]

TOPIC_CAT = ["top_topic"]

RF_SPECS = {
    "stats_only": {
        "num": TEXT_STATS + BOOL_CONTROLS,
        "cat": CAT_CONTROLS,
    },
    "stats_plus_vader": {
        "num": TEXT_STATS + VADER + BOOL_CONTROLS,   # VADER = ["vader_pos"]
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
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    p1 = float(np.mean(y_true))
    return float(max(p1, 1.0 - p1))


def make_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    """
    Create preprocessing + random forest pipeline.
    - Numerics are passed through without scaling (RF doesn't need it).
    - Categoricals are one-hot encoded.
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=50,
        min_samples_split=100,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight=None,  # consider "balanced" if classes become skewed; not needed here
    )

    return Pipeline(steps=[("pre", pre), ("rf", rf)])


def _auc_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def eval_target_spec(train: pd.DataFrame, test: pd.DataFrame, target: str, spec_name: str) -> Dict[str, float]:
    spec = RF_SPECS[spec_name]
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
    # RF supports predict_proba
    y_prob = pipe.predict_proba(X_test)[:, 1]

    return {
        "n_train": int(len(train2)),
        "n_test": int(len(test2)),
        "baseline_acc": float(majority_baseline_accuracy(y_test)),
        "rf_acc": float(accuracy_score(y_test, y_hat)),
        "rf_auc": float(_auc_safe(y_test, y_prob)),
    }


def run_by_regime(
    df: pd.DataFrame,
    regimes: List[str] = ["pre", "power", "post"],
    specs: List[str] = ["stats_only", "stats_plus_vader", "stats_plus_vader_plus_topics"],
) -> pd.DataFrame:
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
    required = set(["created_at", "regime"] + TARGETS)
    for spec in RF_SPECS.values():
        required |= set(spec["num"])
        required |= set(spec["cat"])

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset_model missing columns needed for rf_models: {sorted(missing)}")


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_es_classic.parquet"

    df = pd.read_parquet(data_path)
    _check_required_columns(df)

    results = run_by_regime(df)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 60)

    print("\n=== Random Forest results by regime/target/spec ===")
    print(results.sort_values(["regime", "target", "spec"]).to_string(index=False))

    print("\n=== Pooled (all regimes combined) ===")
    train_all, test_all = time_split(df, train_frac=0.70)
    for target in TARGETS:
        for spec_name in ["stats_only", "stats_plus_vader", "stats_plus_vader_plus_topics"]:
            m = eval_target_spec(train_all, test_all, target, spec_name)
            print(
                f"{target} | {spec_name}: "
                f"n_train={m['n_train']}, n_test={m['n_test']}, "
                f"baseline_acc={m['baseline_acc']:.4f}, rf_acc={m['rf_acc']:.4f}, rf_auc={m['rf_auc']:.4f}"
            )

    #save as a .parquet(0 under runs in results, so we can aggregate all results and analyze them
    out_dir = PROJECT_ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # add metadata columns
    results = results.copy()
    results["market"] = "es"          #  vx    es
    results["feature_track"] = "classic"  # llm    classic
    results["model"] = "rf"        # logit     rf
    results["analysis"] = "baseline"  # baseline   ablation

    out_path = out_dir / "baseline_rf_es_classic.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)