"""
baseline_logit_es_llm.py

Logistic regression benchmarks for ES direction using LLM-derived features.

Input:
- data_processed/dataset_es_llm.parquet
  (baseline dataset_model + LEFT-merged LLM features by post_id)

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

Feature specs (ablation-style):
1) stats_only:
   - simple text statistics + controls (platform, is_retweet)
2) stats_plus_llm:
   - stats_only + LLM numeric/binary features (excluding market_topic)
3) stats_plus_llm_plus_topic:
   - stats_plus_llm + market_topic one-hot
4) llm_only:
   - LLM numeric/binary + topic + controls (no text stats)

Notes:
- We intentionally exclude VADER and embedding topics here; LLM features are the replacement block.
- LLM features may be missing for some rows (e.g., skipped items). We drop rows with NA within each spec.
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

# Controls
CAT_CONTROLS = ["platform"]        # one-hot
BOOL_CONTROLS = ["is_retweet"]     # numeric 0/1

# LLM features
LLM_NUMBOOL = [
    "market_relevance",        # 0/1
    "uncertainty_shock",       # 0/1
    "policy_surprise",         # 0/1
    "novelty",                 # 0..10
    "tone_valence",            # 0..10
    "tail_risk_severity",      # 0..3
]
LLM_TOPIC_CAT = ["market_topic"]   # one-hot

SPECS: Dict[str, Dict[str, List[str]]] = {
    "stats_only": {
        "num": TEXT_STATS + BOOL_CONTROLS,
        "cat": CAT_CONTROLS,
    },
    "stats_plus_llm": {
        "num": TEXT_STATS + BOOL_CONTROLS + LLM_NUMBOOL,
        "cat": CAT_CONTROLS,
    },
    "stats_plus_llm_plus_topic": {
        "num": TEXT_STATS + BOOL_CONTROLS + LLM_NUMBOOL,
        "cat": CAT_CONTROLS + LLM_TOPIC_CAT,
    },
    "llm_only": {
        "num": BOOL_CONTROLS + LLM_NUMBOOL,
        "cat": CAT_CONTROLS + LLM_TOPIC_CAT,
    },
}


# -----------------------
# Helpers
# -----------------------

def time_split(df: pd.DataFrame, train_frac: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-respecting split: sort by created_at then split by fraction."""
    df = df.sort_values("created_at").reset_index(drop=True)
    n = len(df)
    cut = int(np.floor(train_frac * n))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    """Accuracy of predicting the majority class in y_true."""
    p1 = float(np.mean(y_true))
    return max(p1, 1.0 - p1)


def make_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    """Create preprocessing + logistic regression pipeline for a given feature spec."""
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
        n_jobs=None,
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def eval_target_spec(train: pd.DataFrame, test: pd.DataFrame, target: str, spec_name: str) -> Dict[str, float]:
    """Fit logistic regression on train and evaluate on test for one target/spec."""
    spec = SPECS[spec_name]
    num_cols = spec["num"]
    cat_cols = spec["cat"]

    cols_needed = ["created_at", target] + num_cols + cat_cols

    # Drop rows with missing needed fields (esp. LLM features if any were skipped)
    train2 = train[cols_needed].dropna().copy()
    test2 = test[cols_needed].dropna().copy()

    if len(train2) < 200 or len(test2) < 200:
        return {
            "n_train": int(len(train2)),
            "n_test": int(len(test2)),
            "baseline_acc": np.nan,
            "logit_acc": np.nan,
            "logit_auc": np.nan,
        }

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
    specs: List[str] = ["stats_only", "stats_plus_llm", "stats_plus_llm_plus_topic", "llm_only"],
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

    for spec in SPECS.values():
        required |= set(spec["num"])
        required |= set(spec["cat"])

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "dataset_model_llm missing columns needed for baseline_models_llm: "
            f"{sorted(missing)}"
        )


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_es_llm.parquet"

    df = pd.read_parquet(data_path)

    _check_required_columns(df)

    results = run_by_regime(df)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 80)

    print("\n=== Logistic Regression (LLM features) results by regime/target/spec ===")
    print(results.sort_values(["regime", "target", "spec"]).to_string(index=False))

    # Optional pooled results (all regimes combined)
    print("\n=== Pooled (all regimes combined) ===")
    train_all, test_all = time_split(df, train_frac=0.70)
    for target in TARGETS:
        for spec_name in ["stats_only", "stats_plus_llm", "stats_plus_llm_plus_topic", "llm_only"]:
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
    results["feature_track"] = "llm"  # llm    classic
    results["model"] = "logit"        # logit     rf
    results["analysis"] = "baseline"  # baseline   ablation

    out_path = out_dir / "baseline_logit_es_llm.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)