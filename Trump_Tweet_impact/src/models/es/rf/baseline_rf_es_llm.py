"""
baseline_rf_es_llm.py

Random Forest benchmarks for ES direction after posts using LLM-derived features.

Goal
----
Test whether simple non-linearities / interactions improve predictive performance
relative to logistic regression, using LLM features as the semantic block.

Input:
- data_processed/dataset_es_llm.parquet

Regimes:
- pre, power, post (estimated separately)

Targets:
- es_up_30m
- es_up_60m

Validation:
- within each regime, sort by created_at
- train on first 70%, test on last 30% (time-respecting split)

Feature specs
-------------
1) stats_only:
   - text stats + controls (platform, is_retweet)

2) stats_plus_llm:
   - text stats + LLM numeric/binary + controls

3) stats_plus_llm_plus_topic:
   - stats_plus_llm + market_topic (one-hot)

4) llm_only:
   - LLM numeric/binary + market_topic + controls (no text stats)

Notes
-----
- RF is a robustness check for non-linear effects; it is not the primary model.
- Hyperparameters are conservative to reduce overfitting in a weak-signal market setting.
- Rows with missing required features for a spec are dropped (no imputation).
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

BOOL_CONTROLS = ["is_retweet"]
CAT_CONTROLS = ["platform"]

LLM_NUMBOOL = [
    "market_relevance",
    "uncertainty_shock",
    "policy_surprise",
    "novelty",
    "tone_valence",
    "tail_risk_severity",
]

LLM_TOPIC_CAT = ["market_topic"]

RF_SPECS: Dict[str, Dict[str, List[str]]] = {
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
    p1 = float(np.mean(y_true))
    return float(max(p1, 1.0 - p1))


def make_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    """
    Create preprocessing + random forest pipeline.
    - Numerics passed through (RF doesn't need scaling)
    - Categoricals one-hot encoded
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=50,
        min_samples_split=100,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        class_weight=None,  # consider "balanced" only if a regime/horizon is extremely skewed
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

    if len(train2) < 200 or len(test2) < 200:
        # guardrail when effective sample becomes tiny due to missingness
        return {
            "n_train": int(len(train2)),
            "n_test": int(len(test2)),
            "baseline_acc": float("nan"),
            "rf_acc": float("nan"),
            "rf_auc": float("nan"),
        }

    X_train = train2[num_cols + cat_cols]
    y_train = train2[target].astype(int).values

    X_test = test2[num_cols + cat_cols]
    y_test = test2[target].astype(int).values

    pipe = make_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    y_hat = pipe.predict(X_test)
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
    for spec in RF_SPECS.values():
        required |= set(spec["num"])
        required |= set(spec["cat"])

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset_model_llm missing columns needed for rf_models_llm: {sorted(missing)}")


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

    print("\n=== Random Forest (LLM features) results by regime/target/spec ===")
    print(results.sort_values(["regime", "target", "spec"]).to_string(index=False))

    print("\n=== Pooled (all regimes combined) ===")
    train_all, test_all = time_split(df, train_frac=0.70)
    for target in TARGETS:
        for spec_name in ["stats_only", "stats_plus_llm", "stats_plus_llm_plus_topic", "llm_only"]:
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
    results["feature_track"] = "llm"  # llm    classic
    results["model"] = "rf"        # logit     rf
    results["analysis"] = "baseline"  # baseline   ablation

    out_path = out_dir / "baseline_rf_es_llm.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)