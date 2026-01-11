"""
baseline_logit_vx_llm.py

Logistic regression benchmarks for VX volatility spikes after posts, using LLM-derived features.

Input:
- data_processed/dataset_vx_llm.parquet

We estimate models separately by political regime:
- pre
- power
- post

Targets (constructed without leakage)
------------------------------------
We use a binary "volatility spike" indicator based on absolute VX moves:

For horizon h in {30m, 60m}:
- vx_spike_h = 1{ vx_absret_h > Q_train(0.95) }, else 0

where Q_train(0.95) is the 95th percentile computed on the TRAIN split within the same regime.
This makes the spike definition time-respecting and avoids leakage.

Validation:
- within each regime, sort by created_at
- train on first 70%, test on last 30% (time-respecting split)

Feature sets (ablation-style):
1) stats_only:
   - simple text statistics + controls
2) llm_only:
   - LLM numeric/binary + controls + market_topic
3) stats_plus_llm:
   - text stats + LLM numeric/binary + controls
4) stats_plus_llm_plus_topic:
   - stats_plus_llm + market_topic

Outputs (printed):
- spike threshold used (train 95th percentile)
- spike base rate in test set
- majority-class baseline accuracy
- logistic regression accuracy and ROC-AUC on test set
- results table by (regime, target, spec)

Notes:
- VX outcomes can be NaN for some horizons; we drop NaNs only for the specific horizon used.
- AUC is the primary metric; accuracy can be misleading with class imbalance.
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

HORIZONS = [30, 60]  # minutes
SPIKE_Q = 0.95

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

# Controls
BOOL_CONTROLS = ["is_retweet"]   # numeric 0/1
CAT_CONTROLS = ["platform"]      # one-hot

# LLM topic categorical
LLM_TOPIC_CAT = ["market_topic"]

SPECS: Dict[str, Dict[str, List[str]]] = {
    "stats_only": {
        "num": TEXT_STATS + BOOL_CONTROLS,
        "cat": CAT_CONTROLS,
    },
    "llm_only": {
        "num": LLM_NUMBOOL + BOOL_CONTROLS,
        "cat": CAT_CONTROLS + LLM_TOPIC_CAT,
    },
    "stats_plus_llm": {
        "num": TEXT_STATS + LLM_NUMBOOL + BOOL_CONTROLS,
        "cat": CAT_CONTROLS,
    },
    "stats_plus_llm_plus_topic": {
        "num": TEXT_STATS + LLM_NUMBOOL + BOOL_CONTROLS,
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
    return float(max(p1, 1.0 - p1))


def make_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    """Preprocess + logistic regression pipeline for a given feature spec."""
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


def _auc_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC undefined if only one class present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _make_spike_labels(train_abs: pd.Series, test_abs: pd.Series, q: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build spike labels using threshold computed on train only (no leakage).
    Returns (y_train, y_test, threshold).
    """
    thr = float(train_abs.quantile(q))
    y_train = (train_abs > thr).astype(int).to_numpy()
    y_test = (test_abs > thr).astype(int).to_numpy()
    return y_train, y_test, thr


def eval_horizon_spec(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    spec_name: str,
) -> Dict[str, float]:
    """
    Fit logit on train and evaluate on test for one horizon under one feature spec.
    Target = spike indicator from vx_absret_{h} using train 95th percentile.
    """
    abs_col = f"vx_absret_{horizon}m"
    if abs_col not in train.columns or abs_col not in test.columns:
        raise ValueError(f"Missing required VX column: {abs_col}")

    spec = SPECS[spec_name]
    num_cols = spec["num"]
    cat_cols = spec["cat"]

    cols_needed = ["created_at", abs_col] + num_cols + cat_cols

    # Drop NA only for columns needed in this run (keeps VX NaN policy consistent)
    train2 = train[cols_needed].dropna().copy()
    test2 = test[cols_needed].dropna().copy()

    if len(train2) < 300 or len(test2) < 300:
        return {
            "n_train": int(len(train2)),
            "n_test": int(len(test2)),
            "thr_train_q95": float("nan"),
            "test_spike_rate": float("nan"),
            "baseline_acc": float("nan"),
            "logit_acc": float("nan"),
            "logit_auc": float("nan"),
        }

    y_train, y_test, thr = _make_spike_labels(train2[abs_col], test2[abs_col], q=SPIKE_Q)

    X_train = train2[num_cols + cat_cols]
    X_test = test2[num_cols + cat_cols]

    pipe = make_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    y_hat = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    base = majority_baseline_accuracy(y_test)
    spike_rate = float(np.mean(y_test))

    return {
        "n_train": int(len(train2)),
        "n_test": int(len(test2)),
        "thr_train_q95": float(thr),
        "test_spike_rate": float(spike_rate),
        "baseline_acc": float(base),
        "logit_acc": float(accuracy_score(y_test, y_hat)),
        "logit_auc": float(_auc_safe(y_test, y_prob)),
    }


def run_by_regime(
    df: pd.DataFrame,
    regimes: List[str] = ["pre", "power", "post"],
    specs: List[str] = ["stats_only", "llm_only", "stats_plus_llm", "stats_plus_llm_plus_topic"],
) -> pd.DataFrame:
    rows = []
    for reg in regimes:
        sub = df.loc[df["regime"] == reg].copy()
        if len(sub) < 500:
            continue

        train, test = time_split(sub, train_frac=0.70)

        for h in HORIZONS:
            for spec_name in specs:
                m = eval_horizon_spec(train, test, horizon=h, spec_name=spec_name)
                rows.append(
                    {
                        "regime": reg,
                        "target": f"vx_spike_{h}m",
                        "spec": spec_name,
                        **m,
                    }
                )

    return pd.DataFrame(rows)


def _check_required_columns(df: pd.DataFrame) -> None:
    required = {"created_at", "regime"}
    required |= {f"vx_absret_{h}m" for h in HORIZONS}

    for spec in SPECS.values():
        required |= set(spec["num"])
        required |= set(spec["cat"])

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset_model_vx_llm missing columns needed for baseline_models_vx_llm: {sorted(missing)}")


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_vx_llm.parquet"

    df = pd.read_parquet(data_path)
    _check_required_columns(df)

    results = run_by_regime(df)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 80)

    print("\n=== Logistic Regression (VX spike targets, LLM features) ===")
    print(results.sort_values(["regime", "target", "spec"]).to_string(index=False))

    # Optional pooled run (all regimes combined)
    print("\n=== Pooled (all regimes combined) ===")
    train_all, test_all = time_split(df, train_frac=0.70)
    for h in HORIZONS:
        for spec_name in ["stats_only", "llm_only", "stats_plus_llm", "stats_plus_llm_plus_topic"]:
            m = eval_horizon_spec(train_all, test_all, horizon=h, spec_name=spec_name)
            print(
                f"vx_spike_{h}m | {spec_name}: "
                f"n_train={m['n_train']}, n_test={m['n_test']}, "
                f"thr_train_q95={m['thr_train_q95']:.6g}, test_spike_rate={m['test_spike_rate']:.3f}, "
                f"baseline_acc={m['baseline_acc']:.4f}, logit_acc={m['logit_acc']:.4f}, logit_auc={m['logit_auc']:.4f}"
            )


    #save as a .parquet(0 under runs in results, so we can aggregate all results and analyze them
    out_dir = PROJECT_ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # add metadata columns
    results = results.copy()
    results["market"] = "vx"          #  vx    es
    results["feature_track"] = "llm"  # llm    classic
    results["model"] = "logit"        # logit     rf
    results["analysis"] = "baseline"  # baseline   ablation

    out_path = out_dir / "baseline_logit_vx_llm.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)