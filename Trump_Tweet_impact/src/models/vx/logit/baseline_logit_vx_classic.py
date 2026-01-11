"""
baseline_logit_vx_classic.py

Baseline classification models for VX "spike" after posts.

Dataset:
- data_processed/dataset_vx_classic.parquet

Core idea:
VX is a volatility product, so a natural target is *impact magnitude*:
- Define a spike as "VX absolute return is unusually large" at horizon h.
- Specifically: spike_h = 1[ vx_absret_hm > q ], where q is a TRAIN-ONLY percentile.

Targets:
- vx_spike_30m  (from vx_absret_30m)
- vx_spike_60m  (from vx_absret_60m)

Validation:
- within each regime: sort by created_at
- train = first 70%, test = last 30% (time-respecting split)

Feature specs (mirrors ES pipeline idea):
- stats_only
- stats_plus_vader
- stats_plus_vader_plus_topics (if topic columns exist)

Outputs (printed):
- n_train / n_test (after target construction & dropna)
- baseline accuracy (majority class)
- logistic regression accuracy and ROC-AUC on test set

Important:
- We construct spike thresholds using only the training split to avoid leakage.
- VX outcomes can be missing for some horizons; we drop NA only for the chosen horizon/target.
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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

HORIZONS = [30, 60]
TRAIN_FRAC = 0.70
SPIKE_Q = 0.95

# Text stats (baseline)
TEXT_STATS = [
    "n_chars",
    "n_words",
    "n_exclam",
    "n_question",
    "share_upper",
    "has_url",
    "has_hashtag",
]

# VADER: use only vader_pos by default (you can add/remove here)
VADER = ["vader_pos"]

# Controls
BOOL_FEATURES = ["is_retweet"]
CAT_FEATURES = ["platform"]


def detect_topic_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (topic_numeric, topic_categorical) column lists if present.
    - topic flags are numeric (0/1)
    - top_topic is categorical if present
    """
    topic_numeric = []
    topic_categorical = []

    # Flags like topic_..._flag
    topic_numeric.extend([c for c in df.columns if c.startswith("topic_") and c.endswith("_flag")])

    # Optional numeric summary fields (if your topic_embeddings wrote them)
    for c in ["n_topics", "max_topic_similarity", "top_score"]:
        if c in df.columns:
            topic_numeric.append(c)

    # top_topic is categorical if present
    if "top_topic" in df.columns:
        topic_categorical.append("top_topic")

    return sorted(set(topic_numeric)), topic_categorical


def time_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("created_at").reset_index(drop=True)
    n = len(df)
    cut = int(np.floor(train_frac * n))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    p1 = y_true.mean()
    return float(max(p1, 1.0 - p1))


def make_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    transformers = []
    if num_features:
        transformers.append(("num", StandardScaler(), num_features))
    if cat_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def make_spike_target(train_absret: pd.Series, test_absret: pd.Series, q: float) -> Tuple[pd.Series, pd.Series, float]:
    """
    Create binary spike targets using a TRAIN-only threshold (quantile).
    Returns (y_train, y_test, threshold).
    """
    thr = float(train_absret.quantile(q))
    y_train = (train_absret > thr).astype("int8")
    y_test = (test_absret > thr).astype("int8")
    return y_train, y_test, thr


def eval_one_spec(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    spec_name: str,
    num_features: List[str],
    cat_features: List[str],
) -> Dict[str, float | int | str]:
    """
    Evaluate logistic regression for one horizon + feature spec.
    """
    absret_col = f"vx_absret_{horizon}m"
    if absret_col not in train.columns or absret_col not in test.columns:
        raise ValueError(f"Missing {absret_col} in dataset_model_vx.")

    # We must have absret to define target; drop NA only on that column
    train2 = train.dropna(subset=[absret_col]).copy()
    test2 = test.dropna(subset=[absret_col]).copy()

    # Construct targets with train-only threshold
    y_train, y_test, thr = make_spike_target(train2[absret_col], test2[absret_col], SPIKE_Q)

    # Assemble X
    cols_needed = ["created_at"] + num_features + cat_features
    # Be strict: drop rows with missing features (rare if your pipeline is clean)
    trainX = train2[cols_needed].dropna().copy()
    testX = test2[cols_needed].dropna().copy()

    # Align y to the dropped rows (index-safe)
    y_train = y_train.loc[trainX.index].astype(int).values
    y_test = y_test.loc[testX.index].astype(int).values

    # Baseline acc
    base_acc = majority_baseline_accuracy(y_test)

    # If test set has only one class, AUC undefined
    if len(np.unique(y_test)) < 2:
        return {
            "target": f"vx_spike_{horizon}m",
            "spec": spec_name,
            "threshold": thr,
            "n_train": int(len(trainX)),
            "n_test": int(len(testX)),
            "baseline_acc": float(base_acc),
            "logit_acc": float("nan"),
            "logit_auc": float("nan"),
        }

    pipe = make_pipeline(num_features=num_features, cat_features=cat_features)
    pipe.fit(trainX[num_features + cat_features], y_train)

    y_hat = pipe.predict(testX[num_features + cat_features])
    y_prob = pipe.predict_proba(testX[num_features + cat_features])[:, 1]

    return {
        "target": f"vx_spike_{horizon}m",
        "spec": spec_name,
        "threshold": float(thr),
        "n_train": int(len(trainX)),
        "n_test": int(len(testX)),
        "baseline_acc": float(base_acc),
        "logit_acc": float(accuracy_score(y_test, y_hat)),
        "logit_auc": float(roc_auc_score(y_test, y_prob)),
    }


def run_by_regime(df: pd.DataFrame, regimes: List[str] = ["pre", "power", "post"]) -> pd.DataFrame:
    rows = []

    topic_num, topic_cat = detect_topic_columns(df)

    # Build specs (only include topics spec if there are topic columns)
    specs = [
        ("stats_only", TEXT_STATS + BOOL_FEATURES, CAT_FEATURES),
        ("stats_plus_vader", TEXT_STATS + VADER + BOOL_FEATURES, CAT_FEATURES),
    ]
    if topic_num or topic_cat:
        specs.append(("stats_plus_vader_plus_topics", TEXT_STATS + VADER + BOOL_FEATURES + topic_num, CAT_FEATURES + topic_cat))

    for reg in regimes:
        sub = df.loc[df["regime"] == reg].copy()
        if len(sub) < 500:
            continue

        train, test = time_split(sub, train_frac=TRAIN_FRAC)

        for h in HORIZONS:
            for spec_name, num_feats, cat_feats in specs:
                res = eval_one_spec(train, test, horizon=h, spec_name=spec_name, num_features=num_feats, cat_features=cat_feats)
                rows.append({"regime": reg, **res})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    data_path = PROJECT_ROOT / "data_processed" / "dataset_vx_classic.parquet"

    df = pd.read_parquet(data_path)

    # Basic checks
    required = {"created_at", "regime"} | set(TEXT_STATS) | set(BOOL_FEATURES) | set(CAT_FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset_model_vx missing required columns: {sorted(missing)}")

    results = run_by_regime(df)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 60)

    print("\n=== Logistic Regression (VX spike targets) results by regime/target/spec ===")
    print(results.sort_values(["regime", "target", "spec"]).to_string(index=False))

    # Optional pooled (same split logic but pooled data)
    print("\n=== Pooled (all regimes combined) ===")
    train_all, test_all = time_split(df, train_frac=TRAIN_FRAC)

    # detect topics once for pooled too
    topic_num, topic_cat = detect_topic_columns(df)
    pooled_specs = [
        ("stats_only", TEXT_STATS + BOOL_FEATURES, CAT_FEATURES),
        ("stats_plus_vader", TEXT_STATS + VADER + BOOL_FEATURES, CAT_FEATURES),
    ]
    if topic_num or topic_cat:
        pooled_specs.append(("stats_plus_vader_plus_topics", TEXT_STATS + VADER + BOOL_FEATURES + topic_num, CAT_FEATURES + topic_cat))

    for h in HORIZONS:
        for spec_name, num_feats, cat_feats in pooled_specs:
            m = eval_one_spec(train_all, test_all, horizon=h, spec_name=spec_name, num_features=num_feats, cat_features=cat_feats)
            print(
                f"{m['target']} | {spec_name}: "
                f"n_train={m['n_train']}, n_test={m['n_test']}, thr(q{int(SPIKE_Q * 100)})={m['threshold']:.6f}, "
                f"baseline_acc={m['baseline_acc']:.4f}, logit_acc={m['logit_acc']:.4f}, logit_auc={m['logit_auc']:.4f}"
            )



    #save as a .parquet(0 under runs in results, so we can aggregate all results and analyze them
    out_dir = PROJECT_ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # add metadata columns
    results = results.copy()
    results["market"] = "vx"          #  vx    es
    results["feature_track"] = "classic"  # llm    classic
    results["model"] = "logit"        # logit     rf
    results["analysis"] = "baseline"  # baseline   ablation

    out_path = out_dir / "baseline_logit_vx_classic.parquet"
    results.to_parquet(out_path, index=False)

    print("Saved results:", out_path)