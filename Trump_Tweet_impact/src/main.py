"""
src/main.py

End-to-end project orchestrator (run from repo root):
- ingestion (sanity checks)
- preprocessing (build + save event windows)
- features (classic + optional LLM; LLM step is commented by default)
- EDA (optional)
- build model datasets (es/vx; classic/llm)
- models (baselines + ablations + optional xgb)
- results aggregation + analysis/plots

Recommended usage (from project root)
-------------------------------------
# Full pipeline except LLM API call (uses existing features_llm_all.parquet if present):
python -m src.main --all

# Full pipeline including EDA:
python -m src.main --all --eda

# Run only from models onward (if data_processed already exists):
python -m src.main --models --aggregate --analyze

# Build up to datasets (events + features + dataset_*):
python -m src.main --build

Notes
-----
- The LLM step (OpenRouter) is intentionally NOT run by default.
  It can be enabled by uncommenting the line in run_features() below and/or adding a flag.
- Each module is executed via runpy.run_module(..., run_name="__main__") so that each script’s
  `if __name__ == "__main__":` block is triggered.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------
# Ensure repo root is on sys.path even when running as a script
# .../Trump_Tweet_impact/src/main.py -> repo root is parents[1]
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def run_module(mod: str) -> None:
    """Execute a module as if running `python -m <mod>`."""
    print(f"\n[RUN] python -m {mod}")
    runpy.run_module(mod, run_name="__main__")


# ---------------------------------------------------------------------
# Stage module lists
# ---------------------------------------------------------------------
def ingestion_modules() -> List[str]:
    # Ingestion scripts are mainly for sanity checks; the real pipeline uses save_events.
    return [
        "src.ingestion.load_tweets",
        "src.ingestion.load_market",
    ]


def preprocessing_modules() -> List[str]:
    # save_events calls ingestion + event_windows and writes data_processed/events_*.parquet
    return [
        "src.preprocessing.save_events",
    ]


def feature_modules() -> List[str]:
    # Classic features (no API calls)
    return [
        "src.features.text_stats",
        "src.features.vader_sentiment",
        "src.features.topic_embeddings",
        # LLM features (API call) — INTENTIONALLY NOT RUN BY DEFAULT.
        # To enable: uncomment the line below AND ensure you have your OpenRouter key set.
        # "src.features.llm_sentiment_openrouter",
    ]


def eda_modules() -> List[str]:
    return [
        "src.eda.text_eda",
    ]


def build_dataset_modules(markets: List[str]) -> List[str]:
    mods: List[str] = []
    if "es" in markets:
        mods += [
            "src.models.es.build.build_dataset_es_classic",
            "src.models.es.build.build_dataset_es_llm",
        ]
    if "vx" in markets:
        mods += [
            "src.models.vx.build.build_dataset_vx_classic",
            "src.models.vx.build.build_dataset_vx_llm",
        ]
    return mods


def model_modules(markets: List[str], include_xgb: bool) -> List[str]:
    mods: List[str] = []

    if "es" in markets:
        mods += [
            # baselines
            "src.models.es.logit.baseline_logit_es_classic",
            "src.models.es.logit.baseline_logit_es_llm",
            "src.models.es.rf.baseline_rf_es_classic",
            "src.models.es.rf.baseline_rf_es_llm",
            # ablations
            "src.models.es.logit.ablation_logit_es_classic",
            "src.models.es.logit.ablation_logit_es_llm",
            "src.models.es.rf.ablation_rf_es_classic",
            "src.models.es.rf.ablation_rf_es_llm",
        ]

    if "vx" in markets:
        mods += [
            # baselines
            "src.models.vx.logit.baseline_logit_vx_classic",
            "src.models.vx.logit.baseline_logit_vx_llm",
            "src.models.vx.rf.baseline_rf_vx_classic",
            "src.models.vx.rf.baseline_rf_vx_llm",
            # ablations
            "src.models.vx.logit.ablation_logit_vx_classic",
            "src.models.vx.logit.ablation_logit_vx_llm",
            "src.models.vx.rf.ablation_rf_vx_classic",
            "src.models.vx.rf.ablation_rf_vx_llm",
        ]
        if include_xgb:
            mods += [
                "src.models.vx.xgb.baseline_xgb_classic",
                "src.models.vx.xgb.baseline_xgb_llm",
            ]

    return mods


def results_modules(do_aggregate: bool, do_analyze: bool) -> List[str]:
    mods: List[str] = []
    if do_aggregate:
        mods.append("src.results.aggregate_results")
    if do_analyze:
        mods.append("src.results.analyze_aggregates")
    return mods


# ---------------------------------------------------------------------
# Orchestrated stages
# ---------------------------------------------------------------------
def run_ingestion() -> None:
    print("\n=== STAGE: ingestion (sanity checks) ===")
    for m in ingestion_modules():
        run_module(m)


def run_preprocessing() -> None:
    print("\n=== STAGE: preprocessing (event windows -> data_processed/events_*.parquet) ===")
    for m in preprocessing_modules():
        run_module(m)


def run_features() -> None:
    print("\n=== STAGE: features (classic; LLM optional/commented) ===")
    for m in feature_modules():
        run_module(m)

    # Important: This is where the LLM step would go if you want it:
    # run_module("src.features.llm_sentiment_openrouter")


def run_eda() -> None:
    print("\n=== STAGE: EDA (optional) ===")
    for m in eda_modules():
        run_module(m)


def run_build_datasets(markets: List[str]) -> None:
    print("\n=== STAGE: build model datasets (dataset_*_classic/llm.parquet) ===")
    for m in build_dataset_modules(markets):
        run_module(m)


def run_models(markets: List[str], include_xgb: bool) -> None:
    print("\n=== STAGE: models (baselines + ablations + optional xgb) ===")
    for m in model_modules(markets, include_xgb=include_xgb):
        run_module(m)


def run_results(do_aggregate: bool, do_analyze: bool) -> None:
    print("\n=== STAGE: results (aggregate + analyze) ===")
    for m in results_modules(do_aggregate=do_aggregate, do_analyze=do_analyze):
        run_module(m)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Run full pipeline: ingestion->preprocessing->features->build->models->aggregate->analyze")
    ap.add_argument("--ingest", action="store_true", help="Run ingestion sanity checks")
    ap.add_argument("--preprocess", action="store_true", help="Run preprocessing (save_events -> events_*.parquet)")
    ap.add_argument("--features", action="store_true", help="Run feature generation (classic only; LLM step is commented in code)")
    ap.add_argument("--eda", action="store_true", help="Run EDA script(s)")
    ap.add_argument("--build", action="store_true", help="Build model datasets (dataset_*_classic/llm.parquet)")
    ap.add_argument("--models", action="store_true", help="Run baselines + ablations (writes results/runs/*.parquet)")
    ap.add_argument("--aggregate", action="store_true", help="Aggregate results/runs -> results/tables")
    ap.add_argument("--analyze", action="store_true", help="Analyze aggregates -> plots/figures")
    ap.add_argument("--markets", nargs="+", default=["es", "vx"], choices=["es", "vx"], help="Markets to run")
    ap.add_argument("--no-xgb", action="store_true", help="Disable XGBoost baselines (VX only)")

    args = ap.parse_args()

    markets = args.markets
    include_xgb = not args.no_xgb

    # Default behavior: if no flags, run the common "models->aggregate->analyze"
    if not any([args.all, args.ingest, args.preprocess, args.features, args.eda, args.build, args.models, args.aggregate, args.analyze]):
        args.models = True
        args.aggregate = True
        args.analyze = True

    if args.all:
        run_ingestion()
        run_preprocessing()
        run_features()
        if args.eda:
            run_eda()
        run_build_datasets(markets)
        run_models(markets, include_xgb=include_xgb)
        run_results(do_aggregate=True, do_analyze=True)
        print("\n[OK] Done.")
        return

    # Otherwise run requested stages in the intended pipeline order
    if args.ingest:
        run_ingestion()
    if args.preprocess:
        run_preprocessing()
    if args.features:
        run_features()
    if args.eda:
        run_eda()
    if args.build:
        run_build_datasets(markets)
    if args.models:
        run_models(markets, include_xgb=include_xgb)
    if args.aggregate or args.analyze:
        run_results(do_aggregate=args.aggregate, do_analyze=args.analyze)

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
