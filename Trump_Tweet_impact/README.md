# Trump Tweet Impact on Financial Markets

This project studies whether textual information from Donald J. Trump’s social media posts contains predictive signals for short-horizon reactions in U.S. equity and volatility markets.  
The analysis combines social media text data with minute-level futures data and evaluates a range of machine-learning models under realistic, time-respecting validation.

---

## Project Overview

We analyze the short-term impact of Trump’s posts on:

- **Equity markets**: S&P 500 futures (ES)  
- **Volatility markets**: VIX futures (VX)

Two prediction tasks are considered:

1. **Equity return direction** (30- and 60-minute horizons)
2. **Volatility spikes** (large absolute VX moves over the same horizons)

The project compares:
- Linear vs nonlinear models (Logistic Regression, Random Forest, XGBoost)
- “Classic” text features vs **LLM-derived semantic features**
- Equity vs volatility responses
- Baseline vs ablation feature diagnostics

---

## Repository Structure

data_raw/ — raw input files (tweets + market minute data)

trump_archive/ — Twitter + Truth Social sources

market_data/ — ES/VX minute CSVs

data_processed/ — processed parquet datasets + feature tables

events_es.parquet, events_vx.parquet

dataset_es_classic.parquet, dataset_es_llm.parquet

dataset_vx_classic.parquet, dataset_vx_llm.parquet

src/ — all code (pipeline modules)

ingestion/ — load raw datasets

preprocessing/ — build event windows

features/ — text stats, VADER, topics, LLM features

models/ — ES/VX modeling (logit/RF/XGB + ablations)

results/ — aggregation + plotting scripts

main.py — runs the full pipeline end-to-end

results/ — outputs produced by the pipeline

runs/ — per-script model outputs (.parquet)

tables/ — aggregated tables (baselines.parquet, etc.)

graphs/ — final figures for the report

---

## Data Sources

- **Social media**:  
  - Trump Twitter Archive (2009–2021)  
  - Truth Social posts (through Nov 2024, timestamp-reliable only)

- **Market data**:  
  - Minute-level ES and VX futures from BacktestMarket

Raw data must be placed under `data_raw/` exactly as expected by the ingestion scripts.

---

## Installation

```bash
pip install -r requirements.txt


How to Run the Project

All computations start from the data_raw/ directory. No manual preprocessing is required.

From the project root, run:

python -m src.main


This command executes the full pipeline in order:

Ingestion of raw tweet and market data

Construction of event windows

Feature extraction

Model estimation

Result aggregation

Analysis and visualization

All outputs are written automatically to data_processed/, results/runs/, results/tables/, and results/graphs/.

Optional Execution Flags

Specific stages can be run selectively:

python -m src.main --build
python -m src.main --models
python -m src.main --aggregate
python -m src.main --analyze

Large Language Model (LLM) Features

LLM-based semantic features are optional. Precomputed LLM features are already provided in:

data_processed/features_llm_all.parquet


By default, the pipeline uses this file and does not call any external APIs.

The script src/features/llm_sentiment_openrouter.py relies on a paid OpenRouter API and is commented out by default in src/main.py. Users who wish to regenerate LLM features can insert their own API key, uncomment the corresponding line in src/main.py, and rerun the pipeline.

Outputs

Model results: results/runs/*.parquet

Aggregated tables: results/tables/*.parquet

Figures used in the report: results/graphs/*.png

Notes

Train–test splits are time-respecting.

Volatility spike thresholds are computed using training data only.

Accuracy is reported for reference, but ROC-AUC is the primary evaluation metric.

The project emphasizes careful validation and restrained interpretation in a noisy financial prediction setting.