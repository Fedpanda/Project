"""
save_events.py

Build and persist event-window datasets for ES and VX.

Outputs:
- data_processed/events_es.parquet
- data_processed/events_vx.parquet

Notes:
- VX has horizon-specific missingness (NaNs) when no timestamp exists within tolerance.
  We KEEP those NaNs; later, modeling is done horizon-by-horizon (dropna on the chosen target only).
"""

from __future__ import annotations

from pathlib import Path
import sys
import importlib.util

import pandas as pd


def save_events(events_es: pd.DataFrame, events_vx: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events_es.to_parquet(out_dir / "events_es.parquet", index=False)
    events_vx.to_parquet(out_dir / "events_vx.parquet", index=False)


def import_from_path(module_name: str, file_path: Path):
    """
    Import a module from an absolute file path.
    Registers module in sys.modules BEFORE execution (important for dataclasses).
    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_RAW = PROJECT_ROOT / "data_raw"
    OUT_DIR = PROJECT_ROOT / "data_processed"

    load_tweets_mod = import_from_path("load_tweets", PROJECT_ROOT / "src" / "ingestion" / "load_tweets.py")
    load_market_mod = import_from_path("load_market", PROJECT_ROOT / "src" / "ingestion" / "load_market.py")
    ev_mod = import_from_path("event_windows", PROJECT_ROOT / "src" / "preprocessing" / "event_windows.py")

    posts = load_tweets_mod.load_all_posts(DATA_RAW)
    es = load_market_mod.load_es_minute(DATA_RAW / "market_data" / "ES_minute.csv")
    vx = load_market_mod.load_vx_minute(DATA_RAW / "market_data" / "VX_minute.csv")

    # IMPORTANT: tolerance now lives in the config
    # (1m exact; 5m ±1; 30m ±2; 60m ±5)
    tol = {1: 0, 5: 1, 30: 2, 60: 5}

    cfg = ev_mod.EventWindowConfig(
        horizons_min=[1, 5, 30, 60],
        align_direction="forward",
        horizon_tolerance_min=tol,
    )

    events_es = ev_mod.build_es_event_windows(posts, es, cfg)
    events_vx = ev_mod.build_vx_event_windows(posts, vx, cfg)

    save_events(events_es, events_vx, OUT_DIR)

    print("Saved:", OUT_DIR / "events_es.parquet")
    print("Saved:", OUT_DIR / "events_vx.parquet")
    print("ES rows:", len(events_es), "VX rows:", len(events_vx))

    # Quick diagnostics (optional but useful)
    print("\nES return non-missing counts:")
    for h in cfg.horizons_min:
        col = f"es_ret_{h}m"
        if col in events_es.columns:
            print(f"{col}: {events_es[col].notna().sum()}")

    print("\nVX return non-missing counts:")
    for h in cfg.horizons_min:
        col = f"vx_ret_{h}m"
        if col in events_vx.columns:
            print(f"{col}: {events_vx[col].notna().sum()}")

    print("\nVX missingness rates:")
    for h in cfg.horizons_min:
        col = f"vx_ret_{h}m"
        if col in events_vx.columns:
            print(f"{col}: {events_vx[col].isna().mean():.2%}")

