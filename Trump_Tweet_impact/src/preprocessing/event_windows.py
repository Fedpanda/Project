"""
event_windows.py

Event windows + short-horizon reactions.

ES:
- returns + abs returns

VX (Path B):
- returns + abs returns (still useful)
- plus revision relative to a pre-event baseline:
    vx_pre_5min(t0) = mean(VX close over [t0-5min, t0))
    vx_dvx_h = VX(t0+h) - vx_pre_5min(t0)
    vx_absdvx_h = |vx_dvx_h|

Key mechanics
-------------
- Filter posts to US equity RTH based on America/New_York time (09:30–16:00, Mon–Fri).
- Conservative t0 alignment uses merge_asof with direction="forward".
- Horizon matching uses nearest timestamp within a ± tolerance window (by horizon).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class EventWindowConfig:
    horizons_min: List[int] = (1, 5, 30, 60)

    session_tz: str = "America/New_York"
    rth_start: str = "09:30"
    rth_end: str = "16:00"

    # "forward" recommended to avoid look-ahead
    align_direction: str = "forward"

    # ± tolerance window in minutes by horizon
    horizon_tolerance_min: Dict[int, int] = None


def _default_horizon_tolerance() -> Dict[int, int]:
    # Your requested policy
    return {1: 0, 5: 1, 30: 2, 60: 5}


# ---------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------

def filter_posts_to_rth(posts: pd.DataFrame, cfg: EventWindowConfig) -> pd.DataFrame:
    if "created_at" not in posts.columns:
        raise ValueError("posts must contain a 'created_at' column (tz-aware UTC).")

    created_et = posts["created_at"].dt.tz_convert(cfg.session_tz)

    t = created_et.dt.strftime("%H:%M")
    mask = (t >= cfg.rth_start) & (t < cfg.rth_end)
    mask &= (created_et.dt.weekday <= 4)

    return posts.loc[mask].copy()


# ---------------------------------------------------------------------
# Market alignment helpers
# ---------------------------------------------------------------------

def _merge_asof_t0(
    events: pd.DataFrame,
    market: pd.DataFrame,
    ts_col: str,
    direction: str,
    out_ts_col: str,
    out_price_col: str,
) -> pd.DataFrame:
    """
    Conservative t0 alignment using merge_asof.

    Critical: force tz-aware UTC for the matched timestamp.
    """
    if "close" not in market.columns:
        raise ValueError("market must contain a 'close' column.")
    if not isinstance(market.index, pd.DatetimeIndex) or market.index.tz is None:
        raise ValueError("market must have a tz-aware DatetimeIndex (UTC recommended).")

    tmp = events[[ts_col]].copy()
    tmp["_row_id"] = np.arange(len(tmp))

    # Ensure tz-aware UTC on left
    left = tmp.sort_values(ts_col).copy()
    left[ts_col] = pd.to_datetime(left[ts_col], utc=True, errors="coerce")

    # Right: index -> column (may lose tz); force utc after merge
    right = market[["close"]].reset_index().rename(columns={market.index.name or "index": "mkt_ts"})

    merged = pd.merge_asof(
        left,
        right,
        left_on=ts_col,
        right_on="mkt_ts",
        direction=direction,
        allow_exact_matches=True,
    )

    add = merged[["_row_id", "mkt_ts", "close"]].copy()
    add["mkt_ts"] = pd.to_datetime(add["mkt_ts"], utc=True, errors="coerce")  # <— fixes tz-naive
    add = add.rename(columns={"mkt_ts": out_ts_col, "close": out_price_col})
    add = add.sort_values("_row_id").drop(columns=["_row_id"])

    out = events.copy()
    out[out_ts_col] = add[out_ts_col].values
    out[out_price_col] = add[out_price_col].values
    return out


def _match_nearest_within_window(
    market_index: pd.DatetimeIndex,
    target_ts: pd.Series,
    tol_min: int,
) -> pd.Series:
    """
    Match nearest market timestamp to each target within ± tol_min minutes.
    If none exists, return NaT.

    Tie-break: if equal distance, choose the later timestamp.
    """
    if tol_min < 0:
        raise ValueError("tol_min must be >= 0")

    # Ensure tz-aware UTC targets
    t = pd.to_datetime(target_ts, utc=True, errors="coerce")

    # Use integer ns arrays (no .view warnings)
    idx_ns = market_index.asi8  # int64 ns
    t_ns = t.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)

    tol_ns = int(tol_min) * 60 * 1_000_000_000

    out = np.full(len(t_ns), np.iinfo(np.int64).min, dtype=np.int64)  # NaT marker

    for i, tn in enumerate(t_ns):
        if tn == np.iinfo(np.int64).min:
            continue

        lo = tn - tol_ns
        hi = tn + tol_ns

        L = np.searchsorted(idx_ns, lo, side="left")
        R = np.searchsorted(idx_ns, hi, side="right")
        if R <= L:
            continue

        cand = idx_ns[L:R]
        diffs = np.abs(cand - tn)
        min_diff = diffs.min()
        ties = np.where(diffs == min_diff)[0]
        j = int(ties[-1])  # later timestamp on tie
        out[i] = cand[j]

    return pd.to_datetime(out, utc=True)


def _attach_close_at_ts(
    events: pd.DataFrame,
    market: pd.DataFrame,
    ts_col: str,
    out_price_col: str,
) -> pd.DataFrame:
    """
    Attach market close at exact timestamps in events[ts_col] via reindex.
    """
    out = events.copy()
    ts = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    prices = market["close"].reindex(pd.DatetimeIndex(ts))
    out[out_price_col] = prices.values
    out[ts_col] = ts.values  # ensure tz-aware UTC dtype in the column too
    return out


# ---------------------------------------------------------------------
# Generic builder
# ---------------------------------------------------------------------

def build_market_event_windows(
    posts: pd.DataFrame,
    market: pd.DataFrame,
    market_prefix: str,
    cfg: EventWindowConfig = EventWindowConfig(),
    track_effective_horizon: bool = False,
) -> pd.DataFrame:
    required_cols = {"post_id", "created_at", "text", "platform", "is_retweet"}
    missing = required_cols - set(posts.columns)
    if missing:
        raise ValueError(f"posts missing required columns: {sorted(missing)}")

    if not isinstance(market.index, pd.DatetimeIndex) or market.index.tz is None:
        raise ValueError("market must have a tz-aware DatetimeIndex.")
    if "close" not in market.columns:
        raise ValueError("market must contain 'close'.")

    tol_map = cfg.horizon_tolerance_min or _default_horizon_tolerance()

    events = filter_posts_to_rth(posts, cfg).copy()
    events["t0_raw"] = pd.to_datetime(events["created_at"], utc=True, errors="coerce")

    t0_col = f"{market_prefix}_t0_matched"
    p0_col = f"{market_prefix}_p0"

    # t0 matching
    events = _merge_asof_t0(events, market, "t0_raw", cfg.align_direction, t0_col, p0_col)
    events = events.dropna(subset=[t0_col, p0_col]).copy()

    # horizons
    for h in cfg.horizons_min:
        tol = int(tol_map.get(int(h), 0))

        target = pd.to_datetime(events[t0_col], utc=True, errors="coerce") + pd.Timedelta(minutes=int(h))
        th_col = f"{market_prefix}_t{h}_matched"
        ph_col = f"{market_prefix}_p{h}"

        events[th_col] = _match_nearest_within_window(market.index, target, tol_min=tol)
        events = _attach_close_at_ts(events, market, th_col, ph_col)

        ret_col = f"{market_prefix}_ret_{h}m"
        abs_col = f"{market_prefix}_absret_{h}m"
        events[ret_col] = (events[ph_col] / events[p0_col]) - 1.0
        events[abs_col] = events[ret_col].abs()

        if track_effective_horizon:
            eff_col = f"{market_prefix}_eff_h_{h}m"
            t0_series = pd.to_datetime(events[t0_col], utc=True, errors="coerce")
            th_series = pd.to_datetime(events[th_col], utc=True, errors="coerce")
            events[eff_col] = (th_series - t0_series).dt.total_seconds() / 60.0

    # Output (keep NaNs: you may want to analyze match rates; later you can drop NaNs per-horizon)
    out_cols = ["post_id", "created_at", "platform", "is_retweet", "text"]
    out_cols += [f"{market_prefix}_ret_{h}m" for h in cfg.horizons_min]
    out_cols += [f"{market_prefix}_absret_{h}m" for h in cfg.horizons_min]
    if track_effective_horizon:
        out_cols += [f"{market_prefix}_eff_h_{h}m" for h in cfg.horizons_min]

    return events[out_cols].sort_values("created_at").reset_index(drop=True)


# ---------------------------------------------------------------------
# VX Path B: baseline revision ΔVX
# ---------------------------------------------------------------------

def _vx_pre_baseline_5min(vx: pd.DataFrame) -> pd.Series:
    """
    Baseline VX at each market timestamp = mean close over [t-5min, t), excluding t.
    """
    vx = vx.sort_index()
    return vx["close"].rolling("5min", closed="left").mean()


def build_es_event_windows(posts: pd.DataFrame, es: pd.DataFrame, cfg: EventWindowConfig = EventWindowConfig()) -> pd.DataFrame:
    return build_market_event_windows(posts, es, "es", cfg, track_effective_horizon=False)


def build_vx_event_windows(posts: pd.DataFrame, vx: pd.DataFrame, cfg: EventWindowConfig = EventWindowConfig()) -> pd.DataFrame:
    out = build_market_event_windows(posts, vx, "vx", cfg, track_effective_horizon=True)

    # Reconstruct vx_t0_matched so we can align baseline. (We keep this explicit and stable.)
    events0 = filter_posts_to_rth(posts, cfg).copy()
    events0["t0_raw"] = pd.to_datetime(events0["created_at"], utc=True, errors="coerce")
    events0 = _merge_asof_t0(events0, vx, "t0_raw", cfg.align_direction, "vx_t0_matched_tmp", "vx_p0_tmp")
    events0 = events0.dropna(subset=["vx_t0_matched_tmp"]).copy()
    t0_map = events0[["post_id", "created_at", "vx_t0_matched_tmp"]]

    out2 = out.merge(t0_map, on=["post_id", "created_at"], how="left", validate="one_to_one")
    out2 = out2.dropna(subset=["vx_t0_matched_tmp"]).copy()

    vx_pre = _vx_pre_baseline_5min(vx)
    out2["vx_pre_5min"] = vx_pre.reindex(pd.DatetimeIndex(pd.to_datetime(out2["vx_t0_matched_tmp"], utc=True))).values

    for h in cfg.horizons_min:
        ph = f"vx_p{h}"
        if ph not in out2.columns:
            continue
        dvx = f"vx_dvx_{h}m"
        absdvx = f"vx_absdvx_{h}m"
        out2[dvx] = out2[ph] - out2["vx_pre_5min"]
        out2[absdvx] = out2[dvx].abs()

    keep = list(out.columns)
    keep += ["vx_pre_5min"]
    keep += [f"vx_dvx_{h}m" for h in cfg.horizons_min]
    keep += [f"vx_absdvx_{h}m" for h in cfg.horizons_min]
    keep = [c for c in keep if c in out2.columns]

    return out2[keep].sort_values("created_at").reset_index(drop=True)


#This is for testing whether my missing rows are randomly distributed
# or concentrated somewhere
# Assign regime temporarily (same logic as build_dataset)
def assign_regime(created_at: pd.Series) -> pd.Series:
    pre_end = pd.Timestamp("2015-06-15", tz="UTC")
    power_end = pd.Timestamp("2021-01-20", tz="UTC")

    return pd.cut(
        created_at,
        bins=[
            pd.Timestamp.min.tz_localize("UTC"),
            pre_end,
            power_end,
            pd.Timestamp.max.tz_localize("UTC"),
        ],
        labels=["pre", "power", "post"],
        right=True,
    )

# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    import importlib.util

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_RAW = PROJECT_ROOT / "data_raw"

    def import_from_path(module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    load_tweets_mod = import_from_path("load_tweets", PROJECT_ROOT / "src" / "ingestion" / "load_tweets.py")
    load_market_mod = import_from_path("load_market", PROJECT_ROOT / "src" / "ingestion" / "load_market.py")

    posts = load_tweets_mod.load_all_posts(DATA_RAW)
    es = load_market_mod.load_es_minute(DATA_RAW / "market_data" / "ES_minute.csv")
    vx = load_market_mod.load_vx_minute(DATA_RAW / "market_data" / "VX_minute.csv")

    cfg = EventWindowConfig(
        horizons_min=[1, 5, 30, 60],
        align_direction="forward",
        horizon_tolerance_min=_default_horizon_tolerance(),
    )

    events_es = build_es_event_windows(posts, es, cfg)
    print("\n=== ES EVENT WINDOWS ===")
    print("ES rows:", len(events_es))
    print(events_es.filter(like="es_ret_").describe())

    events_vx = build_vx_event_windows(posts, vx, cfg)
    print("\n=== VX EVENT WINDOWS (Path B) ===")
    print("VX rows:", len(events_vx))
    print(events_vx.filter(like="vx_ret_").describe())

    eff_cols = [c for c in events_vx.columns if c.startswith("vx_eff_h_")]
    if eff_cols:
        print("\nVX effective horizons (minutes):")
        print(events_vx[eff_cols].describe())

    print("\nVX missingness:")
    for h in cfg.horizons_min:
        col = f"vx_ret_{h}m"
        if col in events_vx.columns:
            print(f"{col}: {events_vx[col].isna().mean():.2%} NaN")
        col2 = f"vx_dvx_{h}m"
        if col2 in events_vx.columns:
            print(f"{col2}: {events_vx[col2].isna().mean():.2%} NaN")







    events_vx["regime"] = assign_regime(events_vx["created_at"])

    print("\nVX missingness by regime:")
    for h in cfg.horizons_min:
        col = f"vx_ret_{h}m"
        tmp = (
            events_vx
            .groupby("regime")[col]
            .apply(lambda s: s.isna().mean())
        )
        print(f"\n{col}")
        print(tmp)