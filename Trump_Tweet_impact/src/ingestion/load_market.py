"""
load_market.py

Market-data ingestion (ES + VX).

Reads large intraday futures files efficiently, keeps only necessary columns,
creates a timezone-aware UTC timestamp index.

Expected raw format (semicolon-separated, no header):
Date;Time;Open;High;Low;Close;Volume
27/05/2009;08:30:00;32.1;...;32.1;1

Vendor timezone: GMT -6 (fixed offset) -> use "Etc/GMT+6" then convert to UTC.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def _load_ohlcv_minute_vendor_file(
    path: Path,
    tz_source: str = "Etc/GMT+6",  # NOTE: "GMT-6" corresponds to "Etc/GMT+6"
) -> pd.DataFrame:
    """
    Generic loader for vendor OHLCV files (semicolon-separated, no header).
    Returns a DataFrame indexed by UTC timestamps with column 'close'.
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        usecols=["date", "time", "close"],
        dtype={"date": "string", "time": "string", "close": "float64"},
        engine="c",
    )

    ts = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df = df.loc[ts.notna(), ["close"]].copy()
    df["ts_local"] = ts[ts.notna()].values

    df["ts_utc"] = (
        df["ts_local"]
        .dt.tz_localize(tz_source, nonexistent="shift_forward", ambiguous="NaT")
        .dt.tz_convert("UTC")
    )

    df = df.dropna(subset=["ts_utc"]).drop(columns=["ts_local"])
    df = df.set_index("ts_utc").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    return df


def load_es_minute(es_path: Path, tz_source: str = "Etc/GMT+6") -> pd.DataFrame:
    """
    Load ES (E-mini S&P 500 futures) data.
    """
    return _load_ohlcv_minute_vendor_file(es_path, tz_source=tz_source)


def load_vx_minute(vx_path: Path, tz_source: str = "Etc/GMT+6") -> pd.DataFrame:
    """
    Load VX (VIX futures) data.
    """
    return _load_ohlcv_minute_vendor_file(vx_path, tz_source=tz_source)


if __name__ == "__main__":
    root = Path("../../data_raw/market_data")

    es_file = root / "ES_minute.csv"
    vx_file = root / "VX_minute.csv"

    es = load_es_minute(es_file)
    vx = load_vx_minute(vx_file)

    print("ES:", es.head(), es.tail(), es.index.tz, sep="\n")
    print("VX:", vx.head(), vx.tail(), vx.index.tz, sep="\n")