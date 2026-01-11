"""
text_eda.py

EDA for Trump social-media post text.

Input:
- data_processed/dataset_es_classic.parquet

Outputs (ONLY):
- Trump_Tweet_impact/outputs/eda/

Core outputs:
- words_per_post_pct.png
- chars_per_post_pct.png
- avg_words_per_post_by_year.png
- top_words_overall.csv
- top_words_pre.csv / top_words_power.csv / top_words_post.csv (if regime exists)
- usage_by_regime.csv (if regime exists)

Added outputs:
- intensity_over_time_yearly.png          (share_upper, exclamations, questions)
- posting_hour_distribution_pct.png       (hour-of-day posting distribution)
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# CONFIG
# =====================

TEXT_COL = "text"
REGIME_COL = "regime"
DATE_COL = "created_at"

TOP_N_WORDS = 30
MIN_WORD_LEN = 3

STOPWORDS = set([
    "the", "and", "for", "that", "with", "this", "you", "are",
    "was", "have", "has", "but", "not", "they", "will", "from",
    "his", "her", "she", "him", "their", "about", "just", "your",
    "our", "out", "who", "what", "when", "where", "why", "how",
])

# Histogram controls (readability caps)
WORDS_MAX_X = 80
CHARS_MAX_X = 280
N_BINS = 60

# Time aggregation
TIME_AGG = "Y"   # yearly mean; change to "M" if you ever want monthly


# =====================
# Helpers
# =====================

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return [w for w in text.split() if len(w) >= MIN_WORD_LEN and w not in STOPWORDS]


def compute_basic_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    s = df[TEXT_COL].fillna("").astype(str)

    out = df.copy()
    out["n_chars"] = s.str.len()
    out["n_words"] = s.str.split().str.len()
    out["n_exclam"] = s.str.count(r"!")
    out["n_question"] = s.str.count(r"\?")
    out["n_hashtags"] = s.str.count(r"#")
    out["has_url"] = s.str.contains(r"http", regex=True).astype(int)

    alpha = s.str.count(r"[A-Za-z]").replace(0, np.nan)
    out["share_upper"] = (s.str.count(r"[A-Z]") / alpha).fillna(0.0)

    out["is_empty_text"] = s.str.strip().eq("").astype(int)
    out["is_zero_words"] = (out["n_words"] == 0).astype(int)

    return out


def top_words(series: pd.Series, n: int) -> pd.DataFrame:
    counter = Counter()
    for txt in series.dropna().astype(str):
        counter.update(tokenize(txt))
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])


def save_hist_pct(values: pd.Series, title: str, xlabel: str, out_path: Path,
                  bins: int,) -> None:
    """
    Histogram with y-axis in percent of posts.
    Caps values at x_max for readability (tail excluded from plot).
    """
    v_full = values.dropna().values
    #v = v_full[v_full <= x_max]

    weights = np.ones_like(v_full) * (100.0 / len(v_full))
    plt.figure(figsize=(7, 4))
    plt.hist(v_full, bins=bins, weights=weights)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Percentage of posts")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_avg_words_over_time(df_stats: pd.DataFrame, out_path: Path) -> None:
    if DATE_COL not in df_stats.columns:
        return

    tmp = df_stats.copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=[DATE_COL])

    ts = tmp.set_index(DATE_COL)["n_words"].resample(TIME_AGG).mean().dropna()

    plt.figure(figsize=(8, 4))
    plt.plot(ts.index, ts.values)
    plt.title(f"Average words per post over time ({'yearly' if TIME_AGG == 'Y' else 'monthly'} mean)")
    plt.xlabel("Time")
    plt.ylabel("Average words per post")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_intensity_over_time(df_stats: pd.DataFrame, out_path: Path) -> None:
    """
    Plot language intensity proxies over time:
    - share_upper: capitalization intensity
    - n_exclam: exclamation marks per post
    - n_question: question marks per post
    """
    if DATE_COL not in df_stats.columns:
        return

    tmp = df_stats.copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=[DATE_COL])

    grp = tmp.set_index(DATE_COL).resample(TIME_AGG)[["share_upper", "n_exclam", "n_question"]].mean().dropna()

    plt.figure(figsize=(9, 4))
    plt.plot(grp.index, grp["share_upper"].values, label="share_upper (caps intensity)")
    plt.plot(grp.index, grp["n_exclam"].values, label="exclamation marks per post")
    plt.plot(grp.index, grp["n_question"].values, label="question marks per post")
    plt.title(f"Language intensity over time ({'yearly' if TIME_AGG == 'Y' else 'monthly'} mean)")
    plt.xlabel("Time")
    plt.ylabel("Mean value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_posting_hour_distribution(df_stats: pd.DataFrame, out_path: Path) -> None:
    """
    Posting time distribution in 30-minute bins, US Eastern Time (ET),
    restricted to 08:30–17:00 ET, with hourly x-axis ticks.
    Values are shown as percent of posts.
    """
    if DATE_COL not in df_stats.columns:
        return

    # Local copy ONLY for this plot
    tmp = df_stats[[DATE_COL]].copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=[DATE_COL])

    # Convert to ET (DST-aware)
    et = tmp[DATE_COL].dt.tz_convert("America/New_York")

    # Build half-hour bins: e.g. 8.5 = 08:30–08:59, 9.0 = 09:00–09:29, etc.
    half_hour = et.dt.hour + (et.dt.minute >= 30) * 0.5

    # Restrict to 08:30–17:00 (last bin is 16.5 = 16:30–16:59)
    half_hour = half_hour[(half_hour >= 9) & (half_hour < 16.5)]

    # Explicit bin grid to keep spacing stable
    bins = np.arange(9, 16.5, 0.5)  # 8.5, 9.0, ..., 16.5
    counts = half_hour.value_counts().reindex(bins, fill_value=0).sort_index()
    pct = counts / counts.sum() * 100.0

    plt.figure(figsize=(10, 4))
    plt.bar(counts.index, pct.values, width=0.45)
    plt.title("Posting time distribution")
    plt.xlabel("Time of day (ET)")
    plt.ylabel("Percent of posts")

    # Hourly ticks: 09:00, 10:00, ..., 17:00
    tick_pos = np.arange(9, 18, 1)
    plt.xticks(
        ticks=tick_pos,
        labels=[f"{h:02d}:00" for h in tick_pos],
    )

    plt.xlim(8.45, 16.45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_top_words_bar_sns(top_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """
    Save a horizontal bar chart (seaborn) for top words.
    Expects columns: ['word', 'count'].
    """
    if top_df is None or top_df.empty:
        return

    # Sort for readability (largest at bottom)
    tmp = top_df.sort_values("count", ascending=False)

    plt.figure(figsize=(9, 7))
    sns.barplot(
        data=tmp,
        x="count",
        y="word",
        orient="h",
    )
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# =====================
# Main
# =====================

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    processed = PROJECT_ROOT / "data_processed"

    # OUTPUTS ONLY HERE
    out_dir = PROJECT_ROOT / "outputs" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = processed / "dataset_es_classic.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected {data_path}")

    df = pd.read_parquet(data_path)

    if TEXT_COL not in df.columns:
        raise ValueError(f"dataset_es_classic.parquet missing required column: {TEXT_COL}")

    # Keep empty strings (if any) for diagnostics, though you now filter them earlier
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    df_stats = compute_basic_text_stats(df)

    print("\n=== BASIC TEXT SUMMARY ===")
    print(df_stats[[
        "n_chars", "n_words", "n_exclam", "n_question",
        "n_hashtags", "has_url", "share_upper",
        "is_empty_text", "is_zero_words"
    ]].describe())

    # ---------------------
    # Length distributions (percent histograms)
    # ---------------------
    save_hist_pct(
        df_stats["n_words"],
        title="Words per post (relative frequency)",
        xlabel="Number of words",
        out_path=out_dir / "words_per_post_pct.png",
        bins=N_BINS,
        #x_max=WORDS_MAX_X,
    )

    save_hist_pct(
        df_stats["n_chars"],
        title="Characters per post (relative frequency)",
        xlabel="Number of characters",
        out_path=out_dir / "chars_per_post_pct.png",
        bins=N_BINS,
        #x_max=CHARS_MAX_X,
    )

    # ---------------------
    # Averages over time
    # ---------------------
    if DATE_COL in df_stats.columns:
        save_avg_words_over_time(df_stats, out_dir / "avg_words_per_post_by_year.png")
        save_intensity_over_time(df_stats, out_dir / "intensity_over_time_yearly.png")
        save_posting_hour_distribution(df_stats, out_dir / "posting_hour_distribution_pct.png")

    # ---------------------
    # Vocabulary tables (saved ONLY to outputs/eda)
    # ---------------------
    print("\n=== TOP WORDS (OVERALL) ===")
    top_all = top_words(df[TEXT_COL], TOP_N_WORDS)
    print(top_all)
    top_all.to_csv(out_dir / "top_words_overall.csv", index=False)

    if REGIME_COL in df.columns:
        for reg in ["pre", "power", "post"]:
            sub = df[df[REGIME_COL] == reg]
            if len(sub) == 0:
                continue
            print(f"\n=== TOP WORDS ({reg.upper()}) ===")
            top_reg = top_words(sub[TEXT_COL], TOP_N_WORDS)
            print(top_reg)
            top_reg.to_csv(out_dir / f"top_words_{reg}.csv", index=False)

        usage_cols = [
            "n_words", "n_chars", "n_hashtags", "has_url",
            "share_upper", "n_exclam", "n_question",
        ]
        usage = df_stats.groupby(REGIME_COL)[usage_cols].mean().sort_index()
        print("\n=== MEAN USAGE STATS BY REGIME ===")
        print(usage)
        usage.to_csv(out_dir / "usage_by_regime.csv")

    print(f"\nEDA outputs saved ONLY to: {out_dir.resolve()}")

    print("\n=== TOP WORDS (OVERALL) ===")
    top_all = top_words(df[TEXT_COL], TOP_N_WORDS)
    print(top_all)
    top_all.to_csv(out_dir / "top_words_overall.csv", index=False)

    save_top_words_bar_sns(
        top_all,
        title=f"Top {TOP_N_WORDS} words overall",
        out_path=out_dir / "top_words_overall.png",
    )
