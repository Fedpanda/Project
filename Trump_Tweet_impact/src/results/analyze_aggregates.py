"""
analyze_aggregates.py

Purpose
-------
Load the aggregated baseline results table (`baselines.parquet`), create four
filtered “analysis-ready” tables for a specific regime and a small set of
feature specifications, export those tables to LaTeX (`.tex`) for direct
inclusion in a report, and produce two summary figures for AUC comparisons.

What this script does
---------------------
1) Locates the project root from this file’s location (robust to where you run it from).
2) Loads:    <project_root>/results/tables/baselines.parquet
3) Filters four tables (regime = "power"):
   - ES direction:   target = "es_up_60m"
   - VX spike:       target = "vx_spike_60m"
   - ES direction:   target = "es_up_30m"
   - VX spike:       target = "vx_spike_30m"
4) Applies the same spec constraints to each table:
   - Keep only: stats_only, stats_plus_vader, llm_only, stats_plus_llm
   - Exclude rows where (spec_or_feature_set == "stats_only") AND (feature_track == "llm")
5) Writes the 4 filtered tables (selected columns) to LaTeX under:
   <project_root>/results/graphs/
6) Produces two figures for the (power, 60m) comparison:
   - Figure 1: AUC by spec, separate bars per model (ES panel + VX panel)
   - Figure 2: ΔAUC vs baseline = (model=logit, feature_track=classic, spec=stats_only)
              computed separately for ES and VX (same two-panel layout)

Outputs
-------
Tables (.tex)
- results/graphs/es_60m_power.tex
- results/graphs/vx_60m_power.tex
- results/graphs/es_30m_power.tex
- results/graphs/vx_30m_power.tex

Figures (.png)
- results/summary/figures/fig1_auc_power_60m.png
- results/summary/figures/fig2_delta_auc_power_60m.png

Notes
-----
- LaTeX tables include caption + label and can be included via:
    \\input{results/graphs/es_60m_power.tex}
- Numeric values are formatted to 3 decimals by default.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Project paths / IO
# -----------------------------

def project_root() -> Path:
    """
    Resolve the project root directory reliably using the file location.

    Expected location:
        <project_root>/src/.../load_baselines.py

    Therefore, parents[2] points to <project_root>.
    Adjust `parents[2]` if you move this script in the tree.
    """
    return Path(__file__).resolve().parents[2]


def load_baselines() -> pd.DataFrame:
    """
    Load the aggregated baselines parquet table.
    """
    root = project_root()
    data_path = root / "results" / "tables" / "baselines.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {data_path}")

    return pd.read_parquet(data_path)


# -----------------------------
# Filtering logic (generalized)
# -----------------------------

ALLOWED_SPECS = (
    "stats_only",
    "stats_plus_vader",
    "llm_only",
    "stats_plus_llm",
)

OUTPUT_COLS = ["model", "spec_or_feature_set", "baseline_acc", "acc", "auc"]

SPEC_ORDER = ["stats_only", "stats_plus_vader", "llm_only", "stats_plus_llm"]
MODEL_ORDER = ["logit", "rf", "xgb"]

# Lock colors so the legend is stable across figures.
# (You asked for "green bar = xgb" explicitly.)
MODEL_PALETTE = {
    "logit": "tab:blue",
    "rf": "tab:orange",
    "xgb": "tab:green",
}


def filter_results(
    df: pd.DataFrame,
    *,
    target: str,
    regime: str = "power",
    allowed_specs: tuple[str, ...] = ALLOWED_SPECS,
) -> pd.DataFrame:
    """
    Apply your exact filter specification, parameterized by target/regime.
    """
    # Exclude the invalid comparison combination:
    # (spec_or_feature_set == "stats_only") AND (feature_track == "llm")
    exclude = (df["spec_or_feature_set"] == "stats_only") & (df["feature_track"] == "llm")

    mask = (
        (df["target"] == target)
        & (df["regime"] == regime)
        & (df["spec_or_feature_set"].isin(allowed_specs))
        & (~exclude)
    )

    return df.loc[mask].copy()


# -----------------------------
# LaTeX export
# -----------------------------

def save_df_as_latex(
    df: pd.DataFrame,
    out_path: Path,
    *,
    caption: str,
    label: str,
    float_format: str = "{:.3f}",
) -> None:
    """
    Save a DataFrame as a LaTeX table to a .tex file (caption + label included).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numeric values to consistently formatted strings (stable tables).
    df_out = df.copy()
    for c in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[c]):
            df_out[c] = df_out[c].map(lambda x: float_format.format(x) if pd.notna(x) else "")

    latex = df_out.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        longtable=False,
    )

    out_path.write_text(latex, encoding="utf-8")


# -----------------------------
# Plotting helpers
# -----------------------------

def _baseline_auc_classic_logit_stats_only(df: pd.DataFrame) -> float:
    """
    Baseline AUC definition:
      model == 'logit' AND feature_track == 'classic' AND spec_or_feature_set == 'stats_only'

    If multiple rows match (e.g., multiple markets/runs), we take the mean.
    """
    base = df[
        (df["model"] == "logit")
        & (df["feature_track"] == "classic")
        & (df["spec_or_feature_set"] == "stats_only")
    ].copy()

    if base.empty:
        raise ValueError(
            "Baseline row not found for (model=logit, feature_track=classic, spec=stats_only). "
            "Verify that feature_track contains 'classic' and that the baseline spec exists."
        )

    return float(base["auc"].mean())


def _prep_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only requested specs and aggregate to one bar per (spec, model).
    If a model is absent in the data, it will not appear in the plot/legend.
    """
    d = df[df["spec_or_feature_set"].isin(SPEC_ORDER)].copy()

    d = (
        d.groupby(["spec_or_feature_set", "model"], as_index=False)
         .agg(auc=("auc", "mean"))
    )

    return d


def _plot_two_panel_bar(
    *,
    df_es: pd.DataFrame,
    df_vx: pd.DataFrame,
    y_col: str,
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    """
    Create a two-panel figure (ES left, VX right) with seaborn barplots.

    Legend placement:
    - You requested the legend to be *under* the plots.
    - We use a figure-level legend anchored below and reserve space with tight_layout(rect=...).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    sns.barplot(
        data=df_es,
        x="spec_or_feature_set",
        y=y_col,
        hue="model",
        order=SPEC_ORDER,
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        errorbar=None,
        ax=axes[0],
    )
    axes[0].set_title("ES (power, 60m)")
    axes[0].set_xlabel("Spec")
    axes[0].set_ylabel(y_label)
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=df_vx,
        x="spec_or_feature_set",
        y=y_col,
        hue="model",
        order=SPEC_ORDER,
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set_title("VX (power, 60m)")
    axes[1].set_xlabel("Spec")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=20)

    # Remove per-axis legends and add a single figure-level legend underneath.
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()

    fig.legend(
        handles,
        labels,
        title="Model",
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.18),
    )

    # Reserve bottom space for the legend.
    fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    df = load_baselines()

    # Four filtered tables (power regime)
    df_es_60 = filter_results(df, target="es_up_60m", regime="power")
    df_vx_60 = filter_results(df, target="vx_spike_60m", regime="power")
    df_es_30 = filter_results(df, target="es_up_30m", regime="power")
    df_vx_30 = filter_results(df, target="vx_spike_30m", regime="power")

    # -----------------------------
    # Export LaTeX tables
    # -----------------------------
    out_tables_dir = project_root() / "results" / "graphs"

    save_df_as_latex(
        df_es_60[OUTPUT_COLS],
        out_tables_dir / "es_60m_power.tex",
        caption="ES direction (60m), regime = power.",
        label="tab:es_60m_power",
    )
    save_df_as_latex(
        df_vx_60[OUTPUT_COLS],
        out_tables_dir / "vx_60m_power.tex",
        caption="VX spike (60m), regime = power.",
        label="tab:vx_60m_power",
    )
    save_df_as_latex(
        df_es_30[OUTPUT_COLS],
        out_tables_dir / "es_30m_power.tex",
        caption="ES direction (30m), regime = power.",
        label="tab:es_30m_power",
    )
    save_df_as_latex(
        df_vx_30[OUTPUT_COLS],
        out_tables_dir / "vx_30m_power.tex",
        caption="VX spike (30m), regime = power.",
        label="tab:vx_30m_power",
    )

    print("\n[done] Wrote LaTeX tables to:", out_tables_dir.resolve())

    # -----------------------------
    # Figures (Power, 60m)
    # -----------------------------
    fig_dir = project_root() / "results" / "graphs"

    # Figure 1: raw AUC
    es_plot = _prep_for_plot(df_es_60)
    vx_plot = _prep_for_plot(df_vx_60)

    _plot_two_panel_bar(
        df_es=es_plot,
        df_vx=vx_plot,
        y_col="auc",
        title="Figure 1 — AUC comparison (power, 60m)",
        y_label="AUC",
        out_path=fig_dir / "fig1_auc_power_60m.png",
    )

    # Figure 2: ΔAUC vs baseline (logit classic stats_only), computed per asset
    es_base = _baseline_auc_classic_logit_stats_only(df_es_60)
    vx_base = _baseline_auc_classic_logit_stats_only(df_vx_60)

    es_delta = es_plot.copy()
    vx_delta = vx_plot.copy()
    es_delta["delta_auc"] = es_delta["auc"] - es_base
    vx_delta["delta_auc"] = vx_delta["auc"] - vx_base

    _plot_two_panel_bar(
        df_es=es_delta,
        df_vx=vx_delta,
        y_col="delta_auc",
        title="Figure 2 — ΔAUC vs baseline: logit classic stats_only (power, 60m)",
        y_label="ΔAUC",
        out_path=fig_dir / "fig2_delta_auc_power_60m.png",
    )

    print("\n[done] Wrote figures to:", fig_dir.resolve())
    print(" - fig1_auc_power_60m.png")
    print(" - fig2_delta_auc_power_60m.png")


if __name__ == "__main__":
    main()
