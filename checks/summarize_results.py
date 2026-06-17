"""Summarize manuscript-level results from generated analysis outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]


def pearson_summary(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[int, float, float]:
    clean = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return len(clean), float("nan"), float("nan")
    result = stats.pearsonr(clean[x_col], clean[y_col])
    return len(clean), float(result.statistic), float(result.pvalue)


def print_correlation(label: str, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    n_obs, r_value, p_value = pearson_summary(df, x_col, y_col)
    print(f"{label}: n={n_obs}, Pearson r={r_value:.3f}, p={p_value:.4g}")


def summarize_control_cv() -> None:
    path = ROOT / "01_control_stationarity_cv" / "derived_data" / "cv_entropy_stats_sumN.csv"
    stats_df = pd.read_csv(path)
    row = stats_df.iloc[0]
    print(
        "Control CV vs entropy: "
        f"n={int(row['n_groups'])}, Pearson r={row['pearson_r']:.3f}, "
        f"p={row['pearson_p']:.4g}, R2={row['linear_r2']:.3f}"
    )


def summarize_day0_metrics() -> None:
    path = ROOT / "02_drought_control_flux_entropy" / "derived_data" / "day0_network_metrics.csv"
    metrics_df = pd.read_csv(path)
    print_correlation("S_full vs dominant real eigenvalue", metrics_df, "S_full", "dominant_real_eigenvalue")
    print_correlation("S_full vs reactivity", metrics_df, "S_full", "reactivity")
    print_correlation("S_full vs total flux", metrics_df, "S_full", "total_flux")


def summarize_frr() -> None:
    path = ROOT / "03_frr_moisture_respiration" / "derived_data" / "paired_frr_table.csv"
    frr_df = pd.read_csv(path)
    clean = (
        frr_df[frr_df["frr_valid"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["response_proxy", "fluctuation_term_theory_control"])
    )
    fit = stats.linregress(clean["response_proxy"], clean["fluctuation_term_theory_control"])
    print(
        "FRR response vs control fluctuation term: "
        f"n={len(clean)}, Pearson r={fit.rvalue:.3f}, slope={fit.slope:.3f}, R2={fit.rvalue ** 2:.3f}"
    )


def summarize_ensemble_rank() -> None:
    base = ROOT / "02_drought_control_flux_entropy" / "derived_data"
    summaries = [
        pd.read_csv(base / "control_day0_ensemble_summary.csv"),
        pd.read_csv(base / "drought_day0_ensemble_summary.csv"),
    ]
    combined = pd.concat(summaries, ignore_index=True)
    n_networks = len(combined)
    for metric in ["dominant_real_eigenvalue", "reactivity"]:
        counts = combined[f"{metric}_match_class"].value_counts(dropna=False).to_dict()
        reproduced = int(counts.get("reproduced", 0))
        print(f"Ensemble rank, {metric}: reproduced={reproduced}/{n_networks}")


def main() -> None:
    summarize_control_cv()
    summarize_day0_metrics()
    summarize_frr()
    summarize_ensemble_rank()


if __name__ == "__main__":
    main()
