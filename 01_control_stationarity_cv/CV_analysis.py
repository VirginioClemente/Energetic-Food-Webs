"""Control poolability, CV, entropy and canonical-ensemble checks on Dataset_analysis_CV."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from cv_local_common import (
    BIOMASS_COLS,
    INDIVIDUAL_MASS_COLS,
    RATE_COLS,
    aggregate_for_ensemble,
    bootstrap_correlation_ci,
    build_crema_sampler,
    compute_network_metrics,
    load_cv_dataset,
    safe_corr,
    sample_crema_flux_matrix,
    solve_biomass_for_crema_sample,
    sum_n_from_biomasses,
)


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
DERIVED_DIR = SCRIPT_DIR / "derived_data"


def coefficient_of_variation(values: pd.Series | np.ndarray) -> float:
    # CV is computed on the observed replicate values within each group.
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return float("nan")
    return float(np.std(array) / np.mean(array))


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 120


def make_cv_entropy_figure(correlation_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(correlation_df["entropy"], correlation_df["cv_SumN"], s=58, color="#4C6A92", edgecolors="white", linewidths=0.5, alpha=0.9)
    clean = correlation_df[["entropy", "cv_SumN"]].dropna()
    if len(clean) >= 2:
        slope, intercept, _, p_value, _ = stats.linregress(clean["entropy"], clean["cv_SumN"])
        x_values = np.linspace(clean["entropy"].min(), clean["entropy"].max(), 200)
        ax.plot(x_values, intercept + slope * x_values, color="#333333", linewidth=1.2)
        ax.text(
            0.98,
            0.98,
            f"Pearson r = {stats_df.loc[0, 'pearson_r']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
        )
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Observed CV of SumN")
    ax.set_title("Pooled control variability versus network entropy")
    sns.despine()
    save_figure(fig, FIGURES_DIR / "figure_control_cv_vs_entropy_count")


def make_cv_ensemble_figure(summary_df: pd.DataFrame) -> None:
    set_style()
    ordered = summary_df.sort_values("group_id", kind="stable").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, max(4.8, 0.32 * len(ordered) + 1.2)))
    y_positions = np.arange(len(ordered))
    ax.hlines(y_positions, ordered["p01"], ordered["p99"], color="#B9B9B9", linewidth=3.0, alpha=0.9, label="Ensemble 1-99%")
    ax.hlines(y_positions, ordered["p20"], ordered["p80"], color="#6F6F6F", linewidth=6.0, alpha=0.95, label="Ensemble 20-80%")
    ax.scatter(ordered["ensemble_median"], y_positions, color="#444444", s=28, zorder=3, label="Ensemble median")
    ax.scatter(ordered["cv_SumN"], y_positions, color="#111111", s=28, zorder=4, label="Observed CV")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["group_id"])
    ax.set_xlabel("CV")
    ax.set_title("Observed pooled control CV versus canonical ensemble")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, FIGURES_DIR / "figure_control_cv_emp_vs_ensemble_count")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-samples", type=int, default=150)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--m-scale", type=float, default=0.01)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    # Load the plot-level control dataset with biomass, individual mass and rates.
    data = load_cv_dataset()

    # Compute the observed CV of SumN using all plot-day replicates in each group.
    cv_rows = []
    for group_id, group_df in data.groupby("group_id", sort=True):
        cv_rows.append(
            {
                "group_id": group_id,
                "Location": group_df["Location"].iloc[0],
                "Site": group_df["Site"].iloc[0],
                "Management": group_df["Management"].iloc[0],
                "Treatment": group_df["Treatment"].iloc[0],
                "n_obs": int(len(group_df)),
                "mean_SumN": float(group_df["SumN"].mean()),
                "sd_SumN": float(group_df["SumN"].std(ddof=1)),
                "cv_SumN": coefficient_of_variation(group_df["SumN"]),
            }
        )
    empirical_cv_df = pd.DataFrame(cv_rows).sort_values("group_id").reset_index(drop=True)
    empirical_cv_df.to_csv(DERIVED_DIR / "empirical_cv_sumN.csv", index=False)

    # Build one mean biomass vector per group by averaging over Day and Plot.
    aggregated_df = aggregate_for_ensemble(data)
    aggregated_df.to_csv(DERIVED_DIR / "mean_biomass_for_cv_analysis.csv", index=False)

    cv_lookup = empirical_cv_df.set_index("group_id")
    entropy_rows = []
    ensemble_rows = []
    for row_index, row in aggregated_df.iterrows():
        # The flux model uses total biomasses directly from the aggregated dataset.
        biomasses = row[BIOMASS_COLS].to_numpy(dtype=float)
        individual_masses = row[INDIVIDUAL_MASS_COLS].to_numpy(dtype=float)
        metabolism = row[RATE_COLS].to_numpy(dtype=float)
        metrics = compute_network_metrics(biomasses, metabolism, m_scale=args.m_scale)
        entropy_rows.append(
            {
                "group_id": row["group_id"],
                "Location": row["Location"],
                "Site": row["Site"],
                "Management": row["Management"],
                "Treatment": row["Treatment"],
                "entropy": metrics["entropy"],
                "total_flux": metrics["total_flux"],
                "dominant_real_eigenvalue": metrics["dominant_real_eigenvalue"],
                "reactivity": metrics["reactivity"],
            }
        )

        sample_sum_n_values = []
        with tempfile.TemporaryDirectory(prefix=f"crema_section01_{row['group_id']}_") as sample_dir:
            # Sample the weighted ensemble around the observed flux matrix.
            sampler = build_crema_sampler(metrics["flux_matrix"])
            for sample_id in range(args.ensemble_samples):
                sampled_flux = sample_crema_flux_matrix(sampler, seed=10000 + sample_id, output_dir=sample_dir)
                solve_result = solve_biomass_for_crema_sample(
                    sampled_flux,
                    biomasses,
                    metabolism,
                    sampler["F_out_real"],
                    sampler["F_in_real"],
                    m_scale=args.m_scale,
                )
                if solve_result["solver_success"]:
                    sample_sum_n_values.append(sum_n_from_biomasses(solve_result["biomasses"], individual_masses))

        sample_sum_n_values = np.asarray(sample_sum_n_values, dtype=float)
        bootstrap_cvs = []
        n_obs = int(cv_lookup.loc[row["group_id"], "n_obs"])
        if sample_sum_n_values.size >= max(10, n_obs):
            # Match the observed number of replicates when bootstrapping CV.
            rng = np.random.default_rng(20000 + row_index)
            for _ in range(args.bootstrap_reps):
                draw = rng.choice(sample_sum_n_values, size=n_obs, replace=True)
                bootstrap_cvs.append(coefficient_of_variation(draw))
        bootstrap_cvs = np.asarray(bootstrap_cvs, dtype=float)
        ensemble_rows.append(
            {
                "group_id": row["group_id"],
                "cv_SumN": float(cv_lookup.loc[row["group_id"], "cv_SumN"]),
                "ensemble_median": float(np.nanmedian(bootstrap_cvs)) if len(bootstrap_cvs) else float("nan"),
                "p01": float(np.nanpercentile(bootstrap_cvs, 1)) if len(bootstrap_cvs) else float("nan"),
                "p99": float(np.nanpercentile(bootstrap_cvs, 99)) if len(bootstrap_cvs) else float("nan"),
                "p20": float(np.nanpercentile(bootstrap_cvs, 20)) if len(bootstrap_cvs) else float("nan"),
                "p80": float(np.nanpercentile(bootstrap_cvs, 80)) if len(bootstrap_cvs) else float("nan"),
            }
        )

    entropy_df = pd.DataFrame(entropy_rows).sort_values("group_id").reset_index(drop=True)
    entropy_df.to_csv(DERIVED_DIR / "entropy_sumN.csv", index=False)
    ensemble_df = pd.DataFrame(ensemble_rows).sort_values("group_id").reset_index(drop=True)
    ensemble_df.to_csv(DERIVED_DIR / "ensemble_cv_sumN.csv", index=False)

    cv_entropy_df = empirical_cv_df.merge(entropy_df[["group_id", "entropy"]], on="group_id", how="left")
    corr_stats = safe_corr(cv_entropy_df["entropy"], cv_entropy_df["cv_SumN"])
    clean_corr = cv_entropy_df[["entropy", "cv_SumN"]].dropna()
    if len(clean_corr) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_corr["entropy"], clean_corr["cv_SumN"])
    else:
        slope = intercept = r_value = p_value = std_err = float("nan")
    pearson_low, pearson_high = bootstrap_correlation_ci(cv_entropy_df["entropy"], cv_entropy_df["cv_SumN"], "pearson", n_boot=args.bootstrap_reps)
    spearman_low, spearman_high = bootstrap_correlation_ci(cv_entropy_df["entropy"], cv_entropy_df["cv_SumN"], "spearman", n_boot=args.bootstrap_reps)
    corr_summary_df = pd.DataFrame(
        [
            {
                "n_groups": int(corr_stats["n"]),
                "pearson_r": corr_stats["pearson_r"],
                "pearson_p": corr_stats["pearson_p"],
                "pearson_ci_low": pearson_low,
                "pearson_ci_high": pearson_high,
                "spearman_rho": corr_stats["spearman_rho"],
                "spearman_p": corr_stats["spearman_p"],
                "spearman_ci_low": spearman_low,
                "spearman_ci_high": spearman_high,
                "linear_slope": slope,
                "linear_intercept": intercept,
                "linear_r2": r_value ** 2 if np.isfinite(r_value) else float("nan"),
                "linear_p": p_value,
                "linear_std_err": std_err,
            }
        ]
    )
    corr_summary_df.to_csv(DERIVED_DIR / "cv_entropy_stats_sumN.csv", index=False)

    make_cv_entropy_figure(cv_entropy_df, corr_summary_df)
    make_cv_ensemble_figure(ensemble_df)
    print("Section 01 completed.")


if __name__ == "__main__":
    main()
