"""Day-0 control vs drought fluxes, entropy, ensembles and paired analyses."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

from drought_local_common import (
    BIOMASS_COLS,
    DEFAULT_M_SCALE,
    RATE_COLS,
    build_crema_sampler,
    compute_entropy_partition,
    compute_metrics_from_flux_matrix,
    compute_network_metrics,
    load_day0_dataset,
    sample_crema_flux_matrix,
    solve_biomass_for_crema_sample,
)


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
DERIVED_DIR = SCRIPT_DIR / "derived_data"
PALETTE = {"C": "#4C6A92", "D": "#C44E52"}
LABEL_MAP = {
    "S_full": r"$S_{\mathrm{full}}$",
    "dominant_real_eigenvalue": r"$\Re(\lambda_{\max})$",
    "reactivity": r"$\omega_{\max}$",
    "total_flux": r"$F_{\mathrm{tot}}$",
}

EXCLUDED_NETWORKS = {
    "Yo_S3_In_C_d0": "beta fitting does not converge, so S_full is not available",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 120


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def format_metric_label(name: str) -> str:
    return LABEL_MAP.get(name, name.replace("_", " "))


def correlation_annotation(df: pd.DataFrame, x_var: str, y_var: str) -> str:
    clean = df[[x_var, y_var]].dropna()
    if len(clean) < 3:
        return f"n = {len(clean)}\nPearson r = NA\np = NA"
    pearson_r, pearson_p = stats.pearsonr(clean[x_var], clean[y_var])
    if pearson_p < 0.0015:
        p_text = "p < 0.001"
    elif pearson_p < 0.01:
        p_text = f"p = {pearson_p:.4f}"
    else:
        p_text = f"p = {pearson_p:.3f}"
    return f"n = {len(clean)}\nPearson r = {pearson_r:.2f}\n{p_text}"


def add_stat_box(
    ax: plt.Axes,
    text: str,
    *,
    x: float = 0.03,
    y: float = 0.97,
    ha: str = "left",
    va: str = "top",
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va=va,
        ha=ha,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#CFCFCF", "linewidth": 0.8, "alpha": 0.95},
        clip_on=False,
    )


def style_data_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.45)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.45, alpha=0.35)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.margins(x=0.05, y=0.08)


def empirical_below_fraction(observed: float, samples: np.ndarray) -> float:
    clean = np.asarray(samples, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float(np.mean(clean <= observed))


def classify_ensemble_match(empirical_rank: float) -> str:
    if not np.isfinite(empirical_rank):
        return "insufficient_data"
    if 0.05 <= empirical_rank <= 0.95:
        return "reproduced"
    return "not_reproduced"


def summarize_observed_vs_ensemble(observed_df: pd.DataFrame, sample_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    summary_rows = []
    for network_id, sub in sample_df.groupby("network_id", sort=True):
        obs_row = observed_df[observed_df["network_id"] == network_id].iloc[0]
        valid_sub = sub[sub["solver_success"].fillna(False)].copy()
        row = {
            "network_id": network_id,
            "Location": obs_row["Location"],
            "Site": obs_row["Site"],
            "Management": obs_row["Management"],
            "Treatment": obs_row["Treatment"],
            "valid_feasible_samples": int(len(valid_sub)),
        }
        for metric in metrics:
            samples = valid_sub[metric].to_numpy(dtype=float)
            samples = samples[np.isfinite(samples)]
            observed_value = float(obs_row[metric])
            row[f"{metric}_observed"] = observed_value
            row[f"{metric}_valid_samples"] = int(samples.size)
            row[f"{metric}_median"] = float(np.nanmedian(samples)) if samples.size else float("nan")
            row[f"{metric}_p01"] = float(np.nanpercentile(samples, 1)) if samples.size else float("nan")
            row[f"{metric}_p99"] = float(np.nanpercentile(samples, 99)) if samples.size else float("nan")
            row[f"{metric}_p20"] = float(np.nanpercentile(samples, 20)) if samples.size else float("nan")
            row[f"{metric}_p80"] = float(np.nanpercentile(samples, 80)) if samples.size else float("nan")
            row[f"{metric}_empirical_rank"] = empirical_below_fraction(observed_value, samples)
            row[f"{metric}_match_class"] = classify_ensemble_match(row[f"{metric}_empirical_rank"])
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values("network_id").reset_index(drop=True)


def make_entropy_main_panel(metrics_df: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    panels = [
        ("dominant_real_eigenvalue", format_metric_label("dominant_real_eigenvalue")),
        ("reactivity", format_metric_label("reactivity")),
        ("total_flux", format_metric_label("total_flux")),
    ]
    for idx, (ax, (column, label)) in enumerate(zip(axes, panels, strict=True), start=1):
        sns.scatterplot(
            data=metrics_df,
            x="S_full",
            y=column,
            hue="Treatment",
            palette=[PALETTE["C"], PALETTE["D"]],
            s=52,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            ax=ax,
        )
        if metrics_df[["S_full", column]].dropna().shape[0] >= 2:
            sns.regplot(
                data=metrics_df,
                x="S_full",
                y=column,
                scatter=False,
                ci=None,
                color="#3A3A3A",
                line_kws={"linewidth": 1.4, "alpha": 0.95},
                ax=ax,
            )
        ax.set_xlabel(format_metric_label("S_full"))
        ax.set_ylabel(label)
        ax.set_title(f"{chr(64 + idx)}. {label}", loc="left", fontweight="bold")
        stat_box_y = 0.66 if idx == 1 else 0.97
        add_stat_box(ax, correlation_annotation(metrics_df, "S_full", column), y=stat_box_y)
        style_data_axis(ax)
    axes[0].legend(frameon=False, title="Treatment")
    for ax in axes[1:]:
        if ax.legend_ is not None:
            ax.legend_.remove()
    fig.suptitle("Entropy and day-0 network properties", y=1.03, fontsize=13, fontweight="bold")
    save_figure(fig, FIGURES_DIR / "figure_entropy_main_panel")


def make_day0_ensemble_rank_panel(summary_df: pd.DataFrame) -> None:
    set_style()
    ordered = summary_df.sort_values("network_id").reset_index(drop=True)
    x_positions = np.arange(len(ordered))
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.4), sharex=True)
    panel_specs = [
        ("dominant_real_eigenvalue", format_metric_label("dominant_real_eigenvalue")),
        ("reactivity", format_metric_label("reactivity")),
    ]
    mismatch_color = "#B44A3F"
    for idx, (ax, (metric, label)) in enumerate(zip(axes, panel_specs, strict=True), start=1):
        p_col = f"{metric}_empirical_rank"
        class_col = f"{metric}_match_class"
        mismatch_mask = ordered[class_col] == "not_reproduced"
        well_mask = ordered[class_col] == "reproduced"
        insufficient_mask = ordered[class_col] == "insufficient_data"
        treatment_colors = ordered["Treatment"].map(PALETTE)

        ax.axhspan(0.0, 0.05, color="#FBE9E7", zorder=0, alpha=0.8)
        ax.axhspan(0.05, 0.95, color="#EEF5EC", zorder=0, alpha=0.55)
        ax.axhspan(0.95, 1.0, color="#FBE9E7", zorder=0, alpha=0.8)
        ax.axhline(0.05, color="#B56C5B", linewidth=1.0, linestyle="--")
        ax.axhline(0.95, color="#B56C5B", linewidth=1.0, linestyle="--")

        ax.scatter(x_positions[well_mask], ordered.loc[well_mask, p_col], s=44, color=treatment_colors[well_mask], edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(x_positions[mismatch_mask], ordered.loc[mismatch_mask, p_col], s=62, color=mismatch_color, edgecolor="white", linewidth=0.6, marker="D", zorder=5)
        if insufficient_mask.any():
            ax.scatter(x_positions[insufficient_mask], ordered.loc[insufficient_mask, p_col], s=46, color="#888888", edgecolor="white", linewidth=0.5, marker="x", zorder=3)

        for x_pos, row in ordered.loc[mismatch_mask, ["network_id", p_col]].reset_index(drop=True).iterrows():
            ax.annotate(
                row["network_id"],
                (x_positions[mismatch_mask][x_pos], row[p_col]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=6.5,
                color=mismatch_color,
                rotation=90,
            )

        counts = ordered[class_col].value_counts()
        add_stat_box(
            ax,
            "\n".join(
                [
                    f"reproduced = {int(counts.get('reproduced', 0))}",
                    f"not reproduced = {int(counts.get('not_reproduced', 0))}",
                ]
            ),
        )
        ax.set_ylim(-0.02, 1.04)
        ax.set_ylabel("Empirical rank in ensemble")
        ax.set_title(f"{chr(64 + idx)}. {label}", loc="left", fontweight="bold")
        style_data_axis(ax)
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.55)
    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(ordered["network_id"], rotation=90)
    axes[-1].set_xlabel("Day-0 network")
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Reproduced", markerfacecolor="#808080", markeredgecolor="white", markersize=7),
        Line2D([0], [0], marker="D", color="w", label="Not reproduced", markerfacecolor=mismatch_color, markeredgecolor="white", markersize=7),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.suptitle("Observed rank inside the day-0 ensemble distribution", y=1.02, fontsize=13, fontweight="bold")
    save_figure(fig, FIGURES_DIR / "figure_day0_ensemble_empirical_rank")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-samples", type=int, default=150)
    parser.add_argument("--m-scale", type=float, default=DEFAULT_M_SCALE)
    args = parser.parse_args()

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # The derived workbook already contains the day-0 plot means, environment and rates.
    aggregated_df = load_day0_dataset().sort_values("network_id").reset_index(drop=True)
    aggregated_df = aggregated_df[~aggregated_df["network_id"].isin(EXCLUDED_NETWORKS)].copy()

    metrics_rows = []
    for _, row in aggregated_df.iterrows():
        biomasses = row[BIOMASS_COLS].to_numpy(dtype=float)
        metabolism = row[RATE_COLS].to_numpy(dtype=float)
        metrics = compute_network_metrics(biomasses, metabolism, m_scale=args.m_scale)
        entropy_partition = compute_entropy_partition(metrics["F_out"], metrics["F_in"], metrics["entropy"])
        metrics_rows.append(
            {
                "network_id": row["network_id"],
                "pair_id": row["pair_id"],
                "Location": row["Location"],
                "Site": row["Site"],
                "Management": row["Management"],
                "Treatment": row["Treatment"],
                "day": row["day"],
                "moisture": row["moisture"],
                "respiration": row["respiration"],
                "total_biomass": row["total_biomass"],
                "total_flux": metrics["total_flux"],
                "dominant_real_eigenvalue": metrics["dominant_real_eigenvalue"],
                "reactivity": metrics["reactivity"],
                "S_full": metrics["entropy"],
                "beta_out_json": json.dumps(np.asarray(metrics["beta_out"], dtype=float).tolist()),
                "beta_in_json": json.dumps(np.asarray(metrics["beta_in"], dtype=float).tolist()),
                **entropy_partition,
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).sort_values("network_id").reset_index(drop=True)

    ensemble_rows = []
    for _, row in aggregated_df.iterrows():
        biomasses = row[BIOMASS_COLS].to_numpy(dtype=float)
        metabolism = row[RATE_COLS].to_numpy(dtype=float)
        observed_metrics = compute_network_metrics(biomasses, metabolism, m_scale=args.m_scale)
        if not np.isfinite(observed_metrics["flux_matrix"]).all():
            continue
        sampler = build_crema_sampler(observed_metrics["flux_matrix"])
        with tempfile.TemporaryDirectory(prefix=f"crema_section02_{row['network_id']}_") as sample_dir:
            for sample_id in range(args.ensemble_samples):
                sampled_flux = sample_crema_flux_matrix(sampler, seed=30000 + sample_id, output_dir=sample_dir)
                solve_result = solve_biomass_for_crema_sample(
                    sampled_flux,
                    biomasses,
                    metabolism,
                    sampler["F_out_real"],
                    sampler["F_in_real"],
                    m_scale=args.m_scale,
                )
                sample_metrics = {
                    "network_id": row["network_id"],
                    "pair_id": row["pair_id"],
                    "Location": row["Location"],
                    "Site": row["Site"],
                    "Management": row["Management"],
                    "Treatment": row["Treatment"],
                    "sample_id": sample_id,
                    "solver_success": bool(solve_result["solver_success"]),
                }
                if solve_result["solver_success"]:
                    sampled_biomasses = np.asarray(solve_result["biomasses"], dtype=float)
                    sampled_metrics = compute_metrics_from_flux_matrix(sampled_flux, sampled_biomasses, metabolism, m_scale=args.m_scale)
                    if np.isfinite(sampled_biomasses).all() and np.all(sampled_biomasses > 0) and np.isfinite(
                        [
                            sampled_metrics["total_flux"],
                            sampled_metrics["dominant_real_eigenvalue"],
                            sampled_metrics["reactivity"],
                        ]
                    ).all():
                        sample_metrics.update(
                            {
                                "total_flux": float(sampled_metrics["total_flux"]),
                                "dominant_real_eigenvalue": float(sampled_metrics["dominant_real_eigenvalue"]),
                                "reactivity": float(sampled_metrics["reactivity"]),
                                "S_full": float(sampled_metrics["entropy"]),
                            }
                        )
                    else:
                        sample_metrics["solver_success"] = False
                        sample_metrics.update(
                            {
                                "total_flux": float("nan"),
                                "dominant_real_eigenvalue": float("nan"),
                                "reactivity": float("nan"),
                                "S_full": float("nan"),
                            }
                        )
                else:
                    sample_metrics.update(
                        {
                            "total_flux": float("nan"),
                            "dominant_real_eigenvalue": float("nan"),
                            "reactivity": float("nan"),
                            "S_full": float("nan"),
                        }
                    )
                ensemble_rows.append(sample_metrics)

    ensemble_df = pd.DataFrame(ensemble_rows)
    metrics_for_ensemble = ["dominant_real_eigenvalue", "reactivity", "total_flux", "S_full"]
    control_samples_df = ensemble_df[ensemble_df["Treatment"] == "C"].copy()
    drought_samples_df = ensemble_df[ensemble_df["Treatment"] == "D"].copy()
    control_summary_df = summarize_observed_vs_ensemble(metrics_df[metrics_df["Treatment"] == "C"], control_samples_df, metrics_for_ensemble)
    drought_summary_df = summarize_observed_vs_ensemble(metrics_df[metrics_df["Treatment"] == "D"], drought_samples_df, metrics_for_ensemble)
    combined_summary_df = pd.concat([control_summary_df, drought_summary_df], ignore_index=True)

    make_day0_ensemble_rank_panel(combined_summary_df)
    make_entropy_main_panel(metrics_df)

    aggregated_df.to_csv(DERIVED_DIR / "aggregated_plot_means_drought_control_day0.csv", index=False)
    metrics_df.to_csv(DERIVED_DIR / "day0_network_metrics.csv", index=False)
    control_samples_df.to_parquet(DERIVED_DIR / "control_day0_ensemble_samples.parquet", index=False)
    drought_samples_df.to_parquet(DERIVED_DIR / "drought_day0_ensemble_samples.parquet", index=False)
    control_summary_df.to_csv(DERIVED_DIR / "control_day0_ensemble_summary.csv", index=False)
    drought_summary_df.to_csv(DERIVED_DIR / "drought_day0_ensemble_summary.csv", index=False)
    print("Section 02 completed.")


if __name__ == "__main__":
    main()
