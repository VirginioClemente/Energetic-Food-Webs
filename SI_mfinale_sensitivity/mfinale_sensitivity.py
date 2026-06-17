"""Sensitivity analysis for m_finale = d_i * c across aggregated day-0 networks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from mfinale_local_common import BIOMASS_COLS, RATE_COLS, compute_entropy_partition, compute_network_metrics, load_day0_dataset


C_VALUES = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_ROOT = SCRIPT_DIR / "figures"
DERIVED_DIR = SCRIPT_DIR / "derived_data"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 120


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def regression_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    clean = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 4:
        return {
            "n_valid": float(len(clean)),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
        }
    pearson = stats.pearsonr(clean["x"], clean["y"])
    spearman = stats.spearmanr(clean["x"], clean["y"])
    fit = stats.linregress(clean["x"], clean["y"])
    return {
        "n_valid": float(len(clean)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r2": float(fit.rvalue ** 2),
    }


def draw_trajectory(summary_df: pd.DataFrame) -> None:
    subset = summary_df[(summary_df["subset"] == "all") & (summary_df["relationship"].isin([
        "S_full vs reactivity",
        "S_full vs total_flux",
        "S_full vs dominant_real_eigenvalue",
    ]))]
    relationship_labels = {
        "S_full vs reactivity": r"$S_{\mathrm{full}}$ vs $\omega_{\max}$",
        "S_full vs total_flux": r"$S_{\mathrm{full}}$ vs $F_{\mathrm{tot}}$",
        "S_full vs dominant_real_eigenvalue": r"$S_{\mathrm{full}}$ vs $\Re(\lambda_{\max})$",
    }
    subset = subset.assign(relationship_label=subset["relationship"].map(relationship_labels))
    set_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    sns.lineplot(data=subset, x="c", y="pearson_r", hue="relationship_label", marker="o", linewidth=1.8, ax=ax)
    ax.axvline(0.01, color="#8C8C8C", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$c$ in $m_{\mathrm{finale}} = d_i \cdot c$")
    ax.set_ylabel("Pearson r")
    ax.set_title(r"Trajectories of $S_{\mathrm{full}}$ relationships across the $m_{\mathrm{finale}}$ grid")
    ax.legend(frameon=False, loc="best")
    save_figure(fig, FIGURE_ROOT / "figure_mfinale_trajectory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and (FIGURE_ROOT / "figure_mfinale_trajectory.png").exists():
        print("Figure already exists, skipping.")
        return

    day0_df = load_day0_dataset().sort_values("network_id").reset_index(drop=True)

    full_rows = []
    summary_rows = []
    for c_value in C_VALUES:
        current_rows = []
        for _, row in day0_df.iterrows():
            biomasses = row[BIOMASS_COLS].to_numpy(dtype=float)
            metabolism = row[RATE_COLS].to_numpy(dtype=float)
            metrics = compute_network_metrics(biomasses, metabolism, m_scale=c_value)
            entropy_parts = compute_entropy_partition(metrics["F_out"], metrics["F_in"], metrics["entropy"])
            current_rows.append(
                {
                    "c": c_value,
                    "network_id": row["network_id"],
                    "pair_id": row["pair_id"],
                    "Location": row["Location"],
                    "Site": row["Site"],
                    "Management": row["Management"],
                    "Treatment": row["Treatment"],
                    "total_flux": metrics["total_flux"],
                    "S_full": metrics["entropy"],
                    "S_het": entropy_parts["S_het"],
                    "S_Ftot": entropy_parts["S_Ftot"],
                    "dominant_real_eigenvalue": metrics["dominant_real_eigenvalue"],
                    "reactivity": metrics["reactivity"],
                }
            )
        current_df = pd.DataFrame(current_rows)
        full_rows.extend(current_rows)

        for subset_name, subset_df in {
            "all": current_df,
            "control_only": current_df[current_df["Treatment"] == "C"],
            "drought_only": current_df[current_df["Treatment"] == "D"],
        }.items():
            relation_specs = [
                ("S_full vs dominant_real_eigenvalue", "S_full", "dominant_real_eigenvalue"),
                ("S_full vs reactivity", "S_full", "reactivity"),
                ("S_full vs total_flux", "S_full", "total_flux"),
                ("S_het vs dominant_real_eigenvalue", "S_het", "dominant_real_eigenvalue"),
                ("S_het vs reactivity", "S_het", "reactivity"),
                ("S_Ftot vs dominant_real_eigenvalue", "S_Ftot", "dominant_real_eigenvalue"),
                ("S_Ftot vs reactivity", "S_Ftot", "reactivity"),
            ]
            for relation_name, x_col, y_col in relation_specs:
                summary_rows.append(
                    {
                        "subset": subset_name,
                        "relationship": relation_name,
                        "c": c_value,
                        **regression_stats(subset_df[x_col], subset_df[y_col]),
                    }
                )

    full_df = pd.DataFrame(full_rows).sort_values(["c", "network_id"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["subset", "relationship", "c"]).reset_index(drop=True)
    full_df.to_csv(DERIVED_DIR / "mfinale_sensitivity_full.csv", index=False)
    summary_df.to_csv(DERIVED_DIR / "mfinale_sensitivity_summary.csv", index=False)
    draw_trajectory(summary_df)
    print("SI m_finale sensitivity completed.")


if __name__ == "__main__":
    main()
