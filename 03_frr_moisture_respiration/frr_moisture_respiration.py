"""FRR, moisture-beta and respiration-flux analysis using the derived day-0 workbook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy import stats

from frr_local_common import BIOMASS_COLS, DEFAULT_M_SCALE, RATE_COLS, TOPOLOGY, beta_summary_from_metrics, compute_network_metrics, load_day0_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_ROOT = SCRIPT_DIR / "figures"
DERIVED_ROOT = SCRIPT_DIR / "derived_data"

LABEL_MAP = {
    "response_proxy": r"$\Delta F_{\mathrm{tot}} / \Delta \beta$",
    "fluctuation_term_theory_control": r"$-\mathrm{Var}(F_{\mathrm{tot}})\ \mathrm{at}\ \beta_{C}$",
    "Delta_moisture": r"$\Delta \mathrm{Moisture}$",
    "Delta_beta": r"$\Delta \beta$",
    "Delta_total_respiration": r"$\Delta \mathrm{Respiration}_{\mathrm{tot}}$",
    "Delta_Ftot": r"$\Delta F_{\mathrm{tot}}$",
}

FRR_POINT_COLOR = "#567A88"
FRR_LINE_COLOR = "#1F2A30"
FRR_REFERENCE_COLOR = "#8C8C8C"


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 120


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def format_label(label: str) -> str:
    return LABEL_MAP.get(label, label)


def fit_ols(x: pd.Series, y: pd.Series) -> dict[str, float]:
    clean = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return {"n": float(len(clean)), "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    fit = stats.linregress(clean["x"], clean["y"])
    return {"n": float(len(clean)), "slope": float(fit.slope), "intercept": float(fit.intercept), "r2": float(fit.rvalue ** 2)}


def add_fit_annotation(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.9},
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.45)
    ax.grid(axis="x", color="#ECECEC", linewidth=0.45, alpha=0.35)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.margins(x=0.05, y=0.08)


def apply_sci_formatter(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def drop_iqr_outliers(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=columns).copy()
    keep_mask = pd.Series(True, index=clean.index)
    for column in columns:
        q1 = clean[column].quantile(0.25)
        q3 = clean[column].quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        keep_mask &= clean[column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return clean.loc[keep_mask].copy()


def make_scatter_with_fits(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_stem: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    clean = data[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    set_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(clean[x_col], clean[y_col], s=60, color=FRR_POINT_COLOR, edgecolors="white", linewidths=0.5, alpha=0.9, zorder=3)
    fit = fit_ols(clean[x_col], clean[y_col])
    if np.isfinite(fit["slope"]):
        x_grid = np.linspace(clean[x_col].min(), clean[x_col].max(), 200)
        ax.plot(x_grid, fit["intercept"] + fit["slope"] * x_grid, color=FRR_LINE_COLOR, linewidth=1.35)
    pearson_r = float(stats.pearsonr(clean[x_col], clean[y_col]).statistic) if len(clean) >= 3 else float("nan")
    add_fit_annotation(ax, "\n".join([f"n = {len(clean)}", f"Pearson r = {pearson_r:.2f}"]))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left", fontweight="bold")
    apply_sci_formatter(ax)
    style_axis(ax)
    save_figure(fig, FIGURE_ROOT / output_stem)


def make_frr_main_figure(frr_df: pd.DataFrame) -> None:
    clean = frr_df[frr_df["frr_valid"]].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=["response_proxy", "fluctuation_term_theory_control"])
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.scatter(
        clean["response_proxy"],
        clean["fluctuation_term_theory_control"],
        s=66,
        color=FRR_POINT_COLOR,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.9,
        zorder=3,
    )
    fit = fit_ols(clean["response_proxy"], clean["fluctuation_term_theory_control"])
    if np.isfinite(fit["slope"]):
        x_grid = np.linspace(clean["response_proxy"].min(), clean["response_proxy"].max(), 200)
        ax.plot(x_grid, fit["intercept"] + fit["slope"] * x_grid, color=FRR_LINE_COLOR, linewidth=1.35, label="OLS")
    lims = [
        min(clean["response_proxy"].min(), clean["fluctuation_term_theory_control"].min()),
        max(clean["response_proxy"].max(), clean["fluctuation_term_theory_control"].max()),
    ]
    ax.plot(lims, lims, color=FRR_REFERENCE_COLOR, linewidth=1.1, linestyle=":", label="1:1 expectation")
    pearson_r = float(stats.pearsonr(clean["response_proxy"], clean["fluctuation_term_theory_control"]).statistic) if len(clean) >= 3 else float("nan")
    add_fit_annotation(ax, "\n".join([f"n = {int(fit['n'])}", f"Pearson r = {pearson_r:.3f}", f"R² = {fit['r2']:.3f}"]))
    ax.set_xlabel(format_label("response_proxy"))
    ax.set_ylabel(format_label("fluctuation_term_theory_control"))
    ax.set_title("Fluctuation-response relation for paired day-0 drought vs control", loc="left", fontweight="bold")
    apply_sci_formatter(ax)
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, FIGURE_ROOT / "figure_frr_main")


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)

    day0_df = load_day0_dataset().sort_values("network_id").reset_index(drop=True)
    n_links = float(TOPOLOGY.sum())

    metric_rows = []
    for _, row in day0_df.iterrows():
        biomasses = row[BIOMASS_COLS].to_numpy(dtype=float)
        metabolism = row[RATE_COLS].to_numpy(dtype=float)
        metrics = compute_network_metrics(biomasses, metabolism, m_scale=DEFAULT_M_SCALE)
        beta_summary = beta_summary_from_metrics(metrics)
        metric_rows.append(
            {
                "network_id": row["network_id"],
                "pair_id": row["pair_id"],
                "Location": row["Location"],
                "Site": row["Site"],
                "Management": row["Management"],
                "Treatment": row["Treatment"],
                "moisture": row["moisture"],
                "respiration": row["respiration"],
                "total_biomass": row["total_biomass"],
                "total_flux": metrics["total_flux"],
                "dominant_real_eigenvalue": metrics["dominant_real_eigenvalue"],
                "reactivity": metrics["reactivity"],
                "S_full": metrics["entropy"],
                **beta_summary,
            }
        )
    metrics_df = pd.DataFrame(metric_rows).sort_values("network_id").reset_index(drop=True)
    metrics_df.to_csv(DERIVED_ROOT / "day0_network_metrics.csv", index=False)

    paired_df = metrics_df.pivot(index="pair_id", columns="Treatment")
    paired_df.columns = [f"{column}_{treatment}" for column, treatment in paired_df.columns]
    paired_df = paired_df.reset_index()

    frr_df = paired_df.copy()
    frr_df["Delta_Ftot"] = frr_df["total_flux_D"] - frr_df["total_flux_C"]
    frr_df["Delta_beta"] = frr_df["beta_global_D"] - frr_df["beta_global_C"]
    frr_df["Delta_moisture"] = frr_df["moisture_D"] - frr_df["moisture_C"]
    frr_df["Delta_total_respiration"] = frr_df["respiration_D"] - frr_df["respiration_C"]
    frr_df["beta_ref_control"] = frr_df["beta_global_C"]
    frr_df["response_proxy"] = frr_df["Delta_Ftot"] / frr_df["Delta_beta"]
    frr_df["fluctuation_term_theory_control"] = -n_links / np.square(frr_df["beta_ref_control"])
    frr_df["small_delta_beta_flag"] = frr_df["Delta_beta"].abs() < 1e-6
    frr_df["missing_pair_flag"] = frr_df[["total_flux_C", "total_flux_D", "beta_global_C", "beta_global_D"]].isna().any(axis=1)
    frr_df["frr_valid"] = ~(frr_df["small_delta_beta_flag"] | frr_df["missing_pair_flag"])
    frr_df.to_csv(DERIVED_ROOT / "paired_frr_table.csv", index=False)

    make_scatter_with_fits(
        frr_df,
        x_col="Delta_moisture",
        y_col="Delta_beta",
        output_stem="figure_moisture_beta",
        x_label=r"$\Delta \mathrm{Moisture}$",
        y_label=r"$\Delta \beta$",
        title="Moisture shift versus beta shift in paired day-0 groups",
    )
    make_scatter_with_fits(
        frr_df,
        x_col="Delta_total_respiration",
        y_col="Delta_Ftot",
        output_stem="figure_resp_flux",
        x_label=r"$\Delta \mathrm{Respiration}_{\mathrm{tot}}$",
        y_label=r"$\Delta F_{\mathrm{tot}}$",
        title="Respiration shift versus total-flux shift in paired day-0 groups",
    )
    respiration_no_outliers_df = drop_iqr_outliers(frr_df, ["Delta_total_respiration", "Delta_Ftot"])
    make_scatter_with_fits(
        respiration_no_outliers_df,
        x_col="Delta_total_respiration",
        y_col="Delta_Ftot",
        output_stem="figure_resp_flux_no_outliers",
        x_label=r"$\Delta \mathrm{Respiration}_{\mathrm{tot}}$",
        y_label=r"$\Delta F_{\mathrm{tot}}$",
        title="Respiration shift versus total-flux shift without IQR outliers",
    )
    make_frr_main_figure(frr_df)
    print("Section 03 completed.")


if __name__ == "__main__":
    main()
