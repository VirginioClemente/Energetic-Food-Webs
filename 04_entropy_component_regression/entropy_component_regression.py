"""Regression analysis for entropy components versus day-0 stability metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from entropy_local_common import BIOMASS_COLS, DEFAULT_M_SCALE, RATE_COLS, compute_entropy_partition, compute_network_metrics, load_day0_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "regression_output.txt"
DERIVED_DIR = SCRIPT_DIR / "derived_data"


def zscore_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean_value = numeric.mean()
    sd_value = numeric.std(ddof=0)
    return (numeric - mean_value) / sd_value


def fit_model(df: pd.DataFrame, response_col: str, predictors: list[str], include_intercept: bool):
    work = df.copy()
    work["response_z"] = zscore_series(work[response_col])
    for predictor in predictors:
        work[f"{predictor}_z"] = zscore_series(work[predictor])
    work = work[["response_z"] + [f"{predictor}_z" for predictor in predictors]].dropna().copy()
    y = work["response_z"]
    x = work[[col for col in work.columns if col != "response_z"]].copy()
    if include_intercept:
        x = sm.add_constant(x, has_constant="add")
    return sm.OLS(y, x).fit()


def summarize_model(dep_var: str, model, predictors: list[str], include_intercept: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "Dep. Var.": dep_var,
        "Predictors": " + ".join(predictors),
        "Intercept included": bool(include_intercept),
        "n_obs": int(model.nobs),
        "BIC": float(model.bic),
        "R2": float(model.rsquared),
    }
    row["Intercept_coef"] = float(model.params["const"]) if "const" in model.params.index else float("nan")
    row["Intercept_p"] = float(model.pvalues["const"]) if "const" in model.pvalues.index else float("nan")
    for predictor in predictors:
        param_name = f"{predictor}_z"
        row[f"{predictor}_coef"] = float(model.params[param_name])
        row[f"{predictor}_p"] = float(model.pvalues[param_name])
    return row


def build_metric_dataset() -> pd.DataFrame:
    day0_df = load_day0_dataset().sort_values("network_id").reset_index(drop=True)
    base_rows = []
    for _, row in day0_df.iterrows():
        biomasses = row[BIOMASS_COLS].to_numpy(dtype=float)
        metabolism = row[RATE_COLS].to_numpy(dtype=float)
        metrics = compute_network_metrics(biomasses, metabolism, m_scale=DEFAULT_M_SCALE)
        base_rows.append(
            {
                "network_id": row["network_id"],
                "pair_id": row["pair_id"],
                "Location": row["Location"],
                "Site": row["Site"],
                "Management": row["Management"],
                "Treatment": row["Treatment"],
                "S_full": metrics["entropy"],
                "total_flux": metrics["total_flux"],
                "dominant_real_eigenvalue": metrics["dominant_real_eigenvalue"],
                "reactivity": metrics["reactivity"],
                "_F_out": metrics["F_out"],
                "_F_in": metrics["F_in"],
            }
        )
    base_df = pd.DataFrame(base_rows)
    max_total_flux = float(base_df["total_flux"].max())
    rows = []
    for _, row in base_df.iterrows():
        entropy_partition = compute_entropy_partition(row["_F_out"], row["_F_in"], row["S_full"], max_total_flux)
        rows.append(
            {
                "network_id": row["network_id"],
                "pair_id": row["pair_id"],
                "Location": row["Location"],
                "Site": row["Site"],
                "Management": row["Management"],
                "Treatment": row["Treatment"],
                "S_full": row["S_full"],
                "S_het": entropy_partition["S_het"],
                "S_Ftot": entropy_partition["S_Ftot"],
                "alpha_to_max_flux": entropy_partition["alpha_to_max_flux"],
                "total_flux": row["total_flux"],
                "dominant_real_eigenvalue": row["dominant_real_eigenvalue"],
                "reactivity": row["reactivity"],
            }
        )
    return pd.DataFrame(rows).sort_values("network_id").reset_index(drop=True)


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = build_metric_dataset()
    metrics_df.to_csv(DERIVED_DIR / "day0_entropy_components.csv", index=False)

    model_specs = [
        ("|Re(lambda_max)|", metrics_df.assign(response_raw=metrics_df["dominant_real_eigenvalue"].abs())),
        ("Reactivity", metrics_df.assign(response_raw=metrics_df["reactivity"])),
    ]

    joint_rows = []
    sfull_rows = []
    for dep_var, df in model_specs:
        for include_intercept in [True, False]:
            joint_rows.append(
                summarize_model(
                    dep_var,
                    fit_model(df, "response_raw", ["S_Ftot", "S_het"], include_intercept=include_intercept),
                    ["S_Ftot", "S_het"],
                    include_intercept=include_intercept,
                )
            )
            sfull_rows.append(
                summarize_model(
                    dep_var,
                    fit_model(df, "response_raw", ["S_full"], include_intercept=include_intercept),
                    ["S_full"],
                    include_intercept=include_intercept,
                )
            )

    joint_df = pd.DataFrame(joint_rows)
    sfull_df = pd.DataFrame(sfull_rows)

    full_model_table = joint_df[
        ["Dep. Var.", "Intercept included", "n_obs", "S_Ftot_coef", "S_Ftot_p", "S_het_coef", "S_het_p", "BIC"]
    ].copy()
    sfull_model_table = sfull_df[
        ["Dep. Var.", "Intercept included", "n_obs", "S_full_coef", "S_full_p", "BIC"]
    ].copy()
    bic_comparison = full_model_table[["Dep. Var.", "Intercept included", "BIC"]].rename(columns={"BIC": "BIC_S_Ftot_plus_S_het"})
    bic_comparison = bic_comparison.merge(
        sfull_model_table[["Dep. Var.", "Intercept included", "BIC"]].rename(columns={"BIC": "BIC_S_full_only"}),
        on=["Dep. Var.", "Intercept included"],
        how="left",
    )
    bic_comparison["delta_BIC_full_minus_sfull"] = bic_comparison["BIC_S_Ftot_plus_S_het"] - bic_comparison["BIC_S_full_only"]

    full_model_table.to_csv(DERIVED_DIR / "full_model_table.csv", index=False)
    sfull_model_table.to_csv(DERIVED_DIR / "sfull_model_table.csv", index=False)
    bic_comparison.to_csv(DERIVED_DIR / "bic_comparison.csv", index=False)

    lines = [
        "",
        "=== Full models: response ~ S_Ftot + S_het ===",
        full_model_table.to_string(index=False),
        "",
        "=== Models with S_full only: response ~ S_full ===",
        sfull_model_table.to_string(index=False),
        "",
        "=== BIC comparison (full model vs S_full only) ===",
        bic_comparison.to_string(index=False),
        "",
        "Section 04 completed.",
    ]
    output_text = "\n".join(lines)
    print(output_text)
    OUTPUT_PATH.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
