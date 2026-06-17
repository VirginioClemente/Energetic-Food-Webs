# Soil Food-Web Energetics Analysis

This repository contains the analysis code, generated outputs, and documentation
supporting the manuscript:

**Maximum entropy networks predict fluctuations and stability of food web energetics**

The pipeline reconstructs energetic soil food webs, fits maximum-entropy CReMa
ensembles, computes entropy and dynamical stability metrics, and tests the
fluctuation-response relation (FRR) for paired control and drought treatments.

## Repository Structure

```text
.
|-- data/                             Dataset availability note
|-- data_derived/                     Curated-dataset availability notes
|-- 01_control_stationarity_cv/       Control CV and entropy analysis
|-- 02_drought_control_flux_entropy/  Day-0 entropy, stability, and ensembles
|-- 03_frr_moisture_respiration/      FRR, moisture, and respiration analyses
|-- 04_entropy_component_regression/  Entropy component regressions
|-- SI_mfinale_sensitivity/           Supporting sensitivity analysis
|-- checks/                           Lightweight result-summary checks
|-- run_all.py                        Runs all analysis sections in order
|-- requirements.txt                  Python dependencies
`-- .gitignore                       Files that should not be committed
```

Each analysis section contains:

- `run.py`: terminal entry point for the section.
- `*_local_common.py`: helper functions used only by that section.
- main analysis script: figure generation, derived tables, and statistics.
- `derived_data/`: CSV or Parquet outputs produced by the section.
- `figures/`: generated PDF and PNG figures, where applicable.

## Data Availability

The source and curated datasets required to rerun the full pipeline are not
included in this public repository. They are available from the corresponding
author upon reasonable request.

The expected raw source files are:

- `topology.txt`: 13 x 13 resource-consumer adjacency matrix.
- `data_set_published.xlsx`: published environmental measurements.
- `merged_PublishedData.xlsx`: animal count and biomass records.
- `FoodWeb_vectorsFinal.xlsx`: food-web vectors, rates, and basal biomasses.
- `individual_plot_tot_biomass_flux.xlsx`: plot-level biomass and flux data.

The expected curated workbooks used directly by the analysis scripts are:

- `Dataset_analysis_CV.xlsx`: control-only plot-day data used by Section 01.
- `individual_plot_tot_biomass_date0_resp_moist_mean_over_plot.xlsx`:
  plot-averaged day-0 networks used by Sections 02, 03, 04, and SI.
- `individual_plot_tot_biomass_date0_resp_moist.xlsx`: plot-level day-0 data
  with respiration and moisture.

The repository retains generated section-level outputs and figures so the main
reported results can be inspected without access to the underlying datasets.

## Analysis Sections

### 01 - Control Stationarity, CV, and Entropy

Entry point:

```bash
python 01_control_stationarity_cv/run.py
```

This section computes the empirical coefficient of variation (CV) of total
consumer abundance (`SumN`) across control plot-day observations, reconstructs
one mean biomass vector per control group, solves food-web fluxes, computes
network entropy, samples CReMa ensembles, and compares empirical CV with
ensemble-predicted CV.

Main outputs:

- `derived_data/empirical_cv_sumN.csv`
- `derived_data/mean_biomass_for_cv_analysis.csv`
- `derived_data/entropy_sumN.csv`
- `derived_data/ensemble_cv_sumN.csv`
- `derived_data/cv_entropy_stats_sumN.csv`
- `figures/figure_control_cv_vs_entropy_count.*`
- `figures/figure_control_cv_emp_vs_ensemble_count.*`

### 02 - Day-0 Drought vs Control Entropy and Stability

Entry point:

```bash
python 02_drought_control_flux_entropy/run.py
```

This section uses the day-0 plot-averaged networks, excludes the network whose
entropy fit does not converge, computes entropy, total flux, dominant real
eigenvalue, and reactivity for the retained networks, partitions entropy into
heterogeneity and total-flux components, and samples CReMa ensembles around the
observed networks.

Main outputs:

- `derived_data/aggregated_plot_means_drought_control_day0.csv`
- `derived_data/day0_network_metrics.csv`
- `derived_data/control_day0_ensemble_samples.parquet`
- `derived_data/drought_day0_ensemble_samples.parquet`
- `derived_data/control_day0_ensemble_summary.csv`
- `derived_data/drought_day0_ensemble_summary.csv`
- `figures/figure_entropy_main_panel.*`
- `figures/figure_day0_ensemble_empirical_rank.*`

### 03 - Fluctuation-Response Relation, Moisture, and Respiration

Entry point:

```bash
python 03_frr_moisture_respiration/run.py
```

This section recomputes day-0 metrics, derives the inverse-scale parameter
summary used in the FRR, pairs control and drought networks, and tests whether
the finite-difference response ratio `Delta F_tot / Delta beta` is predicted by
the control-state fluctuation term `-Var(F_tot)`.

The curated data contain 30 possible control-drought pairs. One pair
(`Sc_S1_In`) has missing rate information and is flagged as invalid, so the FRR
test uses 29 valid pairs, matching the manuscript.

Main outputs:

- `derived_data/day0_network_metrics.csv`
- `derived_data/paired_frr_table.csv`
- `figures/figure_frr_main.*`
- `figures/figure_moisture_beta.*`
- `figures/figure_resp_flux.*`
- `figures/figure_resp_flux_no_outliers.*`

### 04 - Entropy Component Regression

Entry point:

```bash
python 04_entropy_component_regression/run.py
```

This section partitions entropy into `S_het` and `S_Ftot`, fits standardized
ordinary-least-squares regressions for stability responses, and compares the
two-component specification with the single `S_full` model using BIC.

Main outputs:

- `derived_data/day0_entropy_components.csv`
- `derived_data/full_model_table.csv`
- `derived_data/sfull_model_table.csv`
- `derived_data/bic_comparison.csv`
- `regression_output.txt`

### SI - Mortality-Scaling Sensitivity

Entry point:

```bash
python SI_mfinale_sensitivity/run.py
```

This supporting analysis evaluates how entropy, flux, dominant real eigenvalue,
and reactivity relationships change across a grid of mortality-scaling constants
in `m_finale = d_i * c`.

Main outputs:

- `derived_data/mfinale_sensitivity_full.csv`
- `derived_data/mfinale_sensitivity_summary.csv`
- `figures/figure_mfinale_trajectory.*`

## Running the Pipeline

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python run_all.py
```

Before rerunning the full pipeline, place the requested datasets in the expected
`data/` and `data_derived/` paths listed above.

For paper-scale ensemble sampling, use larger values:

```bash
python run_all.py --ensemble-samples 1000 --bootstrap-reps 2000
```

The CReMa ensemble steps can take a long time. The repository includes generated
outputs so the results can be inspected without rerunning the full ensemble
pipeline.

## Quick Consistency Check

To summarize the key manuscript-level results from the included output files:

```bash
python checks/summarize_results.py
```

Expected values from the included outputs are approximately:

- Control CV vs entropy: `n = 30`, Pearson `r = -0.476`, `p = 0.0078`.
- Day-0 entropy vs dominant real eigenvalue: `n = 59`, Pearson `r = -0.415`.
- Day-0 entropy vs reactivity: `n = 59`, Pearson `r = 0.455`.
- Day-0 entropy vs total flux: `n = 59`, Pearson `r = 0.716`.
- FRR valid pairs: `n = 29`, Pearson `r = 0.838`, `R2 = 0.702`.
- Ensemble rank check: dominant real eigenvalue and reactivity are reproduced
  for all 59 evaluable day-0 networks in the included Section 02 summaries.

## Notes for Reuse

- The scripts use relative paths, so commands should be run from the repository
  root or through the section-level `run.py` entry points.
- Full reruns require the datasets, which are available upon reasonable request.
- Generated outputs are intentionally included because ensemble sampling is
  computationally expensive and may vary slightly with package versions.
- Python cache files, editor lock files, local environments, and ad hoc logs are
  ignored by `.gitignore`.

## Citation

If you use or adapt this repository, cite the associated manuscript.
