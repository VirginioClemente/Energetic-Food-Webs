"""Local helpers for the rewritten control CV analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent / "data_derived" / "Dataset_analysis_CV.xlsx"

KEY_COLS = ["Location", "Site", "Management", "Treatment"]
CONSUMER_BIOMASS_COLS = [
    "predmite_B",
    "prednem_B",
    "coll_B",
    "orib_B",
    "detmite_B",
    "plantfeed_B",
    "ppnnem_B",
    "omninem_B",
    "fungnem_B",
    "bactnem_B",
]
BASAL_BIOMASS_COLS = ["fungi_B", "bact_B", "roots_B"]
BIOMASS_COLS = CONSUMER_BIOMASS_COLS + BASAL_BIOMASS_COLS
INDIVIDUAL_MASS_COLS = [
    "predmite_M",
    "prednem_M",
    "coll_M",
    "orib_M",
    "detmite_M",
    "plantfeed_M",
    "ppnnem_M",
    "omninem_M",
    "fungnem_M",
    "bactnem_M",
]
RATE_COLS = [
    "predmite_met",
    "prednem_met",
    "coll_M_met",
    "orib_M_met",
    "detmite_met",
    "plantfeed_met",
    "ppnnem_met",
    "omninem_met",
    "fungnem_met",
    "bactnem_met",
    "fungi_met",
    "bact_met",
    "root_met",
]

TOPOLOGY = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    ],
    dtype=float,
)

D_I = np.asarray([1.84, 1.6, 1.84, 1.2, 1.84, 1.84, 1.08, 4.36, 1.92, 2.68, 1.2, 1.2, 1.0]) / 8760
EFFICIENCY = np.asarray([0.906, 0.906, 0.158, 0.158, 0.158, 0.545, 0.545, 0.536, 0.158, 0.158, 1.0, 1.0, 1.0])
DEFAULT_M_SCALE = 0.01

def build_group_id(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg("_".join, axis=1)


def load_cv_dataset() -> pd.DataFrame:
    # Sheet1 contains the biomasses and individual masses used in section 01.
    data = pd.read_excel(DATASET_PATH, sheet_name="Sheet1")
    rates = pd.read_excel(DATASET_PATH, sheet_name="rates")
    rates = rates.drop(columns=["row_index"]).drop_duplicates(KEY_COLS)
    # Rename the ppnnem rate to avoid clashing with the individual-mass column in Sheet1.
    rates = rates.rename(columns={"ppnnem_M": "ppnnem_met"})
    data = data.merge(rates, on=KEY_COLS, how="left")
    data["group_id"] = build_group_id(data, KEY_COLS)
    return data


def aggregate_for_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    # For the ensemble we average biomasses over Day and Plot within each group.
    aggregations = {column: "mean" for column in BIOMASS_COLS}
    for column in INDIVIDUAL_MASS_COLS + RATE_COLS:
        aggregations[column] = "first"
    aggregated = df.groupby(KEY_COLS, as_index=False, sort=True).agg(aggregations)
    aggregated["group_id"] = build_group_id(aggregated, KEY_COLS)
    return aggregated


def safe_corr(x: pd.Series, y: pd.Series) -> dict[str, float]:
    # Keep the correlation summary simple and robust to missing values.
    clean = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return {
            "n": float(len(clean)),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
        }
    pearson_r = np.corrcoef(clean["x"], clean["y"])[0, 1]
    from scipy import stats

    pearson_p = stats.pearsonr(clean["x"], clean["y"]).pvalue
    spearman_rho, spearman_p = stats.spearmanr(clean["x"], clean["y"])
    return {
        "n": float(len(clean)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def bootstrap_correlation_ci(x: pd.Series, y: pd.Series, method: str, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    from scipy import stats

    clean = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.choice(clean.index.to_numpy(), size=len(clean), replace=True)
        sample = clean.loc[idx]
        if method == "pearson":
            draws.append(stats.pearsonr(sample["x"], sample["y"])[0])
        else:
            draws.append(stats.spearmanr(sample["x"], sample["y"])[0])
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def equation_system(beta: np.ndarray, adjacency: np.ndarray, flow_out: np.ndarray, flow_in: np.ndarray) -> list[float]:
    n_nodes = adjacency.shape[0]
    beta_out = beta[:n_nodes]
    beta_in = beta[n_nodes:]
    residuals = []
    for i in range(n_nodes):
        sum_out = sum(adjacency[i, j] / (beta_out[i] + beta_in[j]) for j in range(n_nodes) if j != i)
        sum_in = sum(adjacency[j, i] / (beta_out[j] + beta_in[i]) for j in range(n_nodes) if j != i)
        residuals.append(sum_out - flow_out[i])
        residuals.append(sum_in - flow_in[i])
    return residuals


def solve_beta(adjacency: np.ndarray, flow_out: np.ndarray, flow_in: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    n_nodes = adjacency.shape[0]
    result = least_squares(
        equation_system,
        np.ones(2 * n_nodes),
        args=(adjacency, flow_out, flow_in),
        bounds=(1e-9, np.inf),
    )
    beta_out = result.x[:n_nodes]
    beta_in = result.x[n_nodes:]
    beta_in[flow_in == 0] = 0
    beta_out[flow_out == 0] = 0
    return beta_out, beta_in, bool(result.success)


def compute_entropy(adjacency: np.ndarray, beta_out: np.ndarray, beta_in: np.ndarray, flow_out: np.ndarray, flow_in: np.ndarray) -> float:
    epsilon = 1e-12
    term_1 = sum(beta_out[i] * flow_out[i] + beta_in[i] * flow_in[i] for i in range(adjacency.shape[0]))
    term_2 = -sum(
        adjacency[i, j] * np.log(max(beta_out[i] + beta_in[j], epsilon))
        for i in range(adjacency.shape[0])
        for j in range(adjacency.shape[0])
        if j != i and (beta_out[i] + beta_in[j]) > 0
    )
    return float(term_1 + term_2)


def equations_dinamiche(number: np.ndarray, flow_out: np.ndarray, flow_in: np.ndarray, metabolism: np.ndarray, growth: np.ndarray, mortality: np.ndarray) -> np.ndarray:
    values = np.zeros(len(number), dtype=float)
    for i in range(len(number)):
        if i < 10:
            values[i] = -metabolism[i] * number[i] + EFFICIENCY[i] * flow_in[i] - flow_out[i] - mortality[i] * number[i] ** 2
        else:
            values[i] = growth[i] * number[i] - flow_out[i] - mortality[i] * number[i] ** 2
    return values


def numerically_solve_equations_dinamiche(initial_values: np.ndarray, flow_out: np.ndarray, flow_in: np.ndarray, metabolism: np.ndarray, growth: np.ndarray, mortality: np.ndarray) -> tuple[np.ndarray, float]:
    result = least_squares(
        fun=equations_dinamiche,
        x0=initial_values,
        jac="3-point",
        args=(flow_out, flow_in, metabolism, growth, mortality),
        bounds=(0, np.inf),
        max_nfev=100000,
        loss="huber",
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    )
    return np.asarray(result.x, dtype=float), float(result.cost)


def compute_jacobian(adjacency: np.ndarray, flux_matrix: np.ndarray, biomasses: np.ndarray, metabolism: np.ndarray, mortality: np.ndarray) -> np.ndarray:
    n_nodes = len(biomasses)
    jacobian = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j and adjacency.sum(0)[i] == 0:
                jacobian[i, j] = -biomasses[i] * mortality[i]
            elif i == j:
                jacobian[i, j] = -mortality[i] * biomasses[i]
            else:
                jacobian[i, j] = EFFICIENCY[i] * flux_matrix[j, i] / biomasses[j] - flux_matrix[i, j] / biomasses[j]
    return jacobian


def numerical_abscissa(matrix: np.ndarray) -> float:
    hermitian = (matrix + matrix.T) / 2
    return float(np.max(np.linalg.eigvals(hermitian)))


def compute_tot_fluxes(flux_matrix: np.ndarray) -> float:
    return float(np.sum(flux_matrix))


class Fluxes:
    def __init__(self, adjacency: np.ndarray, biomasses: np.ndarray, metabolism: np.ndarray, mortality: np.ndarray):
        self.adjacency = adjacency
        self.biomasses = biomasses
        self.metabolism = metabolism
        self.mortality = mortality

    def equation_to_solve(self, flow_in: np.ndarray, adjacency: np.ndarray, biomasses: np.ndarray, metabolism: np.ndarray, mortality: np.ndarray) -> np.ndarray:
        n_nodes = len(biomasses)
        values = np.zeros(n_nodes, dtype=float)
        weights = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            for j in range(n_nodes):
                denominator = sum(adjacency[k, j] * biomasses[k] for k in range(n_nodes))
                if denominator == 0:
                    denominator = 1
                weights[i, j] = adjacency[i, j] * biomasses[i] / denominator
        for i in range(n_nodes):
            if adjacency.sum(0)[i] != 0:
                outgoing = 0.0
                for j in range(n_nodes):
                    outgoing += weights[i, j] * flow_in[j]
                values[i] = EFFICIENCY[i] * flow_in[i] - metabolism[i] * biomasses[i] - outgoing - mortality[i] * biomasses[i] ** 2
        return values

    def solve(self) -> tuple[np.ndarray, float]:
        result = least_squares(
            fun=self.equation_to_solve,
            x0=self.biomasses,
            jac="3-point",
            args=(self.adjacency, self.biomasses, self.metabolism, self.mortality),
            bounds=(0, np.inf),
            max_nfev=1000,
            loss="linear",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        n_nodes = len(self.biomasses)
        weights = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            for j in range(n_nodes):
                denominator = sum(self.adjacency[k, j] * self.biomasses[k] for k in range(n_nodes))
                if denominator == 0:
                    denominator = 1
                weights[i, j] = self.adjacency[i, j] * self.biomasses[i] / denominator
        flow_in = result.x
        flux_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            for j in range(n_nodes):
                flux_matrix[i, j] = flow_in[j] * weights[i, j]
        return flux_matrix, float(result.cost)


def compute_network_metrics(biomasses: np.ndarray, metabolism: np.ndarray, m_scale: float = DEFAULT_M_SCALE) -> dict[str, object]:
    # Solve the observed flux matrix, then derive entropy and stability metrics.
    biomasses = np.asarray(biomasses, dtype=float)
    metabolism = np.asarray(metabolism, dtype=float)
    mortality = D_I * m_scale
    flux_solver = Fluxes(TOPOLOGY, biomasses, metabolism, mortality)
    flux_matrix, flux_cost = flux_solver.solve()
    flow_out = flux_matrix.sum(axis=1)
    flow_in = flux_matrix.sum(axis=0)
    beta_out, beta_in, beta_success = solve_beta(TOPOLOGY, flow_out, flow_in)
    entropy = compute_entropy(TOPOLOGY, beta_out, beta_in, flow_out, flow_in) if beta_success else float("nan")
    jacobian = compute_jacobian(TOPOLOGY, flux_matrix, biomasses, metabolism, mortality)
    eigenvalues = LA.eigvals(np.nan_to_num(jacobian, nan=0.0, posinf=0.0, neginf=0.0))
    return {
        "flux_matrix": flux_matrix,
        "flux_solver_cost": float(flux_cost),
        "entropy": float(entropy),
        "total_flux": compute_tot_fluxes(flux_matrix),
        "dominant_real_eigenvalue": float(np.max(eigenvalues).real),
        "reactivity": numerical_abscissa(jacobian),
    }


def build_crema_sampler(observed_flux_matrix: np.ndarray) -> dict[str, np.ndarray | object]:
    # Fit the weighted ensemble on the observed in/out strength sequence.
    from NEMtropy import DirectedGraph

    flow_out = observed_flux_matrix.sum(axis=1)
    flow_in = observed_flux_matrix.sum(axis=0)
    graph = DirectedGraph(strength_sequence=np.concatenate([flow_out, flow_in]))
    graph.solve_tool(model="crema", method="newton", initial_guess="random", adjacency=TOPOLOGY)
    return {
        "graph": graph,
        "F_out_real": flow_out,
        "F_in_real": flow_in,
    }


def sample_crema_flux_matrix(sampler: dict[str, np.ndarray | object], seed: int, output_dir: str | Path | None = None) -> np.ndarray:
    from NEMtropy.network_functions import build_adjacency_from_edgelist

    graph = sampler["graph"]
    if output_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="crema_sample_")
        tmpdir = Path(tmp_ctx.__enter__())
    else:
        tmp_ctx = None
        tmpdir = Path(output_dir)
        tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        sample_path = tmpdir / "0.txt"
        if sample_path.exists():
            sample_path.unlink()
        graph.ensemble_sampler(1, cpu_n=1, output_dir=f"{tmpdir}/", seed=seed)
        edgelist = np.loadtxt(sample_path)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.__exit__(None, None, None)
    flux_matrix = np.asarray(
        build_adjacency_from_edgelist(
            edgelist=np.atleast_2d(edgelist),
            is_directed=True,
            is_sparse=False,
            is_weighted=True,
        ),
        dtype=float,
    )
    if flux_matrix.shape != TOPOLOGY.shape:
        padded = np.zeros_like(TOPOLOGY)
        rows = min(padded.shape[0], flux_matrix.shape[0])
        cols = min(padded.shape[1], flux_matrix.shape[1])
        padded[:rows, :cols] = flux_matrix[:rows, :cols]
        flux_matrix = padded
    return flux_matrix


def solve_biomass_for_crema_sample(
    sample_flux: np.ndarray,
    initial_biomasses: np.ndarray,
    metabolism: np.ndarray,
    real_flow_out: np.ndarray,
    real_flow_in: np.ndarray,
    m_scale: float = DEFAULT_M_SCALE,
) -> dict[str, object]:
    # Reconstruct one biomass vector from a sampled flux matrix.
    flow_out = np.asarray(sample_flux, dtype=float).sum(axis=1)
    flow_in = np.asarray(sample_flux, dtype=float).sum(axis=0)
    mortality = D_I * m_scale
    numerator = np.concatenate([real_flow_in[:10], real_flow_out[10:]])
    denominator = np.concatenate([flow_in[:10], flow_out[10:]])
    m_sample = numerator / denominator * mortality
    growth = np.append(
        np.zeros(10, dtype=float),
        [
            flow_out[10] / initial_biomasses[10] + m_sample[10] * initial_biomasses[10],
            flow_out[11] / initial_biomasses[11] + m_sample[11] * initial_biomasses[11],
            flow_out[12] / initial_biomasses[12] + m_sample[12] * initial_biomasses[12],
        ],
    )
    biomasses, cost = numerically_solve_equations_dinamiche(initial_biomasses, flow_out, flow_in, metabolism, growth, m_sample)
    return {
        "solver_success": bool(np.isfinite(cost) and cost <= 0.1 and np.all(np.isfinite(biomasses)) and np.all(biomasses > 0)),
        "solver_cost": float(cost),
        "biomasses": biomasses,
    }


def sum_n_from_biomasses(biomasses: np.ndarray, individual_masses: np.ndarray) -> float:
    # SumN is based only on the first 10 consumer groups.
    counts = np.asarray(biomasses[:10], dtype=float) / np.asarray(individual_masses, dtype=float)
    return float(np.sum(counts))
