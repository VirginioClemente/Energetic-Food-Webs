"""Local helpers for the rewritten FRR, moisture and respiration analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent / "data_derived" / "individual_plot_tot_biomass_date0_resp_moist_mean_over_plot.xlsx"

KEY_COLS = ["Location", "Site", "Management", "Treatment"]
PAIR_COLS = ["Location", "Site", "Management"]
BIOMASS_COLS = [
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
    "fungi_B",
    "bact_B",
    "roots_B",
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

EXCLUDED_NETWORKS = {
    "Sc_S1_In_D_d0": "missing information",
}

def load_day0_dataset() -> pd.DataFrame:
    data = pd.read_excel(DATASET_PATH, sheet_name="Sheet1")
    rates = pd.read_excel(DATASET_PATH, sheet_name="rates")
    rates = rates.drop(columns=["row_index"]).drop_duplicates(KEY_COLS)
    rates = rates.rename(columns={"ppnnem_M": "ppnnem_met"})
    data = data.merge(rates, on=KEY_COLS, how="left")
    data = data.rename(columns={"Moist": "moisture", "Resp": "respiration"})
    data["pair_id"] = data[PAIR_COLS].astype(str).agg("_".join, axis=1)
    data["network_id"] = data[KEY_COLS].astype(str).agg("_".join, axis=1) + "_d0"
    data["total_biomass"] = data[BIOMASS_COLS].sum(axis=1)
    data = data[~data["network_id"].isin(EXCLUDED_NETWORKS)].copy()
    return data.reset_index(drop=True)


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


def compute_jacobian(adjacency: np.ndarray, flux_matrix: np.ndarray, biomasses: np.ndarray, mortality: np.ndarray) -> np.ndarray:
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
    return float(np.max(np.linalg.eigvals(hermitian)).real)


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
    if not np.isfinite(biomasses).all() or not np.isfinite(metabolism).all():
        return {
            "flux_matrix": np.full_like(TOPOLOGY, np.nan, dtype=float),
            "F_out": np.full(13, np.nan),
            "F_in": np.full(13, np.nan),
            "beta_out": np.full(13, np.nan),
            "beta_in": np.full(13, np.nan),
            "entropy": float("nan"),
            "total_flux": float("nan"),
            "dominant_real_eigenvalue": float("nan"),
            "reactivity": float("nan"),
        }
    mortality = D_I * m_scale
    flux_solver = Fluxes(TOPOLOGY, biomasses, metabolism, mortality)
    flux_matrix, _ = flux_solver.solve()
    flow_out = flux_matrix.sum(axis=1)
    flow_in = flux_matrix.sum(axis=0)
    beta_out, beta_in, beta_success = solve_beta(TOPOLOGY, flow_out, flow_in)
    entropy = compute_entropy(TOPOLOGY, beta_out, beta_in, flow_out, flow_in) if beta_success else float("nan")
    jacobian = compute_jacobian(TOPOLOGY, flux_matrix, biomasses, mortality)
    jacobian = np.nan_to_num(jacobian, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues = LA.eigvals(jacobian)
    return {
        "flux_matrix": flux_matrix,
        "F_out": flow_out,
        "F_in": flow_in,
        "beta_out": beta_out,
        "beta_in": beta_in,
        "entropy": float(entropy),
        "total_flux": float(np.sum(flux_matrix)),
        "dominant_real_eigenvalue": float(np.max(eigenvalues).real),
        "reactivity": numerical_abscissa(jacobian),
    }


def beta_summary_from_metrics(metrics: dict[str, object]) -> dict[str, object]:
    beta_out = np.asarray(metrics["beta_out"], dtype=float)
    beta_in = np.asarray(metrics["beta_in"], dtype=float)
    lambda_matrix = beta_out[:, None] + beta_in[None, :]
    active_lambda = lambda_matrix[TOPOLOGY == 1]
    active_positive = active_lambda[np.isfinite(active_lambda) & (active_lambda > 0)]
    total_flux = float(metrics["total_flux"])
    return {
        "beta_global": float(TOPOLOGY.sum() / total_flux) if np.isfinite(total_flux) and total_flux > 0 else float("nan"),
        "beta_link_hmean": float(len(active_positive) / np.sum(1.0 / active_positive)) if len(active_positive) else float("nan"),
        "beta_out_json": json.dumps(beta_out.tolist()),
        "beta_in_json": json.dumps(beta_in.tolist()),
    }
