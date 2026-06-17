"""Local helpers for the rewritten drought vs control day-0 analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent / "data_derived" / "individual_plot_tot_biomass_date0_resp_moist_mean_over_plot.xlsx"

KEY_COLS = ["Location", "Site", "Management", "Treatment"]
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


def load_day0_dataset() -> pd.DataFrame:
    # Sheet1 already contains one mean biomass vector per treatment combination at day 0.
    data = pd.read_excel(DATASET_PATH, sheet_name="Sheet1")
    rates = pd.read_excel(DATASET_PATH, sheet_name="rates")
    rates = rates.drop(columns=["row_index"]).drop_duplicates(KEY_COLS)
    # Rename the ppnnem rate so it can sit next to the other metabolic parameters.
    rates = rates.rename(columns={"ppnnem_M": "ppnnem_met"})
    data = data.merge(rates, on=KEY_COLS, how="left")
    data = data.rename(columns={"Moist": "moisture", "Resp": "respiration"})
    data["day"] = "d0"
    data["pair_id"] = data[["Location", "Site", "Management"]].astype(str).agg("_".join, axis=1)
    data["network_id"] = data[["Location", "Site", "Management", "Treatment"]].astype(str).agg("_".join, axis=1) + "_d0"
    data["total_biomass"] = data[BIOMASS_COLS].sum(axis=1)
    return data


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
        bounds=(1e-12, np.inf),
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


def compute_metrics_from_flux_matrix(flux_matrix: np.ndarray, biomasses: np.ndarray, metabolism: np.ndarray, m_scale: float = DEFAULT_M_SCALE) -> dict[str, object]:
    if not np.isfinite(flux_matrix).all() or not np.isfinite(biomasses).all() or not np.isfinite(metabolism).all():
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
    flow_out = flux_matrix.sum(axis=1)
    flow_in = flux_matrix.sum(axis=0)
    beta_out, beta_in, beta_success = solve_beta(TOPOLOGY, flow_out, flow_in)
    entropy = compute_entropy(TOPOLOGY, beta_out, beta_in, flow_out, flow_in) if beta_success else float("nan")
    jacobian = compute_jacobian(TOPOLOGY, flux_matrix, biomasses, mortality)
    # Some ensemble samples can produce extremely small biomasses and non-finite Jacobian entries.
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


def compute_network_metrics(biomasses: np.ndarray, metabolism: np.ndarray, m_scale: float = DEFAULT_M_SCALE) -> dict[str, object]:
    # Biomasses are already totals in the derived dataset, so no extra reconstruction is needed.
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
            "flux_solver_cost": float("nan"),
        }
    mortality = D_I * m_scale
    flux_solver = Fluxes(TOPOLOGY, biomasses, metabolism, mortality)
    flux_matrix, flux_cost = flux_solver.solve()
    metrics = compute_metrics_from_flux_matrix(flux_matrix, biomasses, metabolism, m_scale=m_scale)
    metrics["flux_solver_cost"] = float(flux_cost)
    return metrics


def compute_entropy_partition(flow_out: np.ndarray, flow_in: np.ndarray, entropy_full: float) -> dict[str, float]:
    total_flux = float(np.sum(flow_out))
    if not np.isfinite(total_flux) or total_flux <= 0:
        return {
            "S_het": float("nan"),
            "S_Ftot": float("nan"),
            "S_reconstructed": float("nan"),
            "S_reconstruction_error": float("nan"),
            "beta_norm_converged": False,
        }

    norm_out = flow_out / total_flux
    norm_in = flow_in / total_flux
    beta_out_norm, beta_in_norm, beta_success = solve_beta(TOPOLOGY, norm_out, norm_in)
    if not beta_success:
        return {
            "S_het": float("nan"),
            "S_Ftot": float("nan"),
            "S_reconstructed": float("nan"),
            "S_reconstruction_error": float("nan"),
            "beta_norm_converged": False,
        }

    s_het = compute_entropy(TOPOLOGY, beta_out_norm, beta_in_norm, norm_out, norm_in)
    s_ftot = float(np.sum(TOPOLOGY) * np.log(total_flux))
    s_reconstructed = s_het + s_ftot
    return {
        "S_het": float(s_het),
        "S_Ftot": float(s_ftot),
        "S_reconstructed": float(s_reconstructed),
        "S_reconstruction_error": float(entropy_full - s_reconstructed),
        "beta_norm_converged": True,
    }


def build_crema_sampler(observed_flux_matrix: np.ndarray) -> dict[str, np.ndarray | object]:
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


def solve_biomass_for_crema_sample(sample_flux: np.ndarray, initial_biomasses: np.ndarray, metabolism: np.ndarray, real_flow_out: np.ndarray, real_flow_in: np.ndarray, m_scale: float = DEFAULT_M_SCALE) -> dict[str, object]:
    flow_out = np.asarray(sample_flux, dtype=float).sum(axis=1)
    flow_in = np.asarray(sample_flux, dtype=float).sum(axis=0)
    mortality = D_I * m_scale
    numerator = np.concatenate([real_flow_in[:10], real_flow_out[10:]])
    denominator = np.concatenate([flow_in[:10], flow_out[10:]])
    if np.any(denominator <= 0) or np.any(initial_biomasses[10:] <= 0):
        return {
            "solver_success": False,
            "solver_cost": float("nan"),
            "biomasses": np.full(len(initial_biomasses), np.nan),
        }
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
