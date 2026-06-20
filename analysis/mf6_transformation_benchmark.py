from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cmcrameri.cm as cmc
import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.special import exp1


PROJECT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT / "tables"
FIG_DIR = PROJECT / "figures"
OUT_DIR = PROJECT / "outputs"
MF6_WORKSPACE = OUT_DIR / "mf6_numerical_benchmark"

MF6_EXE = shutil.which("mf6") or r"C:\Users\YFLin\AppData\Local\Programs\MODFLOW6\6.7.0\mf6.7.0_win64\bin\mf6.exe"

SECONDS_PER_DAY = 86400.0
Q_M3S = 1.2e-4
Q_M3D = Q_M3S * SECONDS_PER_DAY
T_REF_M2S = 5.0e-5
T_REF_M2D = T_REF_M2S * SECONDS_PER_DAY
S_REF = 1.0e-3
AQ_THICKNESS_M = 10.0
K_REF_MD = T_REF_M2D / AQ_THICKNESS_M
OFFSET_M = 1.0e-3

NROW = 121
NCOL = 121
DELR_M = 5.0
DELC_M = 5.0
TOP_M = 10.0
BOTM_M = 0.0
CENTER = (0, NROW // 2, NCOL // 2)

TIMES_S = np.geomspace(90.0, 2.0 * SECONDS_PER_DAY, 28)
TIMES_D = TIMES_S / SECONDS_PER_DAY
DTS_D = np.diff(np.r_[0.0, TIMES_D])

OBSERVATION_WELLS = [
    ("R20E", 20.0, 0.0),
    ("R50E", 50.0, 0.0),
    ("R100E", 100.0, 0.0),
    ("R100N", 100.0, 90.0),
]

PATHWAYS = [
    "Theis confined",
    "Hantush-Jacob leaky",
    "Lagging Darcy no-leakage",
    "Lagging Darcy with leakage",
]

DEPRECATED_FIXED_SUPPORT_RADIUS_M = 130.0


def cmc_color(colormap_name: str, sample: float) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in getattr(cmc, colormap_name)(sample))


PATHWAY_COLORS = {
    "Theis confined": cmc_color("batlow", 0.12),
    "Hantush-Jacob leaky": cmc_color("batlow", 0.40),
    "Lagging Darcy no-leakage": cmc_color("batlow", 0.68),
    "Lagging Darcy with leakage": cmc_color("batlow", 0.90),
}

CLASS_COLORS = {
    "homogeneous confined": cmc_color("navia", 0.18),
    "vertical leakage": cmc_color("navia", 0.36),
    "constant head boundary": cmc_color("navia", 0.54),
    "heterogeneous K": cmc_color("navia", 0.72),
    "heterogeneous leakage boundary": cmc_color("navia", 0.90),
}

NEUTRAL_DARK = cmc_color("grayC", 0.14)
NEUTRAL_MID = cmc_color("grayC", 0.48)
NEUTRAL_LIGHT = cmc_color("grayC", 0.90)


@dataclass(frozen=True)
class Mf6Scenario:
    scenario_id: str
    scenario_class: str
    seed: int
    sigma_ln_k: float
    corr_len_cells: float
    leakage_b_m: float | None
    constant_head_edges: bool
    description: str


def lock_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MF6_WORKSPACE.mkdir(parents=True, exist_ok=True)


def obs_cell(radius_m: float, angle_deg: float) -> tuple[int, int, int]:
    angle = math.radians(angle_deg)
    row_offset = int(round((radius_m * math.sin(angle)) / DELC_M))
    col_offset = int(round((radius_m * math.cos(angle)) / DELR_M))
    return (0, NROW // 2 - row_offset, NCOL // 2 + col_offset)


def cell_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(NROW)
    cols = np.arange(NCOL)
    x = (cols - NCOL // 2) * DELR_M
    y = (NROW // 2 - rows) * DELC_M
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx * xx + yy * yy)
    return xx, yy, rr


def build_scenarios() -> list[Mf6Scenario]:
    return [
        Mf6Scenario("mf6_homogeneous_00", "homogeneous confined", 11, 0.0, 0.0, None, False, "Confined homogeneous Darcy-flow baseline."),
        Mf6Scenario("mf6_leakage_00", "vertical leakage", 21, 0.0, 0.0, 80.0, False, "Distributed vertical exchange with leakage factor near 80 m."),
        Mf6Scenario("mf6_leakage_01", "vertical leakage", 22, 0.0, 0.0, 45.0, False, "Stronger distributed vertical exchange with leakage factor near 45 m."),
        Mf6Scenario("mf6_boundary_00", "constant head boundary", 31, 0.0, 0.0, None, True, "Finite domain with first type boundary support."),
        Mf6Scenario("mf6_boundary_01", "constant head boundary", 32, 0.0, 0.0, 140.0, True, "Finite domain with first type boundary and weak distributed exchange."),
        Mf6Scenario("mf6_heterogeneous_00", "heterogeneous K", 41, 0.45, 3.0, None, False, "Moderately correlated lognormal K field."),
        Mf6Scenario("mf6_heterogeneous_01", "heterogeneous K", 42, 0.65, 5.0, None, False, "More variable lognormal K field."),
        Mf6Scenario("mf6_heterogeneous_02", "heterogeneous K", 43, 0.85, 7.0, None, False, "Strongly variable lognormal K field."),
        Mf6Scenario("mf6_combined_00", "heterogeneous leakage boundary", 51, 0.55, 4.0, 90.0, True, "Heterogeneity, vertical exchange, and first type boundary together."),
        Mf6Scenario("mf6_combined_01", "heterogeneous leakage boundary", 52, 0.75, 6.0, 60.0, True, "Stronger heterogeneous leakage and boundary case."),
    ]


def make_k_field(scenario: Mf6Scenario) -> np.ndarray:
    if scenario.sigma_ln_k <= 0.0:
        return np.full((NROW, NCOL), K_REF_MD, dtype=float)
    rng = np.random.default_rng(scenario.seed)
    raw = rng.normal(0.0, 1.0, size=(NROW, NCOL))
    smooth = gaussian_filter(raw, sigma=scenario.corr_len_cells, mode="reflect")
    smooth = (smooth - np.mean(smooth)) / max(float(np.std(smooth)), 1.0e-12)
    ln_k = math.log(K_REF_MD) + scenario.sigma_ln_k * smooth
    return np.exp(ln_k)


def weighted_geomean(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.maximum(np.asarray(values, dtype=float), 1.0e-30)
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    valid = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return float(np.exp(np.mean(np.log(vals[np.isfinite(vals)]))))
    return float(np.exp(np.sum(w[valid] * np.log(vals[valid])) / np.sum(w[valid])))


def effective_support_radius_m(scenario: Mf6Scenario) -> tuple[float, dict[str, float]]:
    hydraulic_diffusivity = T_REF_M2S / max(S_REF, 1.0e-30)
    diffusion_radius = math.sqrt(max(2.25 * hydraulic_diffusivity * TIMES_S[-1], DELR_M**2))
    observation_radius = max(radius for _, radius, _ in OBSERVATION_WELLS)
    leakage_limit = float(scenario.leakage_b_m) if scenario.leakage_b_m is not None else float("inf")
    boundary_limit = max((NCOL // 2) * DELR_M, DELR_M)
    limited_radius = min(diffusion_radius, leakage_limit, boundary_limit)
    lower_radius = max(3.0 * DELR_M, min(observation_radius, boundary_limit) * 0.50)
    support_radius = min(max(limited_radius, lower_radius), boundary_limit)
    return support_radius, {
        "support_diffusion_radius_m": diffusion_radius,
        "support_observation_max_radius_m": observation_radius,
        "support_leakage_limit_m": leakage_limit,
        "support_boundary_limit_m": boundary_limit,
    }


def support_transmissivity_m2s(k_field_md: np.ndarray, scenario: Mf6Scenario) -> tuple[float, str]:
    if scenario.sigma_ln_k <= 0.0:
        return T_REF_M2S, "assigned homogeneous transmissivity; effective support radius reported for hydraulic context"
    _, _, radius = cell_coordinates()
    support_radius, controls = effective_support_radius_m(scenario)
    diffusion_radius = max(controls["support_diffusion_radius_m"], DELR_M)
    weights = np.exp(-((radius / diffusion_radius) ** 2))
    if scenario.leakage_b_m is not None:
        weights *= np.exp(-radius / max(float(scenario.leakage_b_m), DELR_M))
    weights = np.where(radius <= support_radius, weights, 0.0)
    transmissivity_m2s = k_field_md * AQ_THICKNESS_M / SECONDS_PER_DAY
    value = weighted_geomean(transmissivity_m2s, weights)
    return value, "scenario-specific diffusion/leakage/boundary weighted effective support; fixed-radius 130 m sensitivity retained only for comparison"


def run_mf6_scenario(scenario: Mf6Scenario) -> tuple[pd.DataFrame, dict, np.ndarray]:
    k_field = make_k_field(scenario)
    t_ref, support_definition = support_transmissivity_m2s(k_field, scenario)
    sim_ws = MF6_WORKSPACE / scenario.scenario_id
    if sim_ws.exists():
        shutil.rmtree(sim_ws)
    sim_ws.mkdir(parents=True)

    sim = flopy.mf6.MFSimulation(sim_name=scenario.scenario_id, exe_name=MF6_EXE, sim_ws=str(sim_ws), version="mf6")
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=len(DTS_D),
        perioddata=[(float(dt), 1, 1.0) for dt in DTS_D],
    )
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1.0e-8, inner_dvclose=1.0e-8)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=NROW,
        ncol=NCOL,
        delr=DELR_M,
        delc=DELC_M,
        top=TOP_M,
        botm=BOTM_M,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k_field, k33=k_field, save_specific_discharge=True)
    flopy.mf6.ModflowGwfsto(
        gwf,
        storagecoefficient=True,
        iconvert=0,
        ss=S_REF,
        steady_state={0: False},
        transient={idx: True for idx in range(len(DTS_D))},
    )
    wel_spd = {idx: [(CENTER, -Q_M3D)] for idx in range(len(DTS_D))}
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd, save_flows=True)

    if scenario.constant_head_edges:
        chd = []
        for row in range(NROW):
            chd.append(((0, row, 0), 0.0))
            chd.append(((0, row, NCOL - 1), 0.0))
        for col in range(1, NCOL - 1):
            chd.append(((0, 0, col), 0.0))
            chd.append(((0, NROW - 1, col), 0.0))
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, save_flows=True)

    if scenario.leakage_b_m is not None:
        c_days = T_REF_M2D / (scenario.leakage_b_m * scenario.leakage_b_m)
        cell_area = DELR_M * DELC_M
        conductance = cell_area * c_days
        ghb = [((0, row, col), 0.0, conductance) for row in range(NROW) for col in range(NCOL)]
        flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb, save_flows=True)

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="gwf.hds",
        budget_filerecord="gwf.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    ok, buff = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError(f"MODFLOW 6 failed for {scenario.scenario_id}: {' '.join(buff[-12:])}")

    heads = gwf.output.head().get_alldata()[:, 0, :, :]
    rows = []
    for well, radius_m, angle_deg in OBSERVATION_WELLS:
        _, row, col = obs_cell(radius_m, angle_deg)
        drawdown = np.maximum(-heads[:, row, col], 0.0)
        for time_s, dd in zip(TIMES_S, drawdown):
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_class": scenario.scenario_class,
                    "well": well,
                    "radius_m": radius_m,
                    "angle_deg": angle_deg,
                    "time_s": float(time_s),
                    "time_h": float(time_s / 3600.0),
                    "drawdown_m": float(dd),
                }
            )

    meta = {
        "scenario_id": scenario.scenario_id,
        "scenario_class": scenario.scenario_class,
        "seed": scenario.seed,
        "description": scenario.description,
        "sigma_ln_k": scenario.sigma_ln_k,
        "corr_len_cells": scenario.corr_len_cells,
        "leakage_B_m": scenario.leakage_b_m,
        "constant_head_edges": scenario.constant_head_edges,
        "truth_T_ref_m2_s": t_ref,
        "truth_S_ref": S_REF,
        "support_definition": support_definition,
        "K_geomean_m_d": float(np.exp(np.mean(np.log(k_field)))),
        "K_arithmetic_m_d": float(np.mean(k_field)),
        "K_cv": float(np.std(k_field) / np.mean(k_field)),
        "observation_count": int(len(TIMES_S) * len(OBSERVATION_WELLS)),
        "max_drawdown_m": float(max(row["drawdown_m"] for row in rows)),
    }
    return pd.DataFrame(rows), meta, k_field


def theis_drawdown(time_s: np.ndarray, q_m3s: float, trans: float, storage: float, radius_m: float) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    out = np.zeros_like(time_s)
    positive = time_s > 0.0
    u = radius_m * radius_m * storage / (4.0 * trans * time_s[positive])
    out[positive] = q_m3s / (4.0 * math.pi * trans) * exp1(u)
    return np.maximum(out, 0.0)


def hantush_well_function(u: np.ndarray, beta: float) -> np.ndarray:
    x, w = np.polynomial.laguerre.laggauss(44)
    out = np.zeros_like(u)
    positive = u > 0.0
    if not np.any(positive):
        return out
    if beta < 1.0e-8:
        out[positive] = exp1(u[positive])
        return out
    up = u[positive]
    y = up[:, None] + x[None, :]
    integrand = np.exp(-(beta * beta) / (4.0 * y)) / y
    out[positive] = np.exp(-up) * np.sum(w[None, :] * integrand, axis=1)
    return out


def leaky_drawdown(time_s: np.ndarray, q_m3s: float, trans: float, storage: float, leakage_b_m: float, radius_m: float) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    out = np.zeros_like(time_s)
    positive = time_s > 0.0
    u = radius_m * radius_m * storage / (4.0 * trans * time_s[positive])
    beta = radius_m / max(leakage_b_m, 1.0e-12)
    out[positive] = q_m3s / (4.0 * math.pi * trans) * hantush_well_function(u, beta)
    return np.maximum(out, 0.0)


def stehfest_weight(n: int, i: int) -> float:
    half = n // 2
    total = 0.0
    for k in range((i + 1) // 2, min(i, half) + 1):
        numerator = (k**half) * math.factorial(2 * k)
        denominator = (
            math.factorial(half - k)
            * math.factorial(k)
            * math.factorial(k - 1)
            * math.factorial(i - k)
            * math.factorial(2 * k - i)
        )
        total += numerator / denominator
    return ((-1) ** (i + half)) * total


def stehfest_invert(laplace_func, time_value: float, terms: int = 10) -> float:
    if time_value <= 0.0:
        return 0.0
    ln2 = math.log(2.0)
    return float((ln2 / time_value) * sum(stehfest_weight(terms, i) * laplace_func(i * ln2 / time_value) for i in range(1, terms + 1)))


def lagging_laplace(s: float, q_m3s: float, trans: float, storage: float, tau_q: float, tau_s: float, radius_m: float) -> float:
    memory = (1.0 + tau_q * s) / (1.0 + tau_s * s)
    effective = max(s * storage * memory / trans, 1.0e-30)
    arg = radius_m * math.sqrt(effective)
    if arg > 60.0:
        return 0.0
    from scipy.special import k0

    return q_m3s / (2.0 * math.pi * trans * s) * float(k0(arg))


def lagging_drawdown(time_s: np.ndarray, q_m3s: float, trans: float, storage: float, tau_q: float, tau_s: float, radius_m: float) -> np.ndarray:
    values = [
        max(stehfest_invert(lambda p: lagging_laplace(p, q_m3s, trans, storage, tau_q, tau_s, radius_m), float(t)), 0.0)
        for t in time_s
    ]
    return np.asarray(values, dtype=float)


def lagging_leaky_laplace(
    s: float,
    q_m3s: float,
    trans: float,
    storage: float,
    tau_q: float,
    tau_s: float,
    leakage_b_m: float,
    radius_m: float,
) -> float:
    memory = (1.0 + tau_q * s) / (1.0 + tau_s * s)
    leakage = 1.0 / (max(leakage_b_m, 1.0e-12) ** 2)
    effective = max(s * storage * memory / trans + leakage, 1.0e-30)
    arg = radius_m * math.sqrt(effective)
    if arg > 60.0:
        return 0.0
    from scipy.special import k0

    return q_m3s / (2.0 * math.pi * trans * s) * float(k0(arg))


def lagging_leaky_drawdown(
    time_s: np.ndarray,
    q_m3s: float,
    trans: float,
    storage: float,
    tau_q: float,
    tau_s: float,
    leakage_b_m: float,
    radius_m: float,
) -> np.ndarray:
    values = [
        max(
            stehfest_invert(
                lambda p: lagging_leaky_laplace(p, q_m3s, trans, storage, tau_q, tau_s, leakage_b_m, radius_m),
                float(t),
            ),
            0.0,
        )
        for t in time_s
    ]
    return np.asarray(values, dtype=float)


def log_response_residual(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.log((np.asarray(obs) + OFFSET_M) / (np.asarray(pred) + OFFSET_M))


def aic_bic(residual: np.ndarray, k: int) -> tuple[float, float]:
    clean = residual[np.isfinite(residual)]
    n = max(clean.size, 1)
    rss = max(float(np.sum(clean * clean)), 1.0e-18)
    sigma2 = rss / n
    return float(n * math.log(sigma2) + 2 * k), float(n * math.log(sigma2) + k * math.log(n))


def boundary_hit(values: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> bool:
    lower, upper = bounds
    return bool(np.any(values <= lower + 1.0e-3) or np.any(values >= upper - 1.0e-3))


def make_fit_row(
    pathway: str,
    scenario: dict,
    well: str,
    radius_m: float,
    obs: np.ndarray,
    pred: np.ndarray,
    log_values: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    parameter_count: int,
    trans: float,
    storage: float,
    tau_q_s: float = np.nan,
    tau_s_s: float = np.nan,
    leakage_b_m: float = np.nan,
) -> dict:
    residual = log_response_residual(obs, pred)
    aic, bic = aic_bic(residual, parameter_count)
    sd_eta = float(np.std(residual, ddof=1))
    t_factor = trans / scenario["truth_T_ref_m2_s"]
    s_factor = storage / scenario["truth_S_ref"]
    time_factor = (storage / trans) / (scenario["truth_S_ref"] / scenario["truth_T_ref_m2_s"])
    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_class": scenario["scenario_class"],
        "well": well,
        "radius_m": radius_m,
        "pathway": pathway,
        "truth_T_ref_m2_s": scenario["truth_T_ref_m2_s"],
        "truth_S_ref": scenario["truth_S_ref"],
        "T_fit_m2_s": float(trans),
        "S_fit": float(storage),
        "tau_q_fit_s": None if not np.isfinite(tau_q_s) else float(tau_q_s),
        "tau_s_fit_s": None if not np.isfinite(tau_s_s) else float(tau_s_s),
        "leakage_B_fit_m": None if not np.isfinite(leakage_b_m) else float(leakage_b_m),
        "lnM_T": float(math.log(max(t_factor, 1.0e-30))),
        "lnM_S": float(math.log(max(s_factor, 1.0e-30))),
        "lnM_capacity": float(math.log(max(t_factor, 1.0e-30))),
        "lnM_response_time": float(math.log(max(time_factor, 1.0e-30))),
        "bias_eta": float(np.mean(residual)),
        "sd_eta": sd_eta,
        "cov_response_model_factor": float(math.sqrt(math.exp(sd_eta * sd_eta) - 1.0)),
        "rmse_m": float(np.sqrt(np.mean((obs - pred) ** 2))),
        "aic": aic,
        "bic": bic,
        "parameter_count": parameter_count,
        "boundary_hit": boundary_hit(log_values, bounds),
    }


def fit_theis(time_s: np.ndarray, obs: np.ndarray, radius_m: float, scenario: dict, well: str) -> dict:
    bounds = (np.log([1.0e-7, 1.0e-6]), np.log([2.0e-3, 1.0e-1]))

    def residual(values: np.ndarray) -> np.ndarray:
        trans, storage = np.exp(values)
        pred = theis_drawdown(time_s, Q_M3S, trans, storage, radius_m)
        return log_response_residual(obs, pred)

    initial = np.log([scenario["truth_T_ref_m2_s"], S_REF])
    result = least_squares(residual, initial, bounds=bounds, max_nfev=240)
    trans, storage = np.exp(result.x)
    pred = theis_drawdown(time_s, Q_M3S, trans, storage, radius_m)
    return make_fit_row("Theis confined", scenario, well, radius_m, obs, pred, result.x, bounds, 2, trans, storage)


def leakage_initial_guess(scenario: dict, fallback: float = 80.0) -> float:
    value = scenario.get("leakage_B_m")
    if value is None:
        return fallback
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(value) or value <= 0.0:
        return fallback
    return value


def fit_hantush(time_s: np.ndarray, obs: np.ndarray, radius_m: float, scenario: dict, well: str) -> dict:
    bounds = (np.log([1.0e-7, 1.0e-6, 3.0]), np.log([2.0e-3, 1.0e-1, 1.0e5]))

    def residual(values: np.ndarray) -> np.ndarray:
        trans, storage, leakage_b = np.exp(values)
        pred = leaky_drawdown(time_s, Q_M3S, trans, storage, leakage_b, radius_m)
        return log_response_residual(obs, pred)

    starts = [
        np.log([scenario["truth_T_ref_m2_s"], S_REF, leakage_initial_guess(scenario)]),
        np.log([scenario["truth_T_ref_m2_s"], S_REF, 1.0e3]),
        np.log([scenario["truth_T_ref_m2_s"] * 0.7, S_REF * 1.4, 35.0]),
    ]
    best = None
    for initial in starts:
        result = least_squares(residual, initial, bounds=bounds, max_nfev=180)
        score = float(np.sum(result.fun * result.fun))
        if best is None or score < best[0]:
            best = (score, result)
    result = best[1]
    trans, storage, leakage_b = np.exp(result.x)
    pred = leaky_drawdown(time_s, Q_M3S, trans, storage, leakage_b, radius_m)
    return make_fit_row(
        "Hantush-Jacob leaky",
        scenario,
        well,
        radius_m,
        obs,
        pred,
        result.x,
        bounds,
        3,
        trans,
        storage,
        leakage_b_m=leakage_b,
    )


def fit_lagging(time_s: np.ndarray, obs: np.ndarray, radius_m: float, scenario: dict, well: str) -> dict:
    bounds = (np.log([1.0e-7, 1.0e-6, 1.0e-4, 1.0e-4]), np.log([2.0e-3, 1.0e-1, 1.0e5, 1.0e5]))

    def residual(values: np.ndarray) -> np.ndarray:
        trans, storage, tau_q, tau_s = np.exp(values)
        pred = lagging_drawdown(time_s, Q_M3S, trans, storage, tau_q, tau_s, radius_m)
        return log_response_residual(obs, pred)

    starts = [
        np.log([scenario["truth_T_ref_m2_s"], S_REF, 10.0, 100.0]),
        np.log([scenario["truth_T_ref_m2_s"], S_REF, 100.0, 1000.0]),
    ]
    best = None
    for initial in starts:
        result = least_squares(residual, initial, bounds=bounds, max_nfev=120)
        score = float(np.sum(result.fun * result.fun))
        if best is None or score < best[0]:
            best = (score, result)
    result = best[1]
    trans, storage, tau_q, tau_s = np.exp(result.x)
    pred = lagging_drawdown(time_s, Q_M3S, trans, storage, tau_q, tau_s, radius_m)
    return make_fit_row(
        "Lagging Darcy no-leakage",
        scenario,
        well,
        radius_m,
        obs,
        pred,
        result.x,
        bounds,
        4,
        trans,
        storage,
        tau_q_s=tau_q,
        tau_s_s=tau_s,
    )


def fit_lagging_leaky(time_s: np.ndarray, obs: np.ndarray, radius_m: float, scenario: dict, well: str) -> dict:
    bounds = (
        np.log([1.0e-7, 1.0e-6, 1.0e-4, 1.0e-4, 3.0]),
        np.log([2.0e-3, 1.0e-1, 1.0e5, 1.0e5, 1.0e5]),
    )

    def residual(values: np.ndarray) -> np.ndarray:
        trans, storage, tau_q, tau_s, leakage_b = np.exp(values)
        pred = lagging_leaky_drawdown(time_s, Q_M3S, trans, storage, tau_q, tau_s, leakage_b, radius_m)
        return log_response_residual(obs, pred)

    starts = [
        np.log([scenario["truth_T_ref_m2_s"], S_REF, 10.0, 100.0, leakage_initial_guess(scenario)]),
        np.log([scenario["truth_T_ref_m2_s"], S_REF, 100.0, 1000.0, 1.0e3]),
    ]
    best = None
    for initial in starts:
        result = least_squares(residual, initial, bounds=bounds, max_nfev=90)
        score = float(np.sum(result.fun * result.fun))
        if best is None or score < best[0]:
            best = (score, result)
    result = best[1]
    trans, storage, tau_q, tau_s, leakage_b = np.exp(result.x)
    pred = lagging_leaky_drawdown(time_s, Q_M3S, trans, storage, tau_q, tau_s, leakage_b, radius_m)
    return make_fit_row(
        "Lagging Darcy with leakage",
        scenario,
        well,
        radius_m,
        obs,
        pred,
        result.x,
        bounds,
        5,
        trans,
        storage,
        tau_q_s=tau_q,
        tau_s_s=tau_s,
        leakage_b_m=leakage_b,
    )


def fit_all(drawdown: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    case_lookup = cases.set_index("scenario_id").to_dict(orient="index")
    rows = []
    for (scenario_id, well), group in drawdown.groupby(["scenario_id", "well"], sort=False):
        scenario = {"scenario_id": scenario_id, **case_lookup[scenario_id]}
        group = group.sort_values("time_s")
        time_s = group["time_s"].to_numpy(float)
        obs = group["drawdown_m"].to_numpy(float)
        radius_m = float(group["radius_m"].iloc[0])
        rows.append(fit_theis(time_s, obs, radius_m, scenario, well))
        rows.append(fit_hantush(time_s, obs, radius_m, scenario, well))
        rows.append(fit_lagging(time_s, obs, radius_m, scenario, well))
        rows.append(fit_lagging_leaky(time_s, obs, radius_m, scenario, well))
    return pd.DataFrame(rows)


def summarize_log_factors(values: pd.Series) -> tuple[float, float, float, float]:
    arr = values.to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    cov = float(math.sqrt(math.exp(sigma * sigma) - 1.0))
    median_factor = float(math.exp(np.median(arr)))
    return mu, sigma, cov, median_factor


def model_factor_cov_table(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario_class, pathway), group in fits.groupby(["scenario_class", "pathway"], sort=False):
        mu_t, sig_t, cov_t, med_t = summarize_log_factors(group["lnM_T"])
        mu_s, sig_s, cov_s, med_s = summarize_log_factors(group["lnM_S"])
        mu_cap, sig_cap, cov_cap, med_cap = summarize_log_factors(group["lnM_capacity"])
        mu_rt, sig_rt, cov_rt, med_rt = summarize_log_factors(group["lnM_response_time"])
        rows.append(
            {
                "scenario_class": scenario_class,
                "pathway": pathway,
                "response_count": int(group.shape[0]),
                "mu_lnM_T": mu_t,
                "sigma_lnM_T": sig_t,
                "cov_M_T": cov_t,
                "median_M_T": med_t,
                "mu_lnM_S": mu_s,
                "sigma_lnM_S": sig_s,
                "cov_M_S": cov_s,
                "median_M_S": med_s,
                "mu_lnM_capacity": mu_cap,
                "sigma_lnM_capacity": sig_cap,
                "cov_M_capacity": cov_cap,
                "median_M_capacity": med_cap,
                "mu_lnM_response_time": mu_rt,
                "sigma_lnM_response_time": sig_rt,
                "cov_M_response_time": cov_rt,
                "median_M_response_time": med_rt,
                "median_response_cov": float(group["cov_response_model_factor"].median()),
                "median_sd_eta": float(group["sd_eta"].median()),
                "boundary_hit_fraction": float(group["boundary_hit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def decision_summary(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario_class, pathway), group in fits.groupby(["scenario_class", "pathway"], sort=False):
        rows.append(
            {
                "scenario_class": scenario_class,
                "pathway": pathway,
                "median_capacity_factor": float(np.exp(np.median(group["lnM_capacity"]))),
                "p05_capacity_factor": float(np.exp(np.quantile(group["lnM_capacity"], 0.05))),
                "p95_capacity_factor": float(np.exp(np.quantile(group["lnM_capacity"], 0.95))),
                "median_response_time_factor": float(np.exp(np.median(group["lnM_response_time"]))),
                "p05_response_time_factor": float(np.exp(np.quantile(group["lnM_response_time"], 0.05))),
                "p95_response_time_factor": float(np.exp(np.quantile(group["lnM_response_time"], 0.95))),
            }
        )
    return pd.DataFrame(rows)


def field_applicability_table(cov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mappings = [
        (
            "Massachusetts",
            "heterogeneous leakage boundary",
            "fractured bedrock response with spatial variability and leakage/boundary alternatives; use as conservative conditional prior",
        ),
        (
            "Massachusetts",
            "heterogeneous K",
            "field-only lower-complexity reference when leakage support is weak in a specific well",
        ),
        (
            "Lovelock Valley",
            "vertical leakage",
            "basin-fill pumping/recovery response with vertical exchange interpretation; use as conditional prior if leakage diagnostics dominate",
        ),
        (
            "Lovelock Valley",
            "constant head boundary",
            "basin-fill late-time flattening or boundary-support alternative; use as conditional prior if boundary diagnostics dominate",
        ),
    ]
    for case, scenario_class, basis in mappings:
        subset = cov[cov["scenario_class"] == scenario_class]
        for _, row in subset.iterrows():
            rows.append(
                {
                    "field_case": case,
                    "benchmark_class": scenario_class,
                    "pathway": row["pathway"],
                    "conditional_cov_M_T": row["cov_M_T"],
                    "conditional_cov_M_S": row["cov_M_S"],
                    "conditional_cov_response_time": row["cov_M_response_time"],
                    "median_M_T": row["median_M_T"],
                    "median_M_S": row["median_M_S"],
                    "applicability_basis": basis,
                    "claim_boundary": "conditional benchmark-derived transformation prior; not a universal design COV",
                }
            )
    return pd.DataFrame(rows)


def export_figure(fig: plt.Figure, stem: str) -> None:
    for base in [PROJECT, FIG_DIR]:
        for suffix in ["pdf", "png", "svg"]:
            kwargs = {"dpi": 450} if suffix == "png" else {}
            fig.savefig(base / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(0.0, 1.04, f"({label}) {title}", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")


def plot_design(drawdown: pd.DataFrame, cases: pd.DataFrame, k_example: np.ndarray) -> None:
    lock_style()
    fig = plt.figure(figsize=(7.3, 2.65))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 1.08, 1.22],
        left=0.07,
        right=0.98,
        bottom=0.22,
        top=0.82,
        wspace=0.55,
    )
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    ax = axes[0]
    im = ax.imshow(
        k_example,
        extent=[-NCOL * DELR_M / 2, NCOL * DELR_M / 2, -NROW * DELC_M / 2, NROW * DELC_M / 2],
        origin="lower",
        cmap=cmc.batlow,
    )
    ax.scatter([0], [0], marker="s", s=34, color=NEUTRAL_DARK, label="pumping")
    label_offsets = {
        "R20E": (-18.0, -36.0),
        "R50E": (4.0, 28.0),
        "R100E": (10.0, -30.0),
        "R100N": (8.0, 8.0),
    }
    for well, radius, angle in OBSERVATION_WELLS:
        x = radius * math.cos(math.radians(angle))
        y = radius * math.sin(math.radians(angle))
        ax.scatter([x], [y], marker="o", s=24, facecolor="white", edgecolor=NEUTRAL_DARK, linewidth=0.8)
        dx, dy = label_offsets[well]
        ax.text(x + dx, y + dy, well, fontsize=5.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    panel_label(ax, "a", "MODFLOW 6 pumping-test domain")
    cax = fig.add_axes([0.265, 0.28, 0.012, 0.46])
    fig.colorbar(im, cax=cax, label="K (m/d)")

    ax = axes[1]
    counts = cases["scenario_class"].value_counts().reindex(CLASS_COLORS.keys()).dropna()
    short_class = {
        "homogeneous confined": "homogeneous",
        "vertical leakage": "leakage",
        "constant head boundary": "constant head",
        "heterogeneous K": "heterogeneous K",
        "heterogeneous leakage boundary": "combined",
    }
    ypos = np.arange(len(counts))
    ax.barh(ypos, counts.to_numpy(), color=[CLASS_COLORS[idx] for idx in counts.index], edgecolor="none")
    ax.set_yticks(ypos)
    ax.set_yticklabels([short_class[idx] for idx in counts.index])
    ax.invert_yaxis()
    ax.set_xlabel("scenarios")
    panel_label(ax, "b", "controlled classes")

    ax = axes[2]
    sample = drawdown[(drawdown["scenario_id"] == "mf6_combined_00") & (drawdown["well"].isin(["R20E", "R50E", "R100E"]))]
    for well, group in sample.groupby("well", sort=False):
        ax.plot(group["time_h"], group["drawdown_m"], marker="o", markersize=2.6, linewidth=1.0, label=well)
    ax.set_xscale("log")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("drawdown (m)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.03, 0.98))
    panel_label(ax, "c", "numerical drawdown response")
    export_figure(fig, "fig13_mf6_benchmark_design")


def plot_model_factor_cov(cov: pd.DataFrame, decision: pd.DataFrame) -> None:
    lock_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.0), constrained_layout=True)
    class_order = list(CLASS_COLORS.keys())
    x = np.arange(len(class_order))
    width = 0.18

    for ax, metric, ylabel, label, title in [
        (axes[0, 0], "median_response_cov", "response model-factor COV", "a", "response-level factor"),
        (axes[0, 1], "cov_M_T", "T model-factor COV", "b", "transmissivity factor"),
        (axes[1, 0], "cov_M_S", "S model-factor COV", "c", "storage factor"),
    ]:
        for idx, pathway in enumerate(PATHWAYS):
            sub = cov[cov["pathway"] == pathway].set_index("scenario_class").reindex(class_order)
            ax.bar(x + (idx - 1.5) * width, sub[metric].to_numpy(float), width=width, color=PATHWAY_COLORS[pathway], label=pathway)
        ax.set_xticks(x)
        ax.set_xticklabels(class_order, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        panel_label(ax, label, title)
        ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.7)

    ax = axes[1, 1]
    for idx, pathway in enumerate(PATHWAYS):
        sub = decision[decision["pathway"] == pathway].set_index("scenario_class").reindex(class_order)
        med = sub["median_response_time_factor"].to_numpy(float)
        p05 = sub["p05_response_time_factor"].to_numpy(float)
        p95 = sub["p95_response_time_factor"].to_numpy(float)
        ax.errorbar(
            x + (idx - 1.5) * width,
            med,
            yerr=np.vstack([med - p05, p95 - med]),
            fmt="o",
            markersize=3.2,
            linewidth=0.9,
            capsize=2.2,
            color=PATHWAY_COLORS[pathway],
            label=pathway,
        )
    ax.axhline(1.0, color=NEUTRAL_MID, linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(class_order, rotation=28, ha="right")
    ax.set_ylabel("response-time factor")
    panel_label(ax, "d", "decision-level inheritance")
    ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.52, 1.04))
    export_figure(fig, "legacy_mf6_model_factor_cov_helper")


def write_outputs() -> dict:
    ensure_dirs()
    scenarios = build_scenarios()
    drawdown_frames: list[pd.DataFrame] = []
    case_rows: list[dict] = []
    k_examples: dict[str, np.ndarray] = {}
    for scenario in scenarios:
        drawdown, meta, k_field = run_mf6_scenario(scenario)
        drawdown_frames.append(drawdown)
        case_rows.append(meta)
        k_examples[scenario.scenario_id] = k_field

    drawdown = pd.concat(drawdown_frames, ignore_index=True)
    cases = pd.DataFrame(case_rows)
    fits = fit_all(drawdown, cases)
    cov = model_factor_cov_table(fits)
    decision = decision_summary(fits)
    field_app = field_applicability_table(cov)

    drawdown.to_csv(TABLE_DIR / "mf6_benchmark_drawdown.csv", index=False)
    cases.to_csv(TABLE_DIR / "mf6_benchmark_cases.csv", index=False)
    fits.to_csv(TABLE_DIR / "mf6_pathway_fit_results.csv", index=False)
    cov.to_csv(TABLE_DIR / "mf6_model_factor_cov.csv", index=False)
    decision.to_csv(TABLE_DIR / "mf6_decision_factor_summary.csv", index=False)
    field_app.to_csv(TABLE_DIR / "mf6_field_applicable_model_factor.csv", index=False)

    plot_design(drawdown, cases, k_examples["mf6_combined_00"])
    plot_model_factor_cov(cov, decision)

    summary = {
        "mf6_executable": MF6_EXE,
        "flopy_version": flopy.__version__,
        "scenario_count": int(cases.shape[0]),
        "scenario_classes": sorted(cases["scenario_class"].unique().tolist()),
        "observation_wells": len(OBSERVATION_WELLS),
        "response_records": int(fits.shape[0]),
        "pathways": PATHWAYS,
        "cov_M_T_range": [float(cov["cov_M_T"].min()), float(cov["cov_M_T"].max())],
        "cov_M_S_range": [float(cov["cov_M_S"].min()), float(cov["cov_M_S"].max())],
        "response_cov_range": [float(cov["median_response_cov"].min()), float(cov["median_response_cov"].max())],
        "response_time_factor_median_range": [
            float(decision["median_response_time_factor"].min()),
            float(decision["median_response_time_factor"].max()),
        ],
        "claim_boundary": "conditional MODFLOW 6 benchmark-derived transformation model factors; not universal design COV values",
    }
    with (OUT_DIR / "mf6_transformation_benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(write_outputs(), indent=2))

