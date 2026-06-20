from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cmcrameri.cm as cmc
import flopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf6_transformation_benchmark as base  # noqa: E402


PROJECT = HERE.parents[0]
TABLE_DIR = PROJECT / "tables"
FIG_DIR = PROJECT / "figures"
OUT_DIR = PROJECT / "outputs"
WORKSPACE = OUT_DIR / "mf6_strict_benchmark"

SECONDS_PER_DAY = 86400.0
Q_M3S = base.Q_M3S
Q_M3D = base.Q_M3D
T_REF_M2S = base.T_REF_M2S
T_REF_M2D = base.T_REF_M2D
S_REF = base.S_REF
AQ_THICKNESS_M = base.AQ_THICKNESS_M
K_REF_MD = base.K_REF_MD
TOP_M = base.TOP_M
BOTM_M = base.BOTM_M
OFFSET_M = base.OFFSET_M
OBS_BASE = base.OBSERVATION_WELLS
PATHWAYS = base.PATHWAYS

MF6_EXE = shutil.which("mf6") or base.MF6_EXE

CLASS_ORDER = [
    "homogeneous confined",
    "vertical leakage",
    "constant head boundary",
    "heterogeneous K",
    "heterogeneous leakage",
    "heterogeneous boundary",
    "heterogeneous leakage boundary",
]

SHORT_CLASS = {
    "homogeneous confined": "homogeneous",
    "vertical leakage": "leakage",
    "constant head boundary": "boundary",
    "heterogeneous K": "heterogeneous K",
    "heterogeneous leakage": "heterogeneous\nleakage",
    "heterogeneous boundary": "heterogeneous\nboundary",
    "heterogeneous leakage boundary": "combined",
}


def cmc_color(name: str, sample: float) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in getattr(cmc, name)(sample))


CLASS_COLORS = {cls: cmc_color("navia", 0.12 + 0.78 * idx / (len(CLASS_ORDER) - 1)) for idx, cls in enumerate(CLASS_ORDER)}
PATHWAY_COLORS = base.PATHWAY_COLORS
NEUTRAL_DARK = cmc_color("grayC", 0.14)
NEUTRAL_MID = cmc_color("grayC", 0.48)
NEUTRAL_LIGHT = cmc_color("grayC", 0.90)


@dataclass(frozen=True)
class StrictScenario:
    scenario_id: str
    scenario_class: str
    seed: int
    sigma_ln_k: float
    corr_len_m: float
    leakage_b_m: float | None
    constant_head_edges: bool
    dx_m: float = 5.0
    domain_m: float = 605.0
    time_count: int = 28
    obs_rotation_deg: float = 0.0
    ensemble_role: str = "ensemble"
    description: str = ""


def lock_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.4,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
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
    WORKSPACE.mkdir(parents=True, exist_ok=True)


def grid_shape(domain_m: float, dx_m: float) -> tuple[int, int]:
    n = int(round(domain_m / dx_m)) + 1
    if n % 2 == 0:
        n += 1
    return n, n


def times_for_count(time_count: int) -> np.ndarray:
    return np.geomspace(90.0, 2.0 * SECONDS_PER_DAY, int(time_count))


def rotate_observations(rotation_deg: float) -> list[tuple[str, float, float]]:
    out = []
    for name, radius, angle in OBS_BASE:
        out.append((name, radius, angle + rotation_deg))
    return out


def obs_cell(radius_m: float, angle_deg: float, nrow: int, ncol: int, dx_m: float) -> tuple[tuple[int, int, int], float, float, float]:
    angle = math.radians(angle_deg)
    x = radius_m * math.cos(angle)
    y = radius_m * math.sin(angle)
    row_offset = int(round(y / dx_m))
    col_offset = int(round(x / dx_m))
    row = nrow // 2 - row_offset
    col = ncol // 2 + col_offset
    actual_x = (col - ncol // 2) * dx_m
    actual_y = (nrow // 2 - row) * dx_m
    actual_radius = math.hypot(actual_x, actual_y)
    return (0, row, col), actual_radius, actual_x, actual_y


def cell_coordinates(nrow: int, ncol: int, dx_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(nrow)
    cols = np.arange(ncol)
    x = (cols - ncol // 2) * dx_m
    y = (nrow // 2 - rows) * dx_m
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx * xx + yy * yy)
    return xx, yy, rr


def make_k_field(scenario: StrictScenario, nrow: int, ncol: int) -> np.ndarray:
    if scenario.sigma_ln_k <= 0.0:
        return np.full((nrow, ncol), K_REF_MD, dtype=float)
    rng = np.random.default_rng(scenario.seed)
    raw = rng.normal(0.0, 1.0, size=(nrow, ncol))
    sigma_cells = max(scenario.corr_len_m / scenario.dx_m, 0.5)
    smooth = gaussian_filter(raw, sigma=sigma_cells, mode="reflect")
    smooth = (smooth - np.mean(smooth)) / max(float(np.std(smooth)), 1.0e-12)
    return np.exp(math.log(K_REF_MD) + scenario.sigma_ln_k * smooth)


DEPRECATED_FIXED_SUPPORT_RADIUS_M = 130.0


def radial_annular_effective_t(k_field_md: np.ndarray, dx_m: float, radius_limit_m: float) -> float:
    nrow, ncol = k_field_md.shape
    _, _, radius = cell_coordinates(nrow, ncol, dx_m)
    dr = max(dx_m, 2.5)
    annuli = np.arange(dx_m, radius_limit_m + dr, dr)
    resistances = []
    for r0, r1 in zip(annuli[:-1], annuli[1:]):
        mask = (radius >= r0) & (radius < r1)
        if not np.any(mask):
            continue
        k_vals = np.maximum(k_field_md[mask], 1.0e-12)
        k_harm = 1.0 / np.mean(1.0 / k_vals)
        resistances.append(math.log(max(r1, dx_m) / max(r0, dx_m * 0.5)) / k_harm)
    if not resistances:
        return T_REF_M2S
    k_eff = math.log(radius_limit_m / max(dx_m * 0.5, 1.0)) / sum(resistances)
    return float(k_eff * AQ_THICKNESS_M / SECONDS_PER_DAY)


def weighted_geomean(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.maximum(np.asarray(values, dtype=float), 1.0e-30)
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    valid = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return float(np.exp(np.mean(np.log(vals[np.isfinite(vals)]))))
    return float(np.exp(np.sum(w[valid] * np.log(vals[valid])) / np.sum(w[valid])))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    valid = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return float(np.nanmean(vals))
    return float(np.sum(w[valid] * vals[valid]) / np.sum(w[valid]))


def effective_support_controls(k_field_md: np.ndarray, scenario: StrictScenario) -> dict:
    t_global = float(np.exp(np.mean(np.log(np.maximum(k_field_md * AQ_THICKNESS_M / SECONDS_PER_DAY, 1.0e-30)))))
    hydraulic_diffusivity = t_global / max(S_REF, 1.0e-30)
    diffusion_radius = math.sqrt(max(2.25 * hydraulic_diffusivity * 2.0 * SECONDS_PER_DAY, scenario.dx_m**2))
    observation_radius = max(radius for _, radius, _ in rotate_observations(scenario.obs_rotation_deg))
    leakage_limit = float(scenario.leakage_b_m) if scenario.leakage_b_m is not None else float("inf")
    boundary_limit = max(0.5 * scenario.domain_m - scenario.dx_m, scenario.dx_m)
    limited_radius = min(diffusion_radius, leakage_limit, boundary_limit)
    lower_radius = max(3.0 * scenario.dx_m, min(observation_radius, boundary_limit) * 0.50)
    support_radius = min(max(limited_radius, lower_radius), boundary_limit)
    return {
        "support_effective_radius_m": support_radius,
        "support_diffusion_radius_m": diffusion_radius,
        "support_observation_max_radius_m": observation_radius,
        "support_leakage_limit_m": leakage_limit,
        "support_boundary_limit_m": boundary_limit,
        "support_hydraulic_diffusivity_m2_s": hydraulic_diffusivity,
    }


def support_targets(k_field_md: np.ndarray, scenario: StrictScenario) -> dict:
    nrow, ncol = k_field_md.shape
    _, _, radius = cell_coordinates(nrow, ncol, scenario.dx_m)
    controls = effective_support_controls(k_field_md, scenario)
    if scenario.sigma_ln_k <= 0.0:
        return {
            "truth_T_ref_m2_s": T_REF_M2S,
            "truth_T_geomean_m2_s": T_REF_M2S,
            "truth_T_arithmetic_m2_s": T_REF_M2S,
            "truth_T_annular_m2_s": T_REF_M2S,
            "support_definition": "assigned homogeneous transmissivity; effective support radius reported for hydraulic context",
            "support_target_spread": 1.0,
            "support_effective_radius_m": float(controls["support_effective_radius_m"]),
            "support_diffusion_radius_m": float(controls["support_diffusion_radius_m"]),
            "support_observation_max_radius_m": float(controls["support_observation_max_radius_m"]),
            "support_leakage_limit_m": None if not np.isfinite(float(controls["support_leakage_limit_m"])) else float(controls["support_leakage_limit_m"]),
            "support_boundary_limit_m": float(controls["support_boundary_limit_m"]),
            "support_hydraulic_diffusivity_m2_s": float(controls["support_hydraulic_diffusivity_m2_s"]),
            "support_weighting": "assigned true value; no heterogeneous averaging",
            "fixed_radius_130m_T_geomean_m2_s": T_REF_M2S,
            "fixed_radius_130m_T_ratio_to_effective": 1.0,
        }
    diffusion_radius = max(float(controls["support_diffusion_radius_m"]), scenario.dx_m)
    weights = np.exp(-((radius / diffusion_radius) ** 2))
    if scenario.leakage_b_m is not None:
        weights *= np.exp(-radius / max(float(scenario.leakage_b_m), scenario.dx_m))
    weights = np.where(radius <= float(controls["support_effective_radius_m"]), weights, 0.0)
    t_field = np.maximum(k_field_md * AQ_THICKNESS_M / SECONDS_PER_DAY, 1.0e-30)
    t_geo = weighted_geomean(t_field, weights)
    t_arith = weighted_mean(t_field, weights)
    t_ann = radial_annular_effective_t(k_field_md, scenario.dx_m, radius_limit_m=max(float(controls["support_effective_radius_m"]), 2.0 * scenario.dx_m))
    fixed_mask = radius <= min(DEPRECATED_FIXED_SUPPORT_RADIUS_M, float(controls["support_boundary_limit_m"]))
    fixed_vals = np.maximum(t_field[fixed_mask], 1.0e-30)
    fixed_geo = float(np.exp(np.mean(np.log(fixed_vals)))) if fixed_vals.size else t_geo
    spread = float(max(t_geo, t_arith, t_ann, fixed_geo) / max(min(t_geo, t_arith, t_ann, fixed_geo), 1.0e-30))
    return {
        "truth_T_ref_m2_s": t_geo,
        "truth_T_geomean_m2_s": t_geo,
        "truth_T_arithmetic_m2_s": t_arith,
        "truth_T_annular_m2_s": t_ann,
        "support_definition": "scenario-specific diffusion/leakage/boundary weighted effective support; fixed-radius 130 m sensitivity retained only for comparison",
        "support_target_spread": spread,
        "support_effective_radius_m": float(controls["support_effective_radius_m"]),
        "support_diffusion_radius_m": float(controls["support_diffusion_radius_m"]),
        "support_observation_max_radius_m": float(controls["support_observation_max_radius_m"]),
        "support_leakage_limit_m": None if not np.isfinite(float(controls["support_leakage_limit_m"])) else float(controls["support_leakage_limit_m"]),
        "support_boundary_limit_m": float(controls["support_boundary_limit_m"]),
        "support_hydraulic_diffusivity_m2_s": float(controls["support_hydraulic_diffusivity_m2_s"]),
        "support_weighting": "Gaussian hydraulic-diffusion weights clipped by effective radius, leakage attenuation, and boundary/domain limit",
        "fixed_radius_130m_T_geomean_m2_s": fixed_geo,
        "fixed_radius_130m_T_ratio_to_effective": float(fixed_geo / max(t_geo, 1.0e-30)),
    }


def run_mf6_scenario(scenario: StrictScenario) -> tuple[pd.DataFrame, dict, np.ndarray]:
    nrow, ncol = grid_shape(scenario.domain_m, scenario.dx_m)
    times_s = times_for_count(scenario.time_count)
    times_d = times_s / SECONDS_PER_DAY
    dts_d = np.diff(np.r_[0.0, times_d])
    k_field = make_k_field(scenario, nrow, ncol)
    targets = support_targets(k_field, scenario)
    sim_ws = WORKSPACE / scenario.scenario_id
    if sim_ws.exists():
        shutil.rmtree(sim_ws)
    sim_ws.mkdir(parents=True)

    sim = flopy.mf6.MFSimulation(sim_name=scenario.scenario_id, exe_name=MF6_EXE, sim_ws=str(sim_ws), version="mf6")
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=len(dts_d), perioddata=[(float(dt), 1, 1.0) for dt in dts_d])
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1.0e-8, inner_dvclose=1.0e-8)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=nrow, ncol=ncol, delr=scenario.dx_m, delc=scenario.dx_m, top=TOP_M, botm=BOTM_M)
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k_field, k33=k_field, save_specific_discharge=True)
    flopy.mf6.ModflowGwfsto(
        gwf,
        storagecoefficient=True,
        iconvert=0,
        ss=S_REF,
        steady_state={0: False},
        transient={idx: True for idx in range(len(dts_d))},
    )
    center = (0, nrow // 2, ncol // 2)
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data={idx: [(center, -Q_M3D)] for idx in range(len(dts_d))}, save_flows=True)
    if scenario.constant_head_edges:
        chd = []
        for row in range(nrow):
            chd.append(((0, row, 0), 0.0))
            chd.append(((0, row, ncol - 1), 0.0))
        for col in range(1, ncol - 1):
            chd.append(((0, 0, col), 0.0))
            chd.append(((0, nrow - 1, col), 0.0))
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, save_flows=True)
    if scenario.leakage_b_m is not None:
        conductance = (scenario.dx_m * scenario.dx_m) * (T_REF_M2D / (scenario.leakage_b_m * scenario.leakage_b_m))
        ghb = [((0, row, col), 0.0, conductance) for row in range(nrow) for col in range(ncol)]
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
    actual_radii = []
    for well, radius_m, angle_deg in rotate_observations(scenario.obs_rotation_deg):
        cell, actual_radius, actual_x, actual_y = obs_cell(radius_m, angle_deg, nrow, ncol, scenario.dx_m)
        actual_radii.append(actual_radius)
        _, row, col = cell
        drawdown = np.maximum(-heads[:, row, col], 0.0)
        for time_s, dd in zip(times_s, drawdown):
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_class": scenario.scenario_class,
                    "ensemble_role": scenario.ensemble_role,
                    "well": well,
                    "nominal_radius_m": radius_m,
                    "radius_m": actual_radius,
                    "x_m": actual_x,
                    "y_m": actual_y,
                    "time_s": float(time_s),
                    "time_h": float(time_s / 3600.0),
                    "drawdown_m": float(dd),
                }
            )
    meta = {
        "scenario_id": scenario.scenario_id,
        "scenario_class": scenario.scenario_class,
        "ensemble_role": scenario.ensemble_role,
        "seed": scenario.seed,
        "description": scenario.description,
        "sigma_ln_k": scenario.sigma_ln_k,
        "corr_len_m": scenario.corr_len_m,
        "leakage_B_m": scenario.leakage_b_m,
        "constant_head_edges": scenario.constant_head_edges,
        "dx_m": scenario.dx_m,
        "domain_m": scenario.domain_m,
        "time_count": scenario.time_count,
        "obs_rotation_deg": scenario.obs_rotation_deg,
        "truth_S_ref": S_REF,
        "K_geomean_m_d": float(np.exp(np.mean(np.log(k_field)))),
        "K_arithmetic_m_d": float(np.mean(k_field)),
        "K_cv": float(np.std(k_field) / max(np.mean(k_field), 1.0e-30)),
        "observation_count": int(len(rows)),
        "max_drawdown_m": float(max(row["drawdown_m"] for row in rows)),
        "min_actual_radius_m": float(np.min(actual_radii)),
        "max_actual_radius_m": float(np.max(actual_radii)),
        **targets,
    }
    return pd.DataFrame(rows), meta, k_field


def scenario_dict(meta: dict) -> dict:
    return {
        "scenario_id": meta["scenario_id"],
        "scenario_class": meta["scenario_class"],
        "truth_T_ref_m2_s": meta["truth_T_ref_m2_s"],
        "truth_S_ref": meta["truth_S_ref"],
        "leakage_B_m": meta.get("leakage_B_m"),
    }


def fit_all(drawdown: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    lookup = cases.set_index("scenario_id").to_dict(orient="index")
    rows = []
    for (scenario_id, well), group in drawdown.groupby(["scenario_id", "well"], sort=False):
        meta = {"scenario_id": scenario_id, **lookup[scenario_id]}
        sdict = scenario_dict(meta)
        group = group.sort_values("time_s")
        time_s = group["time_s"].to_numpy(float)
        obs = group["drawdown_m"].to_numpy(float)
        radius_m = float(group["radius_m"].iloc[0])
        for fit_func in [base.fit_theis, base.fit_hantush, base.fit_lagging, base.fit_lagging_leaky]:
            row = fit_func(time_s, obs, radius_m, sdict, well)
            row.update(
                {
                    "ensemble_role": meta["ensemble_role"],
                    "dx_m": meta["dx_m"],
                    "domain_m": meta["domain_m"],
                    "time_count": meta["time_count"],
                    "sigma_ln_k": meta["sigma_ln_k"],
                    "corr_len_m": meta["corr_len_m"],
                    "support_target_spread": meta["support_target_spread"],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_expanded_ensemble() -> list[StrictScenario]:
    scenarios: list[StrictScenario] = []
    rotations = [0.0, 17.0, 33.0, 49.0, 66.0, 83.0, 101.0, 119.0, 137.0, 151.0]
    for idx, rot in enumerate(rotations):
        scenarios.append(StrictScenario(f"strict_homogeneous_{idx:02d}", "homogeneous confined", 100 + idx, 0.0, 0.0, None, False, obs_rotation_deg=rot, description="Homogeneous baseline orientation/support realization."))
    leakage_values = [35.0, 45.0, 60.0, 80.0, 110.0, 150.0, 35.0, 60.0, 110.0, 150.0]
    for idx, b_m in enumerate(leakage_values):
        scenarios.append(StrictScenario(f"strict_leakage_{idx:02d}", "vertical leakage", 200 + idx, 0.0, 0.0, b_m, False, obs_rotation_deg=rotations[idx], description="Homogeneous leaky aquifer realization."))
    boundary_domains = [405.0, 505.0, 605.0, 805.0, 1005.0, 405.0, 505.0, 605.0, 805.0, 1005.0]
    for idx, domain_m in enumerate(boundary_domains):
        scenarios.append(StrictScenario(f"strict_boundary_{idx:02d}", "constant head boundary", 300 + idx, 0.0, 0.0, None, True, domain_m=domain_m, obs_rotation_deg=rotations[idx], description="Finite first type boundary realization."))
    hetero_design = [(0.3, 10.0), (0.3, 30.0), (0.3, 80.0), (0.7, 10.0), (0.7, 30.0), (0.7, 80.0), (1.1, 10.0), (1.1, 30.0), (1.1, 80.0), (0.7, 50.0)]
    for idx, (sigma, corr) in enumerate(hetero_design):
        scenarios.append(StrictScenario(f"strict_heterogeneous_{idx:02d}", "heterogeneous K", 400 + idx, sigma, corr, None, False, obs_rotation_deg=rotations[idx], description="Lognormal heterogeneous K realization."))
    for idx, (sigma, corr) in enumerate(hetero_design):
        b_m = [45.0, 80.0, 140.0, 45.0, 80.0, 140.0, 45.0, 80.0, 140.0, 100.0][idx]
        scenarios.append(StrictScenario(f"strict_heterogeneous_leakage_{idx:02d}", "heterogeneous leakage", 500 + idx, sigma, corr, b_m, False, obs_rotation_deg=rotations[idx], description="Heterogeneous K with distributed leakage realization."))
    for idx, (sigma, corr) in enumerate(hetero_design):
        domain_m = [405.0, 605.0, 1005.0, 405.0, 605.0, 1005.0, 405.0, 605.0, 1005.0, 805.0][idx]
        scenarios.append(StrictScenario(f"strict_heterogeneous_boundary_{idx:02d}", "heterogeneous boundary", 600 + idx, sigma, corr, None, True, domain_m=domain_m, obs_rotation_deg=rotations[idx], description="Heterogeneous K with first type boundary realization."))
    for idx, (sigma, corr) in enumerate(hetero_design):
        b_m = [45.0, 80.0, 140.0, 45.0, 80.0, 140.0, 45.0, 80.0, 140.0, 100.0][idx]
        domain_m = [405.0, 605.0, 1005.0, 405.0, 605.0, 1005.0, 405.0, 605.0, 1005.0, 805.0][idx]
        scenarios.append(StrictScenario(f"strict_combined_{idx:02d}", "heterogeneous leakage boundary", 700 + idx, sigma, corr, b_m, True, domain_m=domain_m, obs_rotation_deg=rotations[idx], description="Heterogeneous K with leakage and first type boundary realization."))
    return scenarios


def build_sensitivity_cases() -> list[StrictScenario]:
    cases: list[StrictScenario] = []
    idx = 0
    for dx in [10.0, 5.0, 2.5]:
        cases.append(StrictScenario(f"sens_grid_homogeneous_{idx:02d}", "homogeneous confined", 800 + idx, 0.0, 0.0, None, False, dx_m=dx, domain_m=605.0, ensemble_role="grid_sensitivity", description="Homogeneous grid refinement."))
        idx += 1
        cases.append(StrictScenario(f"sens_grid_leakage_{idx:02d}", "vertical leakage", 800 + idx, 0.0, 0.0, 80.0, False, dx_m=dx, domain_m=605.0, ensemble_role="grid_sensitivity", description="Leaky grid refinement."))
        idx += 1
    for domain in [405.0, 605.0, 1005.0]:
        cases.append(StrictScenario(f"sens_domain_boundary_{idx:02d}", "constant head boundary", 800 + idx, 0.0, 0.0, None, True, dx_m=5.0, domain_m=domain, ensemble_role="domain_sensitivity", description="Boundary domain-size sensitivity."))
        idx += 1
    for time_count in [20, 28, 48]:
        cases.append(StrictScenario(f"sens_time_homogeneous_{idx:02d}", "homogeneous confined", 800 + idx, 0.0, 0.0, None, False, dx_m=5.0, time_count=time_count, ensemble_role="time_sensitivity", description="Homogeneous output-time sensitivity."))
        idx += 1
    return cases


def analytical_baseline_verification(drawdown: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    lookup = cases.set_index("scenario_id").to_dict(orient="index")
    rows = []
    for (scenario_id, well), group in drawdown.groupby(["scenario_id", "well"], sort=False):
        meta = lookup[scenario_id]
        scenario_class = meta["scenario_class"]
        if scenario_class not in {"homogeneous confined", "vertical leakage", "constant head boundary"}:
            continue
        group = group.sort_values("time_s")
        time_s = group["time_s"].to_numpy(float)
        obs = group["drawdown_m"].to_numpy(float)
        radius_m = float(group["radius_m"].iloc[0])
        if scenario_class == "homogeneous confined":
            pred = base.theis_drawdown(time_s, Q_M3S, T_REF_M2S, S_REF, radius_m)
            reference = "Theis analytical"
            exactness = "direct baseline"
        elif scenario_class == "vertical leakage":
            pred = base.leaky_drawdown(time_s, Q_M3S, T_REF_M2S, S_REF, float(meta["leakage_B_m"]), radius_m)
            reference = "Hantush-Jacob analytical"
            exactness = "direct baseline"
        else:
            pred = base.theis_drawdown(time_s, Q_M3S, T_REF_M2S, S_REF, radius_m)
            reference = "Theis no-boundary reference"
            exactness = "diagnostic only; square first-type boundary is not a single image-well analytical solution"
        eta = base.log_response_residual(obs, pred)
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_class": scenario_class,
                "ensemble_role": meta["ensemble_role"],
                "well": well,
                "reference_solution": reference,
                "exactness": exactness,
                "radius_m": radius_m,
                "bias_eta": float(np.mean(eta)),
                "sd_eta": float(np.std(eta, ddof=1)),
                "rmse_m": float(np.sqrt(np.mean((obs - pred) ** 2))),
                "max_abs_error_m": float(np.max(np.abs(obs - pred))),
                "median_abs_error_m": float(np.median(np.abs(obs - pred))),
                "pass_flag": bool(np.std(eta, ddof=1) < 0.08) if scenario_class != "constant head boundary" else None,
            }
        )
    return pd.DataFrame(rows)


def lognormal_cov(values: np.ndarray) -> tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"),) * 5
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    cov = float(math.sqrt(math.exp(sigma * sigma) - 1.0))
    median = float(math.exp(np.median(arr)))
    bias = float(math.exp(mu))
    return mu, sigma, cov, median, bias


def bootstrap_ci(group: pd.DataFrame, column: str, n_boot: int = 300) -> tuple[float, float]:
    scenario_ids = np.array(sorted(group["scenario_id"].unique()))
    if scenario_ids.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(20260608)
    values = []
    by_scenario = {sid: group[group["scenario_id"] == sid] for sid in scenario_ids}
    for _ in range(n_boot):
        sampled = rng.choice(scenario_ids, size=scenario_ids.size, replace=True)
        sampled_group = pd.concat([by_scenario[sid] for sid in sampled], ignore_index=True)
        values.append(lognormal_cov(sampled_group[column].to_numpy())[2])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def model_factor_cov_ci(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ensemble = fits[fits["ensemble_role"] == "ensemble"].copy()
    for (scenario_class, pathway), group in ensemble.groupby(["scenario_class", "pathway"], sort=False):
        mu_t, sig_t, cov_t, med_t, bias_t = lognormal_cov(group["lnM_T"].to_numpy())
        mu_s, sig_s, cov_s, med_s, bias_s = lognormal_cov(group["lnM_S"].to_numpy())
        mu_rt, sig_rt, cov_rt, med_rt, bias_rt = lognormal_cov(group["lnM_response_time"].to_numpy())
        ci_t = bootstrap_ci(group, "lnM_T")
        ci_s = bootstrap_ci(group, "lnM_S")
        ci_rt = bootstrap_ci(group, "lnM_response_time")
        rows.append(
            {
                "scenario_class": scenario_class,
                "pathway": pathway,
                "scenario_count": int(group["scenario_id"].nunique()),
                "fit_count": int(group.shape[0]),
                "mu_lnM_T": mu_t,
                "sigma_lnM_T": sig_t,
                "cov_M_T": cov_t,
                "cov_M_T_ci_low": ci_t[0],
                "cov_M_T_ci_high": ci_t[1],
                "median_M_T": med_t,
                "bias_factor_T": bias_t,
                "mu_lnM_S": mu_s,
                "sigma_lnM_S": sig_s,
                "cov_M_S": cov_s,
                "cov_M_S_ci_low": ci_s[0],
                "cov_M_S_ci_high": ci_s[1],
                "median_M_S": med_s,
                "bias_factor_S": bias_s,
                "cov_M_response_time": cov_rt,
                "cov_M_response_time_ci_low": ci_rt[0],
                "cov_M_response_time_ci_high": ci_rt[1],
                "median_M_response_time": med_rt,
                "bias_factor_response_time": bias_rt,
                "median_response_cov": float(group["cov_response_model_factor"].median()),
                "support_target_spread_median": float(group["support_target_spread"].median()),
                "boundary_hit_fraction": float(group["boundary_hit"].mean()),
            }
        )
    return pd.DataFrame(rows)


def convergence_table(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ensemble = fits[fits["ensemble_role"] == "ensemble"].copy()
    for (scenario_class, pathway), group in ensemble.groupby(["scenario_class", "pathway"], sort=False):
        scenario_ids = sorted(group["scenario_id"].unique())
        for idx in range(2, len(scenario_ids) + 1):
            sub = group[group["scenario_id"].isin(scenario_ids[:idx])]
            rows.append(
                {
                    "scenario_class": scenario_class,
                    "pathway": pathway,
                    "n_scenarios": idx,
                    "cov_M_T": lognormal_cov(sub["lnM_T"].to_numpy())[2],
                    "cov_M_S": lognormal_cov(sub["lnM_S"].to_numpy())[2],
                    "bias_factor_T": lognormal_cov(sub["lnM_T"].to_numpy())[4],
                    "bias_factor_S": lognormal_cov(sub["lnM_S"].to_numpy())[4],
                }
            )
    conv = pd.DataFrame(rows)
    if conv.empty:
        return conv
    stable_rows = []
    for (scenario_class, pathway), group in conv.groupby(["scenario_class", "pathway"], sort=False):
        tail = group.sort_values("n_scenarios").tail(3)
        final_t = float(tail["cov_M_T"].iloc[-1])
        final_s = float(tail["cov_M_S"].iloc[-1])
        stable_rows.append(
            {
                "scenario_class": scenario_class,
                "pathway": pathway,
                "n_scenarios": int(tail["n_scenarios"].iloc[-1]),
                "tail_relative_range_cov_M_T": float((tail["cov_M_T"].max() - tail["cov_M_T"].min()) / max(final_t, 1.0e-12)),
                "tail_relative_range_cov_M_S": float((tail["cov_M_S"].max() - tail["cov_M_S"].min()) / max(final_s, 1.0e-12)),
                "converged_flag": bool(
                    ((tail["cov_M_T"].max() - tail["cov_M_T"].min()) / max(final_t, 1.0e-12) < 0.15)
                    and ((tail["cov_M_S"].max() - tail["cov_M_S"].min()) / max(final_s, 1.0e-12) < 0.15)
                ),
            }
        )
    stable = pd.DataFrame(stable_rows)
    return conv.merge(stable, on=["scenario_class", "pathway", "n_scenarios"], how="left")


def sensitivity_summary(fits: pd.DataFrame, verification: pd.DataFrame) -> pd.DataFrame:
    sens = fits[fits["ensemble_role"].str.contains("sensitivity", na=False)].copy()
    rows = []
    for (role, scenario_class, pathway), group in sens.groupby(["ensemble_role", "scenario_class", "pathway"], sort=False):
        rows.append(
            {
                "sensitivity_type": role,
                "scenario_class": scenario_class,
                "pathway": pathway,
                "case_count": int(group["scenario_id"].nunique()),
                "dx_values": ";".join(str(v) for v in sorted(group["dx_m"].unique())),
                "domain_values": ";".join(str(v) for v in sorted(group["domain_m"].unique())),
                "time_count_values": ";".join(str(v) for v in sorted(group["time_count"].unique())),
                "cov_M_T_range": float(group.groupby("scenario_id")["lnM_T"].apply(lambda x: lognormal_cov(x.to_numpy())[2]).max() - group.groupby("scenario_id")["lnM_T"].apply(lambda x: lognormal_cov(x.to_numpy())[2]).min()),
                "cov_M_S_range": float(group.groupby("scenario_id")["lnM_S"].apply(lambda x: lognormal_cov(x.to_numpy())[2]).max() - group.groupby("scenario_id")["lnM_S"].apply(lambda x: lognormal_cov(x.to_numpy())[2]).min()),
                "median_response_cov": float(group["cov_response_model_factor"].median()),
            }
        )
    out = pd.DataFrame(rows)
    if verification.empty:
        return out
    ver = (
        verification.groupby(["ensemble_role", "scenario_class", "reference_solution"], dropna=False)
        .agg(mean_sd_eta=("sd_eta", "mean"), max_sd_eta=("sd_eta", "max"), mean_rmse_m=("rmse_m", "mean"))
        .reset_index()
    )
    ver = ver.rename(columns={"ensemble_role": "sensitivity_type"})
    return out.merge(ver, on=["sensitivity_type", "scenario_class"], how="left")


def field_applicability(cov_ci: pd.DataFrame, conv: pd.DataFrame) -> pd.DataFrame:
    conv_final = conv.dropna(subset=["converged_flag"]).copy()
    conv_final = conv_final.sort_values("n_scenarios").groupby(["scenario_class", "pathway"], as_index=False).tail(1)
    conv_lookup = conv_final.set_index(["scenario_class", "pathway"]).to_dict(orient="index")
    mappings = [
        ("Massachusetts", "heterogeneous K", "spatial variability without persistent leakage support", "sensitivity"),
        ("Massachusetts", "heterogeneous boundary", "fractured bedrock with possible first type boundary behavior", "sensitivity"),
        ("Massachusetts", "heterogeneous leakage boundary", "conservative class for spatial variability plus leakage/boundary alternatives", "prior"),
        ("Lovelock Valley", "vertical leakage", "basin-fill vertical exchange candidate", "sensitivity"),
        ("Lovelock Valley", "constant head boundary", "late-time flattening or boundary-support candidate", "sensitivity"),
        ("Lovelock Valley", "heterogeneous leakage boundary", "conservative upper-bound class if basin heterogeneity and leakage/boundary both remain plausible", "sensitivity"),
    ]
    rows = []
    for field_case, scenario_class, basis, use in mappings:
        subset = cov_ci[cov_ci["scenario_class"] == scenario_class]
        for _, row in subset.iterrows():
            conv_info = conv_lookup.get((scenario_class, row["pathway"]), {})
            converged = conv_info.get("converged_flag")
            ci_width_t = row["cov_M_T_ci_high"] - row["cov_M_T_ci_low"]
            if converged is False or ci_width_t > max(row["cov_M_T"], 1.0e-12):
                allowable_use = "sensitivity only"
            else:
                allowable_use = use
            rows.append(
                {
                    "field_case": field_case,
                    "benchmark_class": scenario_class,
                    "pathway": row["pathway"],
                    "matching_basis": basis,
                    "allowable_use": allowable_use,
                    "recommended_cov_M_T_low": row["cov_M_T_ci_low"],
                    "recommended_cov_M_T_high": row["cov_M_T_ci_high"],
                    "recommended_cov_M_S_low": row["cov_M_S_ci_low"],
                    "recommended_cov_M_S_high": row["cov_M_S_ci_high"],
                    "median_M_T": row["median_M_T"],
                    "median_M_S": row["median_M_S"],
                    "converged_flag": converged,
                    "mismatch_flags": "field geometry and true support scale remain unverified; use conditional class only",
                    "claim_boundary": "scenario-conditioned transformation factor; not a universal design COV",
                }
            )
    return pd.DataFrame(rows)


def export(fig: plt.Figure, stem: str) -> None:
    for folder in [PROJECT, FIG_DIR]:
        for suffix in ["pdf", "png", "svg"]:
            kwargs = {"dpi": 450} if suffix == "png" else {}
            fig.savefig(folder / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(0.0, 1.04, f"({label}) {title}", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")


def plot_design(cases: pd.DataFrame) -> None:
    lock_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    counts = cases[cases["ensemble_role"] == "ensemble"]["scenario_class"].value_counts().reindex(CLASS_ORDER).fillna(0)
    y = np.arange(len(counts))
    ax.barh(y, counts.to_numpy(), color=[CLASS_COLORS[c] for c in counts.index], edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_CLASS[c] for c in counts.index])
    ax.invert_yaxis()
    ax.set_xlabel("ensemble scenarios")
    panel_label(ax, "a", "expanded scenario classes")
    ax.grid(axis="x", color=NEUTRAL_LIGHT, linewidth=0.7)
    ax = axes[1]
    hetero = cases[(cases["ensemble_role"] == "ensemble") & (cases["sigma_ln_k"] > 0)]
    ax.scatter(hetero["corr_len_m"], hetero["sigma_ln_k"], c=[CLASS_COLORS[c] for c in hetero["scenario_class"]], s=28, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("correlation length (m)")
    ax.set_ylabel(r"$\sigma_{\ln K}$")
    panel_label(ax, "b", "heterogeneous field design")
    ax.grid(color=NEUTRAL_LIGHT, linewidth=0.7)
    export(fig, "fig13_mf6_benchmark_design")


def plot_verification_sensitivity(verification: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    lock_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.75), constrained_layout=True)
    ax = axes[0]
    direct = verification[verification["exactness"] == "direct baseline"]
    grouped = direct.groupby(["scenario_class", "reference_solution"])["sd_eta"].median().reset_index()
    labels = [SHORT_CLASS.get(c, c) for c in grouped["scenario_class"]]
    ax.bar(np.arange(len(grouped)), grouped["sd_eta"], color=[CLASS_COLORS[c] for c in grouped["scenario_class"]], edgecolor="none")
    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("median direct-solution SD_eta")
    panel_label(ax, "a", "analytical baseline verification")
    ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.7)
    ax = axes[1]
    sens = sensitivity[sensitivity["pathway"].isin(["Theis confined", "Hantush-Jacob leaky", "Lagging Darcy with leakage"])]
    for pathway in sens["pathway"].unique():
        sub = sens[sens["pathway"] == pathway]
        ax.scatter(sub["median_response_cov"], sub["cov_M_T_range"], s=30, color=PATHWAY_COLORS[pathway], label=pathway)
    ax.set_xlabel("median response factor COV")
    ax.set_ylabel("within-test T factor COV range")
    panel_label(ax, "b", "grid/domain/time sensitivity")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=NEUTRAL_LIGHT, linewidth=0.7)
    export(fig, "fig14_mf6_verification_sensitivity")


def plot_convergence_cov(cov_ci: pd.DataFrame, conv: pd.DataFrame) -> None:
    lock_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.35), constrained_layout=True)
    ax = axes[0, 0]
    for cls in ["heterogeneous K", "heterogeneous leakage boundary", "vertical leakage"]:
        sub = conv[(conv["scenario_class"] == cls) & (conv["pathway"] == "Lagging Darcy with leakage")].sort_values("n_scenarios")
        ax.plot(sub["n_scenarios"], sub["cov_M_T"], marker="o", markersize=2.5, linewidth=1.0, color=CLASS_COLORS[cls], label=SHORT_CLASS[cls].replace("\n", " "))
    ax.set_xlabel("scenarios")
    ax.set_ylabel("cumulative T factor COV")
    panel_label(ax, "a", "ensemble convergence examples")
    ax.legend(frameon=False)
    ax.grid(color=NEUTRAL_LIGHT, linewidth=0.7)
    for ax, metric, lo, hi, label, title in [
        (axes[0, 1], "cov_M_T", "cov_M_T_ci_low", "cov_M_T_ci_high", "b", "T model factor COV"),
        (axes[1, 0], "cov_M_S", "cov_M_S_ci_low", "cov_M_S_ci_high", "c", "S model factor COV"),
        (axes[1, 1], "cov_M_response_time", "cov_M_response_time_ci_low", "cov_M_response_time_ci_high", "d", "response-time factor COV"),
    ]:
        x = np.arange(len(CLASS_ORDER), dtype=float)
        offsets = np.linspace(-0.27, 0.27, len(PATHWAYS))
        for offset, pathway in zip(offsets, PATHWAYS):
            plot_data = cov_ci[cov_ci["pathway"] == pathway].set_index("scenario_class").reindex(CLASS_ORDER).dropna(subset=[metric])
            vals = plot_data[metric].to_numpy(float)
            err = np.vstack([vals - plot_data[lo].to_numpy(float), plot_data[hi].to_numpy(float) - vals])
            class_positions = np.array([CLASS_ORDER.index(cls) for cls in plot_data.index], dtype=float) + offset
            ax.errorbar(
                class_positions,
                vals,
                yerr=err,
                fmt="o",
                markersize=2.7,
                capsize=1.8,
                linewidth=0.9,
                color=PATHWAY_COLORS[pathway],
                label=pathway if label == "b" else None,
                alpha=0.96,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_CLASS[c] for c in CLASS_ORDER], rotation=28, ha="right")
        ax.set_ylabel("COV")
        panel_label(ax, label, title)
        if label == "b":
            ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=5.8)
        ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.7)
    export(fig, "fig15_mf6_convergence_cov")


def plot_field_applicability(field_app: pd.DataFrame) -> None:
    lock_style()
    fig, ax = plt.subplots(figsize=(7.3, 2.8), constrained_layout=True)
    agg = (
        field_app.groupby(["field_case", "benchmark_class"], as_index=False)
        .agg(max_cov_T=("recommended_cov_M_T_high", "max"), max_cov_S=("recommended_cov_M_S_high", "max"))
        .sort_values(["field_case", "benchmark_class"])
    )
    labels = [f"{r.field_case}\n{SHORT_CLASS[r.benchmark_class].replace(chr(10), ' ')}" for r in agg.itertuples()]
    x = np.arange(agg.shape[0])
    width = 0.35
    ax.bar(x - width / 2, agg["max_cov_T"], width=width, color=cmc_color("batlow", 0.25), label="T")
    ax.bar(x + width / 2, agg["max_cov_S"], width=width, color=cmc_color("batlow", 0.75), label="S")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("upper 95% CI COV")
    panel_label(ax, "a", "field applicability by benchmark class")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.7)
    export(fig, "fig16_mf6_field_applicability")


def run_pipeline() -> dict:
    ensure_dirs()
    scenarios = build_expanded_ensemble()
    sensitivity_cases = build_sensitivity_cases()
    all_scenarios = scenarios + sensitivity_cases
    drawdown_frames: list[pd.DataFrame] = []
    case_rows: list[dict] = []
    for scenario in all_scenarios:
        drawdown, meta, _ = run_mf6_scenario(scenario)
        drawdown_frames.append(drawdown)
        case_rows.append(meta)
    drawdown = pd.concat(drawdown_frames, ignore_index=True)
    cases = pd.DataFrame(case_rows)
    fits = fit_all(drawdown, cases)
    verification = analytical_baseline_verification(drawdown, cases)
    cov_ci = model_factor_cov_ci(fits)
    conv = convergence_table(fits)
    sens = sensitivity_summary(fits, verification)
    field_app = field_applicability(cov_ci, conv)

    drawdown.to_csv(TABLE_DIR / "mf6_strict_benchmark_drawdown.csv", index=False)
    cases.to_csv(TABLE_DIR / "mf6_strict_benchmark_cases.csv", index=False)
    fits.to_csv(TABLE_DIR / "mf6_strict_pathway_fit_results.csv", index=False)
    verification.to_csv(TABLE_DIR / "mf6_strict_analytical_baseline_verification.csv", index=False)
    sens.to_csv(TABLE_DIR / "mf6_strict_grid_domain_time_sensitivity.csv", index=False)
    cov_ci.to_csv(TABLE_DIR / "mf6_strict_model_factor_cov_ci.csv", index=False)
    conv.to_csv(TABLE_DIR / "mf6_strict_convergence.csv", index=False)
    field_app.to_csv(TABLE_DIR / "mf6_strict_field_applicability.csv", index=False)

    plot_design(cases)
    plot_verification_sensitivity(verification, sens)
    plot_convergence_cov(cov_ci, conv)
    plot_field_applicability(field_app)

    conv_final = conv.dropna(subset=["converged_flag"]).sort_values("n_scenarios").groupby(["scenario_class", "pathway"], as_index=False).tail(1)
    summary = {
        "mf6_executable": MF6_EXE,
        "flopy_version": flopy.__version__,
        "ensemble_scenario_count": int(len(scenarios)),
        "sensitivity_scenario_count": int(len(sensitivity_cases)),
        "scenario_count_by_class": cases[cases["ensemble_role"] == "ensemble"]["scenario_class"].value_counts().to_dict(),
        "fit_records": int(fits.shape[0]),
        "baseline_direct_max_sd_eta": float(verification[verification["exactness"] == "direct baseline"]["sd_eta"].max()),
        "cov_M_T_range": [float(cov_ci["cov_M_T"].min()), float(cov_ci["cov_M_T"].max())],
        "cov_M_S_range": [float(cov_ci["cov_M_S"].min()), float(cov_ci["cov_M_S"].max())],
        "not_converged_count": int((conv_final["converged_flag"] == False).sum()),
        "all_converged": bool(conv_final["converged_flag"].all()),
        "claim_boundary": "scenario-conditioned numerical evidence with bootstrap intervals and convergence flags; not a universal design COV database",
    }
    (OUT_DIR / "mf6_strict_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_pipeline()


