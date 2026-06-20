from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import flopy
import numpy as np
import pandas as pd
import psutil
from scipy.ndimage import gaussian_filter
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf6_strict_benchmark as strict  # noqa: E402
import mf6_transformation_benchmark as base  # noqa: E402


PROJECT = HERE.parents[0]
TABLE_DIR = PROJECT / "tables"
OUT_DIR = PROJECT / "outputs"

REGISTRY_CSV = TABLE_DIR / "mf6_mega_benchmark_registry.csv"
PILOT_REGISTRY_CSV = TABLE_DIR / "mf6_mega_benchmark_pilot_registry.csv"
PILOT_CASES_CSV = TABLE_DIR / "mf6_mega_benchmark_pilot_cases.csv"
PILOT_FITS_CSV = TABLE_DIR / "mf6_mega_benchmark_pilot_fits.csv"
FULL_CASES_CSV = TABLE_DIR / "mf6_mega_benchmark_cases.csv"
FULL_FITS_CSV = TABLE_DIR / "mf6_mega_benchmark_fits.csv"
FULL_DRAWDOWN_CSV = TABLE_DIR / "mf6_mega_benchmark_drawdown.csv"
FULL_COV_BY_CLASS_CSV = TABLE_DIR / "mf6_mega_model_factor_cov_by_class.csv"
FULL_COV_BY_LAYOUT_CSV = TABLE_DIR / "mf6_mega_model_factor_cov_by_class_layout.csv"
FULL_COV_QUALITY_GATED_CSV = TABLE_DIR / "mf6_mega_model_factor_cov_quality_gated_by_class.csv"
FULL_COV_BIC_BEST_CSV = TABLE_DIR / "mf6_mega_model_factor_cov_bic_best_by_class.csv"
PILOT_SUMMARY_JSON = OUT_DIR / "mf6_mega_benchmark_pilot_summary.json"
CAPACITY_JSON = OUT_DIR / "mf6_mega_benchmark_capacity.json"
RUN_SUMMARY_JSON = OUT_DIR / "mf6_mega_benchmark_run_summary.json"

WORKSPACE_ROOT = OUT_DIR / "mf6_mega_benchmark_work"
CHECKPOINT_ROOT = OUT_DIR / "mf6_mega_benchmark_checkpoints"
DRAW_DIR = CHECKPOINT_ROOT / "drawdown"
CASE_DIR = CHECKPOINT_ROOT / "cases"
FIT_DIR = CHECKPOINT_ROOT / "fits"
LOG_DIR = CHECKPOINT_ROOT / "logs"

SECONDS_PER_DAY = base.SECONDS_PER_DAY
Q_M3D = base.Q_M3D
T_REF_M2S = base.T_REF_M2S
T_REF_M2D = base.T_REF_M2D
S_REF = base.S_REF
AQ_THICKNESS_M = base.AQ_THICKNESS_M
K_REF_MD = base.K_REF_MD
TOP_M = base.TOP_M
BOTM_M = base.BOTM_M
MF6_EXE = shutil.which("mf6") or base.MF6_EXE

CLASS_ORDER = [
    "homogeneous confined",
    "vertical leakage",
    "finite no-flow boundary",
    "constant head boundary",
    "heterogeneous K",
    "anisotropic heterogeneous K",
    "heterogeneous storage",
    "heterogeneous leakage",
    "heterogeneous boundary",
    "combined leakage boundary",
]

OBS_LAYOUTS: dict[str, list[tuple[str, float, float]]] = {
    "near4": [("R10E", 10.0, 0.0), ("R20E", 20.0, 0.0), ("R50E", 50.0, 0.0), ("R50N", 50.0, 90.0)],
    "standard4": [("R20E", 20.0, 0.0), ("R50E", 50.0, 0.0), ("R100E", 100.0, 0.0), ("R100N", 100.0, 90.0)],
    "far4": [("R50E", 50.0, 0.0), ("R100E", 100.0, 0.0), ("R200E", 200.0, 0.0), ("R200N", 200.0, 90.0)],
    "mixed4": [("R10E", 10.0, 0.0), ("R50E", 50.0, 0.0), ("R200E", 200.0, 0.0), ("R200N", 200.0, 90.0)],
    "dense5": [("R10E", 10.0, 0.0), ("R20NE", 20.0, 45.0), ("R50E", 50.0, 0.0), ("R100E", 100.0, 0.0), ("R200N", 200.0, 90.0)],
    "sparse3": [("R20E", 20.0, 0.0), ("R100E", 100.0, 0.0), ("R200N", 200.0, 90.0)],
}

DX_OPTIONS = [10.0, 7.5, 5.0]
DOMAIN_OPTIONS = [405.0, 605.0, 805.0, 1005.0, 1205.0]
TIME_COUNT_OPTIONS = [20, 28, 40]
DURATION_OPTIONS_D = [1.0, 2.0, 5.0, 10.0]
NOISE_OPTIONS_M = [0.0, 0.001, 0.003, 0.01]
BOUNDARY_OPTIONS = ["none", "constant_head_all", "constant_head_x", "constant_head_east"]


@dataclass(frozen=True)
class MegaScenario:
    scenario_id: str
    scenario_class: str
    seed: int
    sigma_ln_k: float
    corr_len_m: float
    sigma_ln_s: float
    corr_len_s_m: float
    anisotropy_ratio: float
    leakage_b_m: float | None
    boundary_type: str
    dx_m: float
    domain_m: float
    time_count: int
    duration_days: float
    obs_layout: str
    obs_rotation_deg: float
    noise_sd_m: float
    ensemble_role: str = "mega"
    description: str = ""


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in [DRAW_DIR, CASE_DIR, FIT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def choose(options: list[Any], value: float) -> Any:
    idx = min(int(value * len(options)), len(options) - 1)
    return options[idx]


def scale(lo: float, hi: float, value: float) -> float:
    return float(lo + (hi - lo) * value)


def log_scale(lo: float, hi: float, value: float) -> float:
    return float(math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * value))


def scenario_slug(text: str) -> str:
    return text.replace(" ", "_").replace("-", "_")


def adjusted_domain(domain_m: float, dx_m: float, obs_layout: str) -> float:
    max_radius = max(radius for _, radius, _ in OBS_LAYOUTS[obs_layout])
    minimum = 2.0 * max_radius + 8.0 * dx_m + 25.0
    if domain_m >= minimum:
        return float(domain_m)
    candidates = [value for value in DOMAIN_OPTIONS if value >= minimum]
    if candidates:
        return float(candidates[0])
    return float(math.ceil(minimum / 5.0) * 5.0)


def build_mega_registry(n: int = 10000, seed: int = 20260608) -> pd.DataFrame:
    rows: list[dict] = []
    per_class = int(math.ceil(n / len(CLASS_ORDER)))
    obs_names = list(OBS_LAYOUTS)
    for class_idx, scenario_class in enumerate(CLASS_ORDER):
        sampler = qmc.LatinHypercube(d=15, seed=seed + class_idx * 101)
        sample = sampler.random(per_class)
        for local_idx, u in enumerate(sample):
            sigma_ln_k = 0.0
            corr_len_m = 0.0
            sigma_ln_s = 0.0
            corr_len_s_m = 0.0
            anisotropy_ratio = 1.0
            leakage_b_m: float | None = None
            boundary_type = "none"
            description = "Homogeneous confined baseline."

            if scenario_class == "vertical leakage":
                leakage_b_m = log_scale(25.0, 250.0, u[0])
                description = "Homogeneous aquifer with distributed vertical exchange."
            elif scenario_class == "finite no-flow boundary":
                description = "Finite no-flow model boundary stress test."
            elif scenario_class == "constant head boundary":
                boundary_type = choose(BOUNDARY_OPTIONS[1:], u[1])
                leakage_b_m = None if u[2] < 0.75 else log_scale(80.0, 300.0, u[3])
                description = "Finite first type boundary stress test."
            elif scenario_class == "heterogeneous K":
                sigma_ln_k = scale(0.15, 1.50, u[0])
                corr_len_m = log_scale(8.0, 180.0, u[1])
                description = "Lognormal hydraulic-conductivity field."
            elif scenario_class == "anisotropic heterogeneous K":
                sigma_ln_k = scale(0.15, 1.50, u[0])
                corr_len_m = log_scale(8.0, 180.0, u[1])
                anisotropy_ratio = log_scale(0.20, 5.00, u[2])
                description = "Lognormal K field with horizontal anisotropy."
            elif scenario_class == "heterogeneous storage":
                sigma_ln_k = scale(0.15, 1.20, u[0])
                corr_len_m = log_scale(8.0, 160.0, u[1])
                sigma_ln_s = scale(0.10, 1.00, u[2])
                corr_len_s_m = log_scale(8.0, 160.0, u[3])
                description = "Lognormal K and storage-coefficient fields."
            elif scenario_class == "heterogeneous leakage":
                sigma_ln_k = scale(0.15, 1.50, u[0])
                corr_len_m = log_scale(8.0, 180.0, u[1])
                leakage_b_m = log_scale(25.0, 250.0, u[2])
                description = "Heterogeneous K with distributed vertical exchange."
            elif scenario_class == "heterogeneous boundary":
                sigma_ln_k = scale(0.15, 1.50, u[0])
                corr_len_m = log_scale(8.0, 180.0, u[1])
                boundary_type = choose(BOUNDARY_OPTIONS, u[2])
                description = "Heterogeneous K with finite-domain boundary stress."
            elif scenario_class == "combined leakage boundary":
                sigma_ln_k = scale(0.15, 1.50, u[0])
                corr_len_m = log_scale(8.0, 180.0, u[1])
                sigma_ln_s = scale(0.0, 0.80, u[2])
                corr_len_s_m = log_scale(8.0, 160.0, u[3])
                anisotropy_ratio = log_scale(0.30, 3.30, u[4])
                leakage_b_m = log_scale(25.0, 250.0, u[5])
                boundary_type = choose(BOUNDARY_OPTIONS[1:], u[6])
                description = "Combined K heterogeneity, possible storage variability, leakage, anisotropy, and boundary stress."

            obs_layout = choose(obs_names, u[7])
            dx_m = float(choose(DX_OPTIONS, u[8]))
            domain_m = adjusted_domain(float(choose(DOMAIN_OPTIONS, u[9])), dx_m, obs_layout)
            time_count = int(choose(TIME_COUNT_OPTIONS, u[10]))
            duration_days = float(choose(DURATION_OPTIONS_D, u[11]))
            noise_sd_m = float(choose(NOISE_OPTIONS_M, u[12]))
            obs_rotation_deg = scale(0.0, 180.0, u[13])
            scenario_seed = int(seed + class_idx * 100000 + local_idx)
            scenario_id = f"mega_{scenario_slug(scenario_class)}_{local_idx:05d}"
            rows.append(
                asdict(
                    MegaScenario(
                        scenario_id=scenario_id,
                        scenario_class=scenario_class,
                        seed=scenario_seed,
                        sigma_ln_k=sigma_ln_k,
                        corr_len_m=corr_len_m,
                        sigma_ln_s=sigma_ln_s,
                        corr_len_s_m=corr_len_s_m,
                        anisotropy_ratio=anisotropy_ratio,
                        leakage_b_m=leakage_b_m,
                        boundary_type=boundary_type,
                        dx_m=dx_m,
                        domain_m=domain_m,
                        time_count=time_count,
                        duration_days=duration_days,
                        obs_layout=obs_layout,
                        obs_rotation_deg=obs_rotation_deg,
                        noise_sd_m=noise_sd_m,
                        description=description,
                    )
                )
            )
            if len(rows) >= n:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def scenario_from_dict(row: dict) -> MegaScenario:
    data = dict(row)
    leakage = data.get("leakage_b_m")
    if leakage is None or (isinstance(leakage, float) and np.isnan(leakage)) or str(leakage).lower() == "nan":
        data["leakage_b_m"] = None
    else:
        data["leakage_b_m"] = float(leakage)
    data["seed"] = int(data["seed"])
    data["time_count"] = int(data["time_count"])
    for key in [
        "sigma_ln_k",
        "corr_len_m",
        "sigma_ln_s",
        "corr_len_s_m",
        "anisotropy_ratio",
        "dx_m",
        "domain_m",
        "duration_days",
        "obs_rotation_deg",
        "noise_sd_m",
    ]:
        data[key] = float(data[key])
    return MegaScenario(**data)


def grid_shape(domain_m: float, dx_m: float) -> tuple[int, int]:
    n = int(round(domain_m / dx_m)) + 1
    if n % 2 == 0:
        n += 1
    return n, n


def times_for_scenario(scenario: MegaScenario) -> np.ndarray:
    return np.geomspace(90.0, max(90.0 * 1.01, scenario.duration_days * SECONDS_PER_DAY), int(scenario.time_count))


def observation_wells(scenario: MegaScenario) -> list[tuple[str, float, float]]:
    return [(name, radius, angle + scenario.obs_rotation_deg) for name, radius, angle in OBS_LAYOUTS[scenario.obs_layout]]


def obs_cell(radius_m: float, angle_deg: float, nrow: int, ncol: int, dx_m: float) -> tuple[tuple[int, int, int], float, float, float]:
    angle = math.radians(angle_deg)
    x = radius_m * math.cos(angle)
    y = radius_m * math.sin(angle)
    row_offset = int(round(y / dx_m))
    col_offset = int(round(x / dx_m))
    row = nrow // 2 - row_offset
    col = ncol // 2 + col_offset
    if row < 0 or row >= nrow or col < 0 or col >= ncol:
        raise ValueError(f"Observation radius {radius_m} m at {angle_deg} deg falls outside {nrow} x {ncol} grid.")
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


def correlated_unit_field(seed: int, nrow: int, ncol: int, corr_len_m: float, dx_m: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=(nrow, ncol))
    sigma_cells = max(corr_len_m / dx_m, 0.5)
    smooth = gaussian_filter(raw, sigma=sigma_cells, mode="reflect")
    return (smooth - np.mean(smooth)) / max(float(np.std(smooth)), 1.0e-12)


def make_property_fields(scenario: MegaScenario, nrow: int, ncol: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scenario.sigma_ln_k <= 0.0:
        kx = np.full((nrow, ncol), K_REF_MD, dtype=float)
    else:
        unit = correlated_unit_field(scenario.seed, nrow, ncol, scenario.corr_len_m, scenario.dx_m)
        kx = np.exp(math.log(K_REF_MD) + scenario.sigma_ln_k * unit)
    ky = kx / max(scenario.anisotropy_ratio, 1.0e-12)
    if scenario.sigma_ln_s <= 0.0:
        storage = np.full((nrow, ncol), S_REF, dtype=float)
    else:
        unit_s = correlated_unit_field(scenario.seed + 7919, nrow, ncol, scenario.corr_len_s_m, scenario.dx_m)
        storage = np.exp(math.log(S_REF) + scenario.sigma_ln_s * unit_s)
    return kx, ky, storage


DEPRECATED_FIXED_SUPPORT_RADIUS_M = 130.0


def radial_annular_effective_t(k_eff_md: np.ndarray, dx_m: float, radius_limit_m: float) -> float:
    nrow, ncol = k_eff_md.shape
    _, _, radius = cell_coordinates(nrow, ncol, dx_m)
    dr = max(dx_m, 2.5)
    annuli = np.arange(dx_m, radius_limit_m + dr, dr)
    resistances = []
    for r0, r1 in zip(annuli[:-1], annuli[1:]):
        mask = (radius >= r0) & (radius < r1)
        if not np.any(mask):
            continue
        k_vals = np.maximum(k_eff_md[mask], 1.0e-12)
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


def boundary_limit_m(scenario: MegaScenario) -> float:
    half_domain = max(0.5 * scenario.domain_m - scenario.dx_m, scenario.dx_m)
    if scenario.boundary_type == "none":
        return half_domain
    if scenario.boundary_type in {"constant_head_all", "constant_head_x", "constant_head_east"}:
        return half_domain
    return half_domain


def effective_support_controls(k_eff: np.ndarray, storage: np.ndarray, scenario: MegaScenario) -> dict:
    t_global = float(np.exp(np.mean(np.log(np.maximum(k_eff * AQ_THICKNESS_M / SECONDS_PER_DAY, 1.0e-30)))))
    s_global = float(np.exp(np.mean(np.log(np.maximum(storage, 1.0e-30)))))
    hydraulic_diffusivity = t_global / max(s_global, 1.0e-30)
    diffusion_radius = math.sqrt(max(2.25 * hydraulic_diffusivity * scenario.duration_days * SECONDS_PER_DAY, scenario.dx_m**2))
    observation_radius = max(radius for _, radius, _ in observation_wells(scenario))
    leakage_limit = float(scenario.leakage_b_m) if scenario.leakage_b_m is not None else float("inf")
    domain_limit = boundary_limit_m(scenario)
    limited_radius = min(diffusion_radius, leakage_limit, domain_limit)
    lower_radius = max(3.0 * scenario.dx_m, min(observation_radius, domain_limit) * 0.50)
    support_radius = min(max(limited_radius, lower_radius), domain_limit)
    return {
        "hydraulic_diffusivity_m2_s": hydraulic_diffusivity,
        "support_diffusion_radius_m": diffusion_radius,
        "support_observation_max_radius_m": observation_radius,
        "support_leakage_limit_m": leakage_limit,
        "support_boundary_limit_m": domain_limit,
        "support_effective_radius_m": support_radius,
    }


def effective_support_weights(radius: np.ndarray, controls: dict, scenario: MegaScenario) -> np.ndarray:
    support_radius = max(float(controls["support_effective_radius_m"]), scenario.dx_m)
    diffusion_radius = max(float(controls["support_diffusion_radius_m"]), scenario.dx_m)
    weights = np.exp(-((radius / diffusion_radius) ** 2))
    if scenario.leakage_b_m is not None and np.isfinite(float(scenario.leakage_b_m)):
        weights *= np.exp(-radius / max(float(scenario.leakage_b_m), scenario.dx_m))
    weights = np.where(radius <= support_radius, weights, 0.0)
    weights = np.where(radius <= float(controls["support_boundary_limit_m"]), weights, 0.0)
    return weights


def support_targets(kx: np.ndarray, ky: np.ndarray, storage: np.ndarray, scenario: MegaScenario) -> dict:
    k_eff = np.sqrt(np.maximum(kx, 1.0e-30) * np.maximum(ky, 1.0e-30))
    nrow, ncol = k_eff.shape
    _, _, radius = cell_coordinates(nrow, ncol, scenario.dx_m)
    controls = effective_support_controls(k_eff, storage, scenario)
    weights = effective_support_weights(radius, controls, scenario)
    if not np.any(weights > 0.0):
        weights = np.where(radius <= max(3.0 * scenario.dx_m, controls["support_effective_radius_m"]), 1.0, 0.0)
    t_field = np.maximum(k_eff * AQ_THICKNESS_M / SECONDS_PER_DAY, 1.0e-30)
    s_field = np.maximum(storage, 1.0e-30)
    t_geo = weighted_geomean(t_field, weights)
    t_arith = weighted_mean(t_field, weights)
    t_ann = radial_annular_effective_t(k_eff, scenario.dx_m, radius_limit_m=max(controls["support_effective_radius_m"], 2.0 * scenario.dx_m))
    s_geo = weighted_geomean(s_field, weights)
    fixed_radius = min(DEPRECATED_FIXED_SUPPORT_RADIUS_M, controls["support_boundary_limit_m"])
    fixed_mask = radius <= fixed_radius
    fixed_t_vals = np.maximum(t_field[fixed_mask], 1.0e-30)
    fixed_s_vals = np.maximum(s_field[fixed_mask], 1.0e-30)
    fixed_t_geo = float(np.exp(np.mean(np.log(fixed_t_vals)))) if fixed_t_vals.size else t_geo
    fixed_s_geo = float(np.exp(np.mean(np.log(fixed_s_vals)))) if fixed_s_vals.size else s_geo
    spread = float(max(t_geo, t_arith, t_ann, fixed_t_geo) / max(min(t_geo, t_arith, t_ann, fixed_t_geo), 1.0e-30))
    if scenario.sigma_ln_k <= 0.0 and scenario.sigma_ln_s <= 0.0:
        support_definition = "assigned true T/S; effective support radius reported for hydraulic context"
        spread = 1.0
        t_geo = T_REF_M2S
        t_arith = T_REF_M2S
        t_ann = T_REF_M2S
        s_geo = S_REF
    else:
        support_definition = "scenario-specific diffusion/leakage/boundary weighted effective support; fixed-radius 130 m sensitivity retained only for comparison"
    return {
        "truth_T_ref_m2_s": t_geo,
        "truth_T_geomean_m2_s": t_geo,
        "truth_T_arithmetic_m2_s": t_arith,
        "truth_T_annular_m2_s": t_ann,
        "truth_S_ref": s_geo,
        "support_definition": support_definition,
        "support_target_spread": spread,
        "support_effective_radius_m": float(controls["support_effective_radius_m"]),
        "support_diffusion_radius_m": float(controls["support_diffusion_radius_m"]),
        "support_observation_max_radius_m": float(controls["support_observation_max_radius_m"]),
        "support_leakage_limit_m": None if not np.isfinite(float(controls["support_leakage_limit_m"])) else float(controls["support_leakage_limit_m"]),
        "support_boundary_limit_m": float(controls["support_boundary_limit_m"]),
        "support_hydraulic_diffusivity_m2_s": float(controls["hydraulic_diffusivity_m2_s"]),
        "support_weighting": "Gaussian hydraulic-diffusion weights clipped by effective radius, leakage attenuation, and boundary/domain limit",
        "fixed_radius_130m_T_geomean_m2_s": fixed_t_geo,
        "fixed_radius_130m_S_geomean": fixed_s_geo,
        "fixed_radius_130m_T_ratio_to_effective": float(fixed_t_geo / max(t_geo, 1.0e-30)),
        "K_geomean_m_d": float(np.exp(np.mean(np.log(np.maximum(k_eff, 1.0e-30))))),
        "K_arithmetic_m_d": float(np.mean(k_eff)),
        "K_cv": float(np.std(k_eff) / max(np.mean(k_eff), 1.0e-30)),
        "S_geomean": float(np.exp(np.mean(np.log(np.maximum(storage, 1.0e-30))))),
        "S_arithmetic": float(np.mean(storage)),
        "S_cv": float(np.std(storage) / max(np.mean(storage), 1.0e-30)),
    }


def boundary_cells(boundary_type: str, nrow: int, ncol: int) -> list[tuple[tuple[int, int, int], float]]:
    cells: list[tuple[tuple[int, int, int], float]] = []
    if boundary_type == "none":
        return cells
    if boundary_type == "constant_head_all":
        for row in range(nrow):
            cells.append(((0, row, 0), 0.0))
            cells.append(((0, row, ncol - 1), 0.0))
        for col in range(1, ncol - 1):
            cells.append(((0, 0, col), 0.0))
            cells.append(((0, nrow - 1, col), 0.0))
    elif boundary_type == "constant_head_x":
        for row in range(nrow):
            cells.append(((0, row, 0), 0.0))
            cells.append(((0, row, ncol - 1), 0.0))
    elif boundary_type == "constant_head_east":
        for row in range(nrow):
            cells.append(((0, row, ncol - 1), 0.0))
    else:
        raise ValueError(f"Unsupported boundary_type={boundary_type!r}")
    return cells


def workspace_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def safe_rmtree(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise RuntimeError(f"Refusing to delete path outside workspace root: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


def run_mf6_scenario(scenario: MegaScenario, keep_workspace: bool = False) -> tuple[pd.DataFrame, dict]:
    nrow, ncol = grid_shape(scenario.domain_m, scenario.dx_m)
    times_s = times_for_scenario(scenario)
    times_d = times_s / SECONDS_PER_DAY
    dts_d = np.diff(np.r_[0.0, times_d])
    kx, ky, storage = make_property_fields(scenario, nrow, ncol)
    targets = support_targets(kx, ky, storage, scenario)
    sim_ws = WORKSPACE_ROOT / scenario.scenario_id
    if sim_ws.exists():
        safe_rmtree(sim_ws, WORKSPACE_ROOT)
    sim_ws.mkdir(parents=True)

    sim = flopy.mf6.MFSimulation(sim_name=scenario.scenario_id, exe_name=MF6_EXE, sim_ws=str(sim_ws), version="mf6")
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=len(dts_d), perioddata=[(float(dt), 1, 1.0) for dt in dts_d])
    flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1.0e-8, inner_dvclose=1.0e-8)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=False)
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=nrow, ncol=ncol, delr=scenario.dx_m, delc=scenario.dx_m, top=TOP_M, botm=BOTM_M)
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=kx, k22=ky, k33=kx, save_specific_discharge=False)
    flopy.mf6.ModflowGwfsto(
        gwf,
        storagecoefficient=True,
        iconvert=0,
        ss=storage,
        steady_state={0: False},
        transient={idx: True for idx in range(len(dts_d))},
    )
    center = (0, nrow // 2, ncol // 2)
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data={idx: [(center, -Q_M3D)] for idx in range(len(dts_d))}, save_flows=False)
    chd = boundary_cells(scenario.boundary_type, nrow, ncol)
    if chd:
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, save_flows=False)
    if scenario.leakage_b_m is not None:
        t_cell_m2d = np.sqrt(np.maximum(kx, 1.0e-30) * np.maximum(ky, 1.0e-30)) * AQ_THICKNESS_M
        conductance = (scenario.dx_m * scenario.dx_m) * (t_cell_m2d / (scenario.leakage_b_m * scenario.leakage_b_m))
        ghb = [((0, row, col), 0.0, float(conductance[row, col])) for row in range(nrow) for col in range(ncol)]
        flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb, save_flows=False)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds", saverecord=[("HEAD", "ALL")])

    sim.write_simulation(silent=True)
    ok, buff = sim.run_simulation(silent=True)
    if not ok:
        raise RuntimeError(f"MODFLOW 6 failed for {scenario.scenario_id}: {' '.join(buff[-12:])}")

    heads = gwf.output.head().get_alldata()[:, 0, :, :]
    noise_rng = np.random.default_rng(scenario.seed + 104729)
    rows = []
    actual_radii = []
    for well, radius_m, angle_deg in observation_wells(scenario):
        cell, actual_radius, actual_x, actual_y = obs_cell(radius_m, angle_deg, nrow, ncol, scenario.dx_m)
        actual_radii.append(actual_radius)
        _, row, col = cell
        drawdown = np.maximum(-heads[:, row, col], 0.0)
        if scenario.noise_sd_m > 0.0:
            drawdown = np.maximum(drawdown + noise_rng.normal(0.0, scenario.noise_sd_m, size=drawdown.size), 0.0)
        for time_s, dd in zip(times_s, drawdown):
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_class": scenario.scenario_class,
                    "ensemble_role": scenario.ensemble_role,
                    "well": well,
                    "obs_layout": scenario.obs_layout,
                    "nominal_radius_m": radius_m,
                    "radius_m": actual_radius,
                    "x_m": actual_x,
                    "y_m": actual_y,
                    "time_s": float(time_s),
                    "time_h": float(time_s / 3600.0),
                    "drawdown_m": float(dd),
                }
            )

    bytes_before_cleanup = workspace_bytes(sim_ws)
    meta = {
        **asdict(scenario),
        "leakage_B_m": scenario.leakage_b_m,
        "constant_head_edges": scenario.boundary_type != "none",
        "grid_nrow": nrow,
        "grid_ncol": ncol,
        "grid_cell_count": int(nrow * ncol),
        "observation_count": int(len(rows)),
        "well_count": int(len(observation_wells(scenario))),
        "min_actual_radius_m": float(np.min(actual_radii)),
        "max_actual_radius_m": float(np.max(actual_radii)),
        "max_drawdown_m": float(max(row["drawdown_m"] for row in rows)),
        "workspace_bytes_before_cleanup": int(bytes_before_cleanup),
        **targets,
    }
    if not keep_workspace:
        safe_rmtree(sim_ws, WORKSPACE_ROOT)
    return pd.DataFrame(rows), meta


def checkpoint_paths(scenario_id: str) -> dict[str, Path]:
    return {
        "drawdown": DRAW_DIR / f"{scenario_id}.csv",
        "case": CASE_DIR / f"{scenario_id}.json",
        "fit": FIT_DIR / f"{scenario_id}.csv",
        "log": LOG_DIR / f"{scenario_id}.json",
    }


def checkpoint_success(scenario_id: str) -> bool:
    log_path = checkpoint_paths(scenario_id)["log"]
    if not log_path.exists():
        return False
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("status") == "success"


def run_one_scenario(row: dict, keep_workspace: bool = False, force: bool = False) -> dict:
    scenario = scenario_from_dict(row)
    paths = checkpoint_paths(scenario.scenario_id)
    if checkpoint_success(scenario.scenario_id) and not force:
        return {"scenario_id": scenario.scenario_id, "status": "skipped_existing"}
    start = time.perf_counter()
    try:
        drawdown, meta = run_mf6_scenario(scenario, keep_workspace=keep_workspace)
        cases = pd.DataFrame([meta])
        fits = strict.fit_all(drawdown, cases)
        for col in [
            "obs_layout",
            "boundary_type",
            "duration_days",
            "noise_sd_m",
            "sigma_ln_s",
            "corr_len_s_m",
            "anisotropy_ratio",
            "well_count",
            "grid_cell_count",
        ]:
            fits[col] = meta[col]
        elapsed_s = time.perf_counter() - start
        meta["elapsed_s"] = elapsed_s
        atomic_csv(paths["drawdown"], drawdown)
        atomic_json(paths["case"], meta)
        atomic_csv(paths["fit"], fits)
        log = {
            "scenario_id": scenario.scenario_id,
            "status": "success",
            "elapsed_s": elapsed_s,
            "workspace_bytes_before_cleanup": meta["workspace_bytes_before_cleanup"],
            "drawdown_rows": int(drawdown.shape[0]),
            "fit_rows": int(fits.shape[0]),
        }
        atomic_json(paths["log"], log)
        return log
    except Exception as exc:  # noqa: BLE001
        elapsed_s = time.perf_counter() - start
        sim_ws = WORKSPACE_ROOT / scenario.scenario_id
        if sim_ws.exists() and not keep_workspace:
            try:
                safe_rmtree(sim_ws, WORKSPACE_ROOT)
            except Exception:
                pass
        log = {
            "scenario_id": scenario.scenario_id,
            "status": "failed",
            "elapsed_s": elapsed_s,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        atomic_json(paths["log"], log)
        return log


def completed_count(registry: pd.DataFrame) -> int:
    return int(sum(checkpoint_success(str(sid)) for sid in registry["scenario_id"]))


def run_registry(registry: pd.DataFrame, workers: int, limit: int | None, keep_workspace: bool, force: bool) -> list[dict]:
    pending = []
    for row in registry.to_dict(orient="records"):
        if force or not checkpoint_success(str(row["scenario_id"])):
            pending.append(row)
    if limit is not None:
        pending = pending[: int(limit)]
    if not pending:
        return []
    results: list[dict] = []
    workers = max(1, int(workers))
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(run_one_scenario, row, keep_workspace, force): str(row["scenario_id"])
            for row in pending
        }
        for done_idx, future in enumerate(concurrent.futures.as_completed(future_to_id), start=1):
            result = future.result()
            results.append(result)
            if done_idx == 1 or done_idx % 10 == 0 or done_idx == len(pending):
                ok = sum(1 for item in results if item.get("status") == "success")
                fail = sum(1 for item in results if item.get("status") == "failed")
                elapsed = time.perf_counter() - start
                print(f"[mega] {done_idx}/{len(pending)} finished; success={ok}, failed={fail}, elapsed={elapsed:.1f}s", flush=True)
    return results


def load_logs_for_ids(scenario_ids: list[str]) -> pd.DataFrame:
    rows = []
    for scenario_id in scenario_ids:
        path = checkpoint_paths(scenario_id)["log"]
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def aggregate_for_ids(scenario_ids: list[str], cases_path: Path | None = None, fits_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = []
    fits = []
    for scenario_id in scenario_ids:
        paths = checkpoint_paths(scenario_id)
        if paths["case"].exists():
            cases.append(json.loads(paths["case"].read_text(encoding="utf-8")))
        if paths["fit"].exists():
            fits.append(pd.read_csv(paths["fit"]))
    case_frame = pd.DataFrame(cases)
    fit_frame = pd.concat(fits, ignore_index=True) if fits else pd.DataFrame()
    if cases_path is not None and not case_frame.empty:
        atomic_csv(cases_path, case_frame)
    if fits_path is not None and not fit_frame.empty:
        atomic_csv(fits_path, fit_frame)
    return case_frame, fit_frame


def aggregate_drawdown_for_ids(scenario_ids: list[str], drawdown_path: Path) -> pd.DataFrame:
    frames = []
    for scenario_id in scenario_ids:
        path = checkpoint_paths(scenario_id)["drawdown"]
        if path.exists():
            frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frame.empty:
        atomic_csv(drawdown_path, frame)
    return frame


def summarize_log_factor(series: pd.Series) -> dict:
    arr = series.to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mu": float("nan"), "sigma": float("nan"), "cov": float("nan"), "median_factor": float("nan")}
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "mu": mu,
        "sigma": sigma,
        "cov": float(math.sqrt(math.exp(sigma * sigma) - 1.0)),
        "median_factor": float(math.exp(np.median(arr))),
    }


def model_factor_cov_summary(fits: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in fits.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["response_count"] = int(group.shape[0])
        for metric, source in [
            ("T", "lnM_T"),
            ("S", "lnM_S"),
            ("capacity", "lnM_capacity"),
            ("response_time", "lnM_response_time"),
        ]:
            stats = summarize_log_factor(group[source])
            row[f"mu_lnM_{metric}"] = stats["mu"]
            row[f"sd_lnM_{metric}"] = stats["sigma"]
            row[f"cov_M_{metric}"] = stats["cov"]
            row[f"median_M_{metric}"] = stats["median_factor"]
        row["mean_response_model_factor_cov"] = float(group["cov_response_model_factor"].mean())
        row["median_sd_eta"] = float(group["sd_eta"].median())
        row["boundary_hit_fraction"] = float(group["boundary_hit"].astype(bool).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_run(registry: pd.DataFrame, scenario_ids: list[str], workers: int, summary_path: Path) -> dict:
    logs = load_logs_for_ids(scenario_ids)
    success = logs[logs["status"] == "success"].copy() if not logs.empty and "status" in logs else pd.DataFrame()
    failed = logs[logs["status"] == "failed"].copy() if not logs.empty and "status" in logs else pd.DataFrame()
    free_gb = shutil.disk_usage(PROJECT).free / (1024.0**3)
    memory = psutil.virtual_memory()
    if success.empty:
        summary = {
            "scenario_count": int(len(scenario_ids)),
            "success_count": 0,
            "failed_count": int(len(failed)),
            "feasible_for_full_10000": False,
            "reason": "No successful scenario in this run.",
        }
        atomic_json(summary_path, summary)
        return summary
    elapsed = success["elapsed_s"].astype(float)
    workspace_bytes_sample = success["workspace_bytes_before_cleanup"].astype(float)
    checkpoint_bytes = []
    for sid in success["scenario_id"].astype(str):
        paths = checkpoint_paths(sid)
        checkpoint_bytes.append(sum(path.stat().st_size for path in paths.values() if path.exists()))
    checkpoint_bytes = np.asarray(checkpoint_bytes, dtype=float)
    per_scenario_checkpoint_mb = float(np.mean(checkpoint_bytes) / (1024.0**2)) if checkpoint_bytes.size else 0.0
    per_scenario_workspace_mb = float(np.mean(workspace_bytes_sample) / (1024.0**2))
    median_runtime_s = float(elapsed.median())
    p95_runtime_s = float(elapsed.quantile(0.95))
    failure_rate = float(len(failed) / max(len(logs), 1))
    estimated_full_runtime_h = float(median_runtime_s * 10000.0 / max(workers, 1) / 3600.0)
    estimated_workspace_if_kept_gb = float(per_scenario_workspace_mb * 10000.0 / 1024.0)
    estimated_checkpoint_gb = float(per_scenario_checkpoint_mb * 10000.0 / 1024.0)
    feasible = (
        failure_rate <= 0.05
        and p95_runtime_s <= 180.0
        and estimated_checkpoint_gb + 30.0 < free_gb
        and memory.available / (1024.0**3) >= 8.0
    )
    summary = {
        "scenario_count": int(len(scenario_ids)),
        "success_count": int(len(success)),
        "failed_count": int(len(failed)),
        "failure_rate": failure_rate,
        "workers": int(workers),
        "median_elapsed_s_per_scenario": median_runtime_s,
        "p95_elapsed_s_per_scenario": p95_runtime_s,
        "mean_workspace_mb_if_kept": per_scenario_workspace_mb,
        "mean_checkpoint_mb_after_cleanup": per_scenario_checkpoint_mb,
        "estimated_full_10000_runtime_h_at_same_workers": estimated_full_runtime_h,
        "estimated_full_10000_workspace_if_kept_gb": estimated_workspace_if_kept_gb,
        "estimated_full_10000_checkpoint_gb_after_cleanup": estimated_checkpoint_gb,
        "disk_free_gb": free_gb,
        "available_ram_gb": float(memory.available / (1024.0**3)),
        "feasible_for_full_10000": bool(feasible),
        "feasibility_rule": "failure_rate <= 5%, p95 runtime <= 180 s/scenario, checkpoint estimate + 30 GB < free disk, and >= 8 GB available RAM",
    }
    if not failed.empty:
        summary["failed_scenarios"] = failed[["scenario_id", "error"]].head(10).to_dict(orient="records")
    atomic_json(summary_path, summary)
    return summary


def capacity_report() -> dict:
    ensure_dirs()
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(PROJECT)
    gpu_summary = "not checked"
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_summary = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        gpu_summary = f"unavailable: {exc}"
    report = {
        "cpu_logical": os.cpu_count(),
        "ram_total_gb": float(memory.total / (1024.0**3)),
        "ram_available_gb": float(memory.available / (1024.0**3)),
        "disk_free_gb": float(disk.free / (1024.0**3)),
        "mf6_executable": MF6_EXE,
        "gpu": gpu_summary,
        "recommended_initial_workers": min(4, max(1, (os.cpu_count() or 2) // 2)),
        "note": "GPU is recorded for capacity tracking but MODFLOW 6 and the current nonlinear pathway fits run on CPU.",
    }
    atomic_json(CAPACITY_JSON, report)
    return report


def command_registry(args: argparse.Namespace) -> None:
    ensure_dirs()
    registry = build_mega_registry(args.n, args.seed)
    atomic_csv(REGISTRY_CSV, registry)
    print(f"Wrote {len(registry)} scenarios to {REGISTRY_CSV}")


def command_pilot(args: argparse.Namespace) -> None:
    ensure_dirs()
    registry = build_mega_registry(args.n, args.seed)
    atomic_csv(REGISTRY_CSV, registry)
    per_class = max(1, int(math.ceil(args.pilot_size / len(CLASS_ORDER))))
    pilot = registry.groupby("scenario_class", sort=False).head(per_class).head(args.pilot_size).reset_index(drop=True)
    atomic_csv(PILOT_REGISTRY_CSV, pilot)
    print(f"Pilot registry: {len(pilot)} scenarios, workers={args.workers}")
    run_registry(pilot, args.workers, args.limit, args.keep_workspaces, args.force)
    scenario_ids = pilot["scenario_id"].astype(str).tolist()
    aggregate_for_ids(scenario_ids, PILOT_CASES_CSV, PILOT_FITS_CSV)
    summary = summarize_run(pilot, scenario_ids, args.workers, PILOT_SUMMARY_JSON)
    print(json.dumps(json_safe(summary), indent=2))


def command_run(args: argparse.Namespace) -> None:
    ensure_dirs()
    if REGISTRY_CSV.exists() and not args.rebuild_registry:
        registry = pd.read_csv(REGISTRY_CSV)
    else:
        registry = build_mega_registry(args.n, args.seed)
        atomic_csv(REGISTRY_CSV, registry)
    if args.n and len(registry) > args.n:
        registry = registry.head(args.n).copy()
    print(f"Full registry: {len(registry)} scenarios, completed={completed_count(registry)}, workers={args.workers}, limit={args.limit}")
    run_registry(registry, args.workers, args.limit, args.keep_workspaces, args.force)
    scenario_ids = registry["scenario_id"].astype(str).tolist()
    summary = summarize_run(registry, scenario_ids, args.workers, RUN_SUMMARY_JSON)
    print(json.dumps(json_safe(summary), indent=2))


def command_status(args: argparse.Namespace) -> None:
    ensure_dirs()
    if not REGISTRY_CSV.exists():
        print(f"No registry found at {REGISTRY_CSV}")
        return
    registry = pd.read_csv(REGISTRY_CSV)
    if RUN_SUMMARY_JSON.exists() and not args.scan:
        summary = json.loads(RUN_SUMMARY_JSON.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "registry_count": len(registry),
                    "completed_success": int(summary.get("success_count", 0)),
                    "failed": int(summary.get("failed_count", 0)),
                    "source": str(RUN_SUMMARY_JSON),
                    "scan_mode": "summary",
                },
                indent=2,
            )
        )
        return
    scenario_ids = registry["scenario_id"].astype(str).tolist()
    logs = load_logs_for_ids(scenario_ids)
    done = completed_count(registry)
    failed = 0 if logs.empty or "status" not in logs else int((logs["status"] == "failed").sum())
    print(json.dumps({"registry_count": len(registry), "completed_success": done, "failed": failed, "scan_mode": "checkpoint"}, indent=2))


def command_aggregate(args: argparse.Namespace) -> None:
    ensure_dirs()
    if not REGISTRY_CSV.exists():
        raise FileNotFoundError(f"No registry found at {REGISTRY_CSV}")
    registry = pd.read_csv(REGISTRY_CSV)
    scenario_ids = [str(sid) for sid in registry["scenario_id"] if checkpoint_success(str(sid))]
    cases, fits = aggregate_for_ids(scenario_ids, FULL_CASES_CSV, FULL_FITS_CSV)
    if args.drawdown:
        aggregate_drawdown_for_ids(scenario_ids, FULL_DRAWDOWN_CSV)
    if not fits.empty:
        fits["delta_bic"] = fits["bic"] - fits.groupby(["scenario_id", "well"])["bic"].transform("min")
        fits["quality_gate"] = (~fits["boundary_hit"].astype(bool)) & (fits["delta_bic"] <= 10.0)
        atomic_csv(FULL_FITS_CSV, fits)
        by_class = model_factor_cov_summary(fits, ["scenario_class", "pathway"])
        by_layout = model_factor_cov_summary(fits, ["scenario_class", "obs_layout", "pathway"])
        atomic_csv(FULL_COV_BY_CLASS_CSV, by_class)
        atomic_csv(FULL_COV_BY_LAYOUT_CSV, by_layout)
        gated = fits[fits["quality_gate"]].copy()
        if not gated.empty:
            atomic_csv(FULL_COV_QUALITY_GATED_CSV, model_factor_cov_summary(gated, ["scenario_class", "pathway"]))
        best_idx = fits.groupby(["scenario_id", "well"])["bic"].idxmin()
        best = fits.loc[best_idx].copy()
        best = best[~best["boundary_hit"].astype(bool)]
        if not best.empty:
            atomic_csv(FULL_COV_BIC_BEST_CSV, model_factor_cov_summary(best, ["scenario_class", "pathway"]))
    print(
        json.dumps(
            {
                "success_scenarios_aggregated": len(scenario_ids),
                "case_rows": int(cases.shape[0]),
                "fit_rows": int(fits.shape[0]),
                "cases_csv": str(FULL_CASES_CSV),
                "fits_csv": str(FULL_FITS_CSV),
                "cov_by_class_csv": str(FULL_COV_BY_CLASS_CSV),
                "cov_by_class_layout_csv": str(FULL_COV_BY_LAYOUT_CSV),
                "quality_gated_cov_by_class_csv": str(FULL_COV_QUALITY_GATED_CSV),
                "bic_best_cov_by_class_csv": str(FULL_COV_BIC_BEST_CSV),
                "drawdown_csv": str(FULL_DRAWDOWN_CSV) if args.drawdown else "not requested",
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpointed 10,000+ scenario MODFLOW 6 transformation-uncertainty benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capacity")
    cap.set_defaults(func=lambda args: print(json.dumps(json_safe(capacity_report()), indent=2)))

    reg = sub.add_parser("registry")
    reg.add_argument("--n", type=int, default=10000)
    reg.add_argument("--seed", type=int, default=20260608)
    reg.set_defaults(func=command_registry)

    pilot = sub.add_parser("pilot")
    pilot.add_argument("--n", type=int, default=10000)
    pilot.add_argument("--seed", type=int, default=20260608)
    pilot.add_argument("--pilot-size", type=int, default=60)
    pilot.add_argument("--workers", type=int, default=4)
    pilot.add_argument("--limit", type=int, default=None)
    pilot.add_argument("--keep-workspaces", action="store_true")
    pilot.add_argument("--force", action="store_true")
    pilot.set_defaults(func=command_pilot)

    run = sub.add_parser("run")
    run.add_argument("--n", type=int, default=10000)
    run.add_argument("--seed", type=int, default=20260608)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--keep-workspaces", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--rebuild-registry", action="store_true")
    run.set_defaults(func=command_run)

    status = sub.add_parser("status")
    status.add_argument("--scan", action="store_true")
    status.set_defaults(func=command_status)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--drawdown", action="store_true")
    aggregate.set_defaults(func=command_aggregate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


