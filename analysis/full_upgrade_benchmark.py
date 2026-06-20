from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import exp1, k0, k1


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT

Q_TRUE = 1.2e-4
T_TRUE = 5.0e-5
S_TRUE = 1.0e-3
R_OBS = 20.0
OFFSET = 1.0e-3
RW_M = 0.01
LAG_TAU_Q_TRUE = 100.0
LAG_TAU_S_TRUE = 1000.0
LEAKAGE_B_TRUE = 35.0
LAGUERRE_X, LAGUERRE_W = np.polynomial.laguerre.laggauss(48)
PATHWAY_ORDER = [
    "Theis schedule-aware",
    "Theis fixed-rate",
    "Hantush-Jacob leaky",
    "Cooper-Jacob late-time",
    "Lagging Darcy no-leakage",
    "Lagging Darcy with leakage",
]
FIGURE12_PATHWAYS = [
    "Theis schedule-aware",
    "Hantush-Jacob leaky",
    "Lagging Darcy no-leakage",
    "Lagging Darcy with leakage",
]


def cmc_color(colormap_name: str, sample: float) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in getattr(cmc, colormap_name)(sample))


BENCHMARK_COLORS = {
    "Theis schedule-aware": cmc_color("batlow", 0.12),
    "Theis fixed-rate": cmc_color("batlow", 0.30),
    "Hantush-Jacob leaky": cmc_color("batlow", 0.50),
    "Cooper-Jacob late-time": cmc_color("batlow", 0.70),
    "Lagging Darcy no-leakage": cmc_color("batlow", 0.78),
    "Lagging Darcy with leakage": cmc_color("batlow", 0.90),
}
PARAMETER_ERROR_COLORS = {
    "T": cmc_color("navia", 0.28),
    "S": cmc_color("navia", 0.72),
}
NEUTRAL_DARK = cmc_color("grayC", 0.18)
NEUTRAL_LIGHT = cmc_color("grayC", 0.86)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    seed: int
    split: str
    truth_model: str
    schedule: tuple[tuple[float, float], ...]
    noise_sd_m: float
    delay_s: float
    description: str
    tau_q_truth_s: float = 0.0
    tau_s_truth_s: float = 0.0
    leakage_b_truth_m: float = np.nan


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def theis_drawdown(time_s: np.ndarray, q_m3s: float, trans: float, storage: float, radius_m: float) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    out = np.zeros_like(time_s)
    positive = time_s > 0.0
    u = radius_m**2 * storage / (4.0 * trans * time_s[positive])
    out[positive] = q_m3s / (4.0 * math.pi * trans) * exp1(u)
    return out


def hantush_well_function(u: np.ndarray, beta: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    positive = u > 0.0
    if not np.any(positive):
        return out
    if beta < 1.0e-8:
        out[positive] = exp1(u[positive])
        return out
    up = u[positive]
    y = up[:, None] + LAGUERRE_X[None, :]
    integrand = np.exp(-(beta * beta) / (4.0 * y)) / y
    out[positive] = np.exp(-up) * np.sum(LAGUERRE_W[None, :] * integrand, axis=1)
    return out


def leaky_drawdown(
    time_s: np.ndarray,
    q_m3s: float,
    trans: float,
    storage: float,
    leakage_b_m: float,
    radius_m: float,
) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    out = np.zeros_like(time_s)
    positive = time_s > 0.0
    if not np.any(positive):
        return out
    u = radius_m**2 * storage / (4.0 * trans * time_s[positive])
    beta = radius_m / leakage_b_m
    out[positive] = q_m3s / (4.0 * math.pi * trans) * hantush_well_function(u, beta)
    return out


def schedule_drawdown(
    time_s: np.ndarray,
    schedule: tuple[tuple[float, float], ...],
    trans: float,
    storage: float,
    radius_m: float = R_OBS,
    delay_s: float = 0.0,
) -> np.ndarray:
    total = np.zeros_like(time_s, dtype=float)
    for start_s, delta_q in schedule:
        elapsed = np.maximum(time_s - start_s - delay_s, 0.0)
        total += theis_drawdown(elapsed, delta_q, trans, storage, radius_m)
    return total


def schedule_leaky_drawdown(
    time_s: np.ndarray,
    schedule: tuple[tuple[float, float], ...],
    trans: float,
    storage: float,
    leakage_b_m: float,
    radius_m: float = R_OBS,
) -> np.ndarray:
    total = np.zeros_like(time_s, dtype=float)
    for start_s, delta_q in schedule:
        elapsed = np.maximum(time_s - start_s, 0.0)
        total += leaky_drawdown(elapsed, delta_q, trans, storage, leakage_b_m, radius_m)
    return total


def vi(n: int, i: int) -> float:
    half = n // 2
    total = 0.0
    for k in range(max(1, int(np.floor((i + 1) / 2))), min(i, half) + 1):
        total += k**half * math.factorial(2 * k) / (
            math.factorial(half - k)
            * math.factorial(k)
            * math.factorial(k - 1)
            * math.factorial(i - k)
            * math.factorial(2 * k - i)
        )
    return float((-1) ** (i + half) * total)


STEHFEST_WEIGHTS = np.array([vi(6, i) for i in range(1, 7)], dtype=float)


def stehfest(f, t: float) -> float:
    log2_t = np.log(2.0) / t
    return float(log2_t * sum(STEHFEST_WEIGHTS[i - 1] * f(i * log2_t) for i in range(1, 7)))


def lagging_darcy_drawdown(
    time_s: float,
    q_m3s: float,
    trans: float,
    storage: float,
    tau_q: float,
    tau_s: float,
    radius_m: float = R_OBS,
    well_radius_m: float = RW_M,
) -> float:
    if time_s <= 0.0:
        return 0.0

    def f(p: float) -> float:
        with np.errstate(all="ignore"):
            root = np.sqrt(trans + p * trans * tau_s)
            lag_root = np.sqrt(1.0 + p * tau_q)
            arg_r = (np.sqrt(p) * radius_m * np.sqrt(storage) * lag_root) / root
            arg_w = (np.sqrt(p) * well_radius_m * np.sqrt(storage) * lag_root) / root
            numerator = q_m3s * root * k0(arg_r)
            denominator = (
                p**2 * np.pi * well_radius_m**2 * root * k0(arg_w)
                + 2.0 * p**1.5 * np.pi * well_radius_m * np.sqrt(storage) * trans * lag_root * k1(arg_w)
            )
            if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
                return np.nan
            return numerator / denominator

    value = np.real(stehfest(f, float(time_s)))
    if not np.isfinite(value):
        return np.nan
    return float(value)


def schedule_lagging_darcy_drawdown(
    time_s: np.ndarray,
    schedule: tuple[tuple[float, float], ...],
    trans: float,
    storage: float,
    tau_q: float,
    tau_s: float,
    radius_m: float = R_OBS,
) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    total = np.zeros_like(time_s, dtype=float)
    for start_s, delta_q in schedule:
        elapsed = np.maximum(time_s - start_s, 0.0)
        total += np.array(
            [
                lagging_darcy_drawdown(float(t), delta_q, trans, storage, tau_q, tau_s, radius_m)
                for t in elapsed
            ],
            dtype=float,
        )
    return total


def lagging_leaky_laplace(
    p: float,
    q_m3s: float,
    trans: float,
    storage: float,
    tau_q: float,
    tau_s: float,
    leakage_b_m: float,
    radius_m: float = R_OBS,
    well_radius_m: float = RW_M,
) -> float:
    with np.errstate(all="ignore"):
        memory_ratio = (1.0 + p * tau_q) / (1.0 + p * tau_s)
        alpha_sq = (p * storage / trans + 1.0 / (leakage_b_m * leakage_b_m)) * memory_ratio
        alpha = np.sqrt(alpha_sq)
        arg_r = alpha * radius_m
        arg_w = alpha * well_radius_m
        numerator = q_m3s * k0(arg_r)
        denominator = (
            np.pi * well_radius_m**2 * p**2 * k0(arg_w)
            + 2.0 * np.pi * well_radius_m * trans * p * alpha * k1(arg_w)
        )
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
            return np.nan
        return float(numerator / denominator)


def lagging_leaky_drawdown(
    time_s: float,
    q_m3s: float,
    trans: float,
    storage: float,
    tau_q: float,
    tau_s: float,
    leakage_b_m: float,
    radius_m: float = R_OBS,
    well_radius_m: float = RW_M,
) -> float:
    if time_s <= 0.0:
        return 0.0
    value = np.real(
        stehfest(
            lambda p: lagging_leaky_laplace(
                p,
                q_m3s,
                trans,
                storage,
                tau_q,
                tau_s,
                leakage_b_m,
                radius_m,
                well_radius_m,
            ),
            float(time_s),
        )
    )
    if not np.isfinite(value):
        return np.nan
    return float(value)


def schedule_lagging_leaky_drawdown(
    time_s: np.ndarray,
    schedule: tuple[tuple[float, float], ...],
    trans: float,
    storage: float,
    tau_q: float,
    tau_s: float,
    leakage_b_m: float,
    radius_m: float = R_OBS,
) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=float)
    total = np.zeros_like(time_s, dtype=float)
    for start_s, delta_q in schedule:
        elapsed = np.maximum(time_s - start_s, 0.0)
        total += np.array(
            [
                lagging_leaky_drawdown(float(t), delta_q, trans, storage, tau_q, tau_s, leakage_b_m, radius_m)
                for t in elapsed
            ],
            dtype=float,
        )
    return total


def log_response_residual(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    invalid = (~np.isfinite(pred)) | (pred + OFFSET <= 0.0)
    residual = np.log((obs + OFFSET) / (np.where(invalid, np.nan, pred) + OFFSET))
    residual[~np.isfinite(residual)] = 1.0e6
    return residual


def aic_bic(residual: np.ndarray, k: int) -> tuple[float, float]:
    clean = residual[np.isfinite(residual)]
    n = max(clean.size, 1)
    rss = max(float(np.sum(clean**2)), 1.0e-18)
    sigma2 = rss / n
    return float(n * np.log(sigma2) + 2 * k), float(n * np.log(sigma2) + k * np.log(n))


def fit_schedule_theis(time_s: np.ndarray, obs: np.ndarray, schedule: tuple[tuple[float, float], ...]) -> dict:
    bounds = (np.log([1.0e-7, 1.0e-6]), np.log([1.0e-2, 1.0e-1]))

    def residual(values: np.ndarray) -> np.ndarray:
        trans, storage = np.exp(values)
        pred = schedule_drawdown(time_s, schedule, trans, storage)
        return log_response_residual(obs, pred)

    result = least_squares(residual, np.log([T_TRUE, S_TRUE]), bounds=bounds, max_nfev=400)
    trans, storage = np.exp(result.x)
    pred = schedule_drawdown(time_s, schedule, trans, storage)
    return make_fit("Theis schedule-aware", trans, storage, np.nan, np.nan, pred, obs, result.x, bounds, 2)


def fit_constant_theis(time_s: np.ndarray, obs: np.ndarray, q_assumed: float = Q_TRUE) -> dict:
    schedule = ((0.0, q_assumed),)
    bounds = (np.log([1.0e-7, 1.0e-6]), np.log([1.0e-2, 1.0e-1]))

    def residual(values: np.ndarray) -> np.ndarray:
        trans, storage = np.exp(values)
        pred = schedule_drawdown(time_s, schedule, trans, storage)
        return log_response_residual(obs, pred)

    result = least_squares(residual, np.log([T_TRUE, S_TRUE]), bounds=bounds, max_nfev=400)
    trans, storage = np.exp(result.x)
    pred = schedule_drawdown(time_s, schedule, trans, storage)
    return make_fit("Theis fixed-rate", trans, storage, np.nan, np.nan, pred, obs, result.x, bounds, 2)


def fit_lagging_darcy(time_s: np.ndarray, obs: np.ndarray, schedule: tuple[tuple[float, float], ...]) -> dict:
    bounds = (np.log([1.0e-7, 1.0e-6, 1.0e-4, 1.0e-4]), np.log([1.0e-2, 1.0e-1, 1.0e5, 1.0e5]))

    def residual(values: np.ndarray) -> np.ndarray:
        trans, storage, tau_q, tau_s = np.exp(values)
        pred = schedule_lagging_darcy_drawdown(time_s, schedule, trans, storage, tau_q, tau_s)
        return log_response_residual(obs, pred)

    initial = np.log([T_TRUE, S_TRUE, LAG_TAU_Q_TRUE, LAG_TAU_S_TRUE])
    result = least_squares(residual, initial, bounds=bounds, max_nfev=260)
    trans, storage, tau_q, tau_s = np.exp(result.x)
    pred = schedule_lagging_darcy_drawdown(time_s, schedule, trans, storage, tau_q, tau_s)
    return make_fit("Lagging Darcy no-leakage", trans, storage, tau_q, tau_s, pred, obs, result.x, bounds, 4)


def fit_lagging_leaky(time_s: np.ndarray, obs: np.ndarray, schedule: tuple[tuple[float, float], ...]) -> dict:
    bounds = (
        np.log([1.0e-7, 1.0e-6, 1.0e-4, 1.0e-4, 1.0]),
        np.log([1.0e-2, 1.0e-1, 1.0e5, 1.0e5, 1.0e5]),
    )

    def residual(values: np.ndarray) -> np.ndarray:
        trans, storage, tau_q, tau_s, leakage_b = np.exp(values)
        pred = schedule_lagging_leaky_drawdown(time_s, schedule, trans, storage, tau_q, tau_s, leakage_b)
        return log_response_residual(obs, pred)

    starts = [
        np.log([T_TRUE, S_TRUE, LAG_TAU_Q_TRUE, LAG_TAU_S_TRUE, LEAKAGE_B_TRUE]),
        np.log([T_TRUE, S_TRUE, 1.0e-4, 1.0e-4, LEAKAGE_B_TRUE]),
        np.log([T_TRUE, S_TRUE, LAG_TAU_Q_TRUE, LAG_TAU_S_TRUE, 1.0e3]),
        np.log([T_TRUE * 0.9, S_TRUE * 1.1, 10.0, 100.0, 1.0e3]),
    ]
    best = None
    for initial in starts:
        result = least_squares(residual, initial, bounds=bounds, max_nfev=160)
        score = float(np.sum(result.fun * result.fun))
        if best is None or score < best[0]:
            best = (score, result)
    assert best is not None
    result = best[1]
    trans, storage, tau_q, tau_s, leakage_b = np.exp(result.x)
    pred = schedule_lagging_leaky_drawdown(time_s, schedule, trans, storage, tau_q, tau_s, leakage_b)
    return make_fit(
        "Lagging Darcy with leakage",
        trans,
        storage,
        tau_q,
        tau_s,
        pred,
        obs,
        result.x,
        bounds,
        5,
        leakage_b_m=leakage_b,
    )


def fit_leaky_hantush(time_s: np.ndarray, obs: np.ndarray, schedule: tuple[tuple[float, float], ...]) -> dict:
    bounds = (np.log([1.0e-7, 1.0e-6, 1.0]), np.log([1.0e-2, 1.0e-1, 1.0e5]))

    def residual(values: np.ndarray) -> np.ndarray:
        trans, storage, leakage_b = np.exp(values)
        pred = schedule_leaky_drawdown(time_s, schedule, trans, storage, leakage_b)
        return log_response_residual(obs, pred)

    starts = [
        np.log([T_TRUE, S_TRUE, LEAKAGE_B_TRUE]),
        np.log([T_TRUE, S_TRUE, 1.0e3]),
        np.log([T_TRUE * 0.8, S_TRUE * 1.2, 20.0]),
        np.log([T_TRUE * 1.2, S_TRUE * 0.8, 80.0]),
    ]
    best = None
    for initial in starts:
        result = least_squares(residual, initial, bounds=bounds, max_nfev=220)
        score = float(np.sum(result.fun * result.fun))
        if best is None or score < best[0]:
            best = (score, result)
    assert best is not None
    result = best[1]
    trans, storage, leakage_b = np.exp(result.x)
    pred = schedule_leaky_drawdown(time_s, schedule, trans, storage, leakage_b)
    return make_fit(
        "Hantush-Jacob leaky",
        trans,
        storage,
        np.nan,
        np.nan,
        pred,
        obs,
        result.x,
        bounds,
        3,
        leakage_b_m=leakage_b,
    )


def fit_cooper_jacob(time_s: np.ndarray, obs: np.ndarray, q_assumed: float = Q_TRUE) -> dict:
    late = time_s >= np.quantile(time_s, 0.55)
    x = np.log(time_s[late])
    y = obs[late]
    slope, intercept = np.polyfit(x, y, 1)
    unstable = bool(slope <= 1.0e-12)
    safe_slope = max(slope, 1.0e-12)
    trans = max(q_assumed / (4.0 * math.pi * safe_slope), 1.0e-9)
    exponent = intercept / safe_slope + 0.5772156649
    if exponent > 50.0 or exponent < -50.0:
        unstable = True
    storage = max(4.0 * trans / (R_OBS**2 * math.exp(float(np.clip(exponent, -50.0, 50.0)))), 1.0e-9)
    pred = schedule_drawdown(time_s, ((0.0, q_assumed),), trans, storage)
    fit = make_fit("Cooper-Jacob late-time", trans, storage, np.nan, np.nan, pred, obs, np.log([trans, storage]), None, 2)
    if unstable:
        fit["boundary_hit"] = True
        fit["reliability_class"] = "D"
    return fit


def make_fit(
    pathway: str,
    trans: float,
    storage: float,
    tau_q_s: float,
    tau_s_s: float,
    pred: np.ndarray,
    obs: np.ndarray,
    log_values: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    parameter_count: int,
    leakage_b_m: float = np.nan,
) -> dict:
    residual = log_response_residual(obs, pred)
    aic, bic = aic_bic(residual, parameter_count)
    boundary_hit = False
    if bounds is not None:
        lower, upper = bounds
        boundary_hit = bool(np.any(log_values <= lower + 1.0e-3) or np.any(log_values >= upper - 1.0e-3))
    delta_log_t = float(np.log(trans / T_TRUE))
    delta_log_s = float(np.log(storage / S_TRUE))
    return {
        "pathway": pathway,
        "T_fit_m2_s": float(trans),
        "S_fit": float(storage),
        "tau_q_fit_s": None if not np.isfinite(tau_q_s) else float(tau_q_s),
        "tau_s_fit_s": None if not np.isfinite(tau_s_s) else float(tau_s_s),
        "leakage_B_fit_m": None if not np.isfinite(leakage_b_m) else float(leakage_b_m),
        "delta_logT": delta_log_t,
        "delta_logS": delta_log_s,
        "sd_epsilon": float(np.std(residual, ddof=1)),
        "bias_eta": float(np.mean(residual)),
        "rmse_m": float(np.sqrt(np.mean((obs - pred) ** 2))),
        "aic": aic,
        "bic": bic,
        "parameter_count": parameter_count,
        "boundary_hit": boundary_hit,
        "reliability_class": reliability_class(delta_log_t, delta_log_s, float(np.std(residual, ddof=1)), boundary_hit),
    }


def reliability_class(delta_log_t: float, delta_log_s: float, sd_epsilon: float, boundary_hit: bool) -> str:
    max_error = max(abs(delta_log_t), abs(delta_log_s))
    if max_error > 1.0 or sd_epsilon > 3.0:
        return "D"
    if max_error < 0.05 and sd_epsilon < 0.05 and not boundary_hit:
        return "A"
    if max_error < 0.15 and sd_epsilon < 0.80 and not boundary_hit:
        return "B"
    if max_error < 0.75 and sd_epsilon < 1.20:
        return "C"
    return "C"


def build_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    pumping = ((0.0, Q_TRUE),)
    step_rate = ((0.0, Q_TRUE), (2.0 * 3600.0, -0.45 * Q_TRUE), (5.0 * 3600.0, 0.25 * Q_TRUE))
    recovery = ((0.0, Q_TRUE), (6.0 * 3600.0, -Q_TRUE))
    families = [
        ("theis_clean", "Theis homogeneous truth", pumping, 0.0, 0.0, 1),
        ("leaky_clean", "Hantush-Jacob clean leaky truth", pumping, 0.0, 0.0, 1),
        ("theis_noise", "Theis plus measurement noise", pumping, 0.01, 0.0, 12),
        ("leaky_truth", "Hantush-Jacob leaky truth", pumping, 0.006, 0.0, 12),
        ("step_rate", "Step-rate pumping schedule", step_rate, 0.006, 0.0, 12),
        ("recovery", "Pumping plus recovery schedule", recovery, 0.006, 0.0, 12),
        ("lagging_darcy_truth", "Lagging Darcy truth", pumping, 0.006, 0.0, 12),
        ("lagging_leaky_truth", "Lagging Darcy leaky truth", pumping, 0.006, 0.0, 12),
    ]
    for family, truth_model, schedule, noise, delay, count in families:
        for seed in range(count):
            split = "verification" if family == "theis_clean" else ("calibration" if seed < count // 2 else "holdout")
            scenarios.append(
                Scenario(
                    scenario_id=f"{family}_{seed:02d}",
                    family=family,
                    seed=seed,
                    split=split,
                    truth_model=truth_model,
                    schedule=schedule,
                    noise_sd_m=noise,
                    delay_s=delay,
                    description=truth_model,
                    tau_q_truth_s=LAG_TAU_Q_TRUE if family in {"lagging_darcy_truth", "lagging_leaky_truth"} else 0.0,
                    tau_s_truth_s=LAG_TAU_S_TRUE if family in {"lagging_darcy_truth", "lagging_leaky_truth"} else 0.0,
                    leakage_b_truth_m=LEAKAGE_B_TRUE if family in {"leaky_clean", "leaky_truth", "lagging_leaky_truth"} else np.nan,
                )
            )
    return scenarios


def simulate_scenario(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(scenario.seed + 20260604)
    if scenario.family == "recovery":
        time_s = np.logspace(np.log10(120.0), np.log10(18.0 * 3600.0), 58)
    else:
        time_s = np.logspace(np.log10(120.0), np.log10(10.0 * 3600.0), 48)
    if scenario.family == "lagging_darcy_truth":
        truth = schedule_lagging_darcy_drawdown(
            time_s,
            scenario.schedule,
            T_TRUE,
            S_TRUE,
            scenario.tau_q_truth_s,
            scenario.tau_s_truth_s,
        )
    elif scenario.family == "lagging_leaky_truth":
        truth = schedule_lagging_leaky_drawdown(
            time_s,
            scenario.schedule,
            T_TRUE,
            S_TRUE,
            scenario.tau_q_truth_s,
            scenario.tau_s_truth_s,
            scenario.leakage_b_truth_m,
        )
    elif scenario.family in {"leaky_clean", "leaky_truth"}:
        truth = schedule_leaky_drawdown(
            time_s,
            scenario.schedule,
            T_TRUE,
            S_TRUE,
            scenario.leakage_b_truth_m,
        )
    else:
        truth = schedule_drawdown(time_s, scenario.schedule, T_TRUE, S_TRUE, delay_s=scenario.delay_s)
    noisy = np.maximum(truth + rng.normal(0.0, scenario.noise_sd_m, size=truth.size), 0.0)
    return time_s, noisy


def fit_all_scenarios(scenarios: list[Scenario]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    case_rows = []
    for scenario in scenarios:
        time_s, obs = simulate_scenario(scenario)
        fits = [
            fit_constant_theis(time_s, obs, q_assumed=Q_TRUE),
            fit_schedule_theis(time_s, obs, scenario.schedule),
            fit_leaky_hantush(time_s, obs, scenario.schedule),
            fit_cooper_jacob(time_s, obs, q_assumed=Q_TRUE),
            fit_lagging_darcy(time_s, obs, scenario.schedule),
            fit_lagging_leaky(time_s, obs, scenario.schedule),
        ]
        case_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "family": scenario.family,
                "split": scenario.split,
                "truth_model": scenario.truth_model,
                "noise_sd_m": scenario.noise_sd_m,
                "delay_truth_s": scenario.delay_s,
                "tau_q_truth_s": scenario.tau_q_truth_s,
                "tau_s_truth_s": scenario.tau_s_truth_s,
                "leakage_b_truth_m": None if not np.isfinite(scenario.leakage_b_truth_m) else scenario.leakage_b_truth_m,
                "observation_count": time_s.size,
                "max_observed_drawdown_m": float(np.max(obs)),
            }
        )
        for fit in fits:
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "family": scenario.family,
                    "split": scenario.split,
                    "truth_model": scenario.truth_model,
                    "truth_T_m2_s": T_TRUE,
                    "truth_S": S_TRUE,
                    "truth_delay_s": scenario.delay_s,
                    "truth_tau_q_s": scenario.tau_q_truth_s,
                    "truth_tau_s_s": scenario.tau_s_truth_s,
                    "truth_leakage_B_m": scenario.leakage_b_truth_m,
                    **fit,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(case_rows)


def coverage_table(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pathway in sorted(errors["pathway"].unique()):
        calibration = errors[(errors["pathway"] == pathway) & (errors["split"] == "calibration")]
        holdout = errors[(errors["pathway"] == pathway) & (errors["split"] == "holdout")]
        if calibration.empty or holdout.empty:
            continue
        for parameter in ["delta_logT", "delta_logS"]:
            lo, hi = np.quantile(calibration[parameter], [0.05, 0.95])
            covered = ((holdout[parameter] >= lo) & (holdout[parameter] <= hi)).mean()
            rows.append(
                {
                    "pathway": pathway,
                    "parameter": parameter.replace("delta_log", ""),
                    "calibration_p05": float(lo),
                    "calibration_p95": float(hi),
                    "holdout_coverage_fraction": float(covered),
                    "holdout_count": int(holdout.shape[0]),
                    "target_coverage": "0.80-0.90 for P05-P95 diagnostic intervals",
                }
            )
    return pd.DataFrame(rows)


def family_diagnostics(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    class_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    for (family, truth_model, pathway), group in errors.groupby(["family", "truth_model", "pathway"]):
        counts = group["reliability_class"].value_counts().to_dict()
        worst = max(group["reliability_class"], key=lambda item: class_order[item])
        rows.append(
            {
                "family": family,
                "truth_model": truth_model,
                "pathway": pathway,
                "scenario_count": int(group.shape[0]),
                "median_abs_delta_logT": float(group["delta_logT"].abs().median()),
                "median_abs_delta_logS": float(group["delta_logS"].abs().median()),
                "median_sd_epsilon": float(group["sd_epsilon"].median()),
                "class_A_count": int(counts.get("A", 0)),
                "class_B_count": int(counts.get("B", 0)),
                "class_C_count": int(counts.get("C", 0)),
                "class_D_count": int(counts.get("D", 0)),
                "boundary_hit_count": int(group["boundary_hit"].sum()),
                "worst_class": worst,
            }
        )
    order = {pathway: idx for idx, pathway in enumerate(PATHWAY_ORDER)}
    out = pd.DataFrame(rows)
    out["pathway_order"] = out["pathway"].map(order)
    return out.sort_values(["family", "pathway_order"]).drop(columns=["pathway_order"]).reset_index(drop=True)


def pathway_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Theis fixed-rate", "implemented", "Confined homogeneous analytical baseline fit with constant pumping rate."],
            ["Theis schedule-aware", "implemented", "Theis superposition pathway using the prescribed step or recovery schedule."],
            ["Hantush-Jacob leaky", "implemented", "Conventional leaky confined aquifer pathway with T, S, and leakage factor B."],
            ["Cooper-Jacob late-time", "implemented", "Late-time straight-line approximation, used as a conventional screening pathway."],
            ["Lagging Darcy no-leakage", "implemented", "Lagging Darcy no-leakage limit with T, S, tau_q, and tau_s estimated from the response."],
            ["Lagging Darcy with leakage", "implemented", "General Lin-Yeh lagging leaky pathway with T, S, tau_q, tau_s, and leakage factor B."],
            ["Delayed-yield or unconfined drainage", "conceptual", "Conventional pathway to implement before design-grade claims."],
            ["Wellbore storage/skin", "conceptual", "Required to stress-test early-time response interpretation."],
            ["Generalized radial flow", "conceptual", "Required to test non-integer or fractured flow dimension ambiguity."],
            ["Dual-porosity/fracture-matrix", "conceptual", "Required to test exchange-driven temporal memory against lagging diagnostics."],
            ["Numerical inverse model", "conceptual", "Required for heterogeneous known-truth fields and support-scale targets."],
        ],
        columns=["pathway", "implementation_status", "role"],
    )


def uncertainty_source_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["inherent variability", "True spatial, stratigraphic, or fracture-network variability.", "Field T/S spread and declared synthetic support targets.", "Partly diagnosed; not independently isolated in field data and not represented by heterogeneous synthetic fields."],
            ["measurement error", "Water-level sensor noise, baseline correction, timing, and distance uncertainty.", "Noise perturbation benchmark and Lovelock detection threshold.", "Synthetic perturbation implemented; primary-site instrument budget still unavailable."],
            ["pumping schedule/preprocessing error", "Error from rate simplification, recovery handling, or environmental correction.", "Step-rate and recovery superposition benchmark.", "Primary variable-rate reconstruction still limited by available retained inputs."],
            ["transformation uncertainty", "Residual scatter introduced when drawdown is converted to apparent parameters.", "Log response factor, SD_epsilon, Delta_logT, Delta_logS in synthetic cases.", "Field cases provide residual scatter but no truth error."],
            ["model-form uncertainty", "Mismatch among Theis, Hantush-Jacob leaky, late-time, schedule-aware, lagging Darcy no-leakage, lagging Darcy with leakage, and future conventional pathways.", "AIC/BIC, pathway errors, reliability classes.", "Delayed-yield, wellbore-storage, GRF, and dual-porosity pathways remain conceptual here."],
            ["statistical/identifiability uncertainty", "Finite observations, parameter tradeoff, boundary hits, and prior dependence.", "Local MAP correlations, synthetic boundary-hit flags, reliability downgrades.", "Full posterior/profile-likelihood analysis remains a future expansion."],
            ["support-scale uncertainty", "Difference between observation support and target hydraulic-property support.", "Pathway definitions and synthetic support target declaration.", "No dense independent field support target exists."],
            ["decision uncertainty", "Effect of transformation intervals on engineering margins.", "Diagnostic drawdown threshold exceedance and interval width.", "Decision endpoint is diagnostic rather than project-specific design."],
        ],
        columns=["component", "definition", "evidence_in_full_upgrade", "remaining_limitation"],
    )


def reliability_outputs(errors: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    class_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    for pathway, group in errors.groupby("pathway"):
        counts = group["reliability_class"].value_counts().to_dict()
        worst = max(group["reliability_class"], key=lambda item: class_order[item])
        implemented = "implemented"
        reason = "Synthetic benchmark reliability based on Delta_logT, Delta_logS, SD_epsilon, and boundary hits."
        if pathway == "Lagging Darcy no-leakage":
            reason += " Tau parameters remain diagnostic unless independently constrained by additional evidence."
        rows.append(
            {
                "pathway": pathway,
                "implementation_status": implemented,
                "class_A_count": int(counts.get("A", 0)),
                "class_B_count": int(counts.get("B", 0)),
                "class_C_count": int(counts.get("C", 0)),
                "class_D_count": int(counts.get("D", 0)),
                "worst_class": worst,
                "interpretation": reason,
            }
        )
    field_rows = [
        {
            "pathway": "Field lagging Darcy response diagnostic",
            "implementation_status": "field diagnostic",
            "class_A_count": 0,
            "class_B_count": 1,
            "class_C_count": 1,
            "class_D_count": 0,
            "worst_class": "C",
            "interpretation": "Response scatter improves strongly, but spatial transfer and lag-parameter identifiability remain mixed.",
        },
        {
            "pathway": "Field lagging time parameters",
            "implementation_status": "field diagnostic",
            "class_A_count": 0,
            "class_B_count": 0,
            "class_C_count": 1,
            "class_D_count": 0,
            "worst_class": "C",
            "interpretation": "Local MAP tradeoff and weak priors require interpreting tau parameters as diagnostic response features.",
        },
    ]
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(field_rows)], ignore_index=True)


def decision_consequence(errors: pd.DataFrame) -> pd.DataFrame:
    threshold_m = 0.45
    base_pred = theis_drawdown(np.array([8.0 * 3600.0]), Q_TRUE, T_TRUE, S_TRUE, R_OBS)[0]
    rows = []
    for pathway, group in errors.groupby("pathway"):
        calibration = group[group["split"] == "calibration"]
        if calibration.empty:
            calibration = group
        factor_p05, factor_p95 = np.quantile(np.exp(calibration["delta_logT"]), [0.05, 0.95])
        lower_t = T_TRUE * factor_p05
        upper_t = T_TRUE * factor_p95
        pred_low = theis_drawdown(np.array([8.0 * 3600.0]), Q_TRUE, upper_t, S_TRUE, R_OBS)[0]
        pred_high = theis_drawdown(np.array([8.0 * 3600.0]), Q_TRUE, lower_t, S_TRUE, R_OBS)[0]
        rows.append(
            {
                "decision_endpoint": "diagnostic drawdown threshold at 8 h",
                "pathway": pathway,
                "threshold_m": threshold_m,
                "truth_drawdown_m": float(base_pred),
                "p05_based_drawdown_m": float(pred_low),
                "p95_based_drawdown_m": float(pred_high),
                "interval_width_m": float(abs(pred_high - pred_low)),
                "deterministic_exceeds_threshold": bool(base_pred > threshold_m),
                "interval_crosses_threshold": bool(pred_low <= threshold_m <= pred_high),
                "interpretation": "Diagnostic consequence of transformation-factor spread; not a project-specific dewatering design.",
            }
        )
    return pd.DataFrame(rows)


def plot_benchmark(errors: pd.DataFrame, coverage: pd.DataFrame, decision: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    short = {
        "Theis fixed-rate": "Theis fixed Q",
        "Theis schedule-aware": "Theis",
        "Cooper-Jacob late-time": "Cooper-Jacob",
        "Hantush-Jacob leaky": "Hantush leaky",
        "Lagging Darcy no-leakage": "Lagging no leak",
        "Lagging Darcy with leakage": "Lagging leaky",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.85))
    fig.subplots_adjust(left=0.26, right=0.98, top=0.92, bottom=0.17, hspace=0.62, wspace=0.66)

    pathways = [pathway for pathway in FIGURE12_PATHWAYS if pathway in set(errors["pathway"])]
    clean = errors[errors["scenario_id"] == "theis_clean_00"].set_index("pathway").loc[pathways]
    ax = axes[0, 0]
    x = np.arange(len(pathways))
    width = 0.36
    floor = 1.0e-5
    delta_t = np.maximum(clean["delta_logT"].abs().to_numpy(), floor)
    delta_s = np.maximum(clean["delta_logS"].abs().to_numpy(), floor)
    ax.bar(x - width / 2, delta_t, width, color=PARAMETER_ERROR_COLORS["T"], label=r"$|\Delta \log T|$")
    ax.bar(x + width / 2, delta_s, width, color=PARAMETER_ERROR_COLORS["S"], label=r"$|\Delta \log S|$")
    ax.set_yscale("log")
    ax.set_ylim(floor, max(delta_t.max(), delta_s.max()) * 2.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short[pathway] for pathway in pathways], rotation=18, ha="right")
    ax.set_ylabel("absolute parameter error", labelpad=8)
    ax.set_title("(a) clean known-answer error", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2, bbox_to_anchor=(-0.02, 1.02))

    ax = axes[0, 1]
    summary = errors.groupby("pathway", as_index=False)["sd_epsilon"].median().set_index("pathway").loc[pathways].reset_index()
    ax.barh([short[p] for p in summary["pathway"]], summary["sd_epsilon"], color=[BENCHMARK_COLORS[p] for p in summary["pathway"]])
    ax.set_xlabel(r"median $SD_\epsilon$")
    ax.set_title("(b) response scatter", loc="left", fontweight="bold")

    ax = axes[1, 0]
    cov_t = coverage[coverage["parameter"] == "T"].set_index("pathway").loc[pathways].reset_index()
    ax.barh([short[p] for p in cov_t["pathway"]], cov_t["holdout_coverage_fraction"], color=[BENCHMARK_COLORS[p] for p in cov_t["pathway"]])
    ax.axvspan(0.8, 0.9, color=NEUTRAL_LIGHT, zorder=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("holdout coverage")
    ax.set_title("(c) correction interval coverage", loc="left", fontweight="bold")

    ax = axes[1, 1]
    decision_ordered = decision.set_index("pathway").loc[pathways].reset_index()
    ax.barh([short[p] for p in decision_ordered["pathway"]], decision_ordered["interval_width_m"], color=[BENCHMARK_COLORS[p] for p in decision_ordered["pathway"]])
    ax.set_xlabel("drawdown interval width (m)")
    ax.set_title("(d) diagnostic decision consequence", loc="left", fontweight="bold")

    for base in [FIG_DIR, FIG_DIR / "figures"]:
        base.mkdir(exist_ok=True)
        fig.savefig(base / "fig12_synthetic_benchmark_verification.pdf", bbox_inches="tight")
        fig.savefig(base / "fig12_synthetic_benchmark_verification.png", dpi=450, bbox_inches="tight")
        fig.savefig(base / "fig12_synthetic_benchmark_verification.svg", bbox_inches="tight")
    plt.close(fig)


def write_outputs() -> None:
    ensure_dirs()
    scenarios = build_scenarios()
    errors, cases = fit_all_scenarios(scenarios)
    coverage = coverage_table(errors)
    family = family_diagnostics(errors)
    definitions = pathway_definitions()
    uncertainty = uncertainty_source_matrix()
    reliability = reliability_outputs(errors, coverage)
    decision = decision_consequence(errors)

    cases.to_csv(TABLE_DIR / "synthetic_benchmark_cases.csv", index=False)
    errors.to_csv(TABLE_DIR / "pathway_transformation_error.csv", index=False)
    coverage.to_csv(TABLE_DIR / "synthetic_holdout_coverage.csv", index=False)
    family.to_csv(TABLE_DIR / "pathway_family_diagnostics.csv", index=False)
    definitions.to_csv(TABLE_DIR / "pathway_definitions.csv", index=False)
    uncertainty.to_csv(TABLE_DIR / "phoon_uncertainty_source_matrix_full.csv", index=False)
    reliability.to_csv(TABLE_DIR / "reliability_classification.csv", index=False)
    decision.to_csv(TABLE_DIR / "decision_consequence_summary_full.csv", index=False)
    plot_benchmark(errors, coverage, decision)

    summary = {
        "synthetic_scenarios": int(cases.shape[0]),
        "pathway_fits": int(errors.shape[0]),
        "implemented_pathways": sorted(errors["pathway"].unique().tolist()),
        "nonimplemented_pathways": definitions.loc[
            definitions["implementation_status"] != "implemented", "pathway"
        ].tolist(),
        "median_abs_delta_logT_by_pathway": {
            pathway: float(group["delta_logT"].abs().median()) for pathway, group in errors.groupby("pathway")
        },
        "median_sd_epsilon_by_pathway": {
            pathway: float(group["sd_epsilon"].median()) for pathway, group in errors.groupby("pathway")
        },
        "holdout_coverage": coverage.to_dict(orient="records"),
        "family_diagnostics": family.to_dict(orient="records"),
        "claim_boundary": (
            "The synthetic layer verifies known-answer and perturbation behavior for implemented analytical pathways. "
            "It calibrates diagnostic transformation-error factors for these simple cases only. "
            "It now includes a lagging-with-leakage analytical stress test, but it does not yet provide design-grade coverage for delayed-yield, wellbore-storage, generalized-radial-flow, "
            "dual-porosity, or full numerical heterogeneous pathways."
        ),
    }
    (OUTPUT_DIR / "full_upgrade_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    write_outputs()
