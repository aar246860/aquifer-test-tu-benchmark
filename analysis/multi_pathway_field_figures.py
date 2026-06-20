from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import exp1, k0, k1

import cmcrameri.cm as cmc


PROJECT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT / "tables"
FIG_DIR = PROJECT / "figures"
OUT_DIR = PROJECT / "outputs"

Q_M3S = 1.2e-4
RW_M = 0.01
LOG_RESIDUAL_OFFSET_M = 1.0e-3
WELL_ORDER = [
    "MW-13",
    "MW-12",
    "MW-10",
    "MW-08",
    "MW-11",
    "MW-05",
    "MW-06",
    "MW-07",
    "MW-03",
    "MW-04",
    "MW-02",
    "MW-01",
]

PATHWAYS = [
    "Theis confined",
    "Hantush-Jacob leaky",
    "Lagging Darcy no-leakage",
    "Lagging Darcy with leakage",
]

SHORT_LABEL = {
    "Theis confined": "Theis",
    "Hantush-Jacob leaky": "Hantush leaky",
    "Lagging Darcy no-leakage": "Lagging\nwithout leakage",
    "Lagging Darcy with leakage": "Lagging leaky",
}

CMC_DISTANCE_COLORMAP = "batlow"
CMC_HEATMAP_COLORMAP = "lipari"
CMC_LAG_TIME_COLORMAP = "navia"
CMC_NEUTRAL_COLORMAP = "grayC"
CMC_PATHWAY_COLOR_SOURCES = {
    "Theis confined": ("batlow", 0.12),
    "Hantush-Jacob leaky": ("batlow", 0.38),
    "Lagging Darcy no-leakage": ("batlow", 0.66),
    "Lagging Darcy with leakage": ("batlow", 0.90),
}
LAG_TIME_COLOR_KEYS = ("no_leak_tau_q", "no_leak_tau_s", "leaky_tau_q", "leaky_tau_s")
LAG_TIME_COLOR_SAMPLES = (0.12, 0.38, 0.62, 0.84)


def get_cmc_colormap(name: str) -> Colormap:
    """Resolve a Scientific Colour Map from cmcrameri by package name."""
    cmap = getattr(cmc, name, None)
    if cmap is None:
        raise ValueError(f"Unknown cmcrameri colormap: {name}")
    return cmap


def sample_cmc_colors(colormap_name: str, labels: list[str] | tuple[str, ...], samples: tuple[float, ...]) -> dict[str, tuple[float, float, float, float]]:
    if len(labels) != len(samples):
        raise ValueError("CMC color labels and sample positions must have the same length.")
    cmap = get_cmc_colormap(colormap_name)
    return {label: tuple(float(channel) for channel in cmap(sample)) for label, sample in zip(labels, samples)}


def sample_cmc_source_colors(sources: dict[str, tuple[str, float]]) -> dict[str, tuple[float, float, float, float]]:
    return {
        label: tuple(float(channel) for channel in get_cmc_colormap(colormap_name)(sample))
        for label, (colormap_name, sample) in sources.items()
    }


def cmc_color(colormap_name: str, sample: float) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in get_cmc_colormap(colormap_name)(sample))


COLORS = sample_cmc_source_colors(CMC_PATHWAY_COLOR_SOURCES)
LAG_TIME_COLORS = sample_cmc_colors(CMC_LAG_TIME_COLORMAP, LAG_TIME_COLOR_KEYS, LAG_TIME_COLOR_SAMPLES)
DISTANCE_CMAP = get_cmc_colormap(CMC_DISTANCE_COLORMAP)
HEATMAP_CMAP = get_cmc_colormap(CMC_HEATMAP_COLORMAP)
NEUTRAL_DARK = cmc_color(CMC_NEUTRAL_COLORMAP, 0.18)
NEUTRAL_MID = cmc_color(CMC_NEUTRAL_COLORMAP, 0.45)
NEUTRAL_LIGHT = cmc_color(CMC_NEUTRAL_COLORMAP, 0.92)
TEXT_ON_DARK = cmc_color(CMC_NEUTRAL_COLORMAP, 0.98)
TEXT_ON_LIGHT = cmc_color(CMC_NEUTRAL_COLORMAP, 0.08)

LINESTYLES = {
    "Theis confined": "--",
    "Hantush-Jacob leaky": (0, (4, 1.5)),
    "Lagging Darcy no-leakage": "-",
    "Lagging Darcy with leakage": (0, (1.4, 1.2)),
}

PARAM_COUNT = {
    "Theis confined": 2,
    "Hantush-Jacob leaky": 3,
    "Lagging Darcy no-leakage": 4,
    "Lagging Darcy with leakage": 5,
}


def lock_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def export(fig: plt.Figure, stem: str) -> None:
    for base in [FIG_DIR, PROJECT]:
        for suffix in ["pdf", "png", "svg"]:
            kwargs = {"dpi": 450} if suffix == "png" else {}
            fig.savefig(base / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def export_many(fig: plt.Figure, stems: list[str]) -> None:
    for stem in stems:
        for base in [FIG_DIR, PROJECT]:
            for suffix in ["pdf", "png", "svg"]:
                kwargs = {"dpi": 450} if suffix == "png" else {}
                fig.savefig(base / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, text: str | None = None) -> None:
    content = f"({label})" if text is None else f"({label}) {text}"
    ax.text(0.0, 1.04, content, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=8.4)


def heatmap_text_style(cmap: Colormap, norm: Normalize, value: float) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    rgba = cmap(norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    if luminance < 0.47:
        return TEXT_ON_DARK, TEXT_ON_LIGHT
    return TEXT_ON_LIGHT, TEXT_ON_DARK


def theis_drawdown(time_h: np.ndarray, radius_m: float, storage: float, trans: float) -> np.ndarray:
    time_s = np.asarray(time_h, dtype=float) * 3600.0
    u = radius_m * radius_m * storage / (4.0 * trans * time_s)
    return Q_M3S * exp1(u) / (4.0 * np.pi * trans)


def constant_head_image_drawdown(
    time_h: np.ndarray,
    radius_m: float,
    storage: float,
    trans: float,
    image_distance_m: float,
) -> np.ndarray:
    return theis_drawdown(time_h, radius_m, storage, trans) - theis_drawdown(
        time_h, image_distance_m, storage, trans
    )


def hantush_well_function(u: float, beta: float) -> float:
    if not np.isfinite(u) or u <= 0.0 or not np.isfinite(beta) or beta < 0.0:
        return np.nan
    if beta < 1.0e-8:
        return float(exp1(u))
    beta_sq = beta * beta

    def integrand(y: float) -> float:
        return np.exp(-y - beta_sq / (4.0 * y)) / y

    value, _ = quad(integrand, u, np.inf, epsabs=1.0e-8, epsrel=1.0e-6, limit=100)
    return float(value)


def leaky_drawdown(time_h: np.ndarray, radius_m: float, storage: float, trans: float, leakage_b_m: float) -> np.ndarray:
    time_s = np.asarray(time_h, dtype=float) * 3600.0
    u = radius_m * radius_m * storage / (4.0 * trans * time_s)
    beta = radius_m / leakage_b_m
    values = np.array([hantush_well_function(float(ui), float(beta)) for ui in u], dtype=float)
    return Q_M3S * values / (4.0 * np.pi * trans)


def vi(n: int, i: int) -> float:
    half = n // 2
    total = 0.0
    for k in range(max(1, int(math.floor((i + 1) / 2))), min(i, half) + 1):
        total += k**half * math.factorial(2 * k) / (
            math.factorial(half - k)
            * math.factorial(k)
            * math.factorial(k - 1)
            * math.factorial(i - k)
            * math.factorial(2 * k - i)
        )
    return float((-1) ** (i + half) * total)


STEHFEST_WEIGHTS = np.array([vi(6, i) for i in range(1, 7)], dtype=float)


def stehfest(f, time_s: float) -> float:
    log2_t = np.log(2.0) / time_s
    return float(log2_t * sum(STEHFEST_WEIGHTS[i - 1] * f(i * log2_t) for i in range(1, 7)))


def lagging_leaky_laplace(
    p: float,
    radius_m: float,
    storage: float,
    trans: float,
    tau_q_s: float,
    tau_s_s: float,
    leakage_b_m: float,
    well_radius_m: float = RW_M,
    q_m3s: float = Q_M3S,
) -> float:
    with np.errstate(all="ignore"):
        memory_ratio = (1.0 + p * tau_q_s) / (1.0 + p * tau_s_s)
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
    radius_m: float,
    storage: float,
    trans: float,
    tau_q_s: float,
    tau_s_s: float,
    leakage_b_m: float,
    well_radius_m: float = RW_M,
    q_m3s: float = Q_M3S,
) -> float:
    if time_s <= 0.0:
        return 0.0

    value = np.real(
        stehfest(
            lambda p: lagging_leaky_laplace(
                p, radius_m, storage, trans, tau_q_s, tau_s_s, leakage_b_m, well_radius_m, q_m3s
            ),
            float(time_s),
        )
    )
    if not np.isfinite(value):
        return np.nan
    return float(value)


def lagging_leaky_curve(
    time_h: np.ndarray,
    radius_m: float,
    storage: float,
    trans: float,
    tau_q_s: float,
    tau_s_s: float,
    leakage_b_m: float,
) -> np.ndarray:
    time_s = np.asarray(time_h, dtype=float) * 3600.0
    return np.array(
        [
            lagging_leaky_drawdown(float(t), radius_m, storage, trans, tau_q_s, tau_s_s, leakage_b_m)
            for t in time_s
        ],
        dtype=float,
    )


def lagging_no_leakage_drawdown(
    time_s: float,
    radius_m: float,
    storage: float,
    trans: float,
    tau_q_s: float,
    tau_s_s: float,
    well_radius_m: float = RW_M,
    q_m3s: float = Q_M3S,
) -> float:
    return lagging_leaky_drawdown(
        time_s,
        radius_m,
        storage,
        trans,
        tau_q_s,
        tau_s_s,
        leakage_b_m=1.0e12,
        well_radius_m=well_radius_m,
        q_m3s=q_m3s,
    )


def finite_radius_leaky_no_lag_curve(
    time_h: np.ndarray,
    radius_m: float,
    storage: float,
    trans: float,
    leakage_b_m: float,
) -> np.ndarray:
    return lagging_leaky_curve(time_h, radius_m, storage, trans, 0.0, 0.0, leakage_b_m)


def log_response_residual(observed_m: np.ndarray, predicted_m: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed_m, dtype=float)
    predicted = np.asarray(predicted_m, dtype=float)
    eta = np.full(observed.shape, np.nan, dtype=float)
    valid = np.isfinite(predicted) & (predicted + LOG_RESIDUAL_OFFSET_M > 0.0)
    eta[valid] = np.log((observed[valid] + LOG_RESIDUAL_OFFSET_M) / (predicted[valid] + LOG_RESIDUAL_OFFSET_M))
    return eta


def aic_bic(observed_m: np.ndarray, predicted_m: np.ndarray, n_params: int, valid_mask: np.ndarray | None = None) -> tuple[float, float]:
    observed = np.asarray(observed_m, dtype=float)
    predicted = np.asarray(predicted_m, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)
    residual = observed[mask] - predicted[mask]
    if residual.size <= n_params:
        return np.nan, np.nan
    rss = max(float(np.sum(residual * residual)), 1.0e-30)
    n = residual.size
    return n * np.log(rss / n) + 2.0 * n_params, n * np.log(rss / n) + n_params * np.log(n)


def fit_theis(time_h: np.ndarray, observed_m: np.ndarray, radius_m: float, storage_start: float, trans_start: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.log(np.array([1.0e-8, 1.0e-8], dtype=float))
    upper = np.log(np.array([1.0, 1.0e-1], dtype=float))
    starts = [
        np.log(np.array([storage_start, trans_start], dtype=float)),
        np.log(np.array([1.0e-4, 2.0e-5], dtype=float)),
        np.log(np.array([1.0e-3, 5.0e-5], dtype=float)),
        np.log(np.array([5.0e-3, 8.0e-5], dtype=float)),
    ]

    def residual_at(log_params: np.ndarray) -> np.ndarray:
        storage, trans = np.exp(log_params)
        predicted = theis_drawdown(time_h, radius_m, storage, trans)
        eta = log_response_residual(observed_m, predicted)
        eta[~np.isfinite(eta)] = 1.0e6
        return eta

    best_score = np.inf
    best_x = np.clip(starts[0], lower, upper)
    for start in starts:
        result = least_squares(
            residual_at,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=500,
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
        )
        score = float(np.sum(result.fun * result.fun))
        if score < best_score:
            best_score = score
            best_x = result.x
    params = np.exp(best_x)
    return params, theis_drawdown(time_h, radius_m, params[0], params[1])


def near_log_bound(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 0.03) -> bool:
    span = upper - lower
    return bool(np.any((values - lower) / span < tol) or np.any((upper - values) / span < tol))


def fit_lagging_leaky(
    time_h: np.ndarray,
    observed_m: np.ndarray,
    radius_m: float,
    lagging_start: np.ndarray,
    leaky_start: np.ndarray,
    theis_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, int, float]:
    lower = np.log(np.array([1.0e-8, 1.0e-8, 1.0e-4, 1.0e-4, 1.0], dtype=float))
    upper = np.log(np.array([1.0, 1.0e-1, 1.0e7, 1.0e7, 1.0e6], dtype=float))
    lag_s, lag_t, lag_tau_q, lag_tau_s = np.asarray(lagging_start, dtype=float)
    leaky_s, leaky_t, leaky_b = np.asarray(leaky_start, dtype=float)
    theis_s, theis_t = np.asarray(theis_start, dtype=float)
    leaky_b = float(np.clip(leaky_b, 1.0, 1.0e6))
    starts = [
        np.array([lag_s, lag_t, lag_tau_q, lag_tau_s, leaky_b], dtype=float),
        np.array([lag_s, lag_t, lag_tau_q, lag_tau_s, 1.0e6], dtype=float),
        np.array([leaky_s, leaky_t, lag_tau_q, lag_tau_s, leaky_b], dtype=float),
        np.array([theis_s, theis_t, max(lag_tau_q, 10.0), max(lag_tau_s, 100.0), leaky_b], dtype=float),
    ]
    prior_center = np.log(np.clip(starts[0], np.exp(lower), np.exp(upper)))
    prior_scale = np.array([2.0, 1.8, 3.0, 3.0, 3.0], dtype=float)
    prior_weight = 0.18

    def residual_at(log_params: np.ndarray) -> np.ndarray:
        storage, trans, tau_q_s, tau_s_s, leakage_b_m = np.exp(log_params)
        predicted = lagging_leaky_curve(time_h, radius_m, storage, trans, tau_q_s, tau_s_s, leakage_b_m)
        eta = log_response_residual(observed_m, predicted)
        eta[~np.isfinite(eta)] = 1.0e6
        prior = prior_weight * (log_params - prior_center) / prior_scale
        return np.concatenate([eta, prior])

    best_score = np.inf
    best_x = np.clip(np.log(starts[0]), lower, upper)
    best_nfev = 0
    for start in starts:
        start = np.clip(np.log(np.clip(start, np.exp(lower), np.exp(upper))), lower, upper)
        result = least_squares(
            residual_at,
            start,
            bounds=(lower, upper),
            max_nfev=180,
            xtol=1.0e-7,
            ftol=1.0e-7,
            gtol=1.0e-7,
        )
        score = float(np.sum(result.fun * result.fun))
        if score < best_score:
            best_score = score
            best_x = result.x
            best_nfev = int(result.nfev)

    params = np.exp(best_x)
    prediction = lagging_leaky_curve(time_h, radius_m, params[0], params[1], params[2], params[3], params[4])
    data_eta = log_response_residual(observed_m, prediction)
    data_score = float(np.nansum(data_eta * data_eta))
    return params, prediction, near_log_bound(best_x, lower, upper), best_nfev, data_score


def cooper_jacob_late_time(time_h: np.ndarray, observed_m: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.asarray(time_h, dtype=float) * 3600.0
    order = np.argsort(time_s)
    late_start = int(np.floor(2 * time_s.size / 3))
    late_idx = order[late_start:]
    x = np.log(time_s[late_idx])
    y = np.asarray(observed_m, dtype=float)[late_idx]
    valid_late = np.isfinite(x) & np.isfinite(y) & (y > -LOG_RESIDUAL_OFFSET_M)
    if valid_late.sum() < 4:
        return np.array([np.nan, np.nan]), np.full_like(time_h, np.nan, dtype=float), np.zeros_like(time_h, dtype=bool)
    slope, intercept = np.polyfit(x[valid_late], y[valid_late], 1)
    if slope <= 0.0:
        return np.array([np.nan, np.nan]), np.full_like(time_h, np.nan, dtype=float), np.zeros_like(time_h, dtype=bool)
    trans = Q_M3S / (4.0 * np.pi * slope)
    storage = 2.25 * trans / (radius_m * radius_m * np.exp(intercept / slope))
    storage = float(np.clip(storage, 1.0e-8, 1.0))
    prediction = slope * np.log(time_s) + intercept
    u = radius_m * radius_m * storage / (4.0 * trans * time_s)
    valid_time = (np.arange(time_s.size) >= late_start) & np.isfinite(prediction) & (u < 0.05)
    return np.array([storage, trans]), prediction, valid_time


def summarize_pathway(
    well: str,
    pathway: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    eta: np.ndarray,
    valid_time: np.ndarray,
    params: dict[str, float],
    validity_note: str,
    boundary_hit: bool,
) -> dict[str, float | str | bool | int]:
    metric_mask = np.isfinite(predicted)
    if np.any(metric_mask):
        residual = observed[metric_mask] - predicted[metric_mask]
        rmse = float(np.sqrt(np.mean(residual * residual)))
        mae = float(np.mean(np.abs(residual)))
    else:
        rmse = np.nan
        mae = np.nan
    eta_valid = eta[np.isfinite(eta)]
    aic, bic = aic_bic(observed, predicted, PARAM_COUNT[pathway])
    out: dict[str, float | str | bool | int] = {
        "well": well,
        "pathway": pathway,
        "n_points": int(observed.size),
        "n_valid_points": int(np.sum(metric_mask)),
        "rmse_m": rmse,
        "mae_m": mae,
        "sd_eta": float(np.std(eta_valid, ddof=1)) if eta_valid.size > 1 else np.nan,
        "mean_eta": float(np.mean(eta_valid)) if eta_valid.size else np.nan,
        "aic": aic,
        "bic": bic,
        "boundary_hit_flag": bool(boundary_hit),
        "validity_note": validity_note,
    }
    out.update(params)
    return out


def build_field_pathway_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = pd.read_csv(TABLE_DIR / "point_predictions.csv")
    metadata = pd.read_csv(TABLE_DIR / "well_metadata.csv").set_index("well")
    parameters = pd.read_csv(TABLE_DIR / "best_fit_parameters.csv").set_index("well")
    leakage = pd.read_csv(TABLE_DIR / "leakage_boundary_diagnostic.csv").set_index("well")

    prediction_rows: list[dict[str, float | str | bool]] = []
    comparison_rows: list[dict[str, float | str | bool | int]] = []

    for well in WELL_ORDER:
        subset = points.loc[points["well"] == well].sort_values("time_h").copy()
        observed = subset["observed_m"].to_numpy(dtype=float)
        time_h = subset["time_h"].to_numpy(dtype=float)
        radius = float(metadata.loc[well, "distance_m"])

        pathway_payloads: list[tuple[str, np.ndarray, np.ndarray, dict[str, float], str, bool]] = []

        theis_params, theis_pred = fit_theis(
            time_h,
            observed,
            radius,
            float(parameters.loc[well, "S_darcy"]),
            float(parameters.loc[well, "T_darcy_m2s"]),
        )
        pathway_payloads.append(
            (
                "Theis confined",
                theis_pred,
                np.ones_like(time_h, dtype=bool),
                {
                    "S": float(theis_params[0]),
                    "T_m2s": float(theis_params[1]),
                    "B_m": np.nan,
                    "tau_q_s": np.nan,
                    "tau_s_s": np.nan,
                    "fit_nfev": np.nan,
                    "fit_data_score": np.nan,
                },
                "infinite line source confined fit",
                False,
            )
        )

        leaky_pred = leaky_drawdown(
            time_h,
            radius,
            float(leakage.loc[well, "S_leaky"]),
            float(leakage.loc[well, "T_leaky_m2s"]),
            float(leakage.loc[well, "B_leakage_m"]),
        )
        pathway_payloads.append(
            (
                "Hantush-Jacob leaky",
                leaky_pred,
                np.ones_like(time_h, dtype=bool),
                {
                    "S": float(leakage.loc[well, "S_leaky"]),
                    "T_m2s": float(leakage.loc[well, "T_leaky_m2s"]),
                    "B_m": float(leakage.loc[well, "B_leakage_m"]),
                    "tau_q_s": np.nan,
                    "tau_s_s": np.nan,
                    "fit_nfev": float(leakage.loc[well, "nfev_leaky_best_start"]),
                    "fit_data_score": np.nan,
                },
                "conventional Hantush-Jacob leaky fit",
                bool(float(leakage.loc[well, "B_leakage_m"]) > 9.5e5),
            )
        )

        lag_pred = subset["lagging_pred_m"].to_numpy(dtype=float)
        lagging_start = np.array(
            [
                float(parameters.loc[well, "S_lagging"]),
                float(parameters.loc[well, "T_lagging_m2s"]),
                float(parameters.loc[well, "tau_q_lagging_s"]),
                float(parameters.loc[well, "tau_s_lagging_s"]),
            ],
            dtype=float,
        )
        pathway_payloads.append(
            (
                "Lagging Darcy no-leakage",
                lag_pred,
                np.ones_like(time_h, dtype=bool),
                {
                    "S": float(parameters.loc[well, "S_lagging"]),
                    "T_m2s": float(parameters.loc[well, "T_lagging_m2s"]),
                    "B_m": np.nan,
                    "tau_q_s": float(parameters.loc[well, "tau_q_lagging_s"]),
                    "tau_s_s": float(parameters.loc[well, "tau_s_lagging_s"]),
                    "fit_nfev": np.nan,
                    "fit_data_score": np.nan,
                },
                "finite radius lagging Darcy limit without leakage",
                False,
            )
        )

        lagging_leaky_params, lagging_leaky_pred, lagging_leaky_boundary, lagging_leaky_nfev, lagging_leaky_score = fit_lagging_leaky(
            time_h,
            observed,
            radius,
            lagging_start,
            np.array(
                [
                    float(leakage.loc[well, "S_leaky"]),
                    float(leakage.loc[well, "T_leaky_m2s"]),
                    float(leakage.loc[well, "B_leakage_m"]),
                ],
                dtype=float,
            ),
            theis_params,
        )
        pathway_payloads.append(
            (
                "Lagging Darcy with leakage",
                lagging_leaky_pred,
                np.ones_like(time_h, dtype=bool),
                {
                    "S": float(lagging_leaky_params[0]),
                    "T_m2s": float(lagging_leaky_params[1]),
                    "B_m": float(lagging_leaky_params[4]),
                    "tau_q_s": float(lagging_leaky_params[2]),
                    "tau_s_s": float(lagging_leaky_params[3]),
                    "fit_nfev": float(lagging_leaky_nfev),
                    "fit_data_score": float(lagging_leaky_score),
                },
                "formal Lin and Yeh lagging leaky finite radius fit",
                lagging_leaky_boundary,
            )
        )

        for pathway, predicted, valid_time, params, note, boundary_hit in pathway_payloads:
            eta = log_response_residual(observed, predicted)
            comparison_rows.append(summarize_pathway(well, pathway, observed, predicted, eta, valid_time, params, note, boundary_hit))
            for source_idx, (_, row) in enumerate(subset.reset_index(drop=True).iterrows()):
                prediction_rows.append(
                    {
                        "well": well,
                        "distance_m": radius,
                        "azimuth_deg": float(metadata.loc[well, "azimuth_deg"]),
                        "time_h": float(row["time_h"]),
                        "observed_m": float(row["observed_m"]),
                        "pathway": pathway,
                        "pathway_short": SHORT_LABEL[pathway],
                        "predicted_m": float(predicted[source_idx]) if np.isfinite(predicted[source_idx]) else np.nan,
                        "eta": float(eta[source_idx]) if np.isfinite(eta[source_idx]) else np.nan,
                        "valid_time_flag": bool(valid_time[source_idx]),
                        "validity_note": note,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    comparison = pd.DataFrame(comparison_rows)
    baseline_bic = comparison.loc[comparison["pathway"] == "Theis confined", ["well", "bic"]].rename(columns={"bic": "bic_theis"})
    lagging_bic = comparison.loc[comparison["pathway"] == "Lagging Darcy no-leakage", ["well", "bic"]].rename(columns={"bic": "bic_lagging"})
    comparison = comparison.merge(baseline_bic, on="well", how="left").merge(lagging_bic, on="well", how="left")
    comparison["delta_bic_minus_theis"] = comparison["bic"] - comparison["bic_theis"]
    comparison["delta_bic_minus_lagging"] = comparison["bic"] - comparison["bic_lagging"]
    comparison["pathway"] = pd.Categorical(comparison["pathway"], categories=PATHWAYS, ordered=True)
    comparison = comparison.sort_values(["well", "pathway"]).reset_index(drop=True)
    predictions["pathway"] = pd.Categorical(predictions["pathway"], categories=PATHWAYS, ordered=True)
    predictions = predictions.sort_values(["well", "time_h", "pathway"]).reset_index(drop=True)

    summary_rows = []
    for pathway, group in comparison.groupby("pathway", observed=True):
        pred_subset = predictions.loc[predictions["pathway"] == pathway].copy()
        eta_values = pred_subset["eta"].dropna()
        pref_count = int(
            comparison.loc[
                comparison.groupby("well", observed=True)["bic"].idxmin(), "pathway"
            ].eq(pathway).sum()
        )
        summary_rows.append(
            {
                "pathway": str(pathway),
                "median_rmse_m": float(group["rmse_m"].median()),
                "median_sd_eta": float(group["sd_eta"].median()),
                "pooled_sd_eta": float(np.std(eta_values, ddof=1)),
                "bic_preferred_wells": pref_count,
                "median_abs_delta_bic_minus_theis": float(group["delta_bic_minus_theis"].abs().median()),
                "boundary_hit_count": int(group["boundary_hit_flag"].sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    predictions.to_csv(TABLE_DIR / "field_pathway_predictions.csv", index=False)
    comparison.to_csv(TABLE_DIR / "field_pathway_model_comparison.csv", index=False)
    summary.to_csv(TABLE_DIR / "field_pathway_summary.csv", index=False)
    return predictions, comparison, summary


def metric_prediction_subset(predictions: pd.DataFrame, pathway: str) -> pd.DataFrame:
    return predictions[predictions["pathway"] == pathway].copy()


def normalize_lovelock_key(value: object) -> str:
    return str(value).replace("_", "").upper()


def plot_figure2_parallel_site_layout(metadata: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    fig.subplots_adjust(wspace=0.34, bottom=0.22, top=0.86, left=0.075, right=0.94)

    mass = metadata.copy()
    azimuth = np.deg2rad(mass["azimuth_deg"].to_numpy(dtype=float))
    radius = mass["distance_m"].to_numpy(dtype=float)
    pump_x = float(np.mean(mass["x_m"].to_numpy(dtype=float) - radius * np.sin(azimuth)))
    pump_y = float(np.mean(mass["y_m"].to_numpy(dtype=float) - radius * np.cos(azimuth)))
    mass["x_rel_m"] = mass["x_m"].to_numpy(dtype=float) - pump_x
    mass["y_rel_m"] = mass["y_m"].to_numpy(dtype=float) - pump_y
    mass_scale = float(np.sqrt(mass["x_rel_m"] ** 2 + mass["y_rel_m"] ** 2).max())
    mass["x_scaled"] = mass["x_rel_m"] / mass_scale
    mass["y_scaled"] = mass["y_rel_m"] / mass_scale
    sc0 = axes[0].scatter(
        mass["x_scaled"],
        mass["y_scaled"],
        c=mass["distance_m"],
        cmap=DISTANCE_CMAP,
        s=36,
        edgecolors="none",
        zorder=3,
    )
    axes[0].scatter(0.0, 0.0, marker="*", s=105, color=NEUTRAL_DARK, linewidths=0, zorder=4)
    mass_offsets = {
        "MW-08": (-0.060, 0.100),
        "MW-11": (-0.170, 0.045),
        "MW-06": (0.015, 0.080),
        "MW-05": (-0.195, 0.045),
        "MW-07": (0.055, -0.010),
        "MW-03": (0.020, -0.075),
        "MW-04": (-0.175, -0.055),
        "MW-01": (0.035, 0.025),
        "MW-02": (0.035, -0.065),
    }
    for _, row in mass.iterrows():
        dx, dy = mass_offsets.get(row["well"], (0.034, 0.034))
        axes[0].text(row["x_scaled"] + dx, row["y_scaled"] + dy, row["well"], fontsize=5.2, color=NEUTRAL_DARK)
    axes[0].text(0.075, 0.075, "pumping well", fontsize=5.8, color=NEUTRAL_DARK)
    axes[0].text(
        0.02,
        0.03,
        f"scale: max r={mass_scale:.1f} m",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=NEUTRAL_MID,
    )
    axes[0].set_xlabel("scaled easting")
    axes[0].set_ylabel("scaled northing")
    mass_lim = float(max(np.abs(mass["x_scaled"]).max(), np.abs(mass["y_scaled"]).max(), 1.0)) + 0.12
    axes[0].set_xlim(-mass_lim, mass_lim)
    axes[0].set_ylim(-mass_lim, mass_lim)
    axes[0].set_aspect("equal", adjustable="box")
    panel_label(axes[0], "a", "Massachusetts retained wells")
    cb0 = fig.colorbar(sc0, ax=axes[0], fraction=0.050, pad=0.025)
    cb0.set_label("radial distance (m)")

    coord_path = TABLE_DIR / "lovelock_coordinate_audit.csv"
    fit_path = TABLE_DIR / "lovelock_fit_parameters.csv"
    if not coord_path.exists() or not fit_path.exists():
        raise FileNotFoundError("Lovelock coordinate and fit tables are required to plot the paired field support figure.")
    coords = pd.read_csv(coord_path)
    coords = coords[coords["test"] == "MWP2"].copy()
    coords["norm_key"] = coords["well_key"].map(normalize_lovelock_key)
    pump = coords[coords["norm_key"] == "BBING"].iloc[0]
    selected = pd.read_csv(fit_path)[["well_key", "well", "distance_mi"]].drop_duplicates().copy()
    selected["norm_key"] = selected["well_key"].map(normalize_lovelock_key)
    lovelock = selected.merge(coords[["norm_key", "x_model", "y_model", "layer"]], on="norm_key", how="left")
    if lovelock[["x_model", "y_model"]].isna().any().any():
        missing = lovelock.loc[lovelock[["x_model", "y_model"]].isna().any(axis=1), "well"].tolist()
        raise ValueError(f"Missing Lovelock coordinates for retained wells: {missing}")
    lovelock["x_rel_km"] = (lovelock["x_model"] - float(pump["x_model"])) / 1000.0
    lovelock["y_rel_km"] = (lovelock["y_model"] - float(pump["y_model"])) / 1000.0
    lovelock_scale_km = float(lovelock["distance_mi"].max() * 1.609344)
    lovelock["x_scaled"] = lovelock["x_rel_km"] / lovelock_scale_km
    lovelock["y_scaled"] = lovelock["y_rel_km"] / lovelock_scale_km
    sc1 = axes[1].scatter(
        lovelock["x_scaled"],
        lovelock["y_scaled"],
        c=lovelock["distance_mi"],
        cmap=DISTANCE_CMAP,
        s=38,
        edgecolors="none",
        zorder=3,
    )
    axes[1].scatter(0.0, 0.0, marker="*", s=115, color=NEUTRAL_DARK, linewidths=0, zorder=4)
    grouped_labels = (
        lovelock.groupby(["x_scaled", "y_scaled"], as_index=False)
        .agg(well=("well", lambda values: "/".join(sorted(values))), distance_mi=("distance_mi", "mean"))
        .sort_values("distance_mi")
    )
    lovelock_offsets = {
        "FETH": (0.018, 0.040),
        "MOIR": (0.022, 0.028),
        "MD": (0.022, -0.040),
        "LEID": (0.020, 0.034),
        "WWBN": (0.020, -0.050),
        "WD/WS": (0.020, -0.044),
    }
    for _, row in grouped_labels.iterrows():
        dx, dy = lovelock_offsets.get(row["well"], (0.020, 0.020))
        axes[1].text(row["x_scaled"] + dx, row["y_scaled"] + dy, row["well"], fontsize=5.8, color=NEUTRAL_DARK)
    axes[1].text(0.035, 0.055, "pumping Well 10", fontsize=6.2, color=NEUTRAL_DARK)
    axes[1].text(
        0.02,
        0.03,
        f"scale: max r={lovelock['distance_mi'].max():.1f} mi",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=NEUTRAL_MID,
    )
    axes[1].set_xlabel("scaled easting")
    axes[1].set_ylabel("scaled northing")
    lovelock_lim = float(max(np.abs(lovelock["x_scaled"]).max(), np.abs(lovelock["y_scaled"]).max(), 1.0)) + 0.12
    axes[1].set_xlim(-lovelock_lim, lovelock_lim)
    axes[1].set_ylim(-lovelock_lim, lovelock_lim)
    axes[1].set_aspect("equal", adjustable="box")
    panel_label(axes[1], "b", "Lovelock retained wells")
    cb1 = fig.colorbar(sc1, ax=axes[1], fraction=0.050, pad=0.025)
    cb1.set_label("radial distance (mi)")
    symbol_handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=NEUTRAL_DARK, markeredgecolor=NEUTRAL_DARK, markersize=8.5, label="Pumping well"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=cmc_color(CMC_DISTANCE_COLORMAP, 0.55), markeredgecolor="none", markersize=5.8, label="Observation well"),
    ]
    fig.legend(
        handles=symbol_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.045),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
    )
    export(fig, "fig02_site_layout")


def plot_figure3(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(8.0, 7.0), sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.34, top=0.90)
    legend_handles = []
    legend_labels = []
    for ax, well in zip(axes.ravel(), WELL_ORDER):
        sub = predictions[predictions["well"] == well]
        obs = sub.drop_duplicates(["time_h"])[["time_h", "observed_m", "distance_m"]]
        ax.scatter(obs["time_h"], obs["observed_m"], s=8, color=NEUTRAL_DARK, alpha=0.72, linewidths=0, label="Observed")
        for pathway in PATHWAYS:
            psub = sub[sub["pathway"] == pathway].sort_values("time_h")
            line, = ax.plot(
                psub["time_h"],
                psub["predicted_m"],
                color=COLORS[pathway],
                linestyle=LINESTYLES[pathway],
                linewidth=1.05 if pathway.startswith("Lagging Darcy") else 0.85,
                alpha=0.95,
                label=SHORT_LABEL[pathway],
            )
            if well == WELL_ORDER[0]:
                legend_handles.append(line)
                legend_labels.append(SHORT_LABEL[pathway])
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{well}, r={float(obs['distance_m'].iloc[0]):.1f} m", fontsize=7.2)
        ax.tick_params(axis="both", length=2.0)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Drawdown (m)")
    obs_handle = axes[0, 0].collections[0]
    fig.legend(
        [obs_handle, *legend_handles],
        ["Observed", *legend_labels],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    export(fig, "fig03_observed_simulated_drawdown")


def plot_figure4(predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.65), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.25, hspace=0.40, right=0.84, top=0.90)
    cmap = DISTANCE_CMAP
    offset = LOG_RESIDUAL_OFFSET_M
    all_values = pd.concat([predictions["predicted_m"], predictions["observed_m"]])
    all_values = all_values[all_values + offset > 0.0]
    all_log = np.log10(all_values + offset)
    lo, hi = float(np.nanpercentile(all_log, 1)), float(np.nanpercentile(all_log, 99))
    mappable = None
    for idx, (ax, pathway) in enumerate(zip(axes.ravel(), PATHWAYS)):
        sub = metric_prediction_subset(predictions, pathway)
        sub = sub[(sub["predicted_m"] + offset > 0.0) & (sub["observed_m"] + offset > 0.0)]
        mappable = ax.scatter(
            np.log10(sub["predicted_m"] + offset),
            np.log10(sub["observed_m"] + offset),
            c=sub["distance_m"],
            cmap=cmap,
            s=7,
            alpha=0.62,
            linewidths=0,
        )
        ax.plot([lo, hi], [lo, hi], color=NEUTRAL_DARK, linewidth=0.65)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        sd = float(summary.loc[summary["pathway"] == pathway, "pooled_sd_eta"].iloc[0])
        panel_label(ax, chr(ord("a") + idx), f"{SHORT_LABEL[pathway]}\nSD={sd:.3f}")
        if idx >= 2:
            ax.set_xlabel("Predicted log drawdown")
        if idx % 2 == 0:
            ax.set_ylabel("Observed log drawdown")
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.80, pad=0.035)
    cbar.set_label("Distance from pumping well (m)")
    export(fig, "fig04_transformation_scatter")


def plot_figure5(predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    data = [metric_prediction_subset(predictions, pathway)["eta"].dropna().to_numpy() for pathway in PATHWAYS]
    positions = np.arange(len(PATHWAYS))
    parts = ax.violinplot(data, positions=positions, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, pathway in zip(parts["bodies"], PATHWAYS):
        body.set_facecolor(COLORS[pathway])
        body.set_edgecolor(NEUTRAL_LIGHT)
        body.set_alpha(0.74)
    for idx, values in enumerate(data):
        q05, q50, q95 = np.percentile(values, [5, 50, 95])
        ax.plot([idx, idx], [q05, q95], color=NEUTRAL_DARK, linewidth=0.8)
        ax.scatter(idx, q50, s=18, color=NEUTRAL_LIGHT, edgecolor=NEUTRAL_DARK, linewidth=0.4, zorder=3)
        sd = float(summary.loc[summary["pathway"] == PATHWAYS[idx], "pooled_sd_eta"].iloc[0])
        ax.text(idx, q95 + 0.035, f"{sd:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0, color=NEUTRAL_DARK, linewidth=0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT_LABEL[p] for p in PATHWAYS], rotation=18, ha="right")
    ax.set_ylabel("Log response model factor, $\\eta$")
    ax.set_title("Pooled response residual distributions; numbers show pooled $SD_\\epsilon$", loc="left", fontsize=8.5, fontweight="bold")
    export(fig, "fig05_transformation_residual_distributions")


def _phase_labels(sub: pd.DataFrame) -> pd.Series:
    order = sub["time_h"].rank(method="first", pct=True)
    return pd.cut(order, bins=[0, 1 / 3, 2 / 3, 1], labels=["early", "middle", "late"], include_lowest=True).astype(str)


def _binned_median_line(x: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    x_valid = x[valid]
    y_valid = y[valid]
    if x_valid.size < 8:
        return np.array([]), np.array([])
    edges = np.geomspace(x_valid.min(), x_valid.max(), bins + 1)
    mids, meds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x_valid >= lo) & (x_valid < hi)
        if mask.sum() >= 4:
            mids.append(float(np.sqrt(lo * hi)))
            meds.append(float(np.median(y_valid[mask])))
    return np.asarray(mids), np.asarray(meds)


def plot_figure6(predictions: pd.DataFrame) -> pd.DataFrame:
    phase_rows = []
    tmp = predictions.copy()
    tmp["phase"] = ""
    for (well, pathway), idx in tmp.groupby(["well", "pathway"], observed=True).groups.items():
        tmp.loc[idx, "phase"] = _phase_labels(tmp.loc[idx])
    for (pathway, phase), group in tmp.groupby(["pathway", "phase"], observed=True):
        values = group.loc[group["eta"].notna(), "eta"].to_numpy(dtype=float)
        phase_rows.append(
            {
                "pathway": pathway,
                "phase": phase,
                "median_sd_eta_by_well": float(
                    group.groupby("well", observed=True)["eta"].std().dropna().median()
                ),
                "pooled_sd_eta": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
            }
        )
    phase_summary = pd.DataFrame(phase_rows)
    phase_summary.to_csv(TABLE_DIR / "field_pathway_time_window_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.65), gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(wspace=0.42, top=0.80, bottom=0.18, right=0.94)
    for pathway in PATHWAYS:
        sub = predictions[predictions["pathway"] == pathway]
        x, y = _binned_median_line(sub["time_h"].to_numpy(), sub["eta"].to_numpy())
        axes[0].plot(x, y, color=COLORS[pathway], linestyle=LINESTYLES[pathway], linewidth=1.15, label=SHORT_LABEL[pathway])
    axes[0].axhline(0, color=NEUTRAL_DARK, linewidth=0.75)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Time since pumping began (h)")
    axes[0].set_ylabel("Binned median $\\eta(t)$")
    axes[0].legend(frameon=False, loc="best", ncol=1)
    panel_label(axes[0], "a", "temporal residual structure")

    matrix = (
        phase_summary.pivot(index="pathway", columns="phase", values="median_sd_eta_by_well")
        .reindex(PATHWAYS)[["early", "middle", "late"]]
    )
    matrix_values = matrix.to_numpy(dtype=float)
    heatmap_norm = Normalize(vmin=float(np.nanmin(matrix_values)), vmax=float(np.nanmax(matrix_values)))
    im = axes[1].imshow(matrix_values, aspect="auto", cmap=HEATMAP_CMAP, norm=heatmap_norm)
    axes[1].set_yticks(np.arange(len(PATHWAYS)))
    axes[1].set_yticklabels([SHORT_LABEL[p] for p in PATHWAYS])
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels(["Early", "Middle", "Late"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            text_color, halo_color = heatmap_text_style(HEATMAP_CMAP, heatmap_norm, float(val))
            label = axes[1].text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold",
                color=text_color,
            )
            label.set_path_effects([pe.withStroke(linewidth=1.05, foreground=halo_color)])
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.82, pad=0.035)
    cbar.set_label("Median well $SD_\\epsilon$")
    panel_label(axes[1], "b", "time window residuals")
    export(fig, "fig06_residuals_through_time")
    return phase_summary


def plot_figure7(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.2))
    fig.subplots_adjust(wspace=0.34, hspace=0.42, top=0.91)
    positions = np.arange(len(PATHWAYS))
    rng = np.random.default_rng(1707)
    for ax, param, label, letter in [
        (axes[0, 0], "T_m2s", "$\\ln T$ (m$^2$ s$^{-1}$)", "a"),
        (axes[0, 1], "S", "$\\ln S$ (-)", "b"),
    ]:
        for idx, pathway in enumerate(PATHWAYS):
            values = comparison.loc[comparison["pathway"] == pathway, param].dropna().to_numpy(dtype=float)
            jitter = rng.uniform(-0.055, 0.055, size=values.size)
            ax.scatter(np.full(values.size, idx) + jitter, np.log(values), s=19, color=COLORS[pathway], edgecolors="none", alpha=0.84)
            if values.size:
                ax.plot([idx - 0.25, idx + 0.25], [np.median(np.log(values))] * 2, color=NEUTRAL_DARK, linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels([SHORT_LABEL[p] for p in PATHWAYS], rotation=25, ha="right")
        ax.set_ylabel(label)
        panel_label(ax, letter, "inferred parameter spread")

    leaky = comparison[comparison["pathway"] == "Hantush-Jacob leaky"]
    lagging_leaky = comparison[comparison["pathway"] == "Lagging Darcy with leakage"]
    leakage_positions = {
        "Hantush $B$": 0.42,
        "Lagging leaky $B$": 0.58,
    }
    leakage_specs = [
        ("Hantush $B$", leaky["B_m"].dropna().to_numpy(dtype=float), COLORS["Hantush-Jacob leaky"]),
        ("Lagging leaky $B$", lagging_leaky["B_m"].dropna().to_numpy(dtype=float), COLORS["Lagging Darcy with leakage"]),
    ]
    for label_text, values, color in leakage_specs:
        x0 = leakage_positions[label_text]
        jitter = rng.uniform(-0.022, 0.022, size=values.size)
        axes[1, 0].scatter(np.full(values.size, x0) + jitter, values, color=color, s=25, edgecolors="none", alpha=0.84, label=label_text)
        if values.size:
            axes[1, 0].plot([x0 - 0.052, x0 + 0.052], [np.median(values)] * 2, color=NEUTRAL_DARK, linewidth=0.8)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlim(0.30, 0.70)
    axes[1, 0].set_xticks(list(leakage_positions.values()))
    axes[1, 0].set_xticklabels(["Hantush", "Lagging leaky"])
    axes[1, 0].set_ylabel("Length scale (m)")
    panel_label(axes[1, 0], "c", "leakage scales")

    lagging = comparison[comparison["pathway"] == "Lagging Darcy no-leakage"]
    axes[1, 1].scatter(np.zeros(lagging.shape[0]), lagging["tau_q_s"] / 3600.0, color=LAG_TIME_COLORS["no_leak_tau_q"], s=24, edgecolor=NEUTRAL_DARK, linewidth=0.3, label="Without leakage $\\tau_q$")
    axes[1, 1].scatter(np.ones(lagging.shape[0]), lagging["tau_s_s"] / 3600.0, color=LAG_TIME_COLORS["no_leak_tau_s"], s=24, edgecolor=NEUTRAL_DARK, linewidth=0.3, label="Without leakage $\\tau_s$")
    axes[1, 1].scatter(np.full(lagging_leaky.shape[0], 2.0), lagging_leaky["tau_q_s"] / 3600.0, color=LAG_TIME_COLORS["leaky_tau_q"], s=24, edgecolor=NEUTRAL_DARK, linewidth=0.3, label="Leaky $\\tau_q$")
    axes[1, 1].scatter(np.full(lagging_leaky.shape[0], 3.0), lagging_leaky["tau_s_s"] / 3600.0, color=LAG_TIME_COLORS["leaky_tau_s"], s=24, edgecolor=NEUTRAL_DARK, linewidth=0.3, label="Leaky $\\tau_s$")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks([0, 1, 2, 3])
    axes[1, 1].set_xticklabels(["L0 $\\tau_q$", "L0 $\\tau_s$", "LL $\\tau_q$", "LL $\\tau_s$"], rotation=15, ha="right")
    axes[1, 1].set_ylabel("Lag time (h)")
    panel_label(axes[1, 1], "d", "lagging response times")
    export(fig, "fig07_parameter_apparent_variability")


def plot_figure8(comparison: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.1))
    fig.subplots_adjust(wspace=0.36, hspace=0.48, top=0.91)
    x = np.arange(len(PATHWAYS))
    axes[0, 0].bar(x, summary.set_index("pathway").loc[PATHWAYS, "pooled_sd_eta"], color=[COLORS[p] for p in PATHWAYS], alpha=0.85)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([SHORT_LABEL[p] for p in PATHWAYS], rotation=25, ha="right")
    axes[0, 0].set_ylabel("Pooled $SD_\\epsilon$")
    panel_label(axes[0, 0], "a", "residual standard deviation")

    pref = comparison.loc[comparison.groupby("well", observed=True)["bic"].idxmin()].groupby("pathway", observed=True).size().reindex(PATHWAYS, fill_value=0)
    axes[0, 1].bar(x, pref.to_numpy(), color=[COLORS[p] for p in PATHWAYS], alpha=0.85)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([SHORT_LABEL[p] for p in PATHWAYS], rotation=25, ha="right")
    axes[0, 1].set_ylabel("Wells preferred by BIC")
    panel_label(axes[0, 1], "b", "support after complexity penalty")

    cov_rows = []
    for pathway in PATHWAYS:
        group = comparison[comparison["pathway"] == pathway]
        for param in ["T_m2s", "S"]:
            values = group[param].dropna().to_numpy(dtype=float)
            if values.size > 1:
                cov_rows.append(
                    {
                        "pathway": pathway,
                        "parameter": "T" if param == "T_m2s" else "S",
                        "cov": float(np.sqrt(np.exp(np.var(np.log(values), ddof=1)) - 1.0)),
                    }
                )
    cov = pd.DataFrame(cov_rows)
    width = 0.36
    for offset, param, color_scale in [(-width / 2, "T", 0.35), (width / 2, "S", 0.75)]:
        values = cov[cov["parameter"] == param].set_index("pathway").reindex(PATHWAYS)["cov"]
        axes[1, 0].bar(x + offset, values, width, color=[COLORS[p] for p in PATHWAYS], alpha=color_scale, label=param)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([SHORT_LABEL[p] for p in PATHWAYS], rotation=25, ha="right")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Lognormal COV")
    axes[1, 0].legend(frameon=False)
    panel_label(axes[1, 0], "c", "parameter variability")

    delta = comparison.pivot(index="well", columns="pathway", values="delta_bic_minus_lagging").reindex(WELL_ORDER)
    for pathway in PATHWAYS:
        axes[1, 1].plot(
            np.arange(len(WELL_ORDER)),
            delta[pathway],
            marker="o",
            markersize=3.0,
            linewidth=0.85,
            linestyle=LINESTYLES[pathway],
            color=COLORS[pathway],
            label=SHORT_LABEL[pathway],
        )
    axes[1, 1].axhline(0, color=NEUTRAL_DARK, linewidth=0.75)
    axes[1, 1].set_xticks(np.arange(len(WELL_ORDER)))
    axes[1, 1].set_xticklabels(WELL_ORDER, rotation=45, ha="right")
    axes[1, 1].set_ylabel("$\\Delta$BIC vs. lagging without leakage")
    axes[1, 1].legend(frameon=False, loc="best", fontsize=6.3)
    panel_label(axes[1, 1], "d", "relative BIC support")
    export_many(fig, ["fig08_variance_ratio_cov_summary", "fig06_massachusetts_pathway_summary"])


def plot_figure9(comparison: pd.DataFrame, metadata: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.25))
    fig.subplots_adjust(wspace=0.42, right=0.89, top=0.84)
    meta = metadata.set_index("well")
    lag = comparison[comparison["pathway"] == "Lagging Darcy no-leakage"].set_index("well")
    lag_leaky = comparison[comparison["pathway"] == "Lagging Darcy with leakage"].set_index("well")
    hantush = comparison[comparison["pathway"] == "Hantush-Jacob leaky"].set_index("well")
    norm = Normalize(float(meta.loc[WELL_ORDER, "distance_m"].min()), float(meta.loc[WELL_ORDER, "distance_m"].max()))
    cmap = DISTANCE_CMAP
    mappable = axes[0].scatter(
        np.log(lag.loc[WELL_ORDER, "tau_q_s"]),
        np.log(lag.loc[WELL_ORDER, "tau_s_s"]),
        c=[float(meta.loc[well, "distance_m"]) for well in WELL_ORDER],
        cmap=cmap,
        norm=norm,
        s=28,
        edgecolors="none",
        linewidth=0.0,
    )
    axes[0].set_xlabel("$\\ln \\tau_q$")
    axes[0].set_ylabel("$\\ln \\tau_s$")
    panel_label(axes[0], "a", "lagging tradeoff without leakage")

    r = meta.loc[WELL_ORDER, "distance_m"].to_numpy(dtype=float)
    hantush_ratio = r / hantush.loc[WELL_ORDER, "B_m"].to_numpy(dtype=float)
    lag_leaky_ratio = r / lag_leaky.loc[WELL_ORDER, "B_m"].to_numpy(dtype=float)
    axes[1].scatter(
        hantush_ratio,
        lag_leaky_ratio,
        c=[float(meta.loc[well, "distance_m"]) for well in WELL_ORDER],
        cmap=cmap,
        norm=norm,
        s=28,
        edgecolors="none",
        linewidth=0.0,
    )
    axes[1].axvline(0.05, color=COLORS["Hantush-Jacob leaky"], linestyle="--", linewidth=0.8)
    axes[1].axhline(0.05, color=COLORS["Lagging Darcy with leakage"], linestyle="--", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Hantush $r/B$")
    axes[1].set_ylabel("Lagging leaky $r/B$")
    panel_label(axes[1], "b", "leakage activity")

    delta = comparison.pivot(index="well", columns="pathway", values="delta_bic_minus_lagging").reindex(WELL_ORDER)
    x = np.arange(len(WELL_ORDER))
    axes[2].bar(x - 0.18, delta["Hantush-Jacob leaky"], width=0.36, color=COLORS["Hantush-Jacob leaky"], alpha=0.82, label="Hantush")
    axes[2].bar(x + 0.18, delta["Lagging Darcy with leakage"], width=0.36, color=COLORS["Lagging Darcy with leakage"], alpha=0.82, label="Lagging leaky")
    axes[2].axhline(0, color=NEUTRAL_DARK, linewidth=0.75)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(WELL_ORDER, rotation=45, ha="right")
    axes[2].set_ylabel("$\\Delta$BIC vs. lagging without leakage")
    axes[2].legend(frameon=False, loc="best")
    panel_label(axes[2], "c", "leakage support")
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.82, pad=0.025)
    cbar.set_label("Distance from pumping well (m)")
    export(fig, "fig09_lagging_parameter_tradeoff")


TRANSFER_PARAMS = {
    "Theis confined": ["S", "T_m2s"],
    "Hantush-Jacob leaky": ["S", "T_m2s", "B_m"],
    "Lagging Darcy no-leakage": ["S", "T_m2s", "tau_q_s", "tau_s_s"],
    "Lagging Darcy with leakage": ["S", "T_m2s", "B_m", "tau_q_s", "tau_s_s"],
}

TRANSFER_BOUNDS = {
    "S": (1.0e-8, 1.0),
    "T_m2s": (1.0e-8, 1.0e-1),
    "B_m": (1.0, 1.0e6),
    "tau_q_s": (1.0e-4, 1.0e7),
    "tau_s_s": (1.0e-4, 1.0e7),
}


def _predict_log_parameter(train: pd.DataFrame, target_distance: float, parameter: str) -> float:
    values = train[["distance_m", parameter]].dropna().copy()
    values = values[(values["distance_m"] > 0.0) & (values[parameter] > 0.0)]
    lower, upper = TRANSFER_BOUNDS[parameter]
    if values.shape[0] >= 3:
        coeff = np.polyfit(np.log(values["distance_m"].to_numpy(dtype=float)), np.log(values[parameter].to_numpy(dtype=float)), 1)
        pred = coeff[0] * np.log(target_distance) + coeff[1]
    else:
        pred = float(np.log(values[parameter]).median())
    return float(np.clip(pred, np.log(lower), np.log(upper)))


def _simulate_transfer_curve(pathway: str, time_h: np.ndarray, radius_m: float, params: dict[str, float]) -> np.ndarray:
    if pathway == "Theis confined":
        return theis_drawdown(time_h, radius_m, params["S"], params["T_m2s"])
    if pathway == "Hantush-Jacob leaky":
        return leaky_drawdown(time_h, radius_m, params["S"], params["T_m2s"], params["B_m"])
    if pathway == "Lagging Darcy no-leakage":
        return lagging_leaky_curve(time_h, radius_m, params["S"], params["T_m2s"], params["tau_q_s"], params["tau_s_s"], 1.0e12)
    if pathway == "Lagging Darcy with leakage":
        return lagging_leaky_curve(
            time_h,
            radius_m,
            params["S"],
            params["T_m2s"],
            params["tau_q_s"],
            params["tau_s_s"],
            params["B_m"],
        )
    raise ValueError(f"Unsupported pathway: {pathway}")


def build_four_pathway_spatial_transfer(predictions: pd.DataFrame, comparison: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    meta = metadata.set_index("well")
    comp = comparison.merge(metadata[["well", "distance_m"]], on="well", how="left")
    rows = []
    for well in WELL_ORDER:
        point_subset = predictions[predictions["well"] == well].drop_duplicates("time_h").sort_values("time_h")
        observed = point_subset["observed_m"].to_numpy(dtype=float)
        time_h = point_subset["time_h"].to_numpy(dtype=float)
        radius = float(meta.loc[well, "distance_m"])
        for pathway in PATHWAYS:
            train = comp[(comp["pathway"] == pathway) & (comp["well"] != well)].copy()
            params = {
                parameter: float(np.exp(_predict_log_parameter(train, radius, parameter)))
                for parameter in TRANSFER_PARAMS[pathway]
            }
            predicted = _simulate_transfer_curve(pathway, time_h, radius, params)
            valid = np.isfinite(predicted)
            residual = observed[valid] - predicted[valid]
            eta = log_response_residual(observed, predicted)
            rows.append(
                {
                    "excluded_well": well,
                    "pathway": pathway,
                    "pathway_short": SHORT_LABEL[pathway],
                    "distance_m": radius,
                    "rmse_m": float(np.sqrt(np.mean(residual * residual))) if residual.size else np.nan,
                    "sd_eta": float(np.std(eta[np.isfinite(eta)], ddof=1)) if np.sum(np.isfinite(eta)) > 1 else np.nan,
                    **{f"pred_{key}": value for key, value in params.items()},
                }
            )
    transfer = pd.DataFrame(rows)
    transfer["pathway"] = pd.Categorical(transfer["pathway"], categories=PATHWAYS, ordered=True)
    transfer = transfer.sort_values(["excluded_well", "pathway"]).reset_index(drop=True)
    best_index = transfer.groupby("excluded_well", observed=True)["rmse_m"].idxmin()
    transfer["preferred"] = False
    transfer.loc[best_index, "preferred"] = True
    transfer.to_csv(TABLE_DIR / "field_pathway_spatial_transfer.csv", index=False)
    return transfer


def plot_figure10_spatial_transfer(predictions: pd.DataFrame, comparison: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    transfer = build_four_pathway_spatial_transfer(predictions, comparison, metadata)
    transfer["well_order"] = pd.Categorical(transfer["excluded_well"], categories=WELL_ORDER, ordered=True)
    transfer = transfer.sort_values(["well_order", "pathway"])
    x = np.arange(len(WELL_ORDER))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), gridspec_kw={"width_ratios": [1.65, 0.75]})
    fig.subplots_adjust(wspace=0.34, top=0.76, bottom=0.26)
    legend_handles = []
    legend_labels = []
    for idx, pathway in enumerate(PATHWAYS):
        values = transfer.loc[transfer["pathway"] == pathway].set_index("excluded_well").reindex(WELL_ORDER)["rmse_m"]
        bars = axes[0].bar(x + (idx - 1.5) * width, values.to_numpy(dtype=float), width, color=COLORS[pathway], alpha=0.86, label=SHORT_LABEL[pathway])
        legend_handles.append(bars[0])
        legend_labels.append(SHORT_LABEL[pathway])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(WELL_ORDER, rotation=45, ha="right")
    axes[0].set_ylabel("LOO transfer RMSE (m)")
    fig.legend(legend_handles, legend_labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4)
    panel_label(axes[0], "a", "prediction for excluded wells")

    preferred = (
        transfer.loc[transfer.groupby("excluded_well", observed=True)["rmse_m"].idxmin()]
        .groupby("pathway", observed=True)
        .size()
        .reindex(PATHWAYS, fill_value=0)
    )
    mean_rmse = transfer.groupby("pathway", observed=True)["rmse_m"].mean().reindex(PATHWAYS)
    y = np.arange(len(PATHWAYS))
    axes[1].barh(y, preferred.to_numpy(dtype=float), color=[COLORS[p] for p in PATHWAYS], alpha=0.86)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([SHORT_LABEL[p] for p in PATHWAYS])
    axes[1].set_xlabel("Preferred wells")
    for idx, pathway in enumerate(PATHWAYS):
        axes[1].text(
            preferred.iloc[idx] + 0.08,
            idx,
            f"mean {mean_rmse.loc[pathway]:.3f} m",
            va="center",
            fontsize=6.6,
        )
    axes[1].set_xlim(0, max(float(preferred.max()) + 1.2, 2.0))
    panel_label(axes[1], "b", "support count")
    export(fig, "fig10_cross_validation_performance")
    return transfer


def write_report(
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    transfer: pd.DataFrame,
) -> None:
    preferred = comparison.loc[comparison.groupby("well", observed=True)["bic"].idxmin()]
    pref_counts = preferred["pathway"].value_counts().reindex(PATHWAYS, fill_value=0)
    boundary_diagnostic = pd.read_csv(TABLE_DIR / "leakage_boundary_diagnostic.csv")
    best_full_time = summary.sort_values("pooled_sd_eta").iloc[0]
    lagging_leaky = comparison[comparison["pathway"] == "Lagging Darcy with leakage"].copy()
    transfer_preferred = (
        transfer.loc[transfer.groupby("excluded_well", observed=True)["rmse_m"].idxmin()]
        .groupby("pathway", observed=True)
        .size()
        .reindex(PATHWAYS, fill_value=0)
    )
    lagging_leaky_active = lagging_leaky.loc[
        lagging_leaky["B_m"].notna()
        & (pd.read_csv(TABLE_DIR / "well_metadata.csv").set_index("well").loc[lagging_leaky["well"], "distance_m"].to_numpy(dtype=float) / lagging_leaky["B_m"].to_numpy(dtype=float) > 0.05),
        "well",
    ].tolist()
    report = [
        "# Four Interpretation Figure Reproducibility Report",
        "",
        "## Purpose",
        "",
        "Document the field figure workflow for the four interpretation models used in the manuscript.",
        "",
        "## Implemented Field Interpretations",
        "",
    ]
    for pathway in PATHWAYS:
        report.append(f"- {pathway}: implemented in `field_pathway_predictions.csv` and Figures 3-9.")
    report.extend(
        [
            "- Cooper-Jacob late time is retained as a screening diagnostic outside the main interpretation table.",
            "- The constant head boundary analysis is retained as a boundary sensitivity diagnostic in `leakage_boundary_diagnostic.csv`.",
            "",
            "## CMC Package Color Source",
            "",
            f"- Interpretation colors are sampled from formal `cmcrameri.cm.batlow` positions via this model source map: {CMC_PATHWAY_COLOR_SOURCES}; the active Figure 3-10 script no longer uses a hand written CMC style hex model palette.",
            f"- Distance colors are mapped through `cmcrameri.cm.{CMC_DISTANCE_COLORMAP}`.",
            f"- Time window heatmap colors are mapped through `cmcrameri.cm.{CMC_HEATMAP_COLORMAP}`.",
            f"- Lag time marker colors in Figure 7 are sampled from `cmcrameri.cm.{CMC_LAG_TIME_COLORMAP}` at {LAG_TIME_COLOR_SAMPLES}.",
            f"- Neutral dark/light colors are sampled from `cmcrameri.cm.{CMC_NEUTRAL_COLORMAP}` for axes, reference lines, outlines, observations, and text contrast.",
            "",
            "## Generated Tables",
            "",
            "- `tables/field_pathway_predictions.csv`: long point level predictions and log response factors.",
            "- `tables/field_pathway_model_comparison.csv`: well level RMSE, AIC/BIC, SD_eta, parameter values, and validity flags.",
            "- `tables/field_pathway_summary.csv`: model level pooled residual standard deviation and BIC preference counts.",
            "- `tables/field_pathway_time_window_summary.csv`: early/middle/late residual standard deviation by interpretation model.",
            "- `tables/field_pathway_spatial_transfer.csv`: parameter transfer diagnostics that exclude one well at a time for the four main interpretations.",
            "",
            "## Figure Outputs",
            "",
            "- Figure 3 overlays observed drawdown with Theis, Hantush-Jacob leaky, Lagging Darcy without leakage, and Lagging Darcy with leakage predictions.",
            "- Figure 4 uses a 2x2 layout comparing observed and predicted responses for the four main interpretations.",
            "- Figure 5 shows pooled response residual distributions for the four main pathways.",
            "- Figure 6 shows residual time structure by interpretation and early/middle/late residual scatter.",
            "- Figure 7 shows T/S by interpretation, leakage scale B, and lagging response times.",
            "- Figure 8 summarizes residual standard deviation, BIC preference, parameter COV, and relative BIC support.",
            "- Figure 9 focuses on lagging tradeoff, leakage activity, and whether adding leakage to lagging Darcy is supported.",
            "- Figure 10 reports spatial transfer diagnostics for the four interpretations by excluding one well at a time.",
            "",
            "## Main Numerical Results",
            "",
            f"- Primary pathway count: {predictions['pathway'].nunique()} implemented pathways.",
            f"- BIC preference counts: {pref_counts.to_dict()}.",
            f"- Lowest pooled SD_eta for the full observation record: {best_full_time['pathway']} ({best_full_time['pooled_sd_eta']:.3f}).",
            f"- Lagging Darcy with leakage active r/B wells: {lagging_leaky_active}.",
            f"- Spatial transfer preferred counts from excluding one well at a time: {transfer_preferred.to_dict()}.",
            f"- Constant head diagnostic support is evaluated separately from the main comparison for the four interpretations: {boundary_diagnostic.loc[boundary_diagnostic['strong_constant_head_support'], 'well'].tolist()}.",
            f"- Strong Hantush/constant head diagnostic support is treated as a separate leakage or boundary alternative: {boundary_diagnostic.loc[boundary_diagnostic['strong_leaky_boundary_support'], 'well'].tolist()}.",
            "",
            "## Interpretation Scope",
            "",
            "The figures implement Lagging Darcy with leakage as a formal finite radius field interpretation. Leakage and lagging terms are interpreted as transformation diagnostics unless independently constrained by validation, profile likelihood or posterior checks, or site evidence.",
        ]
    )
    report_text = "\n".join(report) + "\n"
    (PROJECT / "four_interpretation_figure_report.md").write_text(report_text, encoding="utf-8")
    (PROJECT / "multi_pathway_figure_report.md").write_text(report_text, encoding="utf-8")
    summary_payload = {
        "implemented_field_pathways": PATHWAYS,
        "auxiliary_diagnostics": ["Cooper-Jacob late time", "Constant head boundary", "Darcy finite radius"],
        "bic_preference_counts": {str(k): int(v) for k, v in pref_counts.items()},
        "lowest_full_time_pooled_sd_eta_pathway": str(best_full_time["pathway"]),
        "lowest_full_time_pooled_sd_eta": float(best_full_time["pooled_sd_eta"]),
        "lagging_leaky_active_r_over_b_wells": [str(well) for well in lagging_leaky_active],
        "tables": [
            "field_pathway_predictions.csv",
            "field_pathway_model_comparison.csv",
            "field_pathway_summary.csv",
            "field_pathway_time_window_summary.csv",
            "field_pathway_spatial_transfer.csv",
        ],
        "figures_rebuilt": [
            "fig03_observed_simulated_drawdown",
            "fig04_transformation_scatter",
            "fig05_transformation_residual_distributions",
            "fig06_residuals_through_time",
            "fig07_parameter_apparent_variability",
            "fig08_variance_ratio_cov_summary",
            "fig09_lagging_parameter_tradeoff",
            "fig10_cross_validation_performance",
        ],
        "interpretation_scope": "Lagging Darcy with leakage is formally implemented; leakage parameters are interpreted with independent constraints when available.",
    }
    (OUT_DIR / "four_interpretation_figure_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (OUT_DIR / "multi_pathway_figure_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    lock_style()
    predictions, comparison, summary = build_field_pathway_tables()
    metadata = pd.read_csv(TABLE_DIR / "well_metadata.csv")
    plot_figure2_parallel_site_layout(metadata)
    plot_figure3(predictions)
    plot_figure4(predictions, summary)
    plot_figure5(predictions, summary)
    phase_summary = plot_figure6(predictions)
    plot_figure7(comparison)
    plot_figure8(comparison, summary)
    plot_figure9(comparison, metadata)
    transfer = plot_figure10_spatial_transfer(predictions, comparison, metadata)
    write_report(predictions, comparison, summary, phase_summary, transfer)


if __name__ == "__main__":
    main()
