from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares
from scipy.special import exp1


PROJECT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT / "tables"
OUT_DIR = PROJECT / "outputs"

Q_M3S = 1.2e-4
LOG_RESIDUAL_OFFSET_M = 1.0e-3


def hantush_well_function(u: float, beta: float) -> float:
    """Return the Hantush leaky-aquifer well function W(u, r/B)."""
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
    pumping = theis_drawdown(time_h, radius_m, storage, trans)
    recharge_image = theis_drawdown(time_h, image_distance_m, storage, trans)
    return pumping - recharge_image


def log_response_residual(observed_m: np.ndarray, predicted_m: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(predicted_m)) or np.any(predicted_m + LOG_RESIDUAL_OFFSET_M <= 0.0):
        return np.full(observed_m.shape, 1.0e6, dtype=float)
    return np.log((observed_m + LOG_RESIDUAL_OFFSET_M) / (predicted_m + LOG_RESIDUAL_OFFSET_M))


def aic_bic(observed_m: np.ndarray, predicted_m: np.ndarray, n_params: int) -> tuple[float, float]:
    residual = observed_m - predicted_m
    rss = max(float(np.sum(residual * residual)), 1.0e-30)
    n = observed_m.size
    return n * np.log(rss / n) + 2.0 * n_params, n * np.log(rss / n) + n_params * np.log(n)


def semilog_slope(values: pd.Series, time_h: pd.Series) -> float:
    x = np.log(time_h.to_numpy(dtype=float))
    y = values.to_numpy(dtype=float)
    if y.size < 3 or np.ptp(x) <= 0.0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def add_phase_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("time_h").copy()
    n = len(result)
    phases = np.array(["early"] * n, dtype=object)
    phases[n // 3 : 2 * n // 3] = "middle"
    phases[2 * n // 3 :] = "late"
    result["phase"] = phases
    return result


def fit_leaky_model(
    time_h: np.ndarray,
    observed_m: np.ndarray,
    radius_m: float,
    storage_start: float,
    trans_start: float,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    lower = np.log(np.array([1.0e-8, 1.0e-8, 0.5], dtype=float))
    upper = np.log(np.array([1.0, 1.0e-1, 1.0e6], dtype=float))
    starts: list[np.ndarray] = []
    for leakage_b_start in [5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 1.0e4, 1.0e5]:
        starts.append(np.log(np.array([storage_start, trans_start, leakage_b_start], dtype=float)))
    for storage_generic, trans_generic in [(1.0e-4, 2.0e-5), (1.0e-3, 5.0e-5), (5.0e-3, 8.0e-5)]:
        for leakage_b_start in [10.0, 50.0, 300.0, 1.0e4]:
            starts.append(np.log(np.array([storage_generic, trans_generic, leakage_b_start], dtype=float)))

    best: tuple[float, np.ndarray, int] | None = None

    def residual_at(log_params: np.ndarray) -> np.ndarray:
        storage, trans, leakage_b = np.exp(log_params)
        predicted = leaky_drawdown(time_h, radius_m, storage, trans, leakage_b)
        return log_response_residual(observed_m, predicted)

    for start in starts:
        result = least_squares(
            residual_at,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=220,
            xtol=1.0e-6,
            ftol=1.0e-6,
            gtol=1.0e-6,
        )
        score = float(np.sum(result.fun * result.fun))
        if best is None or score < best[0]:
            best = (score, result.x, result.nfev)

    if best is None:
        raise RuntimeError("No leaky fit was produced.")

    params = np.exp(best[1])
    predicted = leaky_drawdown(time_h, radius_m, params[0], params[1], params[2])
    aic, bic = aic_bic(observed_m, predicted, 3)
    return params, predicted, aic, bic, best[2]


def fit_constant_head_boundary_model(
    time_h: np.ndarray,
    observed_m: np.ndarray,
    radius_m: float,
    storage_start: float,
    trans_start: float,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    lower_image = max(radius_m * 1.02, radius_m + 0.1)
    upper_image = max(1.0e4, radius_m * 1.0e3)
    lower = np.log(np.array([1.0e-8, 1.0e-8, lower_image], dtype=float))
    upper = np.log(np.array([1.0, 1.0e-1, upper_image], dtype=float))

    starts: list[np.ndarray] = []
    image_starts = [radius_m * factor for factor in [1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]]
    image_starts.extend([100.0, 300.0, 1000.0, 3000.0])
    for image_start in image_starts:
        starts.append(np.log(np.array([storage_start, trans_start, image_start], dtype=float)))
    for storage_generic, trans_generic in [(1.0e-4, 2.0e-5), (1.0e-3, 5.0e-5), (5.0e-3, 8.0e-5)]:
        for image_start in [radius_m * 1.5, radius_m * 3.0, radius_m * 10.0, 300.0, 1000.0]:
            starts.append(np.log(np.array([storage_generic, trans_generic, image_start], dtype=float)))

    best: tuple[float, np.ndarray, int] | None = None

    def residual_at(log_params: np.ndarray) -> np.ndarray:
        storage, trans, image_distance = np.exp(log_params)
        predicted = constant_head_image_drawdown(time_h, radius_m, storage, trans, image_distance)
        return log_response_residual(observed_m, predicted)

    for start in starts:
        result = least_squares(
            residual_at,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=220,
            xtol=1.0e-6,
            ftol=1.0e-6,
            gtol=1.0e-6,
        )
        score = float(np.sum(result.fun * result.fun))
        if best is None or score < best[0]:
            best = (score, result.x, result.nfev)

    if best is None:
        raise RuntimeError("No constant-head boundary fit was produced.")

    params = np.exp(best[1])
    predicted = constant_head_image_drawdown(time_h, radius_m, params[0], params[1], params[2])
    aic, bic = aic_bic(observed_m, predicted, 3)
    return params, predicted, aic, bic, best[2]


def main() -> None:
    points = pd.read_csv(TABLE_DIR / "point_predictions.csv")
    metadata = pd.read_csv(TABLE_DIR / "well_metadata.csv").set_index("well")
    parameters = pd.read_csv(TABLE_DIR / "best_fit_parameters.csv").set_index("well")
    model_metrics = pd.read_csv(TABLE_DIR / "refit_model_metrics_log_response_map.csv").set_index("well")

    rows = []
    for well, subset in points.groupby("well"):
        subset = add_phase_labels(subset)
        observed = subset["observed_m"].to_numpy(dtype=float)
        time_h = subset["time_h"].to_numpy(dtype=float)
        radius = float(metadata.loc[well, "distance_m"])

        slopes = {}
        for phase in ["early", "middle", "late"]:
            phase_data = subset.loc[subset["phase"] == phase]
            slopes[f"{phase}_observed_semilog_slope"] = semilog_slope(phase_data["observed_m"], phase_data["time_h"])
            slopes[f"{phase}_darcy_semilog_slope"] = semilog_slope(phase_data["darcy_pred_m"], phase_data["time_h"])
            slopes[f"{phase}_lagging_semilog_slope"] = semilog_slope(phase_data["lagging_pred_m"], phase_data["time_h"])

        leaky_params, leaky_prediction, aic_leaky, bic_leaky, nfev = fit_leaky_model(
            time_h,
            observed,
            radius,
            float(parameters.loc[well, "S_darcy"]),
            float(parameters.loc[well, "T_darcy_m2s"]),
        )
        boundary_params, boundary_prediction, aic_boundary, bic_boundary, nfev_boundary = fit_constant_head_boundary_model(
            time_h,
            observed,
            radius,
            float(parameters.loc[well, "S_darcy"]),
            float(parameters.loc[well, "T_darcy_m2s"]),
        )

        leakage_b = float(leaky_params[2])
        r_over_b = radius / leakage_b
        image_distance = float(boundary_params[2])
        image_distance_over_r = image_distance / radius
        leaky_eta = log_response_residual(observed, leaky_prediction)
        boundary_eta = log_response_residual(observed, boundary_prediction)
        slope_ratio = slopes["late_observed_semilog_slope"] / slopes["middle_observed_semilog_slope"]
        bic_darcy = float(model_metrics.loc[well, "bic_darcy"])
        bic_lagging = float(model_metrics.loc[well, "bic_lagging"])
        preferred = min(
            [
                ("Darcy", bic_darcy),
                ("Lagging Darcy", bic_lagging),
                ("Leaky", bic_leaky),
                ("Constant-head boundary", bic_boundary),
            ],
            key=lambda item: item[1],
        )[0]

        rows.append(
            {
                "well": well,
                "distance_m": radius,
                "azimuth_deg": float(metadata.loc[well, "azimuth_deg"]),
                "n_points": int(observed.size),
                **slopes,
                "late_middle_observed_slope_ratio": float(slope_ratio),
                "late_flattening_flag": bool(slope_ratio < 0.65),
                "S_leaky": float(leaky_params[0]),
                "T_leaky_m2s": float(leaky_params[1]),
                "B_leakage_m": leakage_b,
                "r_over_B": float(r_over_b),
                "active_leakage_flag": bool(r_over_b > 0.05),
                "S_constant_head": float(boundary_params[0]),
                "T_constant_head_m2s": float(boundary_params[1]),
                "image_distance_m": image_distance,
                "image_distance_over_r": float(image_distance_over_r),
                "active_constant_head_flag": bool(image_distance_over_r < 5.0),
                "rmse_darcy_m": float(np.sqrt(np.mean((observed - subset["darcy_pred_m"].to_numpy(dtype=float)) ** 2))),
                "rmse_lagging_m": float(
                    np.sqrt(np.mean((observed - subset["lagging_pred_m"].to_numpy(dtype=float)) ** 2))
                ),
                "rmse_leaky_m": float(np.sqrt(np.mean((observed - leaky_prediction) ** 2))),
                "rmse_constant_head_m": float(np.sqrt(np.mean((observed - boundary_prediction) ** 2))),
                "sd_eta_leaky": float(np.std(leaky_eta, ddof=1)),
                "sd_eta_constant_head": float(np.std(boundary_eta, ddof=1)),
                "aic_leaky": float(aic_leaky),
                "bic_leaky": float(bic_leaky),
                "aic_constant_head": float(aic_boundary),
                "bic_constant_head": float(bic_boundary),
                "bic_darcy": bic_darcy,
                "bic_lagging": bic_lagging,
                "delta_bic_leaky_minus_darcy": float(bic_leaky - bic_darcy),
                "delta_bic_leaky_minus_lagging": float(bic_leaky - bic_lagging),
                "delta_bic_constant_head_minus_darcy": float(bic_boundary - bic_darcy),
                "delta_bic_constant_head_minus_lagging": float(bic_boundary - bic_lagging),
                "preferred_bic_model": preferred,
                "strong_leaky_boundary_support": bool(
                    slope_ratio < 0.65 and r_over_b > 0.05 and preferred == "Leaky" and (bic_leaky - bic_darcy) < -10.0
                ),
                "strong_constant_head_support": bool(
                    slope_ratio < 0.65
                    and image_distance_over_r < 5.0
                    and preferred == "Constant-head boundary"
                    and (bic_boundary - bic_darcy) < -10.0
                ),
                "nfev_leaky_best_start": int(nfev),
                "nfev_constant_head_best_start": int(nfev_boundary),
            }
        )

    diagnostic = pd.DataFrame(rows).sort_values(["azimuth_deg", "distance_m"])
    diagnostic.to_csv(TABLE_DIR / "leakage_boundary_diagnostic.csv", index=False)

    summary = {
        "n_wells": int(diagnostic.shape[0]),
        "late_flattening_wells": diagnostic.loc[diagnostic["late_flattening_flag"], "well"].tolist(),
        "active_leakage_preferred_wells": diagnostic.loc[
            (diagnostic["preferred_bic_model"] == "Leaky") & diagnostic["active_leakage_flag"], "well"
        ].tolist(),
        "active_constant_head_preferred_wells": diagnostic.loc[
            (diagnostic["preferred_bic_model"] == "Constant-head boundary")
            & diagnostic["active_constant_head_flag"],
            "well",
        ].tolist(),
        "strong_leaky_boundary_support_wells": diagnostic.loc[
            diagnostic["strong_leaky_boundary_support"], "well"
        ].tolist(),
        "strong_constant_head_support_wells": diagnostic.loc[
            diagnostic["strong_constant_head_support"], "well"
        ].tolist(),
        "preferred_bic_counts": diagnostic["preferred_bic_model"].value_counts().to_dict(),
        "median_late_middle_observed_slope_ratio": float(np.nanmedian(diagnostic["late_middle_observed_slope_ratio"])),
        "interpretation": (
            "Late-time flattening is spatially selective rather than field-wide. "
            "The Hantush leaky and constant-head image-well diagnostics are treated as alternative transformation "
            "pathways because both can flatten late-time drawdown. Their support must be interpreted well by well, "
            "not as a global mandatory term in the current lagging Darcy diagnostic."
        ),
    }
    (OUT_DIR / "leakage_boundary_diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
