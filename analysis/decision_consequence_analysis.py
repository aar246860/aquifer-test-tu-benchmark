from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap

import cmcrameri.cm as cmc


PROJECT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT / "tables"
FIG_DIR = PROJECT / "figures"
OUT_DIR = PROJECT / "outputs"

PATHWAYS = [
    "Theis confined",
    "Hantush-Jacob leaky",
    "Lagging Darcy no-leakage",
    "Lagging Darcy with leakage",
]

SHORT_LABEL = {
    "Theis confined": "Theis",
    "Hantush-Jacob leaky": "Hantush\nleaky",
    "Lagging Darcy no-leakage": "Lagging\nno leakage",
    "Lagging Darcy with leakage": "Lagging\nleaky",
}

CASE_OFFSET_M = {
    "Massachusetts": 1.0e-3,
    "Lovelock Valley": 0.0152,
}

CASE_LABEL = {
    "Massachusetts": "Massachusetts",
    "Lovelock Valley": "Lovelock Valley",
}

THRESHOLD_QUANTILE = 0.90
FALSE_SAFE_RISK_TOLERANCE = 0.05
DECISION_SCHEMES = ["strict_bic", "variance_window"]


def get_cmc_colormap(name: str) -> Colormap:
    cmap = getattr(cmc, name, None)
    if cmap is None:
        raise ValueError(f"Unknown cmcrameri colormap: {name}")
    return cmap


def cmc_color(name: str, sample: float) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in get_cmc_colormap(name)(sample))


COLORS = {
    "Theis confined": cmc_color("batlow", 0.12),
    "Hantush-Jacob leaky": cmc_color("batlow", 0.38),
    "Lagging Darcy no-leakage": cmc_color("batlow", 0.66),
    "Lagging Darcy with leakage": cmc_color("batlow", 0.90),
}
CASE_COLORS = {
    "Massachusetts": cmc_color("batlow", 0.24),
    "Lovelock Valley": cmc_color("batlow", 0.78),
}
NO_TU_COLOR = cmc_color("batlow", 0.22)
TU_COLOR = cmc_color("batlow", 0.74)
NEUTRAL_DARK = cmc_color("grayC", 0.12)
NEUTRAL_MID = cmc_color("grayC", 0.45)
NEUTRAL_LIGHT = cmc_color("grayC", 0.92)


def lock_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 6.8,
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


def panel_label(ax: plt.Axes, label: str, text: str) -> None:
    ax.text(
        0.0,
        1.04,
        f"({label}) {text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=8.2,
    )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]
    return float(np.interp(quantile, cumulative, values))


def load_predictions() -> pd.DataFrame:
    mass = pd.read_csv(TABLE_DIR / "field_pathway_predictions.csv")
    mass["case"] = "Massachusetts"
    mass = mass[["case", "well", "time_h", "pathway", "observed_m", "predicted_m", "eta"]]

    lovelock = pd.read_csv(TABLE_DIR / "lovelock_response_predictions.csv")
    lovelock["case"] = "Lovelock Valley"
    lovelock = lovelock.rename(columns={"model": "pathway", "drawdown_m": "observed_m"})
    lovelock["time_h"] = pd.to_numeric(lovelock["elapsed_days"], errors="coerce") * 24.0
    lovelock = lovelock[["case", "well", "time_h", "pathway", "observed_m", "predicted_m", "eta"]]

    predictions = pd.concat([mass, lovelock], ignore_index=True)
    predictions = predictions[predictions["pathway"].isin(PATHWAYS)].copy()
    for column in ["time_h", "observed_m", "predicted_m", "eta"]:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    predictions = predictions.dropna(subset=["time_h", "observed_m", "predicted_m", "eta"])
    predictions["record_id"] = (
        predictions["case"]
        + "|"
        + predictions["well"].astype(str)
        + "|"
        + predictions["time_h"].map(lambda value: f"{value:.10g}")
    )
    return predictions


def compute_bic_weights_from_long(comparison: pd.DataFrame, case: str) -> pd.DataFrame:
    comparison = comparison[comparison["pathway"].isin(PATHWAYS)].copy()
    comparison["bic"] = pd.to_numeric(comparison["bic"], errors="coerce")
    rows = []
    for well, frame in comparison.groupby("well"):
        frame = frame.set_index("pathway").reindex(PATHWAYS).dropna(subset=["bic"])
        min_bic = float(frame["bic"].min())
        evidence = np.exp(-0.5 * (frame["bic"].to_numpy(dtype=float) - min_bic))
        weights = evidence / evidence.sum()
        for pathway, weight in zip(frame.index, weights):
            rows.append({"case": case, "well": well, "pathway": pathway, "bic_weight": float(weight)})
    return pd.DataFrame(rows)


def load_bma_weights() -> pd.DataFrame:
    mass_comparison = pd.read_csv(TABLE_DIR / "field_pathway_model_comparison.csv")
    mass_weights = compute_bic_weights_from_long(mass_comparison, "Massachusetts")

    lovelock = pd.read_csv(TABLE_DIR / "lovelock_model_comparison.csv")
    column_map = {
        "Theis confined": "bic_theis_confined",
        "Hantush-Jacob leaky": "bic_hantush_jacob_leaky",
        "Lagging Darcy no-leakage": "bic_lagging_darcy_no_leakage",
        "Lagging Darcy with leakage": "bic_lagging_darcy_with_leakage",
    }
    rows = []
    for _, row in lovelock.iterrows():
        bics = np.array([float(row[column_map[pathway]]) for pathway in PATHWAYS], dtype=float)
        min_bic = float(np.nanmin(bics))
        evidence = np.exp(-0.5 * (bics - min_bic))
        weights = evidence / evidence.sum()
        for pathway, weight in zip(PATHWAYS, weights):
            rows.append(
                {
                    "case": "Lovelock Valley",
                    "well": row["well"],
                    "pathway": pathway,
                    "bic_weight": float(weight),
                }
            )
    lovelock_weights = pd.DataFrame(rows)
    weights = pd.concat([mass_weights, lovelock_weights], ignore_index=True)
    weights.to_csv(TABLE_DIR / "field_bma_pathway_weights.csv", index=False)
    return weights


def compute_bma_record_table(predictions: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = predictions.merge(weights, on=["case", "well", "pathway"], how="left")
    if predictions["bic_weight"].isna().any():
        missing = predictions[predictions["bic_weight"].isna()][["case", "well", "pathway"]].drop_duplicates()
        raise ValueError(f"Missing BMA weights for {missing.to_dict(orient='records')}")

    residuals = {
        (case, pathway): frame["eta"].to_numpy(dtype=float)
        for (case, pathway), frame in predictions.groupby(["case", "pathway"])
    }

    records: list[dict[str, float | int | str]] = []
    pathway_rows: list[dict[str, float | str]] = []

    for case, case_frame in predictions.groupby("case", sort=False):
        threshold = float(
            np.quantile(
                case_frame.loc[case_frame["pathway"] == "Theis confined", "observed_m"].to_numpy(dtype=float),
                THRESHOLD_QUANTILE,
            )
        )
        offset_m = CASE_OFFSET_M[case]

        for record_id, group in case_frame.groupby("record_id", sort=False):
            group = group.set_index("pathway").reindex(PATHWAYS).reset_index()
            model_predictions = group["predicted_m"].to_numpy(dtype=float)
            model_weights = group["bic_weight"].to_numpy(dtype=float)
            model_weights = model_weights / model_weights.sum()
            observed = float(group["observed_m"].iloc[0])
            well = str(group["well"].iloc[0])
            time_h = float(group["time_h"].iloc[0])

            no_mean = float(np.sum(model_weights * model_predictions))
            no_q05 = weighted_quantile(model_predictions, model_weights, 0.05)
            no_q95 = weighted_quantile(model_predictions, model_weights, 0.95)
            no_interval = no_q95 - no_q05
            no_p_exceed = float(np.sum(model_weights * (model_predictions > threshold)))

            mixture_values: list[np.ndarray] = []
            mixture_weights: list[np.ndarray] = []
            for pathway, prediction, weight in zip(PATHWAYS, model_predictions, model_weights):
                eta_values = residuals[(case, pathway)]
                adjusted = (prediction + offset_m) * np.exp(eta_values) - offset_m
                adjusted = np.maximum(0.0, adjusted)
                mixture_values.append(adjusted)
                mixture_weights.append(np.full(len(adjusted), weight / len(adjusted), dtype=float))
                pathway_rows.append(
                    {
                        "case": case,
                        "record_id": record_id,
                        "well": well,
                        "time_h": time_h,
                        "pathway": pathway,
                        "bic_weight": float(weight),
                        "deterministic_prediction_m": float(prediction),
                        "log_model_factor_sd": float(np.std(eta_values, ddof=1)),
                    }
                )

            tu_values = np.concatenate(mixture_values)
            tu_weights = np.concatenate(mixture_weights)
            tu_mean = float(np.average(tu_values, weights=tu_weights))
            tu_q05 = weighted_quantile(tu_values, tu_weights, 0.05)
            tu_q95 = weighted_quantile(tu_values, tu_weights, 0.95)
            tu_interval = tu_q95 - tu_q05
            tu_p_exceed = float(np.sum(tu_weights[tu_values > threshold]))

            records.append(
                {
                    "case": case,
                    "record_id": record_id,
                    "well": well,
                    "time_h": time_h,
                    "observed_m": observed,
                    "threshold_quantile": THRESHOLD_QUANTILE,
                    "threshold_m": threshold,
                    "observed_exceeds_threshold": bool(observed > threshold),
                    "no_tu_bma_mean_m": no_mean,
                    "tu_aware_bma_mean_m": tu_mean,
                    "no_tu_bma_q05_m": no_q05,
                    "no_tu_bma_q95_m": no_q95,
                    "tu_aware_bma_q05_m": tu_q05,
                    "tu_aware_bma_q95_m": tu_q95,
                    "no_tu_interval_width_m": no_interval,
                    "tu_aware_interval_width_m": tu_interval,
                    "no_tu_exceedance_probability": no_p_exceed,
                    "tu_aware_exceedance_probability": tu_p_exceed,
                    "exceedance_probability_increase": tu_p_exceed - no_p_exceed,
                    "allowable_pumping_overstatement": tu_q95 / no_mean if no_mean > offset_m else np.nan,
                    "no_tu_false_safe": bool((observed > threshold) and (no_p_exceed <= FALSE_SAFE_RISK_TOLERANCE)),
                    "tu_aware_false_safe": bool((observed > threshold) and (tu_p_exceed <= FALSE_SAFE_RISK_TOLERANCE)),
                    "no_tu_false_safe_regret": max(0.0, observed - threshold)
                    if (observed > threshold and no_p_exceed <= FALSE_SAFE_RISK_TOLERANCE)
                    else 0.0,
                    "tu_aware_false_safe_regret": max(0.0, observed - threshold)
                    if (observed > threshold and tu_p_exceed <= FALSE_SAFE_RISK_TOLERANCE)
                    else 0.0,
                }
            )

    record_table = pd.DataFrame(records)
    pathway_table = pd.DataFrame(pathway_rows)
    record_table.to_csv(TABLE_DIR / "field_bma_inheritance_record_predictions.csv", index=False)
    pathway_table.to_csv(TABLE_DIR / "field_bma_inheritance_pathway_contributions.csv", index=False)
    return record_table, pathway_table


def summarize_bma(record_table: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case, frame in record_table.groupby("case", sort=False):
        observed_exceed = frame["observed_exceeds_threshold"].to_numpy(dtype=bool)
        active = frame[(frame["observed_m"] > CASE_OFFSET_M[case]) & (frame["no_tu_bma_mean_m"] > CASE_OFFSET_M[case])]
        no_false = frame["no_tu_false_safe"].to_numpy(dtype=bool)
        tu_false = frame["tu_aware_false_safe"].to_numpy(dtype=bool)
        rows.append(
            {
                "case": case,
                "n_records": int(len(frame)),
                "threshold_quantile": THRESHOLD_QUANTILE,
                "threshold_m": float(frame["threshold_m"].iloc[0]),
                "observed_exceedance_count": int(observed_exceed.sum()),
                "mean_no_tu_exceedance_probability": float(frame["no_tu_exceedance_probability"].mean()),
                "mean_tu_aware_exceedance_probability": float(frame["tu_aware_exceedance_probability"].mean()),
                "mean_exceedance_probability_increase": float(frame["exceedance_probability_increase"].mean()),
                "median_no_tu_interval_width_m": float(frame["no_tu_interval_width_m"].median()),
                "median_tu_aware_interval_width_m": float(frame["tu_aware_interval_width_m"].median()),
                "median_interval_width_ratio": float(
                    np.nanmedian(frame["tu_aware_interval_width_m"] / frame["no_tu_interval_width_m"].replace(0, np.nan))
                ),
                "median_allowable_pumping_overstatement": float(active["allowable_pumping_overstatement"].median()),
                "p95_allowable_pumping_overstatement": float(active["allowable_pumping_overstatement"].quantile(0.95)),
                "no_tu_false_safe_fraction": float(no_false.sum() / max(observed_exceed.sum(), 1)),
                "tu_aware_false_safe_fraction": float(tu_false.sum() / max(observed_exceed.sum(), 1)),
                "no_tu_expected_false_safe_regret_m": float(frame["no_tu_false_safe_regret"].mean()),
                "tu_aware_expected_false_safe_regret_m": float(frame["tu_aware_false_safe_regret"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    weight_summary = (
        weights.groupby(["case", "pathway"], as_index=False)["bic_weight"]
        .mean()
        .rename(columns={"bic_weight": "mean_bic_weight"})
    )
    weight_summary.to_csv(TABLE_DIR / "field_bma_pathway_weight_summary.csv", index=False)
    summary.to_csv(TABLE_DIR / "field_bma_inheritance_summary.csv", index=False)
    return summary


def compute_management_weights_from_long(comparison: pd.DataFrame, case: str) -> pd.DataFrame:
    comparison = comparison[comparison["pathway"].isin(PATHWAYS)].copy()
    comparison["bic"] = pd.to_numeric(comparison["bic"], errors="coerce")
    rows = []
    for well, frame in comparison.groupby("well"):
        frame = frame.set_index("pathway").reindex(PATHWAYS).dropna(subset=["bic"])
        bic = frame["bic"].to_numpy(dtype=float)
        delta = bic - float(np.nanmin(bic))
        strict_evidence = np.exp(-0.5 * delta)
        strict_weights = strict_evidence / strict_evidence.sum()
        # Li and Tsai's variance window logic avoids collapsing all support onto
        # one model when the information-criterion spread itself is large.
        bic_scale = max(float(np.nanstd(delta, ddof=1)), 1.0)
        vw_evidence = np.exp(-0.5 * delta / bic_scale)
        vw_weights = vw_evidence / vw_evidence.sum()
        for pathway, strict, vw, delta_bic in zip(frame.index, strict_weights, vw_weights, delta):
            rows.append(
                {
                    "case": case,
                    "well": well,
                    "pathway": pathway,
                    "strict_bic_weight": float(strict),
                    "variance_window_weight": float(vw),
                    "delta_bic": float(delta_bic),
                    "bic_window_scale": float(bic_scale),
                }
            )
    return pd.DataFrame(rows)


def load_management_weights() -> pd.DataFrame:
    mass_comparison = pd.read_csv(TABLE_DIR / "field_pathway_model_comparison.csv")
    mass_weights = compute_management_weights_from_long(mass_comparison, "Massachusetts")

    lovelock = pd.read_csv(TABLE_DIR / "lovelock_model_comparison.csv")
    column_map = {
        "Theis confined": "bic_theis_confined",
        "Hantush-Jacob leaky": "bic_hantush_jacob_leaky",
        "Lagging Darcy no-leakage": "bic_lagging_darcy_no_leakage",
        "Lagging Darcy with leakage": "bic_lagging_darcy_with_leakage",
    }
    rows = []
    for _, row in lovelock.iterrows():
        bic = np.array([float(row[column_map[pathway]]) for pathway in PATHWAYS], dtype=float)
        delta = bic - float(np.nanmin(bic))
        strict_evidence = np.exp(-0.5 * delta)
        strict_weights = strict_evidence / strict_evidence.sum()
        bic_scale = max(float(np.nanstd(delta, ddof=1)), 1.0)
        vw_evidence = np.exp(-0.5 * delta / bic_scale)
        vw_weights = vw_evidence / vw_evidence.sum()
        for pathway, strict, vw, delta_bic in zip(PATHWAYS, strict_weights, vw_weights, delta):
            rows.append(
                {
                    "case": "Lovelock Valley",
                    "well": row["well"],
                    "pathway": pathway,
                    "strict_bic_weight": float(strict),
                    "variance_window_weight": float(vw),
                    "delta_bic": float(delta_bic),
                    "bic_window_scale": float(bic_scale),
                }
            )
    weights = pd.concat([mass_weights, pd.DataFrame(rows)], ignore_index=True)
    weights.to_csv(TABLE_DIR / "field_bma_management_pathway_weights.csv", index=False)
    return weights


def load_management_parameters() -> pd.DataFrame:
    mass = pd.read_csv(TABLE_DIR / "field_pathway_model_comparison.csv")
    mass_metadata = pd.read_csv(TABLE_DIR / "well_metadata.csv")[["well", "distance_m"]]
    mass = mass.merge(mass_metadata, on="well", how="left")
    mass["case"] = "Massachusetts"
    mass = mass[["case", "well", "pathway", "T_m2s", "S", "distance_m"]]

    lovelock = pd.read_csv(TABLE_DIR / "lovelock_fit_parameters.csv")
    lovelock = lovelock.rename(columns={"model": "pathway"})
    lovelock["case"] = "Lovelock Valley"
    lovelock = lovelock[["case", "well", "pathway", "T_m2s", "S", "distance_m"]]

    parameters = pd.concat([mass, lovelock], ignore_index=True)
    parameters = parameters[parameters["pathway"].isin(PATHWAYS)].copy()
    for column in ["T_m2s", "S", "distance_m"]:
        parameters[column] = pd.to_numeric(parameters[column], errors="coerce")
    parameters = parameters.dropna(subset=["T_m2s", "S", "distance_m"])
    parameters["hydraulic_diffusivity_m2s"] = parameters["T_m2s"] / parameters["S"]
    parameters["characteristic_response_time_h"] = (
        parameters["distance_m"] ** 2 * parameters["S"] / (4.0 * parameters["T_m2s"]) / 3600.0
    )
    parameters.to_csv(TABLE_DIR / "field_bma_management_pathway_parameters.csv", index=False)
    return parameters


def compute_management_diagnostics(parameters: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    frame = parameters.merge(weights, on=["case", "well", "pathway"], how="left")
    if frame[["strict_bic_weight", "variance_window_weight"]].isna().any().any():
        missing = frame[frame["strict_bic_weight"].isna()][["case", "well", "pathway"]].drop_duplicates()
        raise ValueError(f"Missing management BMA weights for {missing.to_dict(orient='records')}")

    rows = []
    for (case, well), group in frame.groupby(["case", "well"], sort=False):
        group = group.set_index("pathway").reindex(PATHWAYS).dropna(subset=["T_m2s", "S"]).reset_index()
        theis = group[group["pathway"] == "Theis confined"].iloc[0]
        for scheme, weight_column in [
            ("strict_bic", "strict_bic_weight"),
            ("variance_window", "variance_window_weight"),
        ]:
            model_weights = group[weight_column].to_numpy(dtype=float)
            model_weights = model_weights / model_weights.sum()
            best = group.iloc[int(np.argmax(model_weights))]

            t_q05 = weighted_quantile(group["T_m2s"].to_numpy(dtype=float), model_weights, 0.05)
            t_q50 = weighted_quantile(group["T_m2s"].to_numpy(dtype=float), model_weights, 0.50)
            t_q95 = weighted_quantile(group["T_m2s"].to_numpy(dtype=float), model_weights, 0.95)
            d_q05 = weighted_quantile(group["hydraulic_diffusivity_m2s"].to_numpy(dtype=float), model_weights, 0.05)
            d_q50 = weighted_quantile(group["hydraulic_diffusivity_m2s"].to_numpy(dtype=float), model_weights, 0.50)
            d_q95 = weighted_quantile(group["hydraulic_diffusivity_m2s"].to_numpy(dtype=float), model_weights, 0.95)
            tc_q05 = weighted_quantile(group["characteristic_response_time_h"].to_numpy(dtype=float), model_weights, 0.05)
            tc_q50 = weighted_quantile(group["characteristic_response_time_h"].to_numpy(dtype=float), model_weights, 0.50)
            tc_q95 = weighted_quantile(group["characteristic_response_time_h"].to_numpy(dtype=float), model_weights, 0.95)

            t_policy = float(best["T_m2s"])
            tc_policy = float(best["characteristic_response_time_h"])
            t_theis = float(theis["T_m2s"])
            tc_theis = float(theis["characteristic_response_time_h"])

            capacity_violation = float(np.sum(model_weights[group["T_m2s"].to_numpy(dtype=float) > t_policy]))
            early_violation = float(
                np.sum(model_weights[group["characteristic_response_time_h"].to_numpy(dtype=float) < tc_policy])
            )
            capacity_regret = float(
                np.sum(
                    model_weights
                    * np.maximum(group["T_m2s"].to_numpy(dtype=float) / t_policy - 1.0, 0.0)
                )
            )
            early_regret = float(
                np.sum(
                    model_weights
                    * np.maximum(tc_policy / group["characteristic_response_time_h"].to_numpy(dtype=float) - 1.0, 0.0)
                )
            )

            rows.append(
                {
                    "case": case,
                    "well": well,
                    "scheme": scheme,
                    "best_pathway": str(best["pathway"]),
                    "theis_T_m2s": t_theis,
                    "single_best_T_m2s": t_policy,
                    "bma_T_q05_m2s": t_q05,
                    "bma_T_q50_m2s": t_q50,
                    "bma_T_q95_m2s": t_q95,
                    "single_best_response_time_h": tc_policy,
                    "theis_response_time_h": tc_theis,
                    "bma_response_time_q05_h": tc_q05,
                    "bma_response_time_q50_h": tc_q50,
                    "bma_response_time_q95_h": tc_q95,
                    "bma_diffusivity_q05_m2s": d_q05,
                    "bma_diffusivity_q50_m2s": d_q50,
                    "bma_diffusivity_q95_m2s": d_q95,
                    "robust_capacity_factor": max(1.0, t_q95 / t_policy),
                    "theis_capacity_factor": max(1.0, t_q95 / t_theis),
                    "capacity_violation_probability": capacity_violation,
                    "capacity_regret_factor": capacity_regret,
                    "early_response_factor": max(1.0, tc_policy / tc_q05),
                    "theis_early_response_factor": max(1.0, tc_theis / tc_q05),
                    "early_response_probability": early_violation,
                    "early_response_regret_factor": early_regret,
                    "diffusivity_spread_ratio": d_q95 / d_q05 if d_q05 > 0 else np.nan,
                    "transmissivity_spread_ratio": t_q95 / t_q05 if t_q05 > 0 else np.nan,
                    "response_time_spread_ratio": tc_q95 / tc_q05 if tc_q05 > 0 else np.nan,
                }
            )

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(TABLE_DIR / "field_bma_management_well_diagnostics.csv", index=False)
    return diagnostics


def summarize_management_diagnostics(diagnostics: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, scheme), frame in diagnostics.groupby(["case", "scheme"], sort=False):
        weight_column = "strict_bic_weight" if scheme == "strict_bic" else "variance_window_weight"
        weight_frame = weights[weights["case"] == case]
        weight_means = (
            weight_frame.groupby("pathway")[weight_column]
            .mean()
            .reindex(PATHWAYS)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        best_counts = frame["best_pathway"].value_counts().reindex(PATHWAYS).fillna(0).astype(int)
        rows.append(
            {
                "case": case,
                "scheme": scheme,
                "n_wells": int(len(frame)),
                "mean_weight_theis": float(weight_means[0]),
                "mean_weight_hantush_leaky": float(weight_means[1]),
                "mean_weight_lagging_no_leakage": float(weight_means[2]),
                "mean_weight_lagging_leaky": float(weight_means[3]),
                "best_pathway_counts": " / ".join(str(value) for value in best_counts.to_numpy()),
                "median_robust_capacity_factor": float(frame["robust_capacity_factor"].median()),
                "p95_robust_capacity_factor": float(frame["robust_capacity_factor"].quantile(0.95)),
                "mean_capacity_violation_probability": float(frame["capacity_violation_probability"].mean()),
                "mean_capacity_regret_factor": float(frame["capacity_regret_factor"].mean()),
                "median_early_response_factor": float(frame["early_response_factor"].median()),
                "p95_early_response_factor": float(frame["early_response_factor"].quantile(0.95)),
                "mean_early_response_probability": float(frame["early_response_probability"].mean()),
                "mean_early_response_regret_factor": float(frame["early_response_regret_factor"].mean()),
                "median_diffusivity_spread_ratio": float(frame["diffusivity_spread_ratio"].median()),
                "p95_diffusivity_spread_ratio": float(frame["diffusivity_spread_ratio"].quantile(0.95)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "field_bma_management_summary.csv", index=False)
    return summary


def plot_bma_figure(summary: pd.DataFrame, weights: pd.DataFrame) -> None:
    weight_summary = (
        weights.groupby(["case", "pathway"], as_index=False)["bic_weight"]
        .mean()
        .rename(columns={"bic_weight": "mean_bic_weight"})
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.05), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    cases = ["Massachusetts", "Lovelock Valley"]
    x = np.arange(len(cases), dtype=float)
    width = 0.34

    panel_label(ax_a, "a", "BMA predictive interval width")
    no_vals = summary.set_index("case").reindex(cases)["median_no_tu_interval_width_m"].to_numpy(dtype=float)
    tu_vals = summary.set_index("case").reindex(cases)["median_tu_aware_interval_width_m"].to_numpy(dtype=float)
    ax_a.bar(x - width / 2, no_vals, width=width, color=NO_TU_COLOR, edgecolor=NEUTRAL_DARK, linewidth=0.6, label="BMA without log factor")
    ax_a.bar(x + width / 2, tu_vals, width=width, color=TU_COLOR, edgecolor=NEUTRAL_DARK, linewidth=0.6, label="BMA with log factor")
    for xpos, value in zip(x - width / 2, no_vals):
        ax_a.text(xpos, value, f"{value:.3g}", ha="center", va="bottom", fontsize=6.2)
    for xpos, value in zip(x + width / 2, tu_vals):
        ax_a.text(xpos, value, f"{value:.3g}", ha="center", va="bottom", fontsize=6.2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cases)
    ax_a.set_ylabel("Median 90% width (m)")
    ax_a.grid(axis="y", color=NEUTRAL_LIGHT, lw=0.6)
    ax_a.legend(frameon=False, loc="upper right")

    panel_label(ax_b, "b", "exceedance probability inherited by BMA")
    no_p = summary.set_index("case").reindex(cases)["mean_no_tu_exceedance_probability"].to_numpy(dtype=float)
    tu_p = summary.set_index("case").reindex(cases)["mean_tu_aware_exceedance_probability"].to_numpy(dtype=float)
    ax_b.bar(x - width / 2, no_p, width=width, color=NO_TU_COLOR, edgecolor=NEUTRAL_DARK, linewidth=0.6, label="BMA without log factor")
    ax_b.bar(x + width / 2, tu_p, width=width, color=TU_COLOR, edgecolor=NEUTRAL_DARK, linewidth=0.6, label="BMA with log factor")
    for xpos, value in zip(x - width / 2, no_p):
        ax_b.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    for xpos, value in zip(x + width / 2, tu_p):
        ax_b.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(cases)
    ax_b.set_ylabel("Mean exceedance probability")
    ax_b.set_ylim(0.0, max(tu_p.max(), no_p.max()) * 1.25)
    ax_b.grid(axis="y", color=NEUTRAL_LIGHT, lw=0.6)

    panel_label(ax_c, "c", "allowable pumping margin from log factor")
    margin_vals = summary.set_index("case").reindex(cases)["median_allowable_pumping_overstatement"].to_numpy(dtype=float)
    tail_vals = summary.set_index("case").reindex(cases)["p95_allowable_pumping_overstatement"].to_numpy(dtype=float)
    ax_c.bar(x - width / 2, margin_vals, width=width, color=CASE_COLORS["Massachusetts"], edgecolor=NEUTRAL_DARK, linewidth=0.6, label="Median")
    ax_c.bar(x + width / 2, tail_vals, width=width, color=CASE_COLORS["Lovelock Valley"], edgecolor=NEUTRAL_DARK, linewidth=0.6, label="95th percentile")
    for xpos, value in zip(x - width / 2, margin_vals):
        ax_c.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    for xpos, value in zip(x + width / 2, tail_vals):
        ax_c.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax_c.axhline(1.0, color=NEUTRAL_MID, lw=0.8, ls=":")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(cases)
    ax_c.set_ylabel("Allowable rate without / with factor")
    ax_c.set_ylim(0.95, max(tail_vals.max() * 1.18, 1.2))
    ax_c.grid(axis="y", color=NEUTRAL_LIGHT, lw=0.6)
    ax_c.legend(frameon=False, loc="upper left")

    panel_label(ax_d, "d", "BIC posterior pathway weights")
    xw = np.arange(len(PATHWAYS), dtype=float)
    for offset, case in [(-width / 2, "Massachusetts"), (width / 2, "Lovelock Valley")]:
        vals = (
            weight_summary[weight_summary["case"] == case]
            .set_index("pathway")
            .reindex(PATHWAYS)["mean_bic_weight"]
            .to_numpy(dtype=float)
        )
        ax_d.bar(
            xw + offset,
            vals,
            width=width,
            color=[COLORS[pathway] for pathway in PATHWAYS],
            alpha=0.95 if case == "Massachusetts" else 0.55,
            edgecolor=NEUTRAL_DARK,
            linewidth=0.6,
            hatch="" if case == "Massachusetts" else "//",
            label=CASE_LABEL[case],
        )
    ax_d.set_xticks(xw)
    ax_d.set_xticklabels([SHORT_LABEL[pathway] for pathway in PATHWAYS])
    ax_d.set_ylabel("Mean BMA weight")
    ax_d.set_ylim(0.0, 1.0)
    ax_d.grid(axis="y", color=NEUTRAL_LIGHT, lw=0.6)
    ax_d.legend(frameon=False, loc="upper left")

    export(fig, "fig12_decision_consequence")


def plot_management_figure(summary: pd.DataFrame, weights: pd.DataFrame) -> None:
    vw_summary = summary[summary["scheme"] == "variance_window"].set_index("case").reindex(
        ["Massachusetts", "Lovelock Valley"]
    )
    strict_summary = summary[summary["scheme"] == "strict_bic"].set_index("case").reindex(
        ["Massachusetts", "Lovelock Valley"]
    )
    vw_weights = (
        weights.groupby(["case", "pathway"], as_index=False)["variance_window_weight"]
        .mean()
        .rename(columns={"variance_window_weight": "mean_weight"})
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.05), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    cases = ["Massachusetts", "Lovelock Valley"]
    x = np.arange(len(cases), dtype=float)
    width = 0.34

    panel_label(ax_a, "a", "variance window BMA model weights")
    xw = np.arange(len(PATHWAYS), dtype=float)
    for offset, case in [(-width / 2, "Massachusetts"), (width / 2, "Lovelock Valley")]:
        vals = (
            vw_weights[vw_weights["case"] == case]
            .set_index("pathway")
            .reindex(PATHWAYS)["mean_weight"]
            .to_numpy(dtype=float)
        )
        ax_a.bar(
            xw + offset,
            vals,
            width=width,
            color=[COLORS[pathway] for pathway in PATHWAYS],
            alpha=0.95 if case == "Massachusetts" else 0.55,
            edgecolor=NEUTRAL_DARK,
            linewidth=0.6,
            hatch="" if case == "Massachusetts" else "//",
            label=CASE_LABEL[case],
        )
    ax_a.set_xticks(xw)
    ax_a.set_xticklabels([SHORT_LABEL[pathway] for pathway in PATHWAYS])
    ax_a.set_ylabel("Mean model weight")
    ax_a.set_ylim(0.0, 0.42)
    ax_a.grid(axis="y", color=NEUTRAL_LIGHT, lw=0.6)
    ax_a.legend(frameon=False, loc="upper right")

    panel_label(ax_b, "b", "hydraulic diffusivity spread")
    diff_med = vw_summary["median_diffusivity_spread_ratio"].to_numpy(dtype=float)
    diff_p95 = vw_summary["p95_diffusivity_spread_ratio"].to_numpy(dtype=float)
    ax_b.bar(x - width / 2, diff_med, width=width, color=NO_TU_COLOR, edgecolor=NEUTRAL_DARK, linewidth=0.6, label="Median")
    ax_b.bar(x + width / 2, diff_p95, width=width, color=TU_COLOR, edgecolor=NEUTRAL_DARK, linewidth=0.6, label="95th percentile")
    for xpos, value in zip(x - width / 2, diff_med):
        ax_b.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    for xpos, value in zip(x + width / 2, diff_p95):
        ax_b.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax_b.axhline(1.0, color=NEUTRAL_MID, lw=0.8, ls=":")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(cases)
    ax_b.set_ylabel(r"$D_{95}/D_{05}$, $D=T/S$")
    ax_b.set_yscale("log")
    ax_b.grid(axis="y", color=NEUTRAL_LIGHT, lw=0.6)
    ax_b.legend(frameon=False, loc="upper left")

    panel_label(ax_c, "c", "hydraulic control capacity factor")
    strict_cap = strict_summary["median_robust_capacity_factor"].to_numpy(dtype=float)
    vw_cap = vw_summary["median_robust_capacity_factor"].to_numpy(dtype=float)
    ax_c.bar(x - width / 2, strict_cap, width=width, color=NO_TU_COLOR, edgecolor=NEUTRAL_DARK, linewidth=0.6, label="Strict BIC")
    ax_c.bar(x + width / 2, vw_cap, width=width, color=TU_COLOR, edgecolor=NEUTRAL_DARK, linewidth=0.6, label="Variance window")
    for xpos, value in zip(x - width / 2, strict_cap):
        ax_c.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    for xpos, value in zip(x + width / 2, vw_cap):
        ax_c.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax_c.axhline(1.0, color=NEUTRAL_MID, lw=0.8, ls=":")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(cases)
    ax_c.set_ylabel("Median robust capacity factor")
    ax_c.set_ylim(0.95, max(vw_cap.max() * 1.35, 2.0))
    ax_c.grid(axis="y", color=NEUTRAL_LIGHT, lw=0.6)
    ax_c.legend(frameon=False, loc="upper left")

    panel_label(ax_d, "d", "single model reporting risk")
    cap_prob = vw_summary["mean_capacity_violation_probability"].to_numpy(dtype=float)
    early_prob = vw_summary["mean_early_response_probability"].to_numpy(dtype=float)
    ax_d.bar(x - width / 2, cap_prob, width=width, color=NO_TU_COLOR, edgecolor=NEUTRAL_DARK, linewidth=0.6, label="Capacity shortfall")
    ax_d.bar(x + width / 2, early_prob, width=width, color=TU_COLOR, edgecolor=NEUTRAL_DARK, linewidth=0.6, label="Earlier response")
    for xpos, value in zip(x - width / 2, cap_prob):
        ax_d.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    for xpos, value in zip(x + width / 2, early_prob):
        ax_d.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(cases)
    ax_d.set_ylabel("Mean BMA probability")
    ax_d.set_ylim(0.0, max(cap_prob.max(), early_prob.max()) * 1.35)
    ax_d.grid(axis="y", color=NEUTRAL_LIGHT, lw=0.6)
    ax_d.legend(frameon=False, loc="upper right")

    export(fig, "fig12_decision_consequence")


def write_management_report(summary: pd.DataFrame, weights: pd.DataFrame) -> None:
    lines = [
        "# Model-Averaged Management Diagnostic",
        "",
        "The diagnostic uses only the four formally fitted aquifer test interpretation models.",
        "Strict BIC weights reproduce a single model tendency. Variance window weights rescale BIC differences by the well specific BIC spread to retain plausible alternatives, following the model window treatment of Li and Tsai.",
        "The decision endpoint is hydraulic control or dewatering capacity, where required capacity is proportional to transmissivity, and monitoring response timing, where characteristic response time is proportional to S/T.",
        "",
        "## Case Summary",
        "",
    ]
    for _, row in summary.sort_values(["case", "scheme"]).iterrows():
        lines.extend(
            [
                f"- {row['case']} ({row['scheme']}): selected model counts in model order = {row['best_pathway_counts']}.",
                f"- {row['case']} ({row['scheme']}): median robust capacity factor = {row['median_robust_capacity_factor']:.2f}; 95th percentile = {row['p95_robust_capacity_factor']:.2f}; mean capacity violation probability = {row['mean_capacity_violation_probability']:.2f}.",
                f"- {row['case']} ({row['scheme']}): median response time factor = {row['median_early_response_factor']:.2f}; 95th percentile = {row['p95_early_response_factor']:.2f}; mean earlier response probability = {row['mean_early_response_probability']:.2f}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `tables/field_bma_management_pathway_weights.csv`",
            "- `tables/field_bma_management_pathway_parameters.csv`",
            "- `tables/field_bma_management_well_diagnostics.csv`",
            "- `tables/field_bma_management_summary.csv`",
            "- `fig12_decision_consequence.pdf/png/svg`",
        ]
    )
    (OUT_DIR / "decision_consequence_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    lock_style()
    # Preserve the record-level BMA outputs for reproducibility, but make the
    # manuscript-facing Figure 12 a management-consequence diagnostic.
    predictions = load_predictions()
    strict_weights = load_bma_weights()
    record_table, _ = compute_bma_record_table(predictions, strict_weights)
    summarize_bma(record_table, strict_weights)

    management_weights = load_management_weights()
    management_parameters = load_management_parameters()
    management_diagnostics = compute_management_diagnostics(management_parameters, management_weights)
    management_summary = summarize_management_diagnostics(management_diagnostics, management_weights)
    plot_management_figure(management_summary, management_weights)
    write_management_report(management_summary, management_weights)


if __name__ == "__main__":
    main()
