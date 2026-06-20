from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf6_mega_benchmark as mega  # noqa: E402
import mf6_transformation_benchmark as base  # noqa: E402


PROJECT = HERE.parents[0]
TABLE_DIR = PROJECT / "tables"
OUT_DIR = PROJECT / "outputs"

ANALYTICAL_LIMIT_CSV = TABLE_DIR / "mf6_mega_analytical_limit_verification.csv"
MF6_CONSISTENCY_CSV = TABLE_DIR / "mf6_mega_mf6_to_analytical_consistency.csv"
MF6_CONSISTENCY_SUMMARY_CSV = TABLE_DIR / "mf6_mega_mf6_to_analytical_consistency_summary.csv"
VARIANCE_ATTRIBUTION_CSV = TABLE_DIR / "mf6_mega_variance_attribution.csv"
OBS_LAYOUT_EFFECTS_CSV = TABLE_DIR / "mf6_mega_observation_layout_effects.csv"
QUALITY_GATE_SUMMARY_CSV = TABLE_DIR / "mf6_mega_quality_gate_summary.csv"
FIELD_GATE_CSV = TABLE_DIR / "mf6_mega_field_applicability_gate.csv"
FIELD_GATE_SUMMARY_CSV = TABLE_DIR / "mf6_mega_field_applicability_gate_summary.csv"
REPORT_MD = OUT_DIR / "mf6_mega_verification_attribution_report.md"

PATHWAYS = [
    "Theis confined",
    "Hantush-Jacob leaky",
    "Lagging Darcy no-leakage",
    "Lagging Darcy with leakage",
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def log_residual(reference: np.ndarray, tested: np.ndarray, offset: float = base.OFFSET_M) -> np.ndarray:
    return np.log((np.asarray(tested, dtype=float) + offset) / (np.asarray(reference, dtype=float) + offset))


def compare_series(reference: np.ndarray, tested: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=float)
    tested = np.asarray(tested, dtype=float)
    diff = tested - reference
    eta = log_residual(reference, tested)
    return {
        "rmse_m": float(np.sqrt(np.mean(diff * diff))),
        "mae_m": float(np.mean(np.abs(diff))),
        "max_abs_error_m": float(np.max(np.abs(diff))),
        "bias_eta": float(np.mean(eta)),
        "sd_eta": float(np.std(eta, ddof=1)) if eta.size > 1 else 0.0,
        "max_abs_eta": float(np.max(np.abs(eta))),
    }


def analytical_limit_verification() -> pd.DataFrame:
    times = np.geomspace(90.0, 2.0 * base.SECONDS_PER_DAY, 40)
    radii = [10.0, 20.0, 50.0, 100.0, 200.0]
    trans = base.T_REF_M2S
    storage = base.S_REF
    rows: list[dict] = []
    for radius in radii:
        theis = base.theis_drawdown(times, base.Q_M3S, trans, storage, radius)
        leaky80 = base.leaky_drawdown(times, base.Q_M3S, trans, storage, 80.0, radius)
        tests = [
            (
                "Hantush B -> infinity recovers Theis",
                "Theis confined",
                "Hantush-Jacob leaky",
                "B=1e12 m",
                theis,
                base.leaky_drawdown(times, base.Q_M3S, trans, storage, 1.0e12, radius),
            ),
            (
                "Lagging tau_q=tau_s recovers Theis",
                "Theis confined",
                "Lagging Darcy no-leakage",
                "tau_q=tau_s=1000 s",
                theis,
                base.lagging_drawdown(times, base.Q_M3S, trans, storage, 1000.0, 1000.0, radius),
            ),
            (
                "Lagging leaky tau_q=tau_s recovers Hantush",
                "Hantush-Jacob leaky",
                "Lagging Darcy with leakage",
                "B=80 m, tau_q=tau_s=1000 s",
                leaky80,
                base.lagging_leaky_drawdown(times, base.Q_M3S, trans, storage, 1000.0, 1000.0, 80.0, radius),
            ),
            (
                "Lagging leaky B -> infinity recovers lagging no-leakage",
                "Lagging Darcy no-leakage",
                "Lagging Darcy with leakage",
                "B=1e12 m, tau_q=100 s, tau_s=1000 s",
                base.lagging_drawdown(times, base.Q_M3S, trans, storage, 100.0, 1000.0, radius),
                base.lagging_leaky_drawdown(times, base.Q_M3S, trans, storage, 100.0, 1000.0, 1.0e12, radius),
            ),
            (
                "Lagging leaky no-lag and B -> infinity recovers Theis",
                "Theis confined",
                "Lagging Darcy with leakage",
                "B=1e12 m, tau_q=tau_s=0 s",
                theis,
                base.lagging_leaky_drawdown(times, base.Q_M3S, trans, storage, 0.0, 0.0, 1.0e12, radius),
            ),
        ]
        for test_name, reference, tested, limiting_case, ref, val in tests:
            metrics = compare_series(ref, val)
            rows.append(
                {
                    "test_name": test_name,
                    "reference_solution": reference,
                    "tested_solution": tested,
                    "limiting_case": limiting_case,
                    "radius_m": radius,
                    "n_times": len(times),
                    "pass_flag": bool(metrics["max_abs_error_m"] < 1.0e-4 and metrics["max_abs_eta"] < 2.0e-2),
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(ANALYTICAL_LIMIT_CSV, index=False)
    return frame


def mf6_consistency_scenarios() -> list[mega.MegaScenario]:
    scenarios: list[mega.MegaScenario] = []
    dx_values = [20.0, 10.0, 5.0, 2.5]
    for dx in dx_values:
        scenarios.append(
            mega.MegaScenario(
                scenario_id=f"deg_theis_dx{str(dx).replace('.', 'p')}",
                scenario_class="homogeneous confined",
                seed=91000 + int(dx * 10),
                sigma_ln_k=0.0,
                corr_len_m=0.0,
                sigma_ln_s=0.0,
                corr_len_s_m=0.0,
                anisotropy_ratio=1.0,
                leakage_b_m=None,
                boundary_type="none",
                dx_m=dx,
                domain_m=805.0,
                time_count=40,
                duration_days=2.0,
                obs_layout="standard4",
                obs_rotation_deg=0.0,
                noise_sd_m=0.0,
                ensemble_role="analytical_consistency",
                description="Homogeneous MODFLOW 6 confined model checked against Theis.",
            )
        )
        scenarios.append(
            mega.MegaScenario(
                scenario_id=f"deg_hantush_dx{str(dx).replace('.', 'p')}",
                scenario_class="vertical leakage",
                seed=92000 + int(dx * 10),
                sigma_ln_k=0.0,
                corr_len_m=0.0,
                sigma_ln_s=0.0,
                corr_len_s_m=0.0,
                anisotropy_ratio=1.0,
                leakage_b_m=80.0,
                boundary_type="none",
                dx_m=dx,
                domain_m=805.0,
                time_count=40,
                duration_days=2.0,
                obs_layout="standard4",
                obs_rotation_deg=0.0,
                noise_sd_m=0.0,
                ensemble_role="analytical_consistency",
                description="Homogeneous MODFLOW 6 leaky model checked against Hantush-Jacob.",
            )
        )
    return scenarios


def mf6_to_analytical_consistency() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for scenario in mf6_consistency_scenarios():
        drawdown, meta = mega.run_mf6_scenario(scenario, keep_workspace=False)
        for well, group in drawdown.groupby("well", sort=False):
            group = group.sort_values("time_s")
            times = group["time_s"].to_numpy(float)
            obs = group["drawdown_m"].to_numpy(float)
            radius = float(group["radius_m"].iloc[0])
            if scenario.scenario_class == "homogeneous confined":
                reference_name = "Theis analytical"
                reference = base.theis_drawdown(times, base.Q_M3S, meta["truth_T_ref_m2_s"], meta["truth_S_ref"], radius)
            elif scenario.scenario_class == "vertical leakage":
                reference_name = "Hantush-Jacob analytical"
                reference = base.leaky_drawdown(times, base.Q_M3S, meta["truth_T_ref_m2_s"], meta["truth_S_ref"], scenario.leakage_b_m or 80.0, radius)
            else:
                raise ValueError(scenario.scenario_class)
            metrics = compare_series(reference, obs)
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_class": scenario.scenario_class,
                    "reference_solution": reference_name,
                    "well": well,
                    "dx_m": scenario.dx_m,
                    "domain_m": scenario.domain_m,
                    "radius_m": radius,
                    "time_count": scenario.time_count,
                    "duration_days": scenario.duration_days,
                    "pass_flag": bool(metrics["rmse_m"] < 0.015 and metrics["sd_eta"] < 0.35),
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["scenario_class", "reference_solution", "dx_m"], as_index=False)
        .agg(
            well_count=("well", "nunique"),
            median_rmse_m=("rmse_m", "median"),
            max_rmse_m=("rmse_m", "max"),
            median_sd_eta=("sd_eta", "median"),
            max_sd_eta=("sd_eta", "max"),
            max_abs_error_m=("max_abs_error_m", "max"),
            pass_fraction=("pass_flag", "mean"),
        )
        .sort_values(["scenario_class", "dx_m"])
    )
    frame.to_csv(MF6_CONSISTENCY_CSV, index=False)
    summary.to_csv(MF6_CONSISTENCY_SUMMARY_CSV, index=False)
    return frame, summary


def add_bins(fits: pd.DataFrame) -> pd.DataFrame:
    out = fits.copy()
    out["leakage_present"] = out["scenario_class"].str.contains("leakage", case=False) | out["pathway"].eq("Hantush-Jacob leaky") | out["pathway"].eq("Lagging Darcy with leakage")
    out["boundary_present"] = out["boundary_type"].fillna("none").ne("none") | out["scenario_class"].str.contains("boundary", case=False)
    out["sigma_ln_k_bin"] = pd.cut(out["sigma_ln_k"], bins=[-0.001, 0.001, 0.5, 1.0, np.inf], labels=["none", "low", "moderate", "high"])
    out["sigma_ln_s_bin"] = pd.cut(out["sigma_ln_s"], bins=[-0.001, 0.001, 0.3, 0.7, np.inf], labels=["none", "low", "moderate", "high"])
    out["corr_len_m_bin"] = pd.cut(out["corr_len_m"], bins=[-0.001, 20.0, 60.0, 120.0, np.inf], labels=["none_or_short", "moderate", "long", "very_long"])
    out["anisotropy_bin"] = pd.cut(out["anisotropy_ratio"], bins=[0.0, 0.75, 1.33, 2.5, np.inf], labels=["low_Ky", "near_iso", "moderate", "high_Kx"])
    out["support_spread_bin"] = pd.cut(out["support_target_spread"], bins=[-0.001, 1.05, 1.25, 1.75, np.inf], labels=["none", "low", "moderate", "high"])
    out["radius_bin"] = pd.cut(out["radius_m"], bins=[0.0, 25.0, 75.0, 150.0, np.inf], labels=["near", "middle", "far", "very_far"])
    return out


def marginal_eta_squared(frame: pd.DataFrame, outcome: str, factor: str) -> dict:
    sub = frame[[outcome, factor]].dropna()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna()
    if sub.empty or sub[factor].nunique() <= 1:
        return {
            "n": int(len(sub)),
            "group_count": int(sub[factor].nunique()) if not sub.empty else 0,
            "eta2_marginal": float("nan"),
            "top_level": None,
            "top_level_mean": float("nan"),
            "bottom_level": None,
            "bottom_level_mean": float("nan"),
        }
    y = sub[outcome].to_numpy(float)
    grand = float(np.mean(y))
    total_ss = float(np.sum((y - grand) ** 2))
    grouped = sub.groupby(factor, observed=True)[outcome].agg(["mean", "count"])
    between_ss = float(np.sum(grouped["count"].to_numpy(float) * (grouped["mean"].to_numpy(float) - grand) ** 2))
    ordered = grouped.sort_values("mean", ascending=False)
    return {
        "n": int(len(sub)),
        "group_count": int(grouped.shape[0]),
        "eta2_marginal": float(between_ss / total_ss) if total_ss > 0 else 0.0,
        "top_level": str(ordered.index[0]),
        "top_level_mean": float(ordered["mean"].iloc[0]),
        "bottom_level": str(ordered.index[-1]),
        "bottom_level_mean": float(ordered["mean"].iloc[-1]),
    }


def variance_attribution() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fits = pd.read_csv(TABLE_DIR / "mf6_mega_benchmark_fits.csv")
    if "delta_bic" not in fits:
        fits["delta_bic"] = fits["bic"] - fits.groupby(["scenario_id", "well"])["bic"].transform("min")
    if "quality_gate" not in fits:
        fits["quality_gate"] = (~fits["boundary_hit"].astype(bool)) & (fits["delta_bic"] <= 10.0)
    fits = add_bins(fits)
    outcomes = ["sd_eta", "lnM_T", "lnM_S", "lnM_response_time"]
    factors = [
        "pathway",
        "scenario_class",
        "obs_layout",
        "radius_bin",
        "dx_m",
        "domain_m",
        "time_count",
        "duration_days",
        "noise_sd_m",
        "boundary_type",
        "leakage_present",
        "boundary_present",
        "sigma_ln_k_bin",
        "sigma_ln_s_bin",
        "corr_len_m_bin",
        "anisotropy_bin",
        "support_spread_bin",
    ]
    rows = []
    for dataset, frame in [("all_fits", fits), ("quality_gated", fits[fits["quality_gate"]].copy())]:
        for outcome in outcomes:
            for factor in factors:
                result = marginal_eta_squared(frame, outcome, factor)
                rows.append({"dataset": dataset, "outcome": outcome, "factor": factor, **result})
    attribution = pd.DataFrame(rows).sort_values(["dataset", "outcome", "eta2_marginal"], ascending=[True, True, False])
    attribution.to_csv(VARIANCE_ATTRIBUTION_CSV, index=False)

    quality = (
        fits.groupby(["scenario_class", "pathway"], observed=True, as_index=False)
        .agg(
            response_count=("scenario_id", "count"),
            unique_scenarios=("scenario_id", "nunique"),
            quality_gate_fraction=("quality_gate", "mean"),
            boundary_hit_fraction=("boundary_hit", "mean"),
            bic_supported_fraction=("delta_bic", lambda x: float(np.mean(pd.to_numeric(x, errors="coerce") <= 10.0))),
            median_delta_bic=("delta_bic", "median"),
            median_sd_eta=("sd_eta", "median"),
        )
        .sort_values(["scenario_class", "pathway"])
    )
    quality.to_csv(QUALITY_GATE_SUMMARY_CSV, index=False)

    gated = fits[fits["quality_gate"]].copy()
    layout_rows = []
    for keys, group in gated.groupby(["scenario_class", "pathway", "obs_layout"], observed=True, sort=False):
        scenario_class, pathway, obs_layout = keys
        layout_rows.append(
            {
                "scenario_class": scenario_class,
                "pathway": pathway,
                "obs_layout": obs_layout,
                "response_count": int(group.shape[0]),
                "unique_scenarios": int(group["scenario_id"].nunique()),
                "median_sd_eta": float(group["sd_eta"].median()),
                "median_abs_lnM_T": float(np.median(np.abs(group["lnM_T"].to_numpy(float)))),
                "median_abs_lnM_S": float(np.median(np.abs(group["lnM_S"].to_numpy(float)))),
                "median_abs_lnM_response_time": float(np.median(np.abs(group["lnM_response_time"].to_numpy(float)))),
            }
        )
    layout = pd.DataFrame(layout_rows).sort_values(["scenario_class", "pathway", "obs_layout"])
    layout.to_csv(OBS_LAYOUT_EFFECTS_CSV, index=False)
    return attribution, quality, layout


def field_case_diagnostics() -> dict[str, dict]:
    mass = pd.read_csv(TABLE_DIR / "field_pathway_summary.csv")
    love = pd.read_csv(TABLE_DIR / "lovelock_pathway_summary.csv")
    bma = pd.read_csv(TABLE_DIR / "field_bma_management_summary.csv")
    mass_bic_summary = " / ".join(f"{row.pathway}:{int(row.bic_preferred_wells)}" for row in mass.itertuples())
    love_bic_summary = " / ".join(f"{row.pathway}:{int(row.bic_preferred_wells)}" for row in love.itertuples())
    return {
        "Massachusetts": {
            "min_pooled_sd_eta_pathway": str(mass.sort_values("pooled_sd_eta").iloc[0]["pathway"]),
            "bic_preferred_summary": mass_bic_summary,
            "lagging_support": int(mass[mass["pathway"].str.contains("Lagging")]["bic_preferred_wells"].sum()),
            "leakage_boundary_hits": int(mass[mass["pathway"].str.contains("leak|Hantush", case=False)]["boundary_hit_count"].sum()),
            "pooled_sd_eta_theis": float(mass.loc[mass["pathway"].eq("Theis confined"), "pooled_sd_eta"].iloc[0]),
            "pooled_sd_eta_best": float(mass["pooled_sd_eta"].min()),
            "decision_capacity_p95": float(bma[(bma["case"].eq("Massachusetts")) & (bma["scheme"].eq("variance_window"))]["p95_robust_capacity_factor"].iloc[0]),
            "decision_response_p95": float(bma[(bma["case"].eq("Massachusetts")) & (bma["scheme"].eq("variance_window"))]["p95_early_response_factor"].iloc[0]),
        },
        "Lovelock Valley": {
            "min_pooled_sd_eta_pathway": str(love.sort_values("pooled_sd_eta").iloc[0]["pathway"]),
            "bic_preferred_summary": love_bic_summary,
            "lagging_support": int(love[love["pathway"].str.contains("Lagging")]["bic_preferred_wells"].sum()),
            "leakage_boundary_hits": int(love[love["pathway"].str.contains("leak|Hantush", case=False)]["boundary_hit_count"].sum()),
            "pooled_sd_eta_theis": float(love.loc[love["pathway"].eq("Theis confined"), "pooled_sd_eta"].iloc[0]),
            "pooled_sd_eta_best": float(love["pooled_sd_eta"].min()),
            "decision_capacity_p95": float(bma[(bma["case"].eq("Lovelock Valley")) & (bma["scheme"].eq("variance_window"))]["p95_robust_capacity_factor"].iloc[0]),
            "decision_response_p95": float(bma[(bma["case"].eq("Lovelock Valley")) & (bma["scheme"].eq("variance_window"))]["p95_early_response_factor"].iloc[0]),
        },
    }


def field_applicability_gate() -> tuple[pd.DataFrame, pd.DataFrame]:
    cov = pd.read_csv(TABLE_DIR / "mf6_mega_model_factor_cov_quality_gated_by_class.csv").rename(
        columns={"scenario_class": "benchmark_class"}
    )
    diagnostics = field_case_diagnostics()
    gate_rows = [
        {
            "field_case": "Massachusetts",
            "benchmark_class": "combined leakage boundary",
            "applicability_level": "primary_conservative",
            "matching_basis": "fractured bedrock field response has spatial variability plus leakage or boundary alternatives; use only as conservative conditional transformation prior",
            "mismatch_flags": "field fracture network, pumping history simplification, and true support scale remain unverified",
        },
        {
            "field_case": "Massachusetts",
            "benchmark_class": "heterogeneous leakage",
            "applicability_level": "primary_sensitivity",
            "matching_basis": "field diagnostics support temporal response and possible leakage without requiring first-type boundary as the main mechanism",
            "mismatch_flags": "boundary equivalent behavior may still be present; field support scale remains unverified",
        },
        {
            "field_case": "Massachusetts",
            "benchmark_class": "heterogeneous boundary",
            "applicability_level": "primary_sensitivity",
            "matching_basis": "field diagnostics include late-time boundary-equivalent behavior in selected wells",
            "mismatch_flags": "leakage and fracture-matrix exchange are not represented in this class",
        },
        {
            "field_case": "Massachusetts",
            "benchmark_class": "heterogeneous K",
            "applicability_level": "lower_complexity_check",
            "matching_basis": "spatially variable response without accepting leakage as a required class",
            "mismatch_flags": "not conservative when leakage or boundary alternatives are supported",
        },
        {
            "field_case": "Lovelock Valley",
            "benchmark_class": "heterogeneous storage",
            "applicability_level": "primary_conservative",
            "matching_basis": "basin-fill pumping/recovery response with no strong leakage-pathway preference and possible storage variability",
            "mismatch_flags": "layering, recovery preprocessing, and regional boundary conditions remain unverified",
        },
        {
            "field_case": "Lovelock Valley",
            "benchmark_class": "heterogeneous K",
            "applicability_level": "primary_sensitivity",
            "matching_basis": "BIC support concentrates in no-leakage interpretations; heterogeneous K is the lower-complexity numerical analogue",
            "mismatch_flags": "does not represent basin-fill storage variability or recovery-specific preprocessing",
        },
        {
            "field_case": "Lovelock Valley",
            "benchmark_class": "finite no-flow boundary",
            "applicability_level": "boundary_sensitivity",
            "matching_basis": "basin setting can be affected by finite support, but field evidence does not establish a specific boundary geometry",
            "mismatch_flags": "homogeneous finite-domain check only; not a field-scale conceptual model",
        },
        {
            "field_case": "Lovelock Valley",
            "benchmark_class": "heterogeneous leakage",
            "applicability_level": "not_primary_unless_local_leakage_supported",
            "matching_basis": "include only as a stress test when a specific well shows leakage support",
            "mismatch_flags": "overall Lovelock BIC support concentrates in no-leakage interpretations",
        },
    ]
    gate = pd.DataFrame(gate_rows)
    gate = gate.merge(cov, on="benchmark_class", how="left")
    for field_case, diag in diagnostics.items():
        mask = gate["field_case"].eq(field_case)
        for key, value in diag.items():
            gate.loc[mask, key] = value
    gate["allowable_use"] = np.where(
        gate["applicability_level"].str.startswith("primary"),
        "conditional field prior if well-level diagnostics match the class",
        "sensitivity or exclusion gate; do not use as default transformation prior",
    )
    gate["claim_boundary"] = "scenario-conditioned transformation factor; not a universal design COV and not proof of true field T/S recovery"
    gate.to_csv(FIELD_GATE_CSV, index=False)

    summary = (
        gate.groupby(["field_case", "applicability_level", "benchmark_class"], as_index=False)
        .agg(
            pathway_count=("pathway", "nunique"),
            median_cov_M_T=("cov_M_T", "median"),
            max_cov_M_T=("cov_M_T", "max"),
            median_cov_M_S=("cov_M_S", "median"),
            max_cov_M_response_time=("cov_M_response_time", "max"),
            field_min_pooled_sd_eta_pathway=("min_pooled_sd_eta_pathway", "first"),
            field_bic_preferred_summary=("bic_preferred_summary", "first"),
            field_lagging_support=("lagging_support", "first"),
            field_decision_capacity_p95=("decision_capacity_p95", "first"),
            field_decision_response_p95=("decision_response_p95", "first"),
            mismatch_flags=("mismatch_flags", "first"),
        )
        .sort_values(["field_case", "applicability_level", "benchmark_class"])
    )
    summary.to_csv(FIELD_GATE_SUMMARY_CSV, index=False)
    return gate, summary


def write_report(
    analytical: pd.DataFrame,
    mf6_summary: pd.DataFrame,
    attribution: pd.DataFrame,
    quality: pd.DataFrame,
    field_summary: pd.DataFrame,
) -> None:
    top_attr = (
        attribution[attribution["dataset"].eq("quality_gated")]
        .sort_values(["outcome", "eta2_marginal"], ascending=[True, False])
        .groupby("outcome")
        .head(5)
    )
    lines = [
        "# MF6 mega benchmark verification and attribution report",
        "",
        "## Analytical limiting checks",
        f"- Rows: {len(analytical)}",
        f"- Passed strict limiting checks: {int(analytical['pass_flag'].sum())}/{len(analytical)}",
        "",
        "## MODFLOW 6 to analytical consistency",
        mf6_summary.to_markdown(index=False),
        "",
        "## Quality gate summary",
        "- Quality gate is `no boundary hit` and `Delta BIC <= 10`.",
        quality.head(20).to_markdown(index=False),
        "",
        "## Main variance-attribution signals",
        top_attr[["outcome", "factor", "eta2_marginal", "top_level", "top_level_mean", "bottom_level", "bottom_level_mean"]].to_markdown(index=False),
        "",
        "## Field applicability gate",
        field_summary.to_markdown(index=False),
        "",
        "## Claim boundary",
        "These tables support scenario-conditioned transformation uncertainty diagnostics. They do not establish universal design COV values or true field recovery of transmissivity, storage, leakage, or lagging response times.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    analytical = analytical_limit_verification()
    mf6_rows, mf6_summary = mf6_to_analytical_consistency()
    attribution, quality, _layout = variance_attribution()
    _gate, field_summary = field_applicability_gate()
    write_report(analytical, mf6_summary, attribution, quality, field_summary)
    summary = {
        "analytical_limit_rows": int(analytical.shape[0]),
        "analytical_limit_passed": int(analytical["pass_flag"].sum()),
        "mf6_consistency_rows": int(mf6_rows.shape[0]),
        "mf6_consistency_confirmed": int(mf6_rows["pass_flag"].sum()),
        "variance_attribution_rows": int(attribution.shape[0]),
        "quality_gate_rows": int(quality.shape[0]),
        "field_gate_rows": int(field_summary.shape[0]),
        "outputs": {
            "analytical_limit_csv": str(ANALYTICAL_LIMIT_CSV),
            "mf6_consistency_csv": str(MF6_CONSISTENCY_CSV),
            "mf6_consistency_summary_csv": str(MF6_CONSISTENCY_SUMMARY_CSV),
            "variance_attribution_csv": str(VARIANCE_ATTRIBUTION_CSV),
            "quality_gate_summary_csv": str(QUALITY_GATE_SUMMARY_CSV),
            "observation_layout_effects_csv": str(OBS_LAYOUT_EFFECTS_CSV),
            "field_gate_csv": str(FIELD_GATE_CSV),
            "field_gate_summary_csv": str(FIELD_GATE_SUMMARY_CSV),
            "report_md": str(REPORT_MD),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


