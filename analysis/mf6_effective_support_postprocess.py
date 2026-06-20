from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf6_mega_benchmark as mega  # noqa: E402


PROJECT = HERE.parents[0]
TABLE_DIR = PROJECT / "tables"
OUT_DIR = PROJECT / "outputs"

CASES_CSV = TABLE_DIR / "mf6_mega_benchmark_cases.csv"
FITS_CSV = TABLE_DIR / "mf6_mega_benchmark_fits.csv"
FIXED_130_SNAPSHOT_CSV = TABLE_DIR / "fixed_radius_130m_reference_snapshot.csv"
EFFECTIVE_SUPPORT_CASES_CSV = TABLE_DIR / "mf6_mega_effective_support_cases.csv"
EFFECTIVE_SUPPORT_SUMMARY_CSV = TABLE_DIR / "mf6_mega_effective_support_summary.csv"
SUMMARY_JSON = OUT_DIR / "mf6_effective_support_postprocess_summary.json"


TARGET_COLUMNS = [
    "truth_T_ref_m2_s",
    "truth_T_geomean_m2_s",
    "truth_T_arithmetic_m2_s",
    "truth_T_annular_m2_s",
    "truth_S_ref",
    "support_definition",
    "support_target_spread",
    "support_effective_radius_m",
    "support_diffusion_radius_m",
    "support_observation_max_radius_m",
    "support_leakage_limit_m",
    "support_boundary_limit_m",
    "support_hydraulic_diffusivity_m2_s",
    "support_weighting",
    "fixed_radius_130m_T_geomean_m2_s",
    "fixed_radius_130m_S_geomean",
    "fixed_radius_130m_T_ratio_to_effective",
    "K_geomean_m_d",
    "K_arithmetic_m_d",
    "K_cv",
    "S_geomean",
    "S_arithmetic",
    "S_cv",
]


def quantile(value: float):
    def _q(series: pd.Series) -> float:
        return float(pd.to_numeric(series, errors="coerce").quantile(value))

    _q.__name__ = f"q{int(value * 100):02d}"
    return _q


def scenario_from_case(row: pd.Series) -> mega.MegaScenario:
    scenario_keys = set(mega.MegaScenario.__dataclass_fields__.keys())
    data = {key: row[key] for key in scenario_keys if key in row.index}
    return mega.scenario_from_dict(data)


def recompute_case_targets(cases: pd.DataFrame) -> pd.DataFrame:
    updated_rows: list[dict] = []
    for idx, row in enumerate(cases.itertuples(index=False), start=1):
        row_dict = row._asdict()
        scenario = scenario_from_case(pd.Series(row_dict))
        nrow, ncol = mega.grid_shape(scenario.domain_m, scenario.dx_m)
        kx, ky, storage = mega.make_property_fields(scenario, nrow, ncol)
        targets = mega.support_targets(kx, ky, storage, scenario)
        for col in TARGET_COLUMNS:
            row_dict[col] = targets.get(col)
        updated_rows.append(row_dict)
        if idx == 1 or idx % 1000 == 0 or idx == len(cases):
            print(f"[effective-support] recomputed targets {idx}/{len(cases)}", flush=True)
    return pd.DataFrame(updated_rows)


def save_fixed_snapshot(cases: pd.DataFrame) -> None:
    cols = [
        "scenario_id",
        "scenario_class",
        "truth_T_ref_m2_s",
        "truth_S_ref",
        "support_definition",
        "support_target_spread",
    ]
    keep = [col for col in cols if col in cases.columns]
    snapshot = cases[keep].copy()
    snapshot = snapshot.rename(
        columns={
            "truth_T_ref_m2_s": "fixed_130m_truth_T_ref_m2_s",
            "truth_S_ref": "fixed_130m_truth_S_ref",
            "support_definition": "fixed_130m_support_definition",
            "support_target_spread": "fixed_130m_support_target_spread",
        }
    )
    snapshot["archive_note"] = "Historical fixed 130 m support reference retained for audit only; not the main reference definition."
    mega.atomic_csv(FIXED_130_SNAPSHOT_CSV, snapshot)


def update_fits(fits: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    updated = fits.copy()
    for col in ["lnM_T", "lnM_S", "lnM_capacity", "lnM_response_time"]:
        if col in updated.columns:
            updated[f"fixed_radius_130m_{col}"] = updated[col]
    for col in ["truth_T_ref_m2_s", "truth_S_ref", "support_target_spread"]:
        if col in updated.columns:
            updated[f"fixed_radius_130m_{col}"] = updated[col]

    support_cols = [
        "scenario_id",
        "truth_T_ref_m2_s",
        "truth_S_ref",
        "support_target_spread",
        "support_definition",
        "support_effective_radius_m",
        "support_diffusion_radius_m",
        "support_observation_max_radius_m",
        "support_leakage_limit_m",
        "support_boundary_limit_m",
        "support_hydraulic_diffusivity_m2_s",
        "support_weighting",
        "fixed_radius_130m_T_geomean_m2_s",
        "fixed_radius_130m_S_geomean",
        "fixed_radius_130m_T_ratio_to_effective",
    ]
    support = cases[[col for col in support_cols if col in cases.columns]].copy()
    updated = updated.drop(columns=[col for col in support.columns if col != "scenario_id" and col in updated.columns])
    updated = updated.merge(support, on="scenario_id", how="left", validate="many_to_one")

    eps = 1.0e-30
    t_fit = np.maximum(pd.to_numeric(updated["T_fit_m2_s"], errors="coerce").to_numpy(float), eps)
    s_fit = np.maximum(pd.to_numeric(updated["S_fit"], errors="coerce").to_numpy(float), eps)
    t_ref = np.maximum(pd.to_numeric(updated["truth_T_ref_m2_s"], errors="coerce").to_numpy(float), eps)
    s_ref = np.maximum(pd.to_numeric(updated["truth_S_ref"], errors="coerce").to_numpy(float), eps)
    updated["lnM_T"] = np.log(t_fit / t_ref)
    updated["lnM_S"] = np.log(s_fit / s_ref)
    updated["lnM_capacity"] = updated["lnM_T"]
    updated["lnM_response_time"] = np.log((s_fit / t_fit) / (s_ref / t_ref))
    if "delta_bic" not in updated.columns:
        updated["delta_bic"] = updated["bic"] - updated.groupby(["scenario_id", "well"])["bic"].transform("min")
    updated["QUALITY_CONTROL"] = (~updated["boundary_hit"].astype(bool)) & (pd.to_numeric(updated["delta_bic"], errors="coerce") <= 10.0)
    return updated


def write_support_summaries(cases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario_id",
        "scenario_class",
        "obs_layout",
        "duration_days",
        "dx_m",
        "domain_m",
        "sigma_ln_k",
        "sigma_ln_s",
        "leakage_b_m",
        "boundary_type",
        "support_effective_radius_m",
        "support_diffusion_radius_m",
        "support_observation_max_radius_m",
        "support_leakage_limit_m",
        "support_boundary_limit_m",
        "support_target_spread",
        "fixed_radius_130m_T_ratio_to_effective",
    ]
    support_cases = cases[[col for col in columns if col in cases.columns]].copy()
    support_cases["fixed_130m_reference_status"] = "sensitivity comparison only"
    mega.atomic_csv(EFFECTIVE_SUPPORT_CASES_CSV, support_cases)

    summary = (
        support_cases.groupby("scenario_class", as_index=False)
        .agg(
            scenario_count=("scenario_id", "count"),
            median_effective_support_radius_m=("support_effective_radius_m", "median"),
            p10_effective_support_radius_m=("support_effective_radius_m", quantile(0.10)),
            p90_effective_support_radius_m=("support_effective_radius_m", quantile(0.90)),
            median_diffusion_radius_m=("support_diffusion_radius_m", "median"),
            median_observation_max_radius_m=("support_observation_max_radius_m", "median"),
            median_support_target_spread=("support_target_spread", "median"),
            median_fixed130_T_ratio_to_effective=("fixed_radius_130m_T_ratio_to_effective", "median"),
        )
        .sort_values("scenario_class")
    )
    mega.atomic_csv(EFFECTIVE_SUPPORT_SUMMARY_CSV, summary)
    return summary


def refresh_cov_tables(fits: pd.DataFrame) -> dict[str, int]:
    mega.atomic_csv(mega.FULL_COV_BY_CLASS_CSV, mega.model_factor_cov_summary(fits, ["scenario_class", "pathway"]))
    mega.atomic_csv(mega.FULL_COV_BY_LAYOUT_CSV, mega.model_factor_cov_summary(fits, ["scenario_class", "obs_layout", "pathway"]))
    gated = fits[fits["QUALITY_CONTROL"]].copy()
    if not gated.empty:
        mega.atomic_csv(mega.FULL_COV_QUALITY_CONTROLD_CSV, mega.model_factor_cov_summary(gated, ["scenario_class", "pathway"]))
    best_idx = fits.groupby(["scenario_id", "well"])["bic"].idxmin()
    best = fits.loc[best_idx].copy()
    best = best[~best["boundary_hit"].astype(bool)]
    if not best.empty:
        mega.atomic_csv(mega.FULL_COV_BIC_BEST_CSV, mega.model_factor_cov_summary(best, ["scenario_class", "pathway"]))
    return {
        "all_fit_rows": int(fits.shape[0]),
        "QUALITY_CONTROL_rows": int(gated.shape[0]),
        "bic_best_rows": int(best.shape[0]),
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases_old = pd.read_csv(CASES_CSV)
    fits_old = pd.read_csv(FITS_CSV)
    save_fixed_snapshot(cases_old)
    cases_new = recompute_case_targets(cases_old)
    fits_new = update_fits(fits_old, cases_new)
    support_summary = write_support_summaries(cases_new)
    cov_counts = refresh_cov_tables(fits_new)
    mega.atomic_csv(CASES_CSV, cases_new)
    mega.atomic_csv(FITS_CSV, fits_new)

    gated_cov = pd.read_csv(mega.FULL_COV_QUALITY_CONTROLD_CSV)
    summary = {
        "case_rows": int(cases_new.shape[0]),
        "fit_rows": int(fits_new.shape[0]),
        "QUALITY_CONTROL_rows": cov_counts["QUALITY_CONTROL_rows"],
        "median_cov_M_T": float(gated_cov["cov_M_T"].median()),
        "median_cov_M_S": float(gated_cov["cov_M_S"].median()),
        "median_cov_M_response_time": float(gated_cov["cov_M_response_time"].median()),
        "median_effective_support_radius_m": float(cases_new["support_effective_radius_m"].median()),
        "p10_effective_support_radius_m": float(cases_new["support_effective_radius_m"].quantile(0.10)),
        "p90_effective_support_radius_m": float(cases_new["support_effective_radius_m"].quantile(0.90)),
        "fixed_130m_reference_status": "sensitivity comparison only; main model factors use scenario-specific effective support",
        "files": {
            "cases": str(CASES_CSV),
            "fits": str(FITS_CSV),
            "support_cases": str(EFFECTIVE_SUPPORT_CASES_CSV),
            "support_summary": str(EFFECTIVE_SUPPORT_SUMMARY_CSV),
            "fixed_130_snapshot": str(FIXED_130_SNAPSHOT_CSV),
            "QUALITY_CONTROLd_cov": str(mega.FULL_COV_QUALITY_CONTROLD_CSV),
        },
        "support_summary_rows": int(support_summary.shape[0]),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()



