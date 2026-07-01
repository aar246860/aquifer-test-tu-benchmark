from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"


def read_csv(name: str) -> list[dict[str, str]]:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Required table has no rows: {path}")
    return rows


def require_columns(rows: list[dict[str, str]], name: str, columns: set[str]) -> None:
    missing = columns.difference(rows[0].keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_list}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    qc = read_csv("mf6_mega_quality_control_summary.csv")
    require_columns(
        qc,
        "mf6_mega_quality_control_summary.csv",
        {"scenario_class", "pathway", "response_count", "unique_scenarios", "median_sd_eta"},
    )

    regression = read_csv("benchmark_transfer_regression_validation.csv")
    require_columns(
        regression,
        "benchmark_transfer_regression_validation.csv",
        {"target", "r2", "mae_abs_log", "n_scenarios"},
    )
    require(len(regression) == 3, "Expected three benchmark-transfer regression targets.")
    scenario_counts = {int(float(row["n_scenarios"])) for row in regression}
    require(scenario_counts == {10000}, "Each regression target must use 10,000 scenarios.")

    management = read_csv("field_bma_management_summary.csv")
    require_columns(
        management,
        "field_bma_management_summary.csv",
        {"case", "scheme", "median_robust_capacity_factor", "p95_early_response_factor"},
    )
    cases = {row["case"] for row in management}
    has_massachusetts = any("Massachusetts" in case for case in cases)
    has_lovelock = any("Lovelock" in case for case in cases)
    require(has_massachusetts and has_lovelock, "Both field cases must be present.")

    cov = read_csv("mf6_mega_model_factor_cov_screened_by_class.csv")
    require_columns(
        cov,
        "mf6_mega_model_factor_cov_screened_by_class.csv",
        {"cov_M_T", "cov_M_S", "cov_M_response_time", "scenario_class", "pathway"},
    )

    r2_summary = ", ".join(f"{row['target']}={float(row['r2']):.2f}" for row in regression)
    case_summary = ", ".join(sorted(cases))
    print("CAGEO quick test passed.")
    print(f"Quality-control rows: {len(qc)}")
    print(f"Regression targets: {r2_summary}")
    print(f"Field cases: {case_summary}")
    print(f"Screened model-factor rows: {len(cov)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CAGEO quick test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
