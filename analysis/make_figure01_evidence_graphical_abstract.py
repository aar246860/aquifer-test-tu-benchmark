"""Generate an evidence-based graphical abstract candidate for Figure 1.

This script builds a paste-ready PNG from existing TU_Lag analysis outputs.
Flow boxes are schematic, but all plotted curves, fields, model-factor
scatter, applicability numbers, and decision diagnostics are regenerated from
the current CSV outputs or the MODFLOW benchmark scenario generator.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import LogNorm, Normalize, to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[0]
TABLE_DIR = PROJECT / "tables"
OUT_DIR = PROJECT / "outputs"
SELECTED_SCENARIOS: dict[str, str] = {}

sys.path.insert(0, str(HERE))
import mf6_mega_benchmark as mega  # noqa: E402


PATHWAYS = [
    "Theis confined",
    "Hantush-Jacob leaky",
    "Lagging Darcy no-leakage",
    "Lagging Darcy with leakage",
]

SHORT = {
    "Theis confined": "Theis",
    "Hantush-Jacob leaky": "Hantush",
    "Lagging Darcy no-leakage": "Lag-noL",
    "Lagging Darcy with leakage": "Lag-L",
}

PATHWAY_COLORS = {
    pathway: getattr(cmc, "batlow")(sample)
    for pathway, sample in zip(PATHWAYS, [0.12, 0.36, 0.64, 0.88])
}

CASE_COLORS = {
    "Massachusetts": getattr(cmc, "navia")(0.24),
    "Lovelock Valley": getattr(cmc, "navia")(0.72),
}

INK = "#1f2429"
MUTED = "#606970"
RULE = "#aab3b8"
PANEL_BG = "#fbfcfc"
SOFT_BLUE = "#dce9ef"
SOFT_GREEN = "#dfeee2"
SOFT_GOLD = "#f3ead0"
SOFT_GRAY = "#edf0f1"


def lighten(color, frac: float = 0.78) -> str:
    rgb = np.array(to_rgb(color))
    return to_hex(rgb * (1 - frac) + frac)


def lock_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.0,
            "legend.fontsize": 5.7,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_frame(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RULE)
        spine.set_linewidth(0.65)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="black",
    )
    ax.text(
        0.09,
        0.985,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color="black",
    )


def inset(parent: plt.Axes, bounds: tuple[float, float, float, float]) -> plt.Axes:
    ax = parent.inset_axes(bounds)
    ax.set_facecolor("white")
    return ax


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str = "#4a575e",
    fontsize: float = 7.0,
) -> patches.FancyBboxPatch:
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.75,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="black",
        linespacing=1.05,
    )
    return patch


def arrow_axes(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], lw: float = 0.9) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": lw, "color": "black", "mutation_scale": 10},
    )


def representative_scenario(cases: pd.DataFrame, scenario_class: str, score_cols: list[str]) -> pd.Series:
    sub = cases[cases["scenario_class"] == scenario_class].copy()
    if scenario_class in {"constant head boundary", "heterogeneous boundary", "combined leakage boundary"}:
        sub = sub[sub["boundary_type"] != "none"].copy()
    if sub.empty:
        raise ValueError(f"No benchmark case found for {scenario_class}")
    score = np.zeros(sub.shape[0])
    for col in score_cols:
        prefer_small = col.startswith("small:")
        col_name = col.split(":", 1)[1] if prefer_small else col
        if col_name not in sub.columns:
            continue
        values = pd.to_numeric(sub[col_name], errors="coerce").fillna(0.0).to_numpy(float)
        denom = max(float(np.nanmax(values) - np.nanmin(values)), 1.0e-12)
        normed = (values - float(np.nanmin(values))) / denom
        score += 1.0 - normed if prefer_small else normed
    sub = sub.assign(_score=score)
    domain_diff = (sub["domain_m"] - 805.0).abs()
    sub = sub[domain_diff == domain_diff.min()].copy()
    return sub.sort_values(["_score", "grid_cell_count"], ascending=[False, True]).iloc[0]


def scenario_to_dataclass(row: pd.Series) -> mega.MegaScenario:
    scenario_keys = set(mega.MegaScenario.__dataclass_fields__.keys())
    return mega.scenario_from_dict({key: row[key] for key in scenario_keys if key in row.index})


def property_field_for(row: pd.Series) -> dict[str, object]:
    scenario = scenario_to_dataclass(row)
    nrow, ncol = mega.grid_shape(scenario.domain_m, scenario.dx_m)
    kx, ky, storage = mega.make_property_fields(scenario, nrow, ncol)
    half_x = (ncol - 1) * scenario.dx_m / 2
    half_y = (nrow - 1) * scenario.dx_m / 2
    return {
        "scenario": scenario,
        "kx": kx,
        "storage": storage,
        "extent": [-half_x, half_x, -half_y, half_y],
    }


def draw_wells(ax: plt.Axes, scenario: mega.MegaScenario, color: str = "white") -> None:
    for _, radius, angle in mega.observation_wells(scenario):
        theta = math.radians(angle)
        ax.scatter(
            radius * math.cos(theta),
            radius * math.sin(theta),
            s=10,
            facecolor=color,
            edgecolor=INK,
            linewidth=0.25,
            zorder=4,
        )
    ax.scatter([0], [0], marker="*", s=42, facecolor=color, edgecolor=INK, linewidth=0.45, zorder=5)
    try:
        kx, ky, storage = mega.make_property_fields(scenario, *mega.grid_shape(scenario.domain_m, scenario.dx_m))
        support = mega.support_targets(kx, ky, storage, scenario)["support_effective_radius_m"]
        ax.add_patch(patches.Circle((0, 0), float(support), fill=False, edgecolor=color, linewidth=0.75, linestyle="--"))
    except Exception:
        pass


def choose_representative_well(predictions: pd.DataFrame, metadata: pd.DataFrame) -> str:
    candidates = metadata.sort_values("drawdown_max_m", ascending=False)["well"].tolist()
    for well in candidates:
        sub = predictions[predictions["well"] == well]
        if sub["pathway"].nunique() == len(PATHWAYS) and sub.shape[0] > 80:
            return str(well)
    return str(predictions["well"].iloc[0])


def panel_a(ax: plt.Axes) -> None:
    panel_frame(ax, "A", "Transformation grammar and diagnostics")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    draw_box(ax, 0.08, 0.82, 0.84, 0.10, "Measured drawdown\nQ, r, b, s_obs(t), t", fc=SOFT_BLUE, fontsize=7.5)
    draw_box(
        ax,
        0.08,
        0.62,
        0.84,
        0.11,
        "Interpretation model M\nTheis | Hantush | lagging Darcy",
        fc=SOFT_GREEN,
        fontsize=7.2,
    )
    draw_box(
        ax,
        0.08,
        0.41,
        0.84,
        0.12,
        "Response diagnostics\nlog model factor, BIC,\nparameter spread",
        fc=SOFT_GOLD,
        fontsize=7.0,
    )
    draw_box(
        ax,
        0.08,
        0.19,
        0.84,
        0.13,
        "Benchmark-conditioned support\n10,000 scenarios\n160,012 pathway fits",
        fc=SOFT_GRAY,
        fontsize=7.0,
    )
    for y0, y1 in [(0.82, 0.73), (0.62, 0.53), (0.41, 0.32)]:
        arrow_axes(ax, (0.50, y0), (0.50, y1), lw=1.0)
    for dx in [-0.18, 0.18]:
        arrow_axes(ax, (0.50, 0.62), (0.50 + dx, 0.53), lw=0.8)
    ax.text(
        0.5,
        0.08,
        "Target: apparent T/S with traceable\ntransformation uncertainty",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=7.0,
        color=INK,
    )


def panel_b(ax: plt.Axes, predictions: pd.DataFrame, comparison: pd.DataFrame, metadata: pd.DataFrame) -> None:
    panel_frame(ax, "B", "Field analytical fits and apparent parameters")
    well = choose_representative_well(predictions, metadata)
    sub = predictions[predictions["well"] == well].copy()

    ax_fit = inset(ax, (0.07, 0.54, 0.42, 0.34))
    obs = sub[["time_h", "observed_m"]].drop_duplicates().sort_values("time_h")
    ax_fit.scatter(obs["time_h"], obs["observed_m"], s=9, color=INK, alpha=0.65, linewidths=0, label="Obs.")
    for pathway in PATHWAYS:
        p = sub[sub["pathway"] == pathway].sort_values("time_h")
        ax_fit.plot(p["time_h"], p["predicted_m"], color=PATHWAY_COLORS[pathway], lw=0.9, label=SHORT[pathway])
    ax_fit.set_xscale("log")
    ax_fit.set_xlabel("Time (h)", labelpad=1)
    ax_fit.set_ylabel("s (m)", labelpad=1)
    ax_fit.set_title(f"Drawdown fit: {well}", loc="left", fontsize=6.7)
    ax_fit.grid(True, which="major", alpha=0.18, linewidth=0.4)

    ax_res = inset(ax, (0.56, 0.54, 0.38, 0.34))
    for pathway in PATHWAYS:
        p = sub[sub["pathway"] == pathway].sort_values("time_h")
        ax_res.plot(p["time_h"], p["eta"], color=PATHWAY_COLORS[pathway], lw=0.85)
    ax_res.axhline(0, color=RULE, lw=0.55)
    ax_res.set_xscale("log")
    ax_res.set_xlabel("Time (h)", labelpad=1)
    ax_res.set_ylabel("eta", labelpad=1)
    ax_res.set_title("Log model factor", loc="left", fontsize=6.7)
    ax_res.grid(True, which="major", alpha=0.16, linewidth=0.4)

    ax_ts = inset(ax, (0.07, 0.10, 0.42, 0.33))
    for pathway in PATHWAYS:
        p = comparison[comparison["pathway"] == pathway]
        ax_ts.scatter(p["T_m2s"], p["S"], s=13, color=PATHWAY_COLORS[pathway], alpha=0.82, edgecolor="none")
    ax_ts.set_xscale("log")
    ax_ts.set_yscale("log")
    ax_ts.set_xlabel("T_app (m2/s)", labelpad=1)
    ax_ts.set_ylabel("S_app", labelpad=1)
    ax_ts.set_title("Massachusetts", loc="left", fontsize=6.7)
    ax_ts.grid(True, which="major", alpha=0.16, linewidth=0.4)

    ax_sum = inset(ax, (0.56, 0.10, 0.38, 0.33))
    summary = pd.read_csv(TABLE_DIR / "field_pathway_summary.csv").set_index("pathway").reindex(PATHWAYS)
    y = np.arange(len(PATHWAYS))
    ax_sum.barh(y, summary["pooled_sd_eta"], color=[PATHWAY_COLORS[p] for p in PATHWAYS], height=0.55)
    ax_sum.set_yticks(y)
    ax_sum.set_yticklabels([SHORT[p] for p in PATHWAYS])
    ax_sum.invert_yaxis()
    ax_sum.set_xlabel("Pooled SD_eta", labelpad=1)
    ax_sum.set_title("Response residual spread", loc="left", fontsize=6.7)
    ax_sum.grid(axis="x", alpha=0.16, linewidth=0.4)



def panel_c(ax: plt.Axes) -> None:
    panel_frame(ax, "C", "Scenario-conditioned model-factor suite")
    cases = pd.read_csv(TABLE_DIR / "mf6_mega_benchmark_cases.csv")
    k_row = representative_scenario(cases, "heterogeneous K", ["sigma_ln_k", "support_target_spread", "small:corr_len_m"])
    s_row = representative_scenario(cases, "heterogeneous storage", ["sigma_ln_s", "small:corr_len_s_m"])
    SELECTED_SCENARIOS["log K field"] = str(k_row["scenario_id"])
    SELECTED_SCENARIOS["log S field"] = str(s_row["scenario_id"])
    k_field = property_field_for(k_row)
    s_field = property_field_for(s_row)

    ax_k = inset(ax, (0.06, 0.53, 0.27, 0.34))
    k_values = np.log10(k_field["kx"])
    im_k = ax_k.imshow(k_values, extent=k_field["extent"], origin="lower", cmap=cmc.lipari)
    draw_wells(ax_k, k_field["scenario"])
    ax_k.set_title("log K field", loc="left", fontsize=6.7)
    ax_k.set_xticks([])
    ax_k.set_yticks([])

    ax_s = inset(ax, (0.06, 0.10, 0.27, 0.34))
    s_values = np.log10(s_field["storage"])
    im_s = ax_s.imshow(s_values, extent=s_field["extent"], origin="lower", cmap=cmc.batlow)
    draw_wells(ax_s, s_field["scenario"])
    ax_s.set_title("log S field", loc="left", fontsize=6.7)
    ax_s.set_xticks([])
    ax_s.set_yticks([])

    fit_cols = ["pathway", "T_fit_m2_s", "S_fit", "truth_T_ref_m2_s", "truth_S_ref", "QUALITY_CONTROL"]
    fits = pd.read_csv(TABLE_DIR / "mf6_mega_benchmark_fits.csv", usecols=fit_cols)
    fits = fits[fits["pathway"].isin(PATHWAYS)].dropna(subset=["T_fit_m2_s", "S_fit", "truth_T_ref_m2_s", "truth_S_ref"])
    if "QUALITY_CONTROL" in fits:
        q = fits[fits["QUALITY_CONTROL"].astype(str).str.lower().isin(["true", "1"])]
        if q.shape[0] > 100:
            fits = q
    fits = fits.sample(n=min(4200, fits.shape[0]), random_state=20260612)

    ax_t = inset(ax, (0.44, 0.53, 0.49, 0.34))
    for pathway in PATHWAYS:
        p = fits[fits["pathway"] == pathway]
        ax_t.scatter(p["truth_T_ref_m2_s"], p["T_fit_m2_s"], s=3.5, color=PATHWAY_COLORS[pathway], alpha=0.26, edgecolor="none")
    lo = min(fits["truth_T_ref_m2_s"].min(), fits["T_fit_m2_s"].min())
    hi = max(fits["truth_T_ref_m2_s"].max(), fits["T_fit_m2_s"].max())
    ax_t.plot([lo, hi], [lo, hi], color=INK, lw=0.55)
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("T_ref", labelpad=1)
    ax_t.set_ylabel("T_app", labelpad=1)
    ax_t.set_title("T model factors", loc="left", fontsize=6.7)
    ax_t.grid(True, which="major", alpha=0.14, linewidth=0.4)

    ax_scat = inset(ax, (0.44, 0.10, 0.49, 0.34))
    for pathway in PATHWAYS:
        p = fits[fits["pathway"] == pathway]
        ax_scat.scatter(p["truth_S_ref"], p["S_fit"], s=3.5, color=PATHWAY_COLORS[pathway], alpha=0.26, edgecolor="none")
    lo = min(fits["truth_S_ref"].min(), fits["S_fit"].min())
    hi = max(fits["truth_S_ref"].max(), fits["S_fit"].max())
    ax_scat.plot([lo, hi], [lo, hi], color=INK, lw=0.55)
    ax_scat.set_xscale("log")
    ax_scat.set_yscale("log")
    ax_scat.set_xlabel("S_ref", labelpad=1)
    ax_scat.set_ylabel("S_app", labelpad=1)
    ax_scat.set_title("S model factors", loc="left", fontsize=6.7)
    ax_scat.grid(True, which="major", alpha=0.14, linewidth=0.4)



def panel_d(ax: plt.Axes) -> None:
    panel_frame(ax, "D", "Field applicability criteria")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    draw_box(ax, 0.05, 0.57, 0.24, 0.16, "Field case\nr_obs, s_obs(t),\nfield response", fc=SOFT_BLUE, fontsize=6.5)
    draw_box(ax, 0.38, 0.72, 0.24, 0.12, "Transferable\nclass?", fc=SOFT_GREEN, fontsize=6.5)
    draw_box(ax, 0.38, 0.52, 0.24, 0.12, "Conservative /\nsensitivity?", fc=SOFT_GREEN, fontsize=6.5)
    draw_box(ax, 0.38, 0.32, 0.24, 0.12, "Complexity\nsupport?", fc=SOFT_GREEN, fontsize=6.5)
    for y in [0.78, 0.58, 0.38]:
        arrow_axes(ax, (0.29, 0.65), (0.38, y), lw=0.85)
    arrow_axes(ax, (0.62, 0.78), (0.70, 0.78), lw=0.85)
    arrow_axes(ax, (0.62, 0.58), (0.70, 0.58), lw=0.85)
    arrow_axes(ax, (0.62, 0.38), (0.70, 0.38), lw=0.85)

    gate = pd.read_csv(TABLE_DIR / "mf6_mega_field_applicability_criteria_summary.csv")
    primary = gate[gate["applicability_level"] == "primary_conservative"].copy()
    primary = primary.set_index("field_case").reindex(["Massachusetts", "Lovelock Valley"]).reset_index()
    metrics = [
        ("median_cov_M_T", "COV_T"),
        ("median_cov_M_S", "COV_S"),
        ("max_cov_M_response_time", "COV_t"),
    ]

    ax_bar = inset(ax, (0.70, 0.22, 0.26, 0.58))
    x = np.arange(len(metrics))
    width = 0.36
    for idx, case in enumerate(["Massachusetts", "Lovelock Valley"]):
        row = primary[primary["field_case"] == case].iloc[0]
        vals = [float(row[m[0]]) for m in metrics]
        ax_bar.bar(x + (idx - 0.5) * width, vals, width=width, color=CASE_COLORS[case], alpha=0.88, label="MA" if idx == 0 else "LV")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([m[1] for m in metrics])
    ax_bar.set_ylabel("conditioned COV", labelpad=1)
    ax_bar.set_title("Primary classes", loc="left", fontsize=6.7)
    ax_bar.grid(axis="y", alpha=0.16, linewidth=0.4)
    ax_bar.legend(frameon=False, loc="upper left", fontsize=5.6)

    for i, case in enumerate(["Massachusetts", "Lovelock Valley"]):
        row = primary[primary["field_case"] == case].iloc[0]
        ax.text(
            0.06,
            0.20 - i * 0.075,
            f"{'MA' if i == 0 else 'LV'} -> {row['benchmark_class']}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.1,
            color=INK,
        )


def panel_e(ax: plt.Axes) -> None:
    panel_frame(ax, "E", "Groundwater reporting inheritance")
    weights = pd.read_csv(TABLE_DIR / "field_bma_management_summary.csv")
    weights = weights[weights["scheme"] == "variance_window"].set_index("case").reindex(["Massachusetts", "Lovelock Valley"])
    diag = pd.read_csv(TABLE_DIR / "field_bma_management_well_diagnostics.csv")
    diag = diag[diag["scheme"] == "variance_window"].copy()

    ax_w = inset(ax, (0.07, 0.56, 0.36, 0.31))
    cols = [
        "mean_weight_theis",
        "mean_weight_hantush_leaky",
        "mean_weight_lagging_no_leakage",
        "mean_weight_lagging_leaky",
    ]
    x = np.arange(len(PATHWAYS))
    width = 0.34
    for idx, case in enumerate(["Massachusetts", "Lovelock Valley"]):
        vals = weights.loc[case, cols].to_numpy(dtype=float)
        ax_w.bar(x + (idx - 0.5) * width, vals, width=width, color=CASE_COLORS[case], alpha=0.88, label="MA" if idx == 0 else "LV")
    ax_w.set_xticks(x)
    ax_w.set_xticklabels([SHORT[p] for p in PATHWAYS], rotation=25, ha="right")
    ax_w.set_ylabel("BMA weight", labelpad=1)
    ax_w.set_title("Variance-window mixture", loc="left", fontsize=6.7)
    ax_w.grid(axis="y", alpha=0.16, linewidth=0.4)
    ax_w.legend(frameon=False, loc="upper right", fontsize=5.6)

    ax_c = inset(ax, (0.53, 0.56, 0.39, 0.31))
    for case in ["Massachusetts", "Lovelock Valley"]:
        p = diag[diag["case"] == case]
        ax_c.scatter(
            p["robust_capacity_factor"],
            p["early_response_factor"],
            s=18,
            color=CASE_COLORS[case],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.25,
            label="MA" if case == "Massachusetts" else "LV",
        )
    ax_c.axvline(1, color=RULE, lw=0.55)
    ax_c.axhline(1, color=RULE, lw=0.55)
    ax_c.set_xlabel("capacity factor", labelpad=1)
    ax_c.set_ylabel("response-time factor", labelpad=1)
    ax_c.set_title("Well-level inheritance", loc="left", fontsize=6.7)
    ax_c.grid(True, alpha=0.14, linewidth=0.4)

    ax_t = inset(ax, (0.07, 0.11, 0.85, 0.31))
    summary = weights.reset_index()
    labels = ["MA", "LV"]
    cap = summary["median_robust_capacity_factor"].to_numpy(dtype=float)
    rsp = summary["median_early_response_factor"].to_numpy(dtype=float)
    p95cap = summary["p95_robust_capacity_factor"].to_numpy(dtype=float)
    p95rsp = summary["p95_early_response_factor"].to_numpy(dtype=float)
    x = np.arange(2)
    ax_t.scatter(x - 0.12, cap, s=34, color=cmc.batlow(0.30), label="median capacity")
    ax_t.scatter(x + 0.12, rsp, s=34, color=cmc.batlow(0.75), label="median timing")
    ax_t.vlines(x - 0.12, cap, p95cap, color=cmc.batlow(0.30), lw=1.1)
    ax_t.vlines(x + 0.12, rsp, p95rsp, color=cmc.batlow(0.75), lw=1.1)
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(labels)
    ax_t.set_ylabel("factor", labelpad=1)
    ax_t.set_title("Median with 95th-percentile tail", loc="left", fontsize=6.7)
    ax_t.grid(axis="y", alpha=0.16, linewidth=0.4)
    ax_t.legend(frameon=False, ncol=2, loc="upper left", fontsize=5.6)


def write_source_report(path: Path) -> None:
    report = [
        "# Figure 1 Evidence Graphical Abstract Sources",
        "",
        "- Output stem: `fig01_evidence_graphical_abstract`.",
        "- Panel A: evidence counts from `evidence_map.yaml` and manuscript state.",
        "- Panel B: Massachusetts well-level fits from `tables/field_pathway_predictions.csv`, `tables/field_pathway_model_comparison.csv`, and `tables/field_pathway_summary.csv`.",
        "- Panel C: MODFLOW benchmark fields from `tables/mf6_mega_benchmark_cases.csv` regenerated through `analysis/mf6_mega_benchmark.py`; model-factor scatter from `tables/mf6_mega_benchmark_fits.csv`.",
        f"  - log K field scenario: `{SELECTED_SCENARIOS.get('log K field', 'not recorded')}`.",
        f"  - log S field scenario: `{SELECTED_SCENARIOS.get('log S field', 'not recorded')}`.",
        "- Panel D: field applicability factors from `tables/mf6_mega_field_applicability_criteria_summary.csv`.",
        "- Panel E: management inheritance diagnostics from `tables/field_bma_management_summary.csv` and `tables/field_bma_management_well_diagnostics.csv`.",
        "",
        "The flow boxes are conceptual; all plotted numerical elements are regenerated from existing project outputs.",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    lock_style()
    predictions = pd.read_csv(TABLE_DIR / "field_pathway_predictions.csv")
    predictions = predictions[predictions["pathway"].isin(PATHWAYS)].copy()
    comparison = pd.read_csv(TABLE_DIR / "field_pathway_model_comparison.csv")
    comparison = comparison[comparison["pathway"].isin(PATHWAYS)].copy()
    metadata = pd.read_csv(TABLE_DIR / "well_metadata.csv")

    fig = plt.figure(figsize=(15.8, 8.2), dpi=180, facecolor="white")
    gs = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.18, 1.0, 1.0, 1.12, 1.12],
        height_ratios=[1, 1],
        left=0.02,
        right=0.985,
        top=0.96,
        bottom=0.045,
        wspace=0.13,
        hspace=0.16,
    )

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:3])
    ax_c = fig.add_subplot(gs[0, 3:5])
    ax_d = fig.add_subplot(gs[1, 1:3])
    ax_e = fig.add_subplot(gs[1, 3:5])

    panel_a(ax_a)
    panel_b(ax_b, predictions, comparison, metadata)
    panel_c(ax_c)
    panel_d(ax_d)
    panel_e(ax_e)

    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e]:
        ax.set_xticks([])
        ax.set_yticks([])

    stem = PROJECT / "fig01_evidence_graphical_abstract"
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.035, **kwargs)
    plt.close(fig)

    OUT_DIR.mkdir(exist_ok=True)
    write_source_report(OUT_DIR / "fig01_evidence_graphical_abstract_sources.md")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()

