from __future__ import annotations

import math
import sys
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mf6_mega_benchmark as mega  # noqa: E402


PROJECT = HERE.parents[0]
TABLE_DIR = PROJECT / "tables"
FIG_DIR = PROJECT / "figures"

PATHWAYS = [
    "Theis confined",
    "Hantush-Jacob leaky",
    "Lagging Darcy no-leakage",
    "Lagging Darcy with leakage",
]

PATHWAY_SHORT = {
    "Theis confined": "Theis",
    "Hantush-Jacob leaky": "Hantush\nleaky",
    "Lagging Darcy no-leakage": "Lagging\nno leak",
    "Lagging Darcy with leakage": "Lagging\nleaky",
}

CLASS_SHORT = {
    "homogeneous confined": "homogeneous",
    "vertical leakage": "leakage",
    "finite no-flow boundary": "no-flow\nboundary",
    "constant head boundary": "constant-head\nboundary",
    "heterogeneous K": "heterogeneous K",
    "anisotropic heterogeneous K": "anisotropic K",
    "heterogeneous storage": "heterogeneous S",
    "heterogeneous leakage": "heterogeneous\nleakage",
    "heterogeneous boundary": "heterogeneous\nboundary",
    "combined leakage boundary": "combined",
}

CLASS_TINY = {
    "combined leakage boundary": "combined",
    "heterogeneous leakage": "het leakage",
    "heterogeneous boundary": "het boundary",
    "heterogeneous storage": "het S",
    "heterogeneous K": "het K",
    "finite no-flow boundary": "no-flow",
}

FIELD_ABBR = {"Massachusetts": "MA", "Lovelock Valley": "LV"}


def cmc_color(name: str, sample: float) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in getattr(cmc, name)(sample))


PATHWAY_COLORS = {
    pathway: cmc_color("batlow", sample)
    for pathway, sample in zip(PATHWAYS, [0.12, 0.38, 0.66, 0.90])
}
FIELD_COLORS = {
    "Massachusetts": cmc_color("batlow", 0.25),
    "Lovelock Valley": cmc_color("batlow", 0.75),
}
NEUTRAL_DARK = cmc_color("grayC", 0.16)
NEUTRAL_MID = cmc_color("grayC", 0.48)
NEUTRAL_LIGHT = cmc_color("grayC", 0.90)


def lock_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.3,
            "axes.labelsize": 7.4,
            "axes.titlesize": 7.8,
            "legend.fontsize": 6.3,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str, text: str) -> None:
    ax.text(
        0.0,
        1.04,
        f"({label}) {text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=7.3,
    )


def export(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for base in [FIG_DIR, PROJECT]:
        for suffix in ["pdf", "png", "svg"]:
            kwargs = {"dpi": 450} if suffix == "png" else {}
            fig.savefig(base / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color) -> None:
    rect = Rectangle(xy, width, height, facecolor=color, edgecolor=NEUTRAL_DARK, linewidth=0.6)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=5.8)


def make_fig13_design() -> None:
    cases = pd.read_csv(TABLE_DIR / "mf6_mega_benchmark_cases.csv")

    def representative_case(scenario_class: str, score_cols: list[str]) -> mega.MegaScenario:
        sub = cases[cases["scenario_class"] == scenario_class].copy()
        if scenario_class in {"constant head boundary", "heterogeneous boundary", "combined leakage boundary"}:
            sub = sub[sub["boundary_type"] != "none"].copy()
        if sub.empty:
            raise ValueError(f"No benchmark case found for {scenario_class}")
        score = np.zeros(sub.shape[0])
        for col in score_cols:
            prefer_small = col.startswith("small:")
            col_name = col.split(":", 1)[1] if prefer_small else col
            if col_name in sub.columns:
                values = sub[col_name].fillna(0.0).to_numpy(float)
                denom = max(float(np.nanmax(values) - np.nanmin(values)), 1.0e-12)
                normalized = (values - float(np.nanmin(values))) / denom
                score += 1.0 - normalized if prefer_small else normalized
        sub = sub.assign(_score=score)
        domain_diff = (sub["domain_m"] - 805.0).abs()
        sub = sub[domain_diff == domain_diff.min()].copy()
        row = sub.sort_values(["_score", "grid_cell_count"], ascending=[False, True]).iloc[0]
        scenario_keys = set(mega.MegaScenario.__dataclass_fields__.keys())
        data = {key: row[key] for key in scenario_keys if key in row.index}
        return mega.scenario_from_dict(data)

    scenarios = {
        "k": representative_case("heterogeneous K", ["sigma_ln_k", "support_target_spread", "small:corr_len_m"]),
        "anisotropy": representative_case("anisotropic heterogeneous K", ["sigma_ln_k", "anisotropy_ratio", "small:corr_len_m"]),
        "storage": representative_case("heterogeneous storage", ["sigma_ln_s", "small:corr_len_s_m"]),
        "combined": representative_case("combined leakage boundary", ["sigma_ln_k", "sigma_ln_s", "support_target_spread", "small:corr_len_m", "small:corr_len_s_m"]),
    }

    fields = {}
    for key, scenario in scenarios.items():
        nrow, ncol = mega.grid_shape(scenario.domain_m, scenario.dx_m)
        kx, ky, storage = mega.make_property_fields(scenario, nrow, ncol)
        half_x = (ncol - 1) * scenario.dx_m / 2
        half_y = (nrow - 1) * scenario.dx_m / 2
        fields[key] = {
            "scenario": scenario,
            "kx": kx,
            "ky": ky,
            "storage": storage,
            "extent": [-half_x, half_x, -half_y, half_y],
        }

    k_arrays = [np.log10(fields[key]["kx"]) for key in ["k", "anisotropy", "combined"]]
    s_arrays = [np.log10(fields[key]["storage"]) for key in ["storage", "combined"]]
    k_vmin = min(float(np.nanmin(array)) for array in k_arrays)
    k_vmax = max(float(np.nanmax(array)) for array in k_arrays)
    s_vmin = min(float(np.nanmin(array)) for array in s_arrays)
    s_vmax = max(float(np.nanmax(array)) for array in s_arrays)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.875, bottom=0.095, top=0.925, wspace=0.34, hspace=0.16)

    def field_note(ax: plt.Axes, text: str) -> None:
        ax.text(
            0.03,
            0.96,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.0,
            color="white",
            bbox={"facecolor": (0, 0, 0, 0.28), "edgecolor": "none", "pad": 1.4},
        )

    def effective_support_radius_for_plot(scenario: mega.MegaScenario) -> float:
        nrow, ncol = mega.grid_shape(scenario.domain_m, scenario.dx_m)
        kx, ky, storage = mega.make_property_fields(scenario, nrow, ncol)
        targets = mega.support_targets(kx, ky, storage, scenario)
        return float(targets["support_effective_radius_m"])

    def draw_wells_and_support(ax: plt.Axes, scenario: mega.MegaScenario, color: str = "white") -> None:
        support_radius = effective_support_radius_for_plot(scenario)
        support = plt.Circle((0, 0), support_radius, fill=False, color=color, linewidth=0.8, linestyle="--")
        ax.add_patch(support)
        for _, radius, angle in mega.observation_wells(scenario):
            theta = math.radians(angle)
            ax.scatter(radius * math.cos(theta), radius * math.sin(theta), s=10, color=color, edgecolor=NEUTRAL_DARK, linewidth=0.25, zorder=4)
        ax.scatter([0], [0], marker="*", s=45, color=color, edgecolor=NEUTRAL_DARK, linewidth=0.45, zorder=5)

    def draw_boundary(ax: plt.Axes, scenario: mega.MegaScenario, color=None) -> None:
        color = color or cmc_color("batlow", 0.82)
        extent = ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]
        xmin, xmax, ymin, ymax = extent
        if scenario.boundary_type == "constant_head_all":
            ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color=color, linewidth=1.3)
        elif scenario.boundary_type == "constant_head_x":
            ax.plot([xmin, xmax], [ymin, ymin], color=color, linewidth=1.3)
            ax.plot([xmin, xmax], [ymax, ymax], color=color, linewidth=1.3)
        elif scenario.boundary_type == "constant_head_east":
            ax.plot([xmax, xmax], [ymin, ymax], color=color, linewidth=1.3)

    ax = axes[0, 0]
    panel_label(ax, "a", "grid, wells, and support")
    display_dx = 10.0
    lim = 405.0
    for value in np.arange(-lim, lim + display_dx, display_dx):
        ax.axhline(value, color=NEUTRAL_LIGHT, linewidth=0.25, zorder=0)
        ax.axvline(value, color=NEUTRAL_LIGHT, linewidth=0.25, zorder=0)
    grid_scenario = scenarios["k"]
    draw_wells_and_support(ax, grid_scenario, color=NEUTRAL_DARK)
    ax.text(0.03, 0.96, "numerical grid\n$\\Delta x$ = 5--10 m\neffective support", transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=NEUTRAL_DARK)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax = axes[0, 1]
    panel_label(ax, "b", "heterogeneous K")
    im_k = ax.imshow(np.log10(fields["k"]["kx"]), extent=fields["k"]["extent"], cmap=cmc.lipari, origin="lower", vmin=k_vmin, vmax=k_vmax)
    draw_wells_and_support(ax, scenarios["k"])
    field_note(ax, "lognormal K")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax = axes[0, 2]
    panel_label(ax, "c", "anisotropic K")
    ax.imshow(np.log10(fields["anisotropy"]["kx"]), extent=fields["anisotropy"]["extent"], cmap=cmc.lipari, origin="lower", vmin=k_vmin, vmax=k_vmax)
    draw_wells_and_support(ax, scenarios["anisotropy"])
    ax.annotate("", xy=(80, -150), xytext=(-80, -150), arrowprops={"arrowstyle": "<->", "color": "white", "linewidth": 1.0})
    field_note(ax, "$K_y/K_x$ varied")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax = axes[1, 0]
    panel_label(ax, "d", "heterogeneous storage")
    im_s = ax.imshow(np.log10(fields["storage"]["storage"]), extent=fields["storage"]["extent"], cmap=cmc.batlow, origin="lower", vmin=s_vmin, vmax=s_vmax)
    draw_wells_and_support(ax, scenarios["storage"])
    field_note(ax, "lognormal S")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax = axes[1, 1]
    panel_label(ax, "e", "combined K and boundary")
    ax.imshow(np.log10(fields["combined"]["kx"]), extent=fields["combined"]["extent"], cmap=cmc.lipari, origin="lower", vmin=k_vmin, vmax=k_vmax)
    draw_wells_and_support(ax, scenarios["combined"])
    draw_boundary(ax, scenarios["combined"])
    field_note(ax, "K + leakage\n+ boundary stress")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax = axes[1, 2]
    panel_label(ax, "f", "combined storage")
    ax.imshow(np.log10(fields["combined"]["storage"]), extent=fields["combined"]["extent"], cmap=cmc.batlow, origin="lower", vmin=s_vmin, vmax=s_vmax)
    draw_wells_and_support(ax, scenarios["combined"])
    draw_boundary(ax, scenarios["combined"])
    field_note(ax, "S field in\ncombined class")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5.8)

    cax_k = fig.add_axes([0.902, 0.565, 0.018, 0.34])
    cax_s = fig.add_axes([0.902, 0.125, 0.018, 0.34])
    cb_k = fig.colorbar(im_k, cax=cax_k)
    cb_k.set_label("log10 K (m d$^{-1}$)")
    cb_s = fig.colorbar(im_s, cax=cax_s)
    cb_s.set_label("log10 S")
    export(fig, "fig13_mf6_benchmark_design")


def make_fig14_verification() -> None:
    limits = pd.read_csv(TABLE_DIR / "mf6_mega_analytical_limit_verification.csv")
    deg = pd.read_csv(TABLE_DIR / "mf6_mega_mf6_to_analytical_consistency_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    short_map = {
        "Hantush B -> infinity recovers Theis": "HJ -> Theis",
        "Lagging leaky B -> infinity recovers lagging no-leakage": "LD+L -> LD",
        "Lagging leaky no-lag and B -> infinity recovers Theis": "LD+L -> Theis",
        "Lagging leaky tau_q=tau_s recovers Hantush": "LD+L -> HJ",
        "Lagging tau_q=tau_s recovers Theis": "LD -> Theis",
    }
    limits["short"] = limits["test_name"].map(short_map).fillna(limits["test_name"])
    test_order = [short_map[key] for key in short_map]
    limit_colors = {label: cmc_color("batlow", 0.12 + 0.78 * i / (len(test_order) - 1)) for i, label in enumerate(test_order)}
    ax = axes[0, 0]
    panel_label(ax, "a", "analytical limiting checks")
    for label in test_order:
        group = limits[limits["short"] == label].sort_values("radius_m")
        ax.plot(group["radius_m"], group["max_abs_eta"], marker="o", linewidth=1.0, markersize=3.2, color=limit_colors[label], label=label)
    ax.axhline(0.02, color=NEUTRAL_DARK, linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("observation distance (m)")
    ax.set_ylabel("max |log factor|")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(color=NEUTRAL_LIGHT, linewidth=0.6)

    ax = axes[0, 1]
    panel_label(ax, "b", "drawdown discrepancy")
    for label in test_order:
        group = limits[limits["short"] == label].sort_values("radius_m")
        ax.plot(group["radius_m"], group["rmse_m"], marker="o", linewidth=1.0, markersize=3.2, color=limit_colors[label], label=label)
    ax.set_xscale("log")
    ax.set_xlabel("observation distance (m)")
    ax.set_ylabel("RMSE (m)")
    ax.grid(color=NEUTRAL_LIGHT, linewidth=0.6)

    ax = axes[1, 0]
    panel_label(ax, "c", "numerical-to-analytical log residual")
    for cls, group in deg.groupby("scenario_class"):
        label = "Theis generator" if cls == "homogeneous confined" else "Hantush generator"
        color = cmc_color("batlow", 0.25 if cls == "homogeneous confined" else 0.70)
        group = group.sort_values("dx_m")
        ax.plot(group["dx_m"], group["median_sd_eta"], marker="o", color=color, label=f"{label}, median")
        ax.plot(group["dx_m"], group["max_sd_eta"], marker="s", color=color, linestyle="--", label=f"{label}, max")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("grid spacing dx (m)")
    ax.set_ylabel("log residual SD")
    ax.legend(frameon=False, ncol=1)
    ax.grid(color=NEUTRAL_LIGHT, linewidth=0.6)

    ax = axes[1, 1]
    panel_label(ax, "d", "drawdown error and pass fraction")
    for cls, group in deg.groupby("scenario_class"):
        color = cmc_color("batlow", 0.25 if cls == "homogeneous confined" else 0.70)
        group = group.sort_values("dx_m")
        ax.plot(group["dx_m"], group["max_abs_error_m"], marker="o", color=color)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("grid spacing dx (m)")
    ax.set_ylabel("max absolute error (m)")
    ax2 = ax.twinx()
    for cls, group in deg.groupby("scenario_class"):
        color = cmc_color("grayC", 0.25 if cls == "homogeneous confined" else 0.55)
        group = group.sort_values("dx_m")
        ax2.plot(group["dx_m"], group["pass_fraction"], marker="^", linestyle=":", color=color)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("pass fraction")
    ax.grid(color=NEUTRAL_LIGHT, linewidth=0.6)
    export(fig, "fig14_mf6_verification_sensitivity")


def make_fig15_cov_attribution() -> None:
    fits = pd.read_csv(TABLE_DIR / "mf6_mega_benchmark_fits.csv")
    fits = fits[fits["quality_gate"].astype(bool)].copy()
    fits["M_T"] = np.exp(fits["lnM_T"])
    fits["M_S"] = np.exp(fits["lnM_S"])
    fits["M_response_time"] = np.exp(fits["lnM_response_time"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    def plot_ecdf(ax: plt.Axes, column: str, xlabel: str) -> None:
        all_values = fits[column].replace([np.inf, -np.inf], np.nan).dropna()
        xmin = max(0.03, float(all_values.quantile(0.005)))
        xmax = min(40.0, float(all_values.quantile(0.995)))
        for pathway in PATHWAYS:
            vals = fits.loc[fits["pathway"] == pathway, column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals > 0) & (vals <= 60)]
            if vals.size == 0:
                continue
            vals = np.sort(vals)
            prob = np.arange(1, vals.size + 1) / vals.size
            ax.plot(vals, prob, color=PATHWAY_COLORS[pathway], linewidth=1.2, label=PATHWAY_SHORT[pathway].replace("\n", " "))
        ax.axvline(1.0, color=NEUTRAL_DARK, linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cumulative probability")
        ax.grid(color=NEUTRAL_LIGHT, linewidth=0.6)

    ax = axes[0, 0]
    panel_label(ax, "a", "transmissivity factor")
    plot_ecdf(ax, "M_T", "$M_T=T_{fit}/T_{ref}$")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    panel_label(ax, "b", "storage factor")
    plot_ecdf(ax, "M_S", "$M_S=S_{fit}/S_{ref}$")

    ax = axes[1, 0]
    panel_label(ax, "c", "response-time factor")
    plot_ecdf(ax, "M_response_time", "$M_{time}$")

    ax = axes[1, 1]
    panel_label(ax, "d", "distance-residual envelope")
    bins = pd.IntervalIndex.from_tuples([(0, 15), (15, 35), (35, 75), (75, 125), (125, 250)])
    fits["radius_group"] = pd.cut(fits["radius_m"], bins)
    centers = np.asarray([(item.left + item.right) / 2 for item in bins])
    for pathway in PATHWAYS:
        group = fits[fits["pathway"] == pathway]
        med = group.groupby("radius_group", observed=False)["sd_eta"].median().reindex(bins).to_numpy()
        q90 = group.groupby("radius_group", observed=False)["sd_eta"].quantile(0.90).reindex(bins).to_numpy()
        ax.plot(centers, med, color=PATHWAY_COLORS[pathway], linewidth=1.2, marker="o", markersize=3.0)
        ax.plot(centers, q90, color=PATHWAY_COLORS[pathway], linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("observation distance (m)")
    ax.set_ylabel("response log residual SD")
    ax.grid(color=NEUTRAL_LIGHT, linewidth=0.6)
    export(fig, "fig15_mf6_convergence_cov")


def parse_bic_summary(text: str) -> dict[str, int]:
    out = {pathway: 0 for pathway in PATHWAYS}
    for part in str(text).split("/"):
        if ":" not in part:
            continue
        name, value = part.strip().rsplit(":", 1)
        out[name.strip()] = int(float(value.strip()))
    return out


def make_fig16_field_gate() -> None:
    gate = pd.read_csv(TABLE_DIR / "mf6_mega_field_applicability_gate_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    ax = axes[0, 0]
    panel_label(ax, "a", "field match to numerical classes")
    level_map = {
        "primary_conservative": 3,
        "primary_sensitivity": 2,
        "lower_complexity_check": 1,
        "boundary_sensitivity": 1,
        "not_primary_unless_local_leakage_supported": 0,
    }
    fields = ["Massachusetts", "Lovelock Valley"]
    gate = gate.copy()
    gate["level_value"] = gate["applicability_level"].map(level_map)
    class_order = [
        "heterogeneous K",
        "heterogeneous storage",
        "heterogeneous leakage",
        "heterogeneous boundary",
        "finite no-flow boundary",
        "combined leakage boundary",
    ]
    class_x = {name: idx for idx, name in enumerate(class_order)}
    gate["class_x"] = gate["benchmark_class"].map(class_x)
    gate = gate.sort_values(["field_case", "level_value", "benchmark_class"], ascending=[True, False, True]).reset_index(drop=True)
    for field, group in gate.groupby("field_case", sort=False):
        group = group.dropna(subset=["class_x"]).sort_values("class_x")
        ax.scatter(group["class_x"], group["level_value"], s=46, color=FIELD_COLORS[field], edgecolor=NEUTRAL_DARK, linewidth=0.35, label=field, zorder=3)
    ax.set_xticks(np.arange(len(class_order)), [CLASS_TINY.get(name, name) for name in class_order], rotation=22, ha="right")
    ax.set_yticks([0, 1, 2, 3], ["not primary", "check", "sensitivity", "primary"])
    ax.set_ylim(-0.35, 3.35)
    ax.set_ylabel("field applicability")
    ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    panel_label(ax, "b", "field conditioned model factor COV")
    marker_map = {
        "primary_conservative": "o",
        "primary_sensitivity": "s",
        "lower_complexity_check": "^",
        "boundary_sensitivity": "^",
        "not_primary_unless_local_leakage_supported": "v",
    }
    for field, group in gate.groupby("field_case", sort=False):
        for row in group.itertuples():
            ax.scatter(
                row.median_cov_M_T,
                row.median_cov_M_S,
                s=42 + 18 * max(float(row.level_value), 0.0),
                marker=marker_map.get(row.applicability_level, "o"),
                color=FIELD_COLORS[field],
                edgecolor=NEUTRAL_DARK,
                linewidth=0.35,
                alpha=0.90 if row.level_value >= 2 else 0.58,
            )
    lim_min = 0.9
    lim_max = max(float(gate["median_cov_M_T"].max()), float(gate["median_cov_M_S"].max())) * 1.10
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=NEUTRAL_LIGHT, linewidth=0.9, linestyle="--")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("$COV(M_T)$ from matched numerical class")
    ax.set_ylabel("$COV(M_S)$ from matched numerical class")
    ax.grid(color=NEUTRAL_LIGHT, linewidth=0.6)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=FIELD_COLORS["Massachusetts"], markeredgecolor=NEUTRAL_DARK, label="Massachusetts"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=FIELD_COLORS["Lovelock Valley"], markeredgecolor=NEUTRAL_DARK, label="Lovelock Valley"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=NEUTRAL_MID, markeredgecolor=NEUTRAL_DARK, label="primary"),
            Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=NEUTRAL_MID, markeredgecolor=NEUTRAL_DARK, label="sensitivity"),
            Line2D([0], [0], marker="^", linestyle="none", markerfacecolor=NEUTRAL_MID, markeredgecolor=NEUTRAL_DARK, alpha=0.60, label="check"),
        ],
        frameon=False,
        loc="lower right",
    )

    ax = axes[1, 0]
    panel_label(ax, "c", "field BIC support")
    summaries = gate.groupby("field_case")["field_bic_preferred_summary"].first()
    xp = np.arange(len(PATHWAYS))
    for field in fields:
        parsed = parse_bic_summary(summaries[field])
        vals = []
        for pathway in PATHWAYS:
            vals.append(parsed.get(pathway, 0))
        ax.plot(xp, vals, marker="o", linewidth=1.2, color=FIELD_COLORS[field], label=field)
    ax.set_xticks(xp, [PATHWAY_SHORT[p].replace("\n", " ") for p in PATHWAYS], rotation=12, ha="right")
    ax.set_ylabel("BIC-preferred wells")
    ax.set_ylim(-0.2, 6.2)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=NEUTRAL_LIGHT, linewidth=0.6)

    ax = axes[1, 1]
    panel_label(ax, "d", "decision inheritance scale")
    decision = gate.groupby("field_case", as_index=False).agg(
        capacity=("field_decision_capacity_p95", "first"),
        response=("field_decision_response_p95", "first"),
    )
    for row in decision.itertuples():
        ax.scatter(row.capacity, row.response, s=62, color=FIELD_COLORS[row.field_case], edgecolor=NEUTRAL_DARK, linewidth=0.35, label=row.field_case)
    ax.axvline(1.0, color=NEUTRAL_LIGHT, linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color=NEUTRAL_LIGHT, linewidth=0.8, linestyle="--")
    ax.set_xlabel("capacity factor, p95")
    ax.set_ylabel("response-time factor, p95")
    ax.set_xlim(0.8, max(decision["capacity"]) * 1.25)
    ax.set_ylim(0.8, max(decision["response"]) * 1.18)
    ax.grid(color=NEUTRAL_LIGHT, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    export(fig, "fig16_mf6_field_applicability")


def main() -> None:
    lock_style()
    make_fig13_design()
    make_fig14_verification()
    make_fig15_cov_attribution()
    make_fig16_field_gate()


if __name__ == "__main__":
    main()

