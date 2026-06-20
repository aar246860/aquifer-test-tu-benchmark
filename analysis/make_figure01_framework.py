"""Generate Figure 1: journal-style transformation uncertainty framework.

The layout follows main-text conceptual figures in geoscience journals: a
compact grammar panel at left, process panels at right, sparse text, thin rules,
and small vector pictorial insets. All visible marks are generated from
Matplotlib vector primitives.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import to_hex, to_rgb
import numpy as np
from cmcrameri import cm as cmc

from figure_layout_qa import (
    assert_layout_pass,
    run_layout_qa,
    tag_artist,
    write_layout_qa_reports,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "fig01_pathway_framework"
QA_JSON = ROOT / "outputs" / "fig01_layout_qa.json"
QA_MD = ROOT / "outputs" / "fig01_layout_qa.md"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.55,
    }
)


def cmap_color(name: str, value: float) -> str:
    return to_hex(getattr(cmc, name)(value), keep_alpha=False)


def soften(color: str, white_fraction: float = 0.94) -> str:
    rgb = np.array(to_rgb(color))
    return to_hex(rgb * (1.0 - white_fraction) + white_fraction, keep_alpha=False)


ACCENTS = {
    "data": cmap_color("navia", 0.30),
    "path": cmap_color("navia", 0.52),
    "diag": cmap_color("batlow", 0.66),
    "bench": cmap_color("batlow", 0.36),
    "field": cmap_color("vik", 0.66),
    "report": cmap_color("batlow", 0.80),
}

PALETTE = {
    "ink": "#1f2429",
    "muted": "#626970",
    "rule": "#bfc5c8",
    "soft_rule": "#e3e7e9",
    "panel": "#fbfcfc",
    "paper": "#ffffff",
    **ACCENTS,
}


def qa_tag(artist, name: str, role: str, panel: str | None = None, **kwargs):
    return tag_artist(artist, name=name, role=role, panel=panel, **kwargs)


def add_panel(
    ax,
    panel_id: str,
    label: str,
    title: str,
    xy: tuple[float, float],
    wh: tuple[float, float],
    accent: str,
):
    x, y = xy
    w, h = wh
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=0.65,
        edgecolor=PALETTE["rule"],
        facecolor=PALETTE["panel"],
        zorder=1,
    )
    ax.add_patch(rect)
    qa_tag(rect, name=f"panel:{panel_id}", role="panel", panel=panel_id)

    qa_tag(
        ax.text(
            x + 0.16,
            y + h - 0.20,
            label,
            ha="left",
            va="top",
            fontsize=8.2,
            weight="bold",
            color=accent,
        ),
        name=f"{panel_id}_label",
        role="panel_number",
        panel=panel_id,
    )
    qa_tag(
        ax.text(
            x + 0.42,
            y + h - 0.20,
            title,
            ha="left",
            va="top",
            fontsize=6.95,
            weight="bold",
            color=PALETTE["ink"],
            linespacing=1.02,
        ),
        name=f"{panel_id}_title",
        role="panel_title",
        panel=panel_id,
    )
    ax.plot([x + 0.16, x + w - 0.16], [y + h - 0.50, y + h - 0.50], color=accent, lw=0.55, zorder=2)
    return x, y, w, h


def add_icon(
    ax,
    icon_name: str,
    extent: tuple[float, float, float, float],
    *,
    alpha: float = 0.92,
    zorder: float = 1.5,
) -> None:
    x0, x1, y0, y1 = extent
    w = x1 - x0
    h = y1 - y0
    cx = x0 + 0.5 * w
    cy = y0 + 0.5 * h

    def sx(v: float) -> float:
        return x0 + v * w

    def sy(v: float) -> float:
        return y0 + v * h

    def add_line(points, color, lw=0.65, ls="-"):
        xs, ys = zip(*[(sx(px), sy(py)) for px, py in points])
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder + 0.1)

    def add_rect(px, py, pw, ph, color, fill=False, lw=0.65):
        ax.add_patch(
            patches.Rectangle(
                (sx(px), sy(py)),
                pw * w,
                ph * h,
                facecolor=color if fill else "none",
                edgecolor=color,
                lw=lw,
                alpha=alpha,
                zorder=zorder,
            )
        )

    color = PALETTE["muted"]
    if "measured_response" in icon_name:
        color = PALETTE["data"]
        add_rect(0.18, 0.22, 0.10, 0.46, color)
        add_rect(0.70, 0.36, 0.10, 0.36, color)
        add_line([(0.12, 0.78), (0.30, 0.74), (0.50, 0.77), (0.78, 0.69)], color, lw=0.75, ls=(0, (4, 3)))
        add_line([(0.16, 0.16), (0.16, 0.56), (0.86, 0.16)], PALETTE["ink"], lw=0.55)
        t = np.linspace(0.20, 0.82, 40)
        s = 0.20 + 0.30 * (1 - np.exp(-4 * (t - 0.20)))
        add_line(list(zip(t, s)), color, lw=1.0)
    elif "analytical_pathways" in icon_name:
        color = PALETTE["path"]
        ax.add_patch(patches.Circle((cx, cy), 0.43 * min(w, h), facecolor="none", edgecolor=color, lw=0.75, alpha=alpha, zorder=zorder))
        for yy in [0.36, 0.48, 0.60]:
            add_line([(0.22, yy), (0.78, yy)], color, lw=0.7)
        add_line([(0.25, 0.26), (0.39, 0.29), (0.52, 0.25), (0.68, 0.30)], color, lw=0.75, ls=(0, (4, 3)))
    elif "diagnostics" in icon_name:
        color = PALETTE["diag"]
        pts = [(0.25, 0.30), (0.38, 0.60), (0.49, 0.42), (0.55, 0.72), (0.66, 0.36), (0.75, 0.58), (0.32, 0.78), (0.62, 0.22)]
        for px, py in pts:
            ax.add_patch(patches.Circle((sx(px), sy(py)), 0.026 * min(w, h), facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder))
        add_line([(0.17, 0.18), (0.86, 0.18)], PALETTE["ink"], lw=0.45)
    elif "benchmark" in icon_name:
        color = PALETTE["bench"]
        for ix in range(4):
            for iy in range(4):
                add_rect(0.18 + ix * 0.15, 0.22 + iy * 0.13, 0.10, 0.08, color, fill=(ix + iy) % 3 == 0, lw=0.45)
        ax.add_patch(patches.Circle((sx(0.70), sy(0.70)), 0.11 * min(w, h), facecolor="none", edgecolor=color, lw=0.75, alpha=alpha, zorder=zorder))
        add_line([(0.70, 0.70), (0.83, 0.83)], color, lw=0.8)
    elif "screening" in icon_name:
        color = PALETTE["field"]
        for i, yy in enumerate([0.72, 0.52, 0.32]):
            add_rect(0.18, yy - 0.04, 0.09, 0.09, color, fill=False, lw=0.65)
            add_line([(0.19, yy), (0.22, yy - 0.025), (0.28, yy + 0.045)], color, lw=0.75)
            add_line([(0.36, yy), (0.82, yy)], color, lw=0.65)
    elif "reporting" in icon_name:
        color = PALETTE["report"]
        add_rect(0.22, 0.18, 0.52, 0.68, color, fill=False, lw=0.75)
        add_line([(0.60, 0.86), (0.74, 0.72), (0.74, 0.86)], color, lw=0.75)
        for i, bh in enumerate([0.16, 0.28, 0.40]):
            add_rect(0.30 + i * 0.10, 0.32, 0.055, bh, color, fill=True, lw=0.45)
        for yy in [0.28, 0.22, 0.16]:
            add_line([(0.30, yy), (0.66, yy)], color, lw=0.55)
    else:
        ax.add_patch(patches.Circle((cx, cy), 0.40 * min(w, h), facecolor="none", edgecolor=color, lw=0.75, alpha=alpha, zorder=zorder))


def straight_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    name: str,
    color: str = "#343a40",
    allow_arrow_overlap: bool = False,
):
    ann = ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=0.95, color=color, shrinkA=3, shrinkB=3, mutation_scale=11),
        zorder=4,
    )
    return qa_tag(
        ann,
        name=name,
        role="arrow",
        arrow_start=start,
        arrow_end=end,
        allow_arrow_overlap=allow_arrow_overlap,
        check_panel_bounds=False,
    )


def tagged_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    name: str,
    panel: str,
    fontsize: float = 6.3,
    color: str | None = None,
    weight: str = "normal",
    ha: str = "left",
    va: str = "center",
    linespacing: float = 1.05,
):
    return qa_tag(
        ax.text(
            x,
            y,
            text,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color or PALETTE["ink"],
            weight=weight,
            linespacing=linespacing,
        ),
        name=name,
        role="major_text",
        panel=panel,
    )


def draw_left_grammar(ax, panel, accent: str):
    x, y, w, h = panel
    rows = [
        ("fig01_icon_1_measured_response.png", "measured response", r"$s(t,r)$ and test geometry", PALETTE["data"]),
        ("fig01_icon_2_analytical_pathways.png", "interpretation pathways", "Theis, leaky, lagging Darcy", PALETTE["path"]),
        ("fig01_icon_3_diagnostics.png", "response diagnostics", r"residual $\eta(t,r)$ and spread", PALETTE["diag"]),
        ("fig01_icon_4_benchmark.png", "benchmark support", "conditional model factors", PALETTE["bench"]),
    ]
    row_top = y + h - 0.80
    gap = 0.80
    for i, (icon, heading, note, color) in enumerate(rows):
        yy = row_top - i * gap
        ax.add_patch(patches.Circle((x + 0.40, yy), 0.20, facecolor=soften(color, 0.88), edgecolor=color, lw=0.65, zorder=1.2))
        add_icon(ax, icon, (x + 0.24, x + 0.56, yy - 0.16, yy + 0.16), alpha=0.82, zorder=1.6)
        tagged_text(ax, x + 0.72, yy + 0.10, heading, name=f"grammar_{i + 1}_heading", panel="grammar", fontsize=6.25, weight="bold")
        tagged_text(ax, x + 0.72, yy - 0.13, note, name=f"grammar_{i + 1}_note", panel="grammar", fontsize=5.45, color=PALETTE["muted"])
        if i < len(rows) - 1:
            ax.plot([x + 0.40, x + 0.40], [yy - 0.28, yy - gap + 0.28], color=PALETTE["soft_rule"], lw=0.8, zorder=1.1)
    tagged_text(
        ax,
        x + 0.20,
        y + 0.26,
        "Drawdown to apparent parameters.",
        name="grammar_purpose",
        panel="grammar",
        fontsize=5.55,
        color=PALETTE["muted"],
    )


def draw_drawdown_curve(ax, x0: float, y0: float, w: float, h: float, color: str):
    ax.plot([x0, x0 + w], [y0, y0], color=PALETTE["ink"], lw=0.55, zorder=3)
    ax.plot([x0, x0], [y0, y0 + h], color=PALETTE["ink"], lw=0.55, zorder=3)
    t = np.linspace(0, 1, 80)
    s = 0.08 + 0.70 * (1.0 - np.exp(-3.5 * t))
    ax.plot(x0 + w * t, y0 + h * s, color=color, lw=1.25, zorder=3)
    obs_t = np.linspace(0.10, 0.94, 7)
    obs_s = 0.08 + 0.70 * (1.0 - np.exp(-3.5 * obs_t)) + np.array([0.025, -0.015, 0.012, -0.008, 0.006, 0.010, -0.010])
    ax.scatter(x0 + w * obs_t, y0 + h * obs_s, s=9, facecolor="white", edgecolor=color, lw=0.75, zorder=4)


def draw_transfer_panel(ax, panel):
    x, y, w, h = panel
    add_icon(ax, "fig01_icon_1_measured_response.png", (x + 0.22, x + 1.08, y + 0.25, y + 1.08), alpha=0.62)
    draw_drawdown_curve(ax, x + 0.36, y + 0.38, 0.80, 0.50, PALETTE["data"])
    tagged_text(ax, x + 0.22, y + 0.16, "drawdown response", name="transfer_drawdown_label", panel="transfer", fontsize=5.55, color=PALETTE["muted"])

    straight_arrow(ax, (x + 1.32, y + 0.68), (x + 1.72, y + 0.68), name="arrow_transfer_drawdown_to_paths")

    path_y = [y + 0.96, y + 0.78, y + 0.60, y + 0.42]
    path_labels = ["Theis", "Hantush-Jacob", "Lagging Darcy", "Lagging Darcy + leakage"]
    for i, (yy, label) in enumerate(zip(path_y, path_labels)):
        ls = "-" if i in (0, 2) else (0, (3, 2))
        ax.plot([x + 1.82, x + 2.38], [yy, yy], color=PALETTE["path"], lw=1.00, ls=ls, zorder=3)
        tagged_text(ax, x + 2.48, yy, label, name=f"transfer_path_{i + 1}", panel="transfer", fontsize=5.15)

    straight_arrow(ax, (x + 3.95, y + 0.68), (x + 4.28, y + 0.68), name="arrow_transfer_paths_to_output")

    rng = np.random.default_rng(4)
    centers = [y + 0.82, y + 0.63, y + 0.46]
    xs = [x + 4.54, x + 4.88, x + 5.22, x + 5.56]
    for j, yy in enumerate(centers):
        vals = np.array(xs) + rng.normal(0, 0.015, len(xs))
        ax.scatter(vals, np.full(len(xs), yy), s=15, color=[PALETTE["data"], PALETTE["path"], PALETTE["diag"], PALETTE["bench"]], zorder=4)
    tagged_text(ax, x + 4.46, y + 1.03, r"apparent $T/S$", name="transfer_output_title", panel="transfer", fontsize=6.05, weight="bold")
    tagged_text(ax, x + 4.46, y + 0.24, "model-dependent spread", name="transfer_output_note", panel="transfer", fontsize=5.50, color=PALETTE["muted"])


def draw_benchmark_panel(ax, panel):
    x, y, w, h = panel
    add_icon(ax, "fig01_icon_4_benchmark.png", (x + 0.18, x + 1.02, y + 0.22, y + 1.04), alpha=0.88)
    tagged_text(ax, x + 1.18, y + 0.86, "10,000 cases", name="benchmark_cases", panel="benchmark", fontsize=6.10, weight="bold")
    tagged_text(ax, x + 1.18, y + 0.60, "analytical-limit checks", name="benchmark_checks", panel="benchmark", fontsize=5.65, color=PALETTE["muted"])
    tagged_text(ax, x + 1.18, y + 0.38, r"conditional $M$ and COV", name="benchmark_factor", panel="benchmark", fontsize=5.65, color=PALETTE["muted"])


def draw_field_panel(ax, panel):
    x, y, w, h = panel
    add_icon(ax, "fig01_icon_5_screening.png", (x + 0.18, x + 0.92, y + 0.24, y + 1.00), alpha=0.86)
    checks = ["support", "response", "scope"]
    for i, label in enumerate(checks):
        yy = y + 0.90 - i * 0.23
        ax.scatter([x + 1.08], [yy], s=12, marker="s", facecolor="white", edgecolor=PALETTE["field"], lw=0.75, zorder=3)
        ax.plot([x + 1.055, x + 1.08, x + 1.125], [yy, yy - 0.030, yy + 0.042], color=PALETTE["field"], lw=0.75, zorder=4)
        tagged_text(ax, x + 1.25, yy, label, name=f"field_check_{i + 1}", panel="field", fontsize=5.80)


def draw_reporting_panel(ax, panel):
    x, y, w, h = panel
    add_icon(ax, "fig01_icon_6_reporting.png", (x + 0.18, x + 0.92, y + 0.24, y + 1.00), alpha=0.86)
    labels = [r"pathway $T/S$", "capacity", "timing"]
    for i, label in enumerate(labels):
        yy = y + 0.88 - i * 0.24
        ax.scatter([x + 1.08], [yy], s=16, color=PALETTE["report"], zorder=4)
        tagged_text(ax, x + 1.25, yy, label, name=f"report_item_{i + 1}", panel="report", fontsize=5.85)


def draw_bottom_flow(ax, bench, field, report):
    bx, by, bw, bh = bench
    fx, fy, fw, fh = field
    rx, ry, rw, rh = report
    straight_arrow(ax, (bx + bw + 0.06, by + 0.34), (fx - 0.06, fy + 0.34), name="arrow_benchmark_to_field")
    straight_arrow(ax, (fx + fw + 0.06, fy + 0.34), (rx - 0.06, ry + 0.34), name="arrow_field_to_report")


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.35, 3.95))
    xlim = (0.0, 12.4)
    ylim = (0.0, 4.85)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    grammar = add_panel(ax, "grammar", "a", "Transformation\ngrammar", (0.35, 0.45), (3.05, 3.95), PALETTE["data"])
    transfer = add_panel(ax, "transfer", "b", "Drawdown-to-parameter transfer", (3.80, 2.45), (8.20, 1.95), PALETTE["path"])
    benchmark = add_panel(ax, "benchmark", "c", "Benchmark-conditioned\nmodel factors", (3.80, 0.45), (3.05, 1.55), PALETTE["bench"])
    field = add_panel(ax, "field", "d", "Field applicability\nscreening", (7.10, 0.45), (2.35, 1.55), PALETTE["field"])
    report = add_panel(ax, "report", "e", "Groundwater\nreporting", (9.70, 0.45), (2.30, 1.55), PALETTE["report"])

    draw_left_grammar(ax, grammar, PALETTE["data"])
    draw_transfer_panel(ax, transfer)
    draw_benchmark_panel(ax, benchmark)
    draw_field_panel(ax, field)
    draw_reporting_panel(ax, report)

    straight_arrow(ax, (3.45, 3.40), (3.76, 3.40), name="arrow_grammar_to_transfer")
    straight_arrow(ax, (5.35, 2.43), (5.35, 2.05), name="arrow_transfer_to_benchmark", allow_arrow_overlap=True)
    straight_arrow(ax, (8.28, 2.43), (8.28, 2.05), name="arrow_transfer_to_field", allow_arrow_overlap=True)
    draw_bottom_flow(ax, benchmark, field, report)

    report_data = run_layout_qa(
        fig,
        figure_name="fig01_pathway_framework",
        panel_margin_px=2.0,
        canvas_margin_px=2.0,
        min_overlap_px=2.0,
    )
    write_layout_qa_reports(report_data, QA_JSON, QA_MD)
    assert_layout_pass(report_data)

    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
