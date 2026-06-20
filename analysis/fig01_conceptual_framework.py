from __future__ import annotations

from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "figures"


def cmc_color(name: str, sample: float) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in getattr(cmc, name)(sample))


DATA = cmc_color("batlow", 0.12)
PATHWAY = cmc_color("batlow", 0.38)
DIAGNOSTIC = cmc_color("batlow", 0.66)
REPORT = cmc_color("batlow", 0.90)
TEXT = cmc_color("grayC", 0.08)
MID = cmc_color("grayC", 0.38)
LIGHT = cmc_color("grayC", 0.93)
BACKGROUND = cmc_color("grayC", 0.985)


def add_panel(ax: plt.Axes, x: float, y: float, w: float, h: float, color, title: str, number: str) -> None:
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=0.9,
            edgecolor=color,
            facecolor=(*color[:3], 0.045),
            zorder=1,
        )
    )
    ax.text(x + 1.8, y + h - 3.1, f"{number}.", ha="left", va="top", color=TEXT, fontsize=8.8, fontweight="bold")
    ax.text(
        x + w / 2 + 1.25,
        y + h - 2.9,
        title,
        ha="center",
        va="top",
        color=TEXT,
        fontsize=9.3,
        fontweight="bold",
        linespacing=1.25,
    )
    ax.plot([x + 1.6, x + w - 1.6], [y + h - 8.2, y + h - 8.2], color=color, lw=0.75)


def draw_arrow(ax: plt.Axes, x1: float, y: float, x2: float, color=TEXT) -> None:
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": color, "mutation_scale": 12, "shrinkA": 0, "shrinkB": 0},
        zorder=5,
    )


def draw_well(ax: plt.Axes, x: float, y: float, h: float, color=TEXT) -> None:
    ax.add_patch(patches.Rectangle((x - 0.45, y), 0.9, h, fill=False, edgecolor=color, linewidth=0.8))
    for yy in np.linspace(y + 0.5, y + h - 0.5, 8):
        ax.plot([x - 0.42, x + 0.42], [yy, yy], color=MID, lw=0.45)


def draw_response_panel(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.text(x + 5.2, y + h - 11.0, "Pumping\nwell", ha="center", va="top", fontsize=7.0, color=TEXT, linespacing=1.2)
    ax.text(x + w - 6.1, y + h - 11.0, "Observation\nwell", ha="center", va="top", fontsize=7.0, color=TEXT, linespacing=1.2)
    draw_well(ax, x + 5.2, y + h - 22.5, 6.1)
    draw_well(ax, x + w - 6.1, y + h - 20.8, 6.8)
    ax.annotate(
        "",
        xy=(x + 5.2, y + h - 15.7),
        xytext=(x + 5.2, y + h - 12.9),
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": DATA, "mutation_scale": 9},
    )
    xs = np.linspace(x + 1.7, x + w - 5.0, 80)
    ys = y + h - 18.6 + 0.7 * np.sin(np.linspace(0, 2.5 * np.pi, 80))
    ax.plot(xs, ys, color=DATA, lw=0.9, ls=(0, (4, 3)), alpha=0.85)

    px, py = x + 3.5, y + 18.0
    ax.arrow(px, py, 0, 9.0, head_width=0.6, head_length=0.8, lw=0.7, color=TEXT, length_includes_head=True)
    ax.arrow(px, py, 15.0, 0, head_width=0.6, head_length=0.8, lw=0.7, color=TEXT, length_includes_head=True)
    curve_x = np.linspace(px, px + 14.0, 80)
    curve_y = py + 8.1 * (1.0 - np.exp(-np.linspace(0, 3.2, 80)))
    ax.plot(curve_x, curve_y, color=DATA, lw=1.2)
    ax.text(px - 1.8, py + 5.0, "Drawdown, s", rotation=90, ha="center", va="center", fontsize=7.2, color=TEXT)
    ax.text(px + 7.4, py - 1.4, "Time, t", ha="center", va="top", fontsize=7.2, color=TEXT)
    ax.plot([x + 1.8, x + w - 1.8], [y + 14.5, y + 14.5], color=DATA, lw=0.65)
    bullets = ["Observed drawdown s(t)", "Pumping schedule", "Well geometry", "Observation window"]
    for i, item in enumerate(bullets):
        yy = y + 11.4 - i * 2.6
        ax.scatter([x + 2.2], [yy], s=9, color=DATA, zorder=4)
        ax.text(x + 4.1, yy, item, ha="left", va="center", fontsize=7.2, color=TEXT)


def draw_pathway_icon(ax: plt.Axes, cx: float, cy: float, color, clock: bool = False, wave: bool = False) -> None:
    ax.add_patch(patches.Circle((cx, cy), 2.75, fill=False, edgecolor=color, linewidth=0.9))
    for offset in [-1.1, -0.35, 0.4, 1.15]:
        ax.plot([cx - 1.7, cx + 1.7], [cy + offset, cy + offset], color=color, lw=0.75)
    if wave:
        xs = np.linspace(cx - 1.7, cx + 1.7, 40)
        ys = cy - 1.65 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, 40))
        ax.plot(xs, ys, color=color, lw=0.75, ls=(0, (4, 3)))
    if clock:
        ax.add_patch(patches.Circle((cx + 1.5, cy - 1.55), 1.15, fill=False, edgecolor=color, linewidth=0.85))
        ax.plot([cx + 1.5, cx + 1.5], [cy - 1.55, cy - 0.78], color=color, lw=0.75)
        ax.plot([cx + 1.5, cx + 2.05], [cy - 1.55, cy - 2.05], color=color, lw=0.75)


def draw_pathways(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    labels = [
        ("Theis confined", False, False),
        ("Hantush--Jacob\nleaky", False, True),
        ("Lagging Darcy\nwithout leakage", True, False),
        ("Lagging Darcy\nwith leakage", True, True),
    ]
    start = y + h - 13.2
    gap = 8.6
    for i, (label, clock, wave) in enumerate(labels):
        cy = start - i * gap
        draw_pathway_icon(ax, x + 5.0, cy, PATHWAY if i < 2 else DIAGNOSTIC, clock=clock, wave=wave)
        ax.text(x + 9.5, cy, label, ha="left", va="center", fontsize=8.7, color=TEXT, linespacing=1.2)


def draw_scatter_icon(ax: plt.Axes, cx: float, cy: float, color) -> None:
    offsets = [(-1.2, 1.0), (-0.3, 1.35), (0.8, 1.0), (-1.1, 0.1), (0.0, 0.25), (1.1, 0.2), (-0.5, -0.7), (0.55, -0.9), (1.35, -0.5)]
    for dx, dy in offsets:
        ax.scatter([cx + dx], [cy + dy], s=8, color=color)


def draw_scale_icon(ax: plt.Axes, cx: float, cy: float, color) -> None:
    ax.plot([cx, cx], [cy - 1.7, cy + 1.5], color=color, lw=0.8)
    ax.plot([cx - 2.0, cx + 2.0], [cy + 0.9, cy + 0.9], color=color, lw=0.8)
    ax.plot([cx - 0.4, cx + 0.4], [cy - 1.7, cy - 1.7], color=color, lw=0.8)
    ax.plot([cx - 1.4, cx - 0.3], [cy + 0.9, cy - 0.5], color=color, lw=0.65)
    ax.plot([cx + 1.4, cx + 0.3], [cy + 0.9, cy - 0.5], color=color, lw=0.65)
    ax.add_patch(patches.Arc((cx - 1.4, cy - 0.45), 1.1, 0.7, theta1=180, theta2=360, edgecolor=color, lw=0.75))
    ax.add_patch(patches.Arc((cx + 1.4, cy - 0.45), 1.1, 0.7, theta1=180, theta2=360, edgecolor=color, lw=0.75))


def draw_wave_icon(ax: plt.Axes, cx: float, cy: float, color) -> None:
    xs = np.linspace(cx - 2.0, cx + 2.0, 80)
    for offset in [-0.9, 0.0, 0.9]:
        ys = cy + offset + 0.25 * np.sin(np.linspace(0, 2.2 * np.pi, 80))
        ax.plot(xs, ys, color=color, lw=0.8)


def draw_bell_icon(ax: plt.Axes, cx: float, cy: float, color) -> None:
    xs = np.linspace(-2.2, 2.2, 100)
    ys = 1.95 * np.exp(-0.5 * xs**2)
    ax.plot(cx + xs, cy - 1.2 + ys, color=color, lw=0.85)
    ax.plot([cx - 2.2, cx + 2.2], [cy - 1.2, cy - 1.2], color=color, lw=0.65)
    ax.plot([cx, cx], [cy - 1.2, cy + 1.35], color=color, lw=0.65, ls=(0, (4, 3)))


def draw_target_icon(ax: plt.Axes, cx: float, cy: float, color) -> None:
    ax.add_patch(patches.Circle((cx, cy), 1.9, fill=False, edgecolor=color, linewidth=0.8))
    ax.add_patch(patches.Circle((cx, cy), 0.85, fill=False, edgecolor=color, linewidth=0.75))
    ax.plot([cx - 2.4, cx + 2.4], [cy, cy], color=color, lw=0.65)
    ax.plot([cx, cx], [cy - 2.4, cy + 2.4], color=color, lw=0.65)


def draw_diagnostics(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    items = [
        ("Response\nresidual SD$_{\\epsilon}$", draw_scatter_icon),
        ("AIC/BIC\ncomplexity check", draw_scale_icon),
        ("Boundary/leakage\ndiagnostics", draw_wave_icon),
        ("Parameter COV", draw_bell_icon),
        ("Identifiability/\nspatial transfer", draw_target_icon),
    ]
    start = y + h - 12.8
    gap = 7.0
    for i, (label, icon_func) in enumerate(items):
        cy = start - i * gap
        icon_func(ax, x + 4.6, cy, DIAGNOSTIC)
        ax.text(x + 9.3, cy, label, ha="left", va="center", fontsize=8.2, color=TEXT, linespacing=1.15)


def draw_document_icon(ax: plt.Axes, x: float, y: float, color) -> None:
    ax.add_patch(patches.Rectangle((x, y), 7.8, 10.6, fill=False, edgecolor=color, linewidth=0.9))
    ax.plot([x + 5.7, x + 7.8, x + 5.7, x + 5.7], [y + 10.6, y + 8.5, y + 8.5, y + 10.6], color=color, lw=0.9)
    for i, hh in enumerate([2.6, 3.7, 5.0]):
        ax.add_patch(patches.Rectangle((x + 1.2 + i * 1.1, y + 6.0), 0.55, hh, facecolor=color, edgecolor=color, lw=0.4))
    for yy in [y + 4.6, y + 3.5, y + 2.4]:
        ax.plot([x + 1.2, x + 6.4], [yy, yy], color=color, lw=0.7)
    ax.plot([x + 4.5, x + 6.4], [y + 6.9, y + 6.9], color=color, lw=0.7)


def draw_reporting(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    draw_document_icon(ax, x + w / 2 - 3.9, y + h - 23.5, REPORT)
    bullets = [
        "Report S and T by\ninterpretation model",
        "Flag variability from\ninterpretation",
        "Avoid direct aquifer\ninference without\ndiagnostics",
    ]
    for i, item in enumerate(bullets):
        yy = y + 20.2 - i * 6.7
        ax.scatter([x + 3.0], [yy], s=10, color=REPORT, zorder=4)
        ax.text(x + 5.0, yy, item, ha="left", va="center", fontsize=8.0, color=TEXT, linespacing=1.18)


def make_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )
    fig, ax = plt.subplots(figsize=(7.55, 4.85))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.text(
        50,
        53.4,
        "Transformation uncertainty framework for pumping test interpretation",
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color=TEXT,
    )

    y, h = 3.6, 44.8
    panels = [
        (2.0, y, 23.0, h, DATA, "Measured\npumping test response", "1"),
        (28.0, y, 22.2, h, PATHWAY, "Four analytical\ninterpretations", "2"),
        (53.3, y, 22.7, h, DIAGNOSTIC, "Transformation\ndiagnostics", "3"),
        (78.8, y, 19.3, h, REPORT, "Engineering\nreporting", "4"),
    ]
    for args in panels:
        add_panel(ax, *args)

    draw_arrow(ax, 25.0, 25.8, 27.6)
    draw_arrow(ax, 50.2, 25.8, 52.9)
    draw_arrow(ax, 76.0, 25.8, 78.4)

    draw_response_panel(ax, 2.0, y, 23.0, h)
    draw_pathways(ax, 28.0, y, 22.2, h)
    draw_diagnostics(ax, 53.3, y, 22.7, h)
    draw_reporting(ax, 78.8, y, 19.3, h)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig = make_figure()
    stems = ["fig01_conceptual_GG", "fig01_pathway_framework"]
    for base in [PROJECT, FIG_DIR]:
        for stem in stems:
            for suffix in ["pdf", "svg", "png"]:
                kwargs = {"dpi": 450} if suffix == "png" else {}
                fig.savefig(base / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
