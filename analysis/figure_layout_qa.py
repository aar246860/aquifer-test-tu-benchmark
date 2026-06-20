"""Reusable layout QA for publication figures generated with Matplotlib.

The checks are intentionally deterministic: artists are tagged with lightweight
metadata, the rendered canvas is inspected through display-space bounding boxes,
and JSON/Markdown reports are written for manuscript QA records.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox


TEXT_ROLES = {
    "title",
    "subtitle",
    "panel_number",
    "panel_title",
    "major_text",
    "minor_text",
    "label",
    "legend_text",
    "footer_text",
}


def tag_artist(
    artist: Artist,
    *,
    name: str,
    role: str,
    panel: str | None = None,
    ignore: bool = False,
    obstacle: bool = False,
    check_panel_bounds: bool = True,
    allow_text_overlap: bool = False,
    allow_arrow_overlap: bool = False,
    arrow_start: tuple[float, float] | None = None,
    arrow_end: tuple[float, float] | None = None,
) -> Artist:
    """Attach QA metadata to a Matplotlib artist and return the artist."""

    meta = {
        "name": name,
        "role": role,
        "panel": panel,
        "ignore": ignore,
        "obstacle": obstacle,
        "check_panel_bounds": check_panel_bounds,
        "allow_text_overlap": allow_text_overlap,
        "allow_arrow_overlap": allow_arrow_overlap,
        "arrow_start": arrow_start,
        "arrow_end": arrow_end,
    }
    setattr(artist, "_figure_layout_qa", meta)
    if hasattr(artist, "set_gid"):
        artist.set_gid(f"qa:{role}:{name}")
    return artist


@dataclass
class QaObject:
    name: str
    role: str
    panel: str | None
    bbox: Bbox
    artist_type: str
    meta: dict[str, Any]
    arrow_start_px: tuple[float, float] | None = None
    arrow_end_px: tuple[float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "panel": self.panel,
            "artist_type": self.artist_type,
            "bbox_px": {
                "x0": round(float(self.bbox.x0), 3),
                "y0": round(float(self.bbox.y0), 3),
                "x1": round(float(self.bbox.x1), 3),
                "y1": round(float(self.bbox.y1), 3),
                "width": round(float(self.bbox.width), 3),
                "height": round(float(self.bbox.height), 3),
            },
            "arrow_start_px": self.arrow_start_px,
            "arrow_end_px": self.arrow_end_px,
        }


def _qa_meta(artist: Artist) -> dict[str, Any] | None:
    meta = getattr(artist, "_figure_layout_qa", None)
    if isinstance(meta, dict):
        return meta
    return None


def _finite_bbox(bbox: Bbox | None) -> Bbox | None:
    if bbox is None:
        return None
    values = [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
    if not all(math.isfinite(float(v)) for v in values):
        return None
    if bbox.width <= 0 or bbox.height <= 0:
        return None
    return bbox


def _artist_bbox(artist: Artist, renderer: Any) -> Bbox | None:
    meta = _qa_meta(artist) or {}
    if meta.get("arrow_start") is not None and meta.get("arrow_end") is not None:
        ax = getattr(artist, "axes", None)
        if ax is None:
            return None
        p0 = ax.transData.transform(meta["arrow_start"])
        p1 = ax.transData.transform(meta["arrow_end"])
        return Bbox.from_extents(p0[0], p0[1], p1[0], p1[1]).padded(4)

    try:
        if isinstance(artist, Text):
            if not artist.get_text().strip():
                return None
            return _finite_bbox(artist.get_window_extent(renderer))
        if isinstance(artist, Patch):
            path = artist.get_path()
            transform = artist.get_transform()
            return _finite_bbox(path.get_extents(transform))
        if isinstance(artist, Line2D):
            path = artist.get_path()
            if len(path.vertices) == 0:
                return None
            return _finite_bbox(path.get_extents(artist.get_transform()).padded(2))
        if isinstance(artist, PathCollection):
            return _finite_bbox(artist.get_window_extent(renderer).padded(2))
        return _finite_bbox(artist.get_window_extent(renderer))
    except Exception:
        return None


def collect_qa_objects(fig) -> list[QaObject]:
    """Collect visible tagged artists from a Matplotlib figure."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    objects: list[QaObject] = []
    for artist in fig.findobj():
        meta = _qa_meta(artist)
        if not meta or meta.get("ignore"):
            continue
        try:
            if not artist.get_visible():
                continue
        except Exception:
            pass
        bbox = _artist_bbox(artist, renderer)
        if bbox is None:
            continue
        arrow_start_px = None
        arrow_end_px = None
        if meta.get("arrow_start") is not None and meta.get("arrow_end") is not None:
            ax = getattr(artist, "axes", None)
            if ax is not None:
                p0 = ax.transData.transform(meta["arrow_start"])
                p1 = ax.transData.transform(meta["arrow_end"])
                arrow_start_px = (round(float(p0[0]), 3), round(float(p0[1]), 3))
                arrow_end_px = (round(float(p1[0]), 3), round(float(p1[1]), 3))
        objects.append(
            QaObject(
                name=str(meta.get("name", artist.get_gid() or artist.__class__.__name__)),
                role=str(meta.get("role", "unknown")),
                panel=meta.get("panel"),
                bbox=bbox,
                artist_type=artist.__class__.__name__,
                meta=meta,
                arrow_start_px=arrow_start_px,
                arrow_end_px=arrow_end_px,
            )
        )
    return objects


def _intersects(a: Bbox, b: Bbox, *, min_overlap_px: float = 2.0) -> bool:
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx > min_overlap_px and dy > min_overlap_px


def _intersection_area(a: Bbox, b: Bbox) -> float:
    dx = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    dy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return dx * dy


def _contains(outer: Bbox, inner: Bbox, *, margin_px: float = 0.0) -> bool:
    return (
        inner.x0 >= outer.x0 + margin_px
        and inner.y0 >= outer.y0 + margin_px
        and inner.x1 <= outer.x1 - margin_px
        and inner.y1 <= outer.y1 - margin_px
    )


def _ccw(a, b, c) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d) -> bool:
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def _point_in_bbox(p: tuple[float, float], bbox: Bbox) -> bool:
    return bbox.x0 <= p[0] <= bbox.x1 and bbox.y0 <= p[1] <= bbox.y1


def _segment_intersects_bbox(p0: tuple[float, float], p1: tuple[float, float], bbox: Bbox) -> bool:
    if _point_in_bbox(p0, bbox) or _point_in_bbox(p1, bbox):
        return True
    corners = [(bbox.x0, bbox.y0), (bbox.x1, bbox.y0), (bbox.x1, bbox.y1), (bbox.x0, bbox.y1)]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return any(_segments_intersect(p0, p1, c, d) for c, d in edges)


def run_layout_qa(
    fig,
    *,
    figure_name: str,
    panel_margin_px: float = 3.0,
    canvas_margin_px: float = 2.0,
    min_overlap_px: float = 2.0,
    check_arrows: bool = True,
) -> dict[str, Any]:
    """Run layout QA and return a machine-readable report."""

    objects = collect_qa_objects(fig)
    panels = {obj.name: obj for obj in objects if obj.role == "panel"}
    text_objects = [obj for obj in objects if obj.role in TEXT_ROLES]
    obstacles = [obj for obj in objects if obj.meta.get("obstacle") and obj.role != "panel"]
    arrows = [obj for obj in objects if obj.role == "arrow" and obj.arrow_start_px and obj.arrow_end_px]
    fig_bbox = fig.bbox
    issues: list[dict[str, Any]] = []

    for obj in objects:
        if not _contains(fig_bbox, obj.bbox, margin_px=canvas_margin_px):
            issues.append(
                {
                    "severity": "major",
                    "type": "outside_canvas",
                    "object": obj.name,
                    "role": obj.role,
                    "message": f"{obj.name} extends outside the figure canvas or violates the {canvas_margin_px}px canvas margin.",
                }
            )

    for text in text_objects:
        if text.panel and text.meta.get("check_panel_bounds", True):
            panel = panels.get(f"panel:{text.panel}") or panels.get(text.panel)
            if panel and not _contains(panel.bbox, text.bbox, margin_px=panel_margin_px):
                issues.append(
                    {
                        "severity": "major",
                        "type": "text_outside_panel",
                        "object": text.name,
                        "panel": text.panel,
                        "message": f"{text.name} violates the {panel_margin_px}px internal margin of panel {text.panel}.",
                    }
                )

    for i, a in enumerate(text_objects):
        if a.meta.get("allow_text_overlap"):
            continue
        for b in text_objects[i + 1 :]:
            if b.meta.get("allow_text_overlap"):
                continue
            if _intersects(a.bbox, b.bbox, min_overlap_px=min_overlap_px):
                issues.append(
                    {
                        "severity": "major",
                        "type": "text_text_overlap",
                        "objects": [a.name, b.name],
                        "overlap_area_px2": round(_intersection_area(a.bbox, b.bbox), 3),
                        "message": f"{a.name} overlaps {b.name}.",
                    }
                )

    for text in text_objects:
        for obstacle in obstacles:
            if text.panel == obstacle.panel and text.name.startswith("icon_"):
                continue
            if _intersects(text.bbox, obstacle.bbox, min_overlap_px=min_overlap_px):
                issues.append(
                    {
                        "severity": "major",
                        "type": "text_obstacle_overlap",
                        "objects": [text.name, obstacle.name],
                        "overlap_area_px2": round(_intersection_area(text.bbox, obstacle.bbox), 3),
                        "message": f"{text.name} overlaps obstacle {obstacle.name}.",
                    }
                )

    if check_arrows:
        for arrow in arrows:
            p0 = arrow.arrow_start_px
            p1 = arrow.arrow_end_px
            if not p0 or not p1:
                continue
            for text in text_objects:
                if text.meta.get("allow_arrow_overlap") or arrow.meta.get("allow_arrow_overlap"):
                    continue
                if _segment_intersects_bbox(p0, p1, text.bbox.padded(2)):
                    issues.append(
                        {
                            "severity": "major",
                            "type": "arrow_text_intersection",
                            "objects": [arrow.name, text.name],
                            "message": f"{arrow.name} crosses or touches text label {text.name}.",
                        }
                    )

    major_count = sum(1 for issue in issues if issue["severity"] == "major")
    return {
        "figure": figure_name,
        "status": "pass" if major_count == 0 else "fail",
        "summary": {
            "objects_checked": len(objects),
            "panels": len(panels),
            "text_objects": len(text_objects),
            "obstacles": len(obstacles),
            "arrows": len(arrows),
            "major_issues": major_count,
            "total_issues": len(issues),
        },
        "thresholds": {
            "panel_margin_px": panel_margin_px,
            "canvas_margin_px": canvas_margin_px,
            "min_overlap_px": min_overlap_px,
        },
        "issues": issues,
        "objects": [obj.as_dict() for obj in objects],
    }


def write_layout_qa_reports(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Figure Layout QA: {report['figure']}",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Objects checked: {report['summary']['objects_checked']}",
        f"- Text objects: {report['summary']['text_objects']}",
        f"- Panels: {report['summary']['panels']}",
        f"- Arrows: {report['summary']['arrows']}",
        f"- Major issues: {report['summary']['major_issues']}",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in report["thresholds"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Issues", ""])
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(f"- **{issue['severity']} / {issue['type']}**: {issue['message']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Notes", "", "- QA uses display-space bounding boxes after canvas rendering."])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_layout_pass(report: dict[str, Any]) -> None:
    if report["status"] != "pass":
        raise SystemExit(f"Figure layout QA failed with {report['summary']['major_issues']} major issue(s).")


def _self_test() -> int:
    import matplotlib.pyplot as plt
    from matplotlib import patches

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel = tag_artist(
        ax.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False)),
        name="panel:test",
        role="panel",
        panel="test",
    )
    _ = panel
    tag_artist(
        ax.text(0.22, 0.58, "A label", fontsize=10, ha="left", va="center"),
        name="label_a",
        role="major_text",
        panel="test",
    )
    tag_artist(
        ax.text(0.55, 0.30, "B label", fontsize=10, ha="left", va="center"),
        name="label_b",
        role="major_text",
        panel="test",
    )
    report = run_layout_qa(fig, figure_name="self_test")
    plt.close(fig)
    return 0 if report["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a self-test for the figure layout QA utilities.")
    parser.add_argument("--self-test", action="store_true", help="Run a small deterministic QA self-test.")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
