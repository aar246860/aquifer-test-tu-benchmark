"""Check that Figure 1 is embedded correctly in the compiled manuscript PDF."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = ROOT / "main.tex"
MAIN_LOG = ROOT / "main.log"
MAIN_PDF = ROOT / "main.pdf"
OUT_DIR = ROOT / "outputs"
FIG_NAME = "fig01_pathway_framework.pdf"
QA_JSON = OUT_DIR / "fig01_pdf_embed_qa.json"
QA_MD = OUT_DIR / "fig01_pdf_embed_qa.md"


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def compile_main() -> dict[str, Any]:
    result = run_command(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", MAIN_TEX.name],
        ROOT,
    )
    return {
        "command": " ".join(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", MAIN_TEX.name]),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


def find_figure_page() -> int | None:
    if not MAIN_LOG.exists():
        return None
    text = MAIN_LOG.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"\[(\d+)\s+<\./" + re.escape(FIG_NAME) + r">", text)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"\[(\d+)\s+<[^>\]]*" + re.escape(FIG_NAME) + r">", text)
    if matches:
        return int(matches[-1])
    return None


def extract_page_png(page: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUT_DIR / f"main_page{page}_fig1_qa"
    for old in OUT_DIR.glob(f"{prefix.name}*.png"):
        old.unlink()
    result = run_command(
        ["pdftoppm", "-f", str(page), "-l", str(page), "-r", "220", "-png", str(MAIN_PDF), str(prefix)],
        ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "pdftoppm failed")
    candidates = sorted(OUT_DIR.glob(f"{prefix.name}*.png"))
    if not candidates:
        raise RuntimeError("pdftoppm did not create a PNG page image")
    canonical = OUT_DIR / f"main_page{page}_fig1_qa.png"
    if candidates[0] != canonical:
        shutil.copyfile(candidates[0], canonical)
    return canonical


def page_text(page: int) -> str:
    result = run_command(["pdftotext", "-f", str(page), "-l", str(page), str(MAIN_PDF), "-"], ROOT)
    if result.returncode != 0:
        return ""
    return result.stdout or ""


def colored_bbox_metrics(png_path: Path) -> dict[str, Any]:
    image = Image.open(png_path).convert("RGB")
    arr = np.asarray(image).astype(np.int16)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    mean = arr.mean(axis=2)
    mask = ((maxc - minc) > 18) & (mean < 252)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "image_width_px": image.width,
            "image_height_px": image.height,
            "colored_pixels": 0,
            "bbox": None,
            "width_ratio": 0.0,
            "height_ratio": 0.0,
            "area_ratio": 0.0,
        }
    bbox = {
        "x0": int(xs.min()),
        "y0": int(ys.min()),
        "x1": int(xs.max()),
        "y1": int(ys.max()),
    }
    bbox["width"] = bbox["x1"] - bbox["x0"] + 1
    bbox["height"] = bbox["y1"] - bbox["y0"] + 1
    return {
        "image_width_px": image.width,
        "image_height_px": image.height,
        "colored_pixels": int(mask.sum()),
        "bbox": bbox,
        "width_ratio": round(bbox["width"] / image.width, 4),
        "height_ratio": round(bbox["height"] / image.height, 4),
        "area_ratio": round((bbox["width"] * bbox["height"]) / (image.width * image.height), 4),
    }


def write_reports(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Figure 1 PDF Embedding QA",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Main PDF: `{MAIN_PDF}`",
        f"- Figure page: {report.get('figure_page')}",
        f"- Page PNG: `{report.get('page_png')}`",
        f"- Caption found: {report['checks']['caption_found']}",
        f"- Colored bbox width ratio: {report['metrics']['width_ratio']}",
        f"- Colored bbox height ratio: {report['metrics']['height_ratio']}",
        f"- Colored bbox area ratio: {report['metrics']['area_ratio']}",
        "",
        "## Issues",
        "",
    ]
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("- None.")
    QA_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pdf_embed_qa(skip_compile: bool = False) -> dict[str, Any]:
    issues: list[str] = []
    compile_result = None
    if not skip_compile:
        compile_result = compile_main()
        if compile_result["returncode"] != 0:
            issues.append("pdflatex failed before PDF embedding QA.")

    page = find_figure_page()
    if page is None:
        issues.append(f"{FIG_NAME} page was not found in main.log.")

    page_png = None
    metrics = {
        "image_width_px": 0,
        "image_height_px": 0,
        "colored_pixels": 0,
        "bbox": None,
        "width_ratio": 0.0,
        "height_ratio": 0.0,
        "area_ratio": 0.0,
    }
    text = ""
    if page is not None and MAIN_PDF.exists() and not any("pdflatex failed" in issue for issue in issues):
        page_png = extract_page_png(page)
        metrics = colored_bbox_metrics(page_png)
        text = page_text(page)

    caption_found = "Pathway-indexed transformation uncertainty framework" in text
    if not caption_found:
        issues.append("Expected Figure 1 caption phrase was not found on the extracted page.")
    if metrics["colored_pixels"] < 4000:
        issues.append("Extracted page contains too few colored pixels for the embedded figure.")
    if metrics["width_ratio"] < 0.45 or metrics["height_ratio"] < 0.16:
        issues.append("Colored figure footprint is smaller than expected for a full-width Figure 1.")

    report = {
        "status": "pass" if not issues else "fail",
        "figure": FIG_NAME,
        "figure_page": page,
        "page_png": str(page_png) if page_png else None,
        "checks": {
            "compiled": None if compile_result is None else compile_result["returncode"] == 0,
            "caption_found": caption_found,
            "main_pdf_exists": MAIN_PDF.exists(),
        },
        "metrics": metrics,
        "issues": issues,
        "compile": compile_result,
    }
    write_reports(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="QA Figure 1 embedding in the compiled main PDF.")
    parser.add_argument("--skip-compile", action="store_true", help="Use the existing main.pdf and main.log.")
    args = parser.parse_args()
    report = run_pdf_embed_qa(skip_compile=args.skip_compile)
    if report["status"] != "pass":
        print(f"Figure 1 PDF embedding QA failed with {len(report['issues'])} issue(s).")
        for issue in report["issues"]:
            print(f"- {issue}")
        return 1
    print("Figure 1 PDF embedding QA passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
