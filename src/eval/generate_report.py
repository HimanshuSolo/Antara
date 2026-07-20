"""Assemble the data-driven parts of the project's report -- the results
table and figure references -- from the CSVs and PNGs the other eval
scripts already produced, so the numbers in the report can never drift
from what the pipeline actually measured.

Narrative sections (Problem, Related Work, Method, Experiments,
Conclusion) are deliberately left as placeholders: unlike the results,
they aren't mechanically derivable from data, so this script doesn't
fabricate them.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from src.eval.evaluate_stratified import summarize

SUBSETS = ["calm", "cyclone"]
METHODS = ["farneback", "film"]


def _load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return [
            {
                "triplet": r["triplet"],
                "psnr": float(r["psnr"]),
                "ssim": float(r["ssim"]),
                "lpips": float(r["lpips"]),
            }
            for r in csv.DictReader(f)
        ]


def build_results_table(results_dir: Path) -> str:
    lines = ["| Subset | Method | PSNR (dB) | SSIM | LPIPS |", "|---|---|---|---|---|"]
    for subset in SUBSETS:
        for method in METHODS:
            rows = _load_rows(results_dir / f"{subset}_{method}.csv")
            if not rows:
                continue
            mean_psnr, mean_ssim, mean_lpips = summarize(rows)
            lines.append(f"| {subset} | {method} | {mean_psnr:.2f} | {mean_ssim:.4f} | {mean_lpips:.4f} |")
    return "\n".join(lines)


def build_report_markdown(results_dir: Path, qualitative_dir: Path, report_dir: Path) -> str:
    sections = [
        "# Antara: Satellite Temporal Super-Resolution via Optical-Flow-Based Frame Interpolation",
        "",
        "<!-- TODO: Problem statement -- see README.md and docs/PLAN.md for the framing already written. -->",
        "",
        "## Related Work",
        "",
        "<!-- TODO: video frame interpolation (Super-SloMo, DAIN, RIFE, FILM) + remote sensing temporal "
        "super-resolution. -->",
        "",
        "## Method",
        "",
        "<!-- TODO: summarize src/baseline/farneback_interpolate.py and "
        "src/deep/film_interpolate.py + finetune_film.py. -->",
        "",
        "## Experiments",
        "",
        "<!-- TODO: data pipeline, IBTrACS-based cyclone/calm split, fine-tuning setup. -->",
        "",
        "## Results",
        "",
        "### Calm vs. cyclone (headline result)",
        "",
        build_results_table(results_dir),
        "",
    ]

    comparison_png = results_dir / "comparison.png"
    if comparison_png.exists():
        rel = os.path.relpath(comparison_png, report_dir)
        sections += [f"![Calm vs. cyclone comparison]({rel})", ""]

    qualitative_pngs = sorted(qualitative_dir.glob("*.png")) if qualitative_dir.exists() else []
    if qualitative_pngs:
        sections += ["### Qualitative comparisons", ""]
        for png in qualitative_pngs:
            rel = os.path.relpath(png, report_dir)
            sections.append(f"![{png.stem}]({rel})")
        sections.append("")

    sections += ["## Conclusion", "", "<!-- TODO -->", ""]
    return "\n".join(sections)


def write_report(results_dir: Path, qualitative_dir: Path, out_path: Path) -> Path:
    markdown = build_report_markdown(results_dir, qualitative_dir, out_path.parent)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="data/processed/stratified")
    parser.add_argument("--qualitative-dir", default="data/processed/qualitative")
    parser.add_argument("--out", default="data/processed/report.md")
    args = parser.parse_args()

    out_path = write_report(Path(args.results_dir), Path(args.qualitative_dir), Path(args.out))
    print(f"Wrote {out_path}")
