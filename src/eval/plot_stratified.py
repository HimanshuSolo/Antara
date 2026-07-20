"""Bar chart of PSNR/SSIM, calm vs. cyclone subset, Farneback vs. FILM --
the project's headline plot per docs/PLAN.md: classical optical-flow
interpolation is expected to degrade sharply on cyclone motion while the
deep model holds up, and a single aggregate number across mixed conditions
would hide that story entirely. Reads the per-subset CSVs already written
by `evaluate_stratified.py` rather than re-running any model.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUBSETS = ["calm", "cyclone"]
METHODS = ["farneback", "film"]


def _read_csv_means(csv_path: Path) -> tuple[float, float]:
    if not csv_path.exists():
        return float("nan"), float("nan")
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return float("nan"), float("nan")
    mean_psnr = sum(float(r["psnr"]) for r in rows) / len(rows)
    mean_ssim = sum(float(r["ssim"]) for r in rows) / len(rows)
    return mean_psnr, mean_ssim


def load_stratified_means(results_dir: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """Reload the {subset}_{method}.csv files evaluate_stratified.py writes
    and average PSNR/SSIM per (subset, method) cell."""
    return {
        subset: {method: _read_csv_means(results_dir / f"{subset}_{method}.csv") for method in METHODS}
        for subset in SUBSETS
    }


def build_stratified_figure(means: dict[str, dict[str, tuple[float, float]]]) -> plt.Figure:
    fig, (ax_psnr, ax_ssim) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(SUBSETS))
    width = 0.35

    for i, method in enumerate(METHODS):
        offset = (i - 0.5) * width
        psnr_vals = [means[subset][method][0] for subset in SUBSETS]
        ssim_vals = [means[subset][method][1] for subset in SUBSETS]
        ax_psnr.bar(x + offset, psnr_vals, width, label=method)
        ax_ssim.bar(x + offset, ssim_vals, width, label=method)

    ax_psnr.set_title("PSNR (dB)")
    ax_ssim.set_title("SSIM")
    for ax in (ax_psnr, ax_ssim):
        ax.set_xticks(x)
        ax.set_xticklabels(SUBSETS)
        ax.legend()

    fig.suptitle("Calm vs. cyclone: Farneback vs. FILM")
    fig.tight_layout()
    return fig


def save_stratified_figure(results_dir: Path, out_path: Path) -> Path:
    means = load_stratified_means(results_dir)
    fig = build_stratified_figure(means)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="data/processed/stratified")
    parser.add_argument("--out", default="data/processed/stratified/comparison.png")
    args = parser.parse_args()

    out_path = save_stratified_figure(Path(args.results_dir), Path(args.out))
    print(f"Wrote {out_path}")
