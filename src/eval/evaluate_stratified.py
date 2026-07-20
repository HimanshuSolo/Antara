"""Run Farneback + FILM over both a cyclone-centered triplet set and a
calm-weather triplet set, and report the stratified comparison -- this is
the project's headline result: classical optical-flow interpolation is
expected to degrade sharply on fast, non-linear cyclone motion while a
(fine-tuned) deep model holds up, which a single aggregate PSNR/SSIM
number across mixed conditions would hide entirely.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.evaluate_baseline import evaluate as evaluate_baseline
from src.eval.evaluate_film import evaluate as evaluate_film


def summarize(rows: list[dict]) -> tuple[float, float]:
    if not rows:
        return float("nan"), float("nan")
    mean_psnr = sum(r["psnr"] for r in rows) / len(rows)
    mean_ssim = sum(r["ssim"] for r in rows) / len(rows)
    return mean_psnr, mean_ssim


def run_stratified(
    calm_dir: Path,
    cyclone_dir: Path,
    film_model_path: Path,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, tuple[float, float]]] = {}

    for subset_name, triplets_dir in [("calm", calm_dir), ("cyclone", cyclone_dir)]:
        if not triplets_dir.exists() or not any(triplets_dir.iterdir()):
            print(f"Skipping '{subset_name}' -- {triplets_dir} is empty/missing")
            continue

        farneback_rows = evaluate_baseline(triplets_dir, out_dir / f"{subset_name}_farneback.csv")
        film_rows = evaluate_film(triplets_dir, film_model_path, out_dir / f"{subset_name}_film.csv")

        results[subset_name] = {
            "farneback": summarize(farneback_rows),
            "film": summarize(film_rows),
        }

    print(f"\n{'subset':10s} {'method':12s} {'PSNR':>8s} {'SSIM':>8s}")
    print("-" * 42)
    for subset_name, methods in results.items():
        for method_name, (mean_psnr, mean_ssim) in methods.items():
            print(f"{subset_name:10s} {method_name:12s} {mean_psnr:8.2f} {mean_ssim:8.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calm-dir", default="data/processed/triplets_calm")
    parser.add_argument("--cyclone-dir", default="data/processed/triplets_cyclone")
    parser.add_argument("--film-model-path", default="models/film_net_fp32.pt")
    parser.add_argument("--out-dir", default="data/processed/stratified")
    args = parser.parse_args()

    run_stratified(
        Path(args.calm_dir),
        Path(args.cyclone_dir),
        Path(args.film_model_path),
        Path(args.out_dir),
    )
