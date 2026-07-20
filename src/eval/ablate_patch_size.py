"""Patch-size ablation: re-extract triplets from the same raw scans at
several sizes and evaluate Farneback + FILM at each -- per docs/PLAN.md's
Week 9-10 ablations. Answers whether interpolation quality depends on how
much spatial context a patch gives the model, or holds steady regardless.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.extract_triplets import build_triplets, default_center, load_radiance
from src.eval.evaluate_baseline import evaluate as evaluate_baseline
from src.eval.evaluate_film import evaluate as evaluate_film
from src.eval.metrics import summarize


def run_patch_size_ablation(
    raw_dir: Path,
    sizes: list[int],
    film_model_path: Path,
    out_dir: Path,
    device: str | None = None,
) -> dict[int, dict[str, tuple[float, float, float]]]:
    nc_paths = sorted(raw_dir.glob("*.nc"))
    if not nc_paths:
        raise SystemExit(f"No .nc files found in {raw_dir} -- run fetch_goes.py first")

    center = default_center(load_radiance(nc_paths[0]).shape)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, tuple[float, float, float]]] = {}
    for size in sizes:
        triplets_dir = out_dir / f"triplets_{size}"
        build_triplets(nc_paths, center, size, triplets_dir)

        farneback_rows = evaluate_baseline(triplets_dir, out_dir / f"farneback_{size}.csv")
        film_rows = evaluate_film(triplets_dir, film_model_path, out_dir / f"film_{size}.csv", device=device)

        results[size] = {
            "farneback": summarize(farneback_rows),
            "film": summarize(film_rows),
        }

    print(f"\n{'size':>6s} {'method':12s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s}")
    print("-" * 47)
    for size, methods in results.items():
        for method_name, (mean_psnr, mean_ssim, mean_lpips) in methods.items():
            print(f"{size:6d} {method_name:12s} {mean_psnr:8.2f} {mean_ssim:8.4f} {mean_lpips:8.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--film-model-path", default="models/film_net_fp32.pt")
    parser.add_argument("--out-dir", default="data/processed/ablation_patch_size")
    parser.add_argument("--device", default=None, help="cuda / cpu -- auto-detects if omitted")
    args = parser.parse_args()

    run_patch_size_ablation(
        Path(args.raw_dir),
        args.sizes,
        Path(args.film_model_path),
        Path(args.out_dir),
        device=args.device,
    )
