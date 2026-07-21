"""Run the Farneback baseline over every extracted triplet and report
PSNR/SSIM against the real held-out middle frame.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.baseline.farneback_interpolate import interpolate_middle_frame
from src.eval.metrics import lpips_distance, psnr, ssim, summarize
from src.eval.plot_comparison import list_triplet_dirs
from src.utils.image import load_triplet_frames


def evaluate(triplets_dir: Path, out_csv: Path) -> list[dict]:
    rows = []
    for triplet_dir in list_triplet_dirs(triplets_dir):
        frames = load_triplet_frames(triplet_dir)
        if frames is None:
            continue
        frame_prev, frame_mid, frame_next = frames

        pred = interpolate_middle_frame(frame_prev, frame_next)
        rows.append({
            "triplet": triplet_dir.name,
            "psnr": psnr(pred, frame_mid),
            "ssim": ssim(pred, frame_mid),
            "lpips": lpips_distance(pred, frame_mid),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["triplet", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-dir", default="data/processed/triplets")
    parser.add_argument("--out-csv", default="data/processed/baseline_results.csv")
    args = parser.parse_args()

    rows = evaluate(Path(args.triplets_dir), Path(args.out_csv))
    if rows:
        avg_psnr, avg_ssim, avg_lpips = summarize(rows)
        print(f"Evaluated {len(rows)} triplets")
        print(f"Mean PSNR: {avg_psnr:.2f} dB")
        print(f"Mean SSIM: {avg_ssim:.4f}")
        print(f"Mean LPIPS: {avg_lpips:.4f}")
    else:
        print("No triplets found.")
