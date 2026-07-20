"""Run the pretrained FILM model over every extracted triplet and report
PSNR/SSIM against the real held-out middle frame.

This uses the checkpoint exactly as downloaded, with zero satellite-
specific training -- the goal is to check whether a model pretrained on
ordinary video transfers at all to satellite cloud imagery before
investing in fine-tuning it.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from src.deep.film_interpolate import interpolate_middle_frame
from src.eval.metrics import lpips_distance, psnr, ssim, summarize


def evaluate(triplets_dir: Path, model_path: Path, out_csv: Path, device: str | None = None) -> list[dict]:
    rows = []
    for triplet_dir in sorted(triplets_dir.iterdir()):
        if not triplet_dir.is_dir():
            continue
        frame_prev = cv2.imread(str(triplet_dir / "t-1.png"), cv2.IMREAD_GRAYSCALE)
        frame_mid = cv2.imread(str(triplet_dir / "t.png"), cv2.IMREAD_GRAYSCALE)
        frame_next = cv2.imread(str(triplet_dir / "t+1.png"), cv2.IMREAD_GRAYSCALE)
        if frame_prev is None or frame_mid is None or frame_next is None:
            continue

        pred = interpolate_middle_frame(frame_prev, frame_next, model_path, device=device)
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
    parser.add_argument("--model-path", default="models/film_net_fp32.pt")
    parser.add_argument("--out-csv", default="data/processed/film_results.csv")
    parser.add_argument("--device", default=None, help="cuda / cpu -- auto-detects if omitted")
    args = parser.parse_args()

    rows = evaluate(Path(args.triplets_dir), Path(args.model_path), Path(args.out_csv), device=args.device)
    if rows:
        avg_psnr, avg_ssim, avg_lpips = summarize(rows)
        print(f"Evaluated {len(rows)} triplets")
        print(f"Mean PSNR: {avg_psnr:.2f} dB")
        print(f"Mean SSIM: {avg_ssim:.4f}")
        print(f"Mean LPIPS: {avg_lpips:.4f}")
    else:
        print("No triplets found.")
