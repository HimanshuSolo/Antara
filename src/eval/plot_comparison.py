"""Qualitative side-by-side comparison for a triplet: real t-1, the
Farneback prediction, the FILM prediction, real t (ground truth), and real
t+1 -- the qualitative complement to the aggregate PSNR/SSIM numbers,
useful for spotting exactly where classical interpolation breaks down
(blur/ghosting on fast-moving cyclone structure) that an averaged metric
can hide. Per docs/PLAN.md's deliverable: "Qualitative side-by-sides ...
especially on cyclone frames".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baseline.farneback_interpolate import interpolate_middle_frame as farneback_interpolate
from src.deep.film_interpolate import interpolate_middle_frame as film_interpolate
from src.eval.metrics import psnr, ssim


def _load_triplet(triplet_dir: Path) -> tuple:
    frame_prev = cv2.imread(str(triplet_dir / "t-1.png"), cv2.IMREAD_GRAYSCALE)
    frame_mid = cv2.imread(str(triplet_dir / "t.png"), cv2.IMREAD_GRAYSCALE)
    frame_next = cv2.imread(str(triplet_dir / "t+1.png"), cv2.IMREAD_GRAYSCALE)
    if frame_prev is None or frame_mid is None or frame_next is None:
        raise FileNotFoundError(f"Missing t-1/t/t+1.png under {triplet_dir}")
    return frame_prev, frame_mid, frame_next


def build_comparison_figure(
    triplet_dir: Path, film_model_path: Path, device: str | None = None
) -> plt.Figure:
    frame_prev, frame_mid, frame_next = _load_triplet(triplet_dir)

    farneback_pred = farneback_interpolate(frame_prev, frame_next)
    film_pred = film_interpolate(frame_prev, frame_next, film_model_path, device=device)

    panels = [
        ("t-1 (real)", frame_prev),
        (
            f"Farneback\nPSNR {psnr(farneback_pred, frame_mid):.1f} / SSIM {ssim(farneback_pred, frame_mid):.2f}",
            farneback_pred,
        ),
        (
            f"FILM\nPSNR {psnr(film_pred, frame_mid):.1f} / SSIM {ssim(film_pred, frame_mid):.2f}",
            film_pred,
        ),
        ("t (ground truth)", frame_mid),
        ("t+1 (real)", frame_next),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(triplet_dir.name)
    fig.tight_layout()
    return fig


def save_comparisons(
    triplets_dir: Path,
    film_model_path: Path,
    out_dir: Path,
    device: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    triplet_dirs = sorted(p for p in triplets_dir.iterdir() if p.is_dir())
    if limit is not None:
        triplet_dirs = triplet_dirs[:limit]

    for triplet_dir in triplet_dirs:
        fig = build_comparison_figure(triplet_dir, film_model_path, device=device)
        out_path = out_dir / f"{triplet_dir.name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-dir", default="data/processed/triplets_cyclone")
    parser.add_argument("--film-model-path", default="models/film_net_fp32.pt")
    parser.add_argument("--out-dir", default="data/processed/qualitative")
    parser.add_argument("--device", default=None, help="cuda / cpu -- auto-detects if omitted")
    parser.add_argument("--limit", type=int, default=3, help="number of triplets to render (sorted order)")
    args = parser.parse_args()

    written = save_comparisons(
        Path(args.triplets_dir),
        Path(args.film_model_path),
        Path(args.out_dir),
        device=args.device,
        limit=args.limit,
    )
    print(f"Wrote {len(written)} comparison panels to {args.out_dir}")
