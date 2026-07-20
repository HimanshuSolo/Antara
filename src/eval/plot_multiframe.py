"""Qualitative multi-frame interpolation panel: real t-1, N evenly-spaced
FILM-synthesized intermediate frames, and real t+1 -- demonstrates Nx
temporal super-resolution from a single real frame pair, rather than
FILM's usual single midpoint. Per docs/PLAN.md's multi-frame stretch goal.

No PSNR/SSIM/LPIPS here: our triplets only have a real ground-truth frame
at the exact midpoint (t) -- that's exactly what evaluate_film.py already
scores. The other synthesized frames (e.g. t=0.25, t=0.75 for
num_frames=3) have no ground truth in this dataset to compare against,
only to look at.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.deep.film_interpolate import interpolate_multi
from src.eval.plot_comparison import list_triplet_dirs, load_triplet


def build_multiframe_figure(
    triplet_dir: Path,
    film_model_path: Path,
    num_frames: int = 3,
    device: str | None = None,
) -> plt.Figure:
    frame_prev, _frame_mid, frame_next = load_triplet(triplet_dir)
    synth_frames = interpolate_multi(frame_prev, frame_next, film_model_path, num_frames=num_frames, device=device)

    panels = [("t-1 (real)", frame_prev)]
    for i, frame in enumerate(synth_frames, start=1):
        t = i / (num_frames + 1)
        panels.append((f"t={t:.2f}\n(synthesized)", frame))
    panels.append(("t+1 (real)", frame_next))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{triplet_dir.name}: {num_frames}x multi-frame interpolation")
    fig.tight_layout()
    return fig


def save_multiframe(
    triplets_dir: Path,
    film_model_path: Path,
    out_dir: Path,
    num_frames: int = 3,
    device: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for triplet_dir in list_triplet_dirs(triplets_dir, limit=limit):
        fig = build_multiframe_figure(triplet_dir, film_model_path, num_frames=num_frames, device=device)
        out_path = out_dir / f"{triplet_dir.name}_multiframe.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-dir", default="data/processed/triplets_cyclone")
    parser.add_argument("--film-model-path", default="models/film_net_fp32.pt")
    parser.add_argument("--out-dir", default="data/processed/multiframe")
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--device", default=None, help="cuda / cpu -- auto-detects if omitted")
    parser.add_argument("--limit", type=int, default=3, help="number of triplets to render (sorted order)")
    args = parser.parse_args()

    written = save_multiframe(
        Path(args.triplets_dir),
        Path(args.film_model_path),
        Path(args.out_dir),
        num_frames=args.num_frames,
        device=args.device,
        limit=args.limit,
    )
    print(f"Wrote {len(written)} multi-frame panels to {args.out_dir}")
