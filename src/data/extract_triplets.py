"""Convert downloaded GOES-16 ABI-L1b radiance NetCDF scans into normalized
8-bit image patches, and group consecutive scans into (t-1, t, t+1)
triplets for frame-interpolation training/eval.

frame_t is never used as model input -- it's the held-out ground truth the
interpolation methods are scored against, which is what makes this whole
project self-supervised: no manual labeling, just real consecutive scans.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import xarray as xr


def load_radiance(nc_path: Path) -> np.ndarray:
    """Load the 'Rad' variable from a GOES ABI-L1b NetCDF file as float32."""
    with xr.open_dataset(nc_path) as ds:
        rad = ds["Rad"].values.astype(np.float32)
    return rad


def to_uint8(rad: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Normalize radiance to an 8-bit image.

    Percentile clipping (rather than raw min/max) keeps a handful of
    sensor outlier pixels from washing out the contrast of the whole frame.
    """
    lo = float(np.nanpercentile(rad, 1)) if lo is None else lo
    hi = float(np.nanpercentile(rad, 99)) if hi is None else hi
    # off-Earth-disk pixels are NaN in the radiance field -- map them to 0
    # explicitly rather than casting NaN to uint8, which is undefined.
    clipped = np.nan_to_num(np.clip(rad, lo, hi), nan=lo)
    normalized = (clipped - lo) / max(hi - lo, 1e-6)
    return (normalized * 255).astype(np.uint8)


def crop_patch(img: np.ndarray, center: tuple[int, int], size: int) -> np.ndarray:
    """Crop a `size` x `size` patch centered at `center` (row, col) pixel coords."""
    cy, cx = center
    half = size // 2
    h, w = img.shape
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    return img[y0:y1, x0:x1]


def scan_to_patch(nc_path: Path, center: tuple[int, int], size: int) -> np.ndarray:
    rad = load_radiance(nc_path)
    return crop_patch(to_uint8(rad), center, size)


def build_triplets(
    nc_paths: list[Path], center: tuple[int, int], size: int, out_dir: Path
) -> list[Path]:
    """Group consecutive scans (sorted by filename, which sorts by scan
    start time since GOES keys are zero-padded) into overlapping triplets
    and save each as a 3-frame PNG set.
    """
    nc_paths = sorted(nc_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(len(nc_paths) - 2):
        triplet_dir = out_dir / f"triplet_{i:04d}"
        triplet_dir.mkdir(exist_ok=True)
        for offset, name in zip((0, 1, 2), ("t-1", "t", "t+1")):
            patch = scan_to_patch(nc_paths[i + offset], center, size)
            cv2.imwrite(str(triplet_dir / f"{name}.png"), patch)
        written.append(triplet_dir)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed/triplets")
    parser.add_argument("--center-row", type=int, default=None, help="defaults to image center")
    parser.add_argument("--center-col", type=int, default=None, help="defaults to image center")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    nc_paths = sorted(Path(args.raw_dir).glob("*.nc"))
    if not nc_paths:
        raise SystemExit(f"No .nc files found in {args.raw_dir} -- run fetch_goes.py first")

    if args.center_row is None or args.center_col is None:
        sample_shape = load_radiance(nc_paths[0]).shape
        center = (sample_shape[0] // 2, sample_shape[1] // 2)
    else:
        center = (args.center_row, args.center_col)

    written = build_triplets(nc_paths, center, args.size, Path(args.out_dir))
    print(f"Wrote {len(written)} triplets to {args.out_dir} (patch center={center}, size={args.size})")
