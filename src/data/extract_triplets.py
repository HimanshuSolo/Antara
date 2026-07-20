"""Convert downloaded GOES-16 ABI-L1b radiance NetCDF scans into normalized
8-bit image patches, and group consecutive scans into (t-1, t, t+1)
triplets for frame-interpolation training/eval.

frame_t is never used as model input -- it's the held-out ground truth the
interpolation methods are scored against, which is what makes this whole
project self-supervised: no manual labeling, just real consecutive scans.
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import xarray as xr

# GOES filenames embed the scan start time as sYYYYDDDHHMMSSf (day-of-year,
# tenths-of-a-second dropped below) -- e.g. s20241001200206 = 2024, day 100,
# 12:00:20.6 UTC.
_SCAN_TIME_RE = re.compile(r"_s(\d{13})\d")


def parse_scan_time(nc_path: Path) -> datetime:
    match = _SCAN_TIME_RE.search(nc_path.name)
    if not match:
        raise ValueError(f"Could not find a scan start timestamp in {nc_path.name}")
    return datetime.strptime(match.group(1), "%Y%j%H%M%S")


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


def default_center(shape: tuple[int, int]) -> tuple[int, int]:
    """Pixel coordinates of the center of a (rows, cols) array."""
    return (shape[0] // 2, shape[1] // 2)


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
    nc_paths: list[Path],
    center: tuple[int, int],
    size: int,
    out_dir: Path,
    max_gap_ratio: float = 1.5,
) -> list[Path]:
    """Group consecutive scans (sorted by filename, which sorts by scan
    start time since GOES keys are zero-padded) into overlapping triplets
    and save each as a 3-frame PNG set.

    A missed/dropped scan (sensor recalibration, downlink gap) would
    otherwise silently produce a degenerate triplet where t-1 to t+1 spans
    far more time than intended -- a bad training/eval example that looks
    fine until you check timestamps. Any triplet whose adjacent gap
    exceeds `max_gap_ratio` times the dataset's median cadence is skipped.
    """
    nc_paths = sorted(nc_paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = [parse_scan_time(p) for p in nc_paths]
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    if not gaps:
        return []
    median_gap = sorted(gaps)[len(gaps) // 2]

    written = []
    skipped = 0
    for i in range(len(nc_paths) - 2):
        gap_prev = times[i + 1] - times[i]
        gap_next = times[i + 2] - times[i + 1]
        if gap_prev > median_gap * max_gap_ratio or gap_next > median_gap * max_gap_ratio:
            skipped += 1
            continue

        triplet_dir = out_dir / f"triplet_{i:04d}"
        triplet_dir.mkdir(exist_ok=True)
        for offset, name in zip((0, 1, 2), ("t-1", "t", "t+1")):
            patch = scan_to_patch(nc_paths[i + offset], center, size)
            cv2.imwrite(str(triplet_dir / f"{name}.png"), patch)
        written.append(triplet_dir)

    if skipped:
        print(f"Skipped {skipped} triplet(s) with an abnormal scan gap (missing scan)")
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
        center = default_center(load_radiance(nc_paths[0]).shape)
    else:
        center = (args.center_row, args.center_col)

    written = build_triplets(nc_paths, center, args.size, Path(args.out_dir))
    print(f"Wrote {len(written)} triplets to {args.out_dir} (patch center={center}, size={args.size})")
