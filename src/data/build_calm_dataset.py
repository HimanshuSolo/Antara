"""Build the calm-weather half of the cyclone/calm evaluation split: the
same geographic region as a cyclone's peak position, cropped from an
off-season window -- so the comparison controls for geography and isolates
weather regime, instead of pairing a storm-centered crop against whatever
happened to sit at the arbitrary full-disk image center.

Defaults to MILTON 2024's peak position (see build_cyclone_dataset.py) and
the April 2024 window already used for the fine-tuning proof-of-concept,
which predates the Atlantic hurricane season (June-November) entirely.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.data.extract_triplets import build_triplets
from src.data.fetch_goes import download_range
from src.data.geo_projection import latlon_to_pixel


def build_calm_dataset(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    raw_dir: Path,
    triplets_dir: Path,
    band: int = 13,
    size: int = 256,
) -> list[Path]:
    nc_paths = download_range(start, end, band, raw_dir)
    if len(nc_paths) < 3:
        raise ValueError(
            f"Only {len(nc_paths)} scans downloaded for {start}-{end} -- "
            "need at least 3 to form a triplet"
        )

    center = latlon_to_pixel(lat, lon, nc_paths[0])
    print(f"Calm patch center in pixel space: {center}")

    return build_triplets(nc_paths, center, size, triplets_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, default=21.8, help="defaults to MILTON 2024's peak latitude")
    parser.add_argument("--lon", type=float, default=-90.9, help="defaults to MILTON 2024's peak longitude")
    parser.add_argument("--start", default="2024-04-09T12:00", help="UTC start, off-season by default")
    parser.add_argument("--end", default="2024-04-09T20:00", help="UTC end, off-season by default")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--triplets-dir", default="data/processed/triplets_calm")
    parser.add_argument("--band", type=int, default=13)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    written = build_calm_dataset(
        args.lat,
        args.lon,
        datetime.fromisoformat(args.start),
        datetime.fromisoformat(args.end),
        Path(args.raw_dir),
        Path(args.triplets_dir),
        band=args.band,
        size=args.size,
    )
    print(f"Wrote {len(written)} calm triplets to {args.triplets_dir}")
