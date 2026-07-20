"""Build a cyclone-centered evaluation set: find a named storm's peak
intensity in IBTrACS, download the real GOES-16 scans covering that
window, and extract triplets centered on the storm's actual position --
instead of guessing at "probably stormy" dates/locations.

This is the principled "fast, non-linear motion" half of the cyclone/calm
evaluation split from docs/PLAN.md. The "calm" half doesn't need this
script at all: any GOES-16 data from outside hurricane season (roughly
June-November for the Atlantic/East Pacific) already qualifies -- the
April 2024 triplets already extracted for fine-tuning work as the calm
comparison set as-is.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

from src.data.extract_triplets import build_triplets
from src.data.fetch_goes import download_range
from src.data.geo_projection import latlon_to_pixel
from src.data.ibtracs import download_ibtracs, load_track


def build_cyclone_dataset(
    storm_name: str,
    season: int,
    ibtracs_csv: Path,
    raw_dir: Path,
    triplets_dir: Path,
    band: int = 13,
    size: int = 256,
    window_hours: float = 2.0,
) -> list[Path]:
    download_ibtracs(ibtracs_csv)

    track = load_track(ibtracs_csv, storm_name, season)
    if not track:
        raise ValueError(f"No IBTrACS track found for {storm_name} ({season})")

    peak = max(track, key=lambda fix: fix.wind_kt or 0)
    print(
        f"{storm_name} peak: {peak.time} at ({peak.lat}, {peak.lon}), "
        f"{peak.wind_kt} kt"
    )

    start = peak.time - timedelta(hours=window_hours)
    end = peak.time + timedelta(hours=window_hours)
    nc_paths = download_range(start, end, band, raw_dir)
    if len(nc_paths) < 3:
        raise ValueError(
            f"Only {len(nc_paths)} scans downloaded for the window around "
            f"{storm_name}'s peak -- need at least 3 to form a triplet"
        )

    center = latlon_to_pixel(peak.lat, peak.lon, nc_paths[0])
    print(f"Storm center in pixel space: {center}")

    return build_triplets(nc_paths, center, size, triplets_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storm-name", default="MILTON")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--ibtracs-csv", default="data/ibtracs_last3years.csv")
    parser.add_argument("--raw-dir", default="data/raw_cyclone")
    parser.add_argument("--triplets-dir", default="data/processed/triplets_cyclone")
    parser.add_argument("--band", type=int, default=13)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--window-hours", type=float, default=2.0)
    args = parser.parse_args()

    written = build_cyclone_dataset(
        args.storm_name,
        args.season,
        Path(args.ibtracs_csv),
        Path(args.raw_dir),
        Path(args.triplets_dir),
        band=args.band,
        size=args.size,
        window_hours=args.window_hours,
    )
    print(f"Wrote {len(written)} cyclone triplets to {args.triplets_dir}")
