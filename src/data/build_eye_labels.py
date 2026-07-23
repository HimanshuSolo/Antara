"""Derive ground-truth storm-eye pixel coordinates for every frame of a
cyclone triplets directory, by combining each triplet's `meta.json` (the
source scan and crop center `extract_triplets.build_triplets` now writes)
with IBTrACS best-track positions interpolated to each frame's exact scan
time.

This is the label source for both the classical eye detector's accuracy
eval and the trained detector's supervised training data -- no manual
annotation involved, matching the rest of this project's self-supervised
approach.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from src.data.geo_projection import latlon_to_pixel
from src.data.ibtracs import StormFix, download_ibtracs, interpolate_position, load_track

FIELDNAMES = ["triplet", "frame", "row", "col"]


def build_eye_labels(triplets_dir: Path, track: list[StormFix]) -> list[dict]:
    """Compute (row, col) eye labels for every frame of every triplet under
    `triplets_dir` that has a `meta.json` (older triplets built before that
    metadata existed are silently skipped, not raised on -- they simply
    can't be labeled)."""
    rows = []
    for triplet_dir in sorted(triplets_dir.glob("triplet_*")):
        meta_path = triplet_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        center_row, center_col = meta["center"]
        half = meta["size"] // 2
        origin_row, origin_col = center_row - half, center_col - half

        for frame, scan_time_iso, source in zip(
            ("t-1", "t", "t+1"), meta["scan_times"], meta["source"]
        ):
            position = interpolate_position(track, datetime.fromisoformat(scan_time_iso))
            if position is None:
                continue
            lat, lon = position
            row, col = latlon_to_pixel(lat, lon, Path(source))
            rows.append({
                "triplet": triplet_dir.name,
                "frame": frame,
                "row": row - origin_row,
                "col": col - origin_col,
            })
    return rows


def write_eye_labels(triplets_dir: Path, storm_name: str, season: int, ibtracs_csv: Path) -> Path:
    download_ibtracs(ibtracs_csv)
    track = load_track(ibtracs_csv, storm_name, season)
    if not track:
        raise ValueError(f"No IBTrACS track found for {storm_name} ({season})")

    rows = build_eye_labels(triplets_dir, track)

    out_path = triplets_dir / "eye_labels.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-dir", default="data/processed/triplets_cyclone")
    parser.add_argument("--storm-name", default="MILTON")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--ibtracs-csv", default="data/ibtracs_last3years.csv")
    args = parser.parse_args()

    out_path = write_eye_labels(
        Path(args.triplets_dir), args.storm_name, args.season, Path(args.ibtracs_csv)
    )
    print(f"Wrote eye labels to {out_path}")
