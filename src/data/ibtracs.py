"""Load cyclone track data from IBTrACS (International Best Track Archive
for Climate Stewardship) -- the standard, freely-available record of
where and when every tracked tropical cyclone actually was, used here to
build a principled "fast, non-linear motion" evaluation subset instead of
guessing at storm dates.

https://www.ncei.noaa.gov/products/international-best-track-archive
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

IBTRACS_LAST_3_YEARS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.last3years.list.v04r01.csv"
)


@dataclass
class StormFix:
    """A single best-track observation: where a named storm was at a point in time."""
    name: str
    basin: str
    time: datetime
    lat: float
    lon: float
    wind_kt: float | None


def download_ibtracs(dest: Path, url: str = IBTRACS_LAST_3_YEARS_URL) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _read_rows(csv_path: Path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        next(reader)  # IBTrACS' second row is units, not data
        yield from reader


def _parse_wind(value: str) -> float | None:
    value = value.strip()
    return float(value) if value else None


def list_storms(
    csv_path: Path,
    season: int,
    basins: tuple[str, ...] = ("NA", "EP"),
    min_wind_kt: float = 64.0,
) -> list[str]:
    """Names of storms that reached at least `min_wind_kt` in a given
    season/basin -- e.g. 64kt = hurricane strength. Useful for picking a
    concrete storm to build a test set around.
    """
    names = []
    for row in _read_rows(csv_path):
        if row["SEASON"] != str(season) or row["BASIN"] not in basins:
            continue
        wind = _parse_wind(row["USA_WIND"])
        if wind is not None and wind >= min_wind_kt and row["NAME"] not in names:
            names.append(row["NAME"])
    return names


def load_track(csv_path: Path, name: str, season: int) -> list[StormFix]:
    """All best-track fixes for a named storm in a given season, sorted by time."""
    fixes = []
    for row in _read_rows(csv_path):
        if row["SEASON"] != str(season) or row["NAME"].upper() != name.upper():
            continue
        try:
            lat, lon = float(row["LAT"]), float(row["LON"])
        except ValueError:
            continue
        fixes.append(StormFix(
            name=row["NAME"],
            basin=row["BASIN"],
            time=datetime.strptime(row["ISO_TIME"], "%Y-%m-%d %H:%M:%S"),
            lat=lat,
            lon=lon,
            wind_kt=_parse_wind(row["USA_WIND"]),
        ))
    fixes.sort(key=lambda fix: fix.time)
    return fixes
