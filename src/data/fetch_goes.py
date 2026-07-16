"""Fetch GOES-16 ABI-L1b Radiance (Full Disk) NetCDF scans from NOAA's
public AWS Open Data bucket -- no credentials needed (unsigned requests).

Bucket layout:
  ABI-L1b-RadF/<year>/<day_of_year>/<hour>/OR_ABI-L1b-RadF-M6C<band>_G16_s<start>_e<end>_c<created>.nc

Scans are ~10 minutes apart for GOES-16 full-disk mode 6. Band 13 (clean
longwave IR, ~10.3um) is used by default: it works day and night (unlike
visible bands) and is ~25MB/scan vs. ~300MB+ for the high-resolution
visible band, which matters on a free-tier compute budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

BUCKET = "noaa-goes16"


def _client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_scans(dt: datetime, band: int, client=None) -> list[str]:
    """List all full-disk scan keys for `band` within `dt`'s hour."""
    client = client or _client()
    prefix = (
        f"ABI-L1b-RadF/{dt.year}/{dt.timetuple().tm_yday:03d}/{dt.hour:02d}/"
        f"OR_ABI-L1b-RadF-M6C{band:02d}"
    )
    resp = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return sorted(obj["Key"] for obj in resp.get("Contents", []))


def list_scans_range(start: datetime, end: datetime, band: int) -> list[str]:
    """List scan keys across every hour in [start, end]."""
    client = _client()
    keys: list[str] = []
    hour = start.replace(minute=0, second=0, microsecond=0)
    while hour <= end:
        keys.extend(list_scans(hour, band, client))
        hour += timedelta(hours=1)
    return keys


def download(key: str, dest_dir: Path, client=None) -> Path:
    """Download a single scan to `dest_dir`, skipping if already present."""
    client = client or _client()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(key).name
    if dest.exists():
        return dest
    client.download_file(BUCKET, key, str(dest))
    return dest


def download_range(start: datetime, end: datetime, band: int, dest_dir: Path) -> list[Path]:
    client = _client()
    keys = list_scans_range(start, end, band)
    return [download(key, dest_dir, client) for key in tqdm(keys, desc=f"band {band}")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC start, e.g. 2024-04-09T12:00")
    parser.add_argument("--end", required=True, help="UTC end, e.g. 2024-04-09T14:00")
    parser.add_argument("--band", type=int, default=13, help="ABI channel number (default 13, clean IR)")
    parser.add_argument("--out", default="data/raw", help="output directory")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    paths = download_range(start, end, args.band, Path(args.out))
    print(f"Downloaded {len(paths)} files to {args.out}")
