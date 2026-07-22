"""Fetch GOES ABI-L1b Radiance (Full Disk) NetCDF scans from NOAA's public
AWS Open Data buckets -- no credentials needed (unsigned requests).

Bucket layout (same across noaa-goes16/18/19):
  ABI-L1b-RadF/<year>/<day_of_year>/<hour>/OR_ABI-L1b-RadF-M6C<band>_G<sat>_s<start>_e<end>_c<created>.nc

Scans are ~10 minutes apart for full-disk mode 6. Band 13 (clean longwave
IR, ~10.3um) is used by default: it works day and night (unlike visible
bands) and is ~25MB/scan vs. ~300MB+ for the high-resolution visible band,
which matters on a free-tier compute budget.

GOES-16 was this project's original data source and is what every
existing dataset/checkpoint here was built from, so it stays the default
bucket for reproducibility -- but GOES-16 stopped operating as GOES-East
on 2025-04-07 (superseded by GOES-19; see `noaa-goes19`), so it no longer
has new data. `latest_scan_pair`, used by the live pipeline API
(`src/api/live.py`), is pointed at `noaa-goes19` explicitly for that
reason.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

BUCKET = "noaa-goes16"

# current operational GOES-East feed (GOES-16's successor) -- see the
# module docstring. This is what a genuinely "live" fetch must use.
LIVE_BUCKET = "noaa-goes19"


def _client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_scans(dt: datetime, band: int, client=None, bucket: str = BUCKET) -> list[str]:
    """List all full-disk scan keys for `band` within `dt`'s hour."""
    client = client or _client()
    prefix = (
        f"ABI-L1b-RadF/{dt.year}/{dt.timetuple().tm_yday:03d}/{dt.hour:02d}/"
        f"OR_ABI-L1b-RadF-M6C{band:02d}"
    )
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
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


def download(key: str, dest_dir: Path, client=None, bucket: str = BUCKET) -> Path:
    """Download a single scan to `dest_dir`, skipping if already present."""
    client = client or _client()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(key).name
    if dest.exists():
        return dest
    client.download_file(bucket, key, str(dest))
    return dest


def latest_scan_pair(
    band: int = 13, now: datetime | None = None, client=None, bucket: str = LIVE_BUCKET
) -> tuple[str, str]:
    """Return the two most recently published full-disk scan keys for `band`.

    There is no real frame *between* these two yet -- that gap is exactly
    what this project's interpolation methods fill in, which makes this
    pair the input for a genuinely "live" demo rather than a canned one.
    Falls back to the previous hour's listing when the current hour hasn't
    published two scans yet (e.g. near the top of the hour). Defaults to
    `LIVE_BUCKET` (GOES-19, the current operational satellite) rather than
    `BUCKET` (GOES-16, retired 2025-04-07 -- see the module docstring).
    """
    client = client or _client()
    now = now or datetime.utcnow()

    keys = list_scans(now, band, client, bucket=bucket)
    if len(keys) < 2:
        keys = list_scans(now - timedelta(hours=1), band, client, bucket=bucket) + keys
    keys = sorted(keys)

    if len(keys) < 2:
        raise RuntimeError(f"Fewer than 2 recent band {band} scans found near {now.isoformat()}")
    return keys[-2], keys[-1]


def download_range(start: datetime, end: datetime, band: int, dest_dir: Path, max_workers: int = 8) -> list[Path]:
    # One scan at a time was badly latency-bound in practice (each request
    # pays a full round trip before the next starts) -- boto3 clients are
    # safe to share across threads, and S3 has no issue serving many
    # unsigned GETs to the same bucket concurrently, so a small thread pool
    # turns this from request-latency-bound into bandwidth-bound.
    client = _client()
    keys = list_scans_range(start, end, band)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(
            tqdm(
                pool.map(lambda key: download(key, dest_dir, client), keys),
                total=len(keys),
                desc=f"band {band}",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC start, e.g. 2024-04-09T12:00")
    parser.add_argument("--end", required=True, help="UTC end, e.g. 2024-04-09T14:00")
    parser.add_argument("--band", type=int, default=13, help="ABI channel number (default 13, clean IR)")
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--max-workers", type=int, default=8, help="concurrent download threads")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    paths = download_range(start, end, args.band, Path(args.out), max_workers=args.max_workers)
    print(f"Downloaded {len(paths)} files to {args.out}")
