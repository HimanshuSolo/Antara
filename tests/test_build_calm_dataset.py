from datetime import datetime
from pathlib import Path

import pytest

from src.data import build_calm_dataset as bcd


def test_raises_when_too_few_scans_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(bcd, "download_range", lambda start, end, band, raw_dir: [Path("a.nc")])

    with pytest.raises(ValueError, match="need at least 3"):
        bcd.build_calm_dataset(
            21.8, -90.9,
            datetime(2024, 4, 9, 12), datetime(2024, 4, 9, 20),
            tmp_path / "raw", tmp_path / "triplets",
        )


def test_crops_triplets_at_the_given_latlon(tmp_path, monkeypatch):
    scans = [Path("s1.nc"), Path("s2.nc"), Path("s3.nc")]
    seen_latlon = {}
    seen_triplet_args = {}

    monkeypatch.setattr(bcd, "download_range", lambda start, end, band, raw_dir: scans)

    def fake_latlon_to_pixel(lat, lon, sample_nc_path):
        seen_latlon["lat"] = lat
        seen_latlon["lon"] = lon
        seen_latlon["sample_nc_path"] = sample_nc_path
        return (50, 60)

    def fake_build_triplets(nc_paths, center, size, out_dir):
        seen_triplet_args["nc_paths"] = nc_paths
        seen_triplet_args["center"] = center
        seen_triplet_args["size"] = size
        seen_triplet_args["out_dir"] = out_dir
        return [out_dir / "triplet_0000"]

    monkeypatch.setattr(bcd, "latlon_to_pixel", fake_latlon_to_pixel)
    monkeypatch.setattr(bcd, "build_triplets", fake_build_triplets)

    result = bcd.build_calm_dataset(
        21.8, -90.9,
        datetime(2024, 4, 9, 12), datetime(2024, 4, 9, 20),
        tmp_path / "raw", tmp_path / "triplets", size=128,
    )

    assert seen_latlon["lat"] == 21.8
    assert seen_latlon["lon"] == -90.9
    assert seen_latlon["sample_nc_path"] == scans[0]
    assert seen_triplet_args["nc_paths"] == scans
    assert seen_triplet_args["center"] == (50, 60)
    assert seen_triplet_args["size"] == 128
    assert result == [tmp_path / "triplets" / "triplet_0000"]
