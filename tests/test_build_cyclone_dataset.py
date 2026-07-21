from datetime import datetime
from pathlib import Path

import pytest

from src.data import build_cyclone_dataset as bcd
from src.data.ibtracs import StormFix


def _fix(time, lat, lon, wind_kt):
    return StormFix(name="MILTON", basin="NA", time=time, lat=lat, lon=lon, wind_kt=wind_kt)


def test_raises_when_no_track_found(tmp_path, monkeypatch):
    monkeypatch.setattr(bcd, "download_ibtracs", lambda csv: csv)
    monkeypatch.setattr(bcd, "load_track", lambda csv, name, season: [])

    with pytest.raises(ValueError, match="No IBTrACS track"):
        bcd.build_cyclone_dataset(
            "MILTON", 2024, tmp_path / "ibtracs.csv", tmp_path / "raw", tmp_path / "triplets"
        )


def test_raises_when_too_few_scans_downloaded(tmp_path, monkeypatch):
    track = [_fix(datetime(2024, 10, 9, 12), 21.8, -90.9, 155.0)]
    monkeypatch.setattr(bcd, "download_ibtracs", lambda csv: csv)
    monkeypatch.setattr(bcd, "load_track", lambda csv, name, season: track)
    monkeypatch.setattr(bcd, "download_range", lambda start, end, band, raw_dir: [Path("a.nc")])

    with pytest.raises(ValueError, match="need at least 3"):
        bcd.build_cyclone_dataset(
            "MILTON", 2024, tmp_path / "ibtracs.csv", tmp_path / "raw", tmp_path / "triplets"
        )


def test_centers_triplets_on_the_peak_wind_fix(tmp_path, monkeypatch):
    track = [
        _fix(datetime(2024, 10, 9, 10), 20.0, -91.0, 65.0),
        _fix(datetime(2024, 10, 9, 12), 21.8, -90.9, 155.0),  # peak
        _fix(datetime(2024, 10, 9, 14), 22.0, -90.5, 90.0),
    ]
    scans = [Path("s1.nc"), Path("s2.nc"), Path("s3.nc")]
    seen_latlon = {}
    seen_triplet_args = {}

    monkeypatch.setattr(bcd, "download_ibtracs", lambda csv: csv)
    monkeypatch.setattr(bcd, "load_track", lambda csv, name, season: track)
    monkeypatch.setattr(bcd, "download_range", lambda start, end, band, raw_dir: scans)

    def fake_latlon_to_pixel(lat, lon, sample_nc_path):
        seen_latlon["lat"] = lat
        seen_latlon["lon"] = lon
        seen_latlon["sample_nc_path"] = sample_nc_path
        return (100, 200)

    def fake_build_triplets(nc_paths, center, size, out_dir):
        seen_triplet_args["nc_paths"] = nc_paths
        seen_triplet_args["center"] = center
        seen_triplet_args["size"] = size
        seen_triplet_args["out_dir"] = out_dir
        return [out_dir / "triplet_0000"]

    monkeypatch.setattr(bcd, "latlon_to_pixel", fake_latlon_to_pixel)
    monkeypatch.setattr(bcd, "build_triplets", fake_build_triplets)

    result = bcd.build_cyclone_dataset(
        "MILTON", 2024, tmp_path / "ibtracs.csv", tmp_path / "raw", tmp_path / "triplets", size=128
    )

    assert seen_latlon["lat"] == 21.8  # the peak-wind fix, not the first/last
    assert seen_latlon["lon"] == -90.9
    assert seen_latlon["sample_nc_path"] == scans[0]
    assert seen_triplet_args["nc_paths"] == scans
    assert seen_triplet_args["center"] == (100, 200)
    assert seen_triplet_args["size"] == 128
    assert result == [tmp_path / "triplets" / "triplet_0000"]
