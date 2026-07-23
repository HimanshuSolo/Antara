import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from src.data import build_eye_labels as bel
from src.data.ibtracs import StormFix


def _fix(time, lat, lon):
    return StormFix(name="MILTON", basin="NA", time=time, lat=lat, lon=lon, wind_kt=100.0)


def _write_meta(triplet_dir: Path, scan_times, center=(100, 100), size=8):
    triplet_dir.mkdir(parents=True)
    meta = {
        "source": [f"{triplet_dir}/{name}.nc" for name in ("t-1", "t", "t+1")],
        "scan_times": [t.isoformat() for t in scan_times],
        "center": list(center),
        "size": size,
    }
    (triplet_dir / "meta.json").write_text(json.dumps(meta))


def test_build_eye_labels_projects_and_offsets_by_crop_origin(tmp_path, monkeypatch):
    triplets_dir = tmp_path / "triplets_cyclone"
    scan_times = [datetime(2024, 10, 7, 10, 0), datetime(2024, 10, 7, 10, 10), datetime(2024, 10, 7, 10, 20)]
    _write_meta(triplets_dir / "triplet_0000", scan_times, center=(100, 100), size=8)

    track = [
        _fix(datetime(2024, 10, 7, 9, 0), 20.0, -90.0),
        _fix(datetime(2024, 10, 7, 11, 0), 22.0, -88.0),
    ]

    # full-disk pixel is always (105, 103) regardless of lat/lon in this
    # fake -- origin for a size=8 patch centered at (100, 100) is (96, 96),
    # so the expected label is (105 - 96, 103 - 96) = (9, 7).
    monkeypatch.setattr(bel, "latlon_to_pixel", lambda lat, lon, sample_nc_path: (105, 103))

    rows = bel.build_eye_labels(triplets_dir, track)

    assert len(rows) == 3
    assert all(r["triplet"] == "triplet_0000" for r in rows)
    assert [r["frame"] for r in rows] == ["t-1", "t", "t+1"]
    assert all((r["row"], r["col"]) == (9, 7) for r in rows)


def test_build_eye_labels_skips_triplets_without_meta(tmp_path):
    triplets_dir = tmp_path / "triplets_cyclone"
    (triplets_dir / "triplet_0000").mkdir(parents=True)  # no meta.json

    rows = bel.build_eye_labels(triplets_dir, track=[])

    assert rows == []


def test_build_eye_labels_skips_frames_outside_track_range(tmp_path, monkeypatch):
    triplets_dir = tmp_path / "triplets_cyclone"
    scan_times = [datetime(2024, 10, 7, 10, 0), datetime(2024, 10, 7, 10, 10), datetime(2024, 10, 7, 10, 20)]
    _write_meta(triplets_dir / "triplet_0000", scan_times)

    # track only covers the first two frames' timestamps
    track = [
        _fix(datetime(2024, 10, 7, 9, 0), 20.0, -90.0),
        _fix(datetime(2024, 10, 7, 10, 15), 21.0, -89.0),
    ]
    monkeypatch.setattr(bel, "latlon_to_pixel", lambda lat, lon, sample_nc_path: (0, 0))

    rows = bel.build_eye_labels(triplets_dir, track)

    assert [r["frame"] for r in rows] == ["t-1", "t"]


def test_write_eye_labels_raises_when_no_track_found(tmp_path, monkeypatch):
    monkeypatch.setattr(bel, "download_ibtracs", lambda csv: csv)
    monkeypatch.setattr(bel, "load_track", lambda csv, name, season: [])

    with pytest.raises(ValueError, match="No IBTrACS track"):
        bel.write_eye_labels(tmp_path, "MILTON", 2024, tmp_path / "ibtracs.csv")


def test_write_eye_labels_writes_a_csv(tmp_path, monkeypatch):
    scan_times = [datetime(2024, 10, 7, 10, 0), datetime(2024, 10, 7, 10, 10), datetime(2024, 10, 7, 10, 20)]
    _write_meta(tmp_path / "triplet_0000", scan_times, center=(50, 50), size=8)

    track = [
        _fix(datetime(2024, 10, 7, 9, 0), 20.0, -90.0),
        _fix(datetime(2024, 10, 7, 11, 0), 22.0, -88.0),
    ]
    monkeypatch.setattr(bel, "download_ibtracs", lambda csv: csv)
    monkeypatch.setattr(bel, "load_track", lambda csv, name, season: track)
    monkeypatch.setattr(bel, "latlon_to_pixel", lambda lat, lon, sample_nc_path: (55, 45))

    out_path = bel.write_eye_labels(tmp_path, "MILTON", 2024, tmp_path / "ibtracs.csv")

    assert out_path == tmp_path / "eye_labels.csv"
    with out_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["row"] == "9"  # 55 - (50 - 4)
    assert rows[0]["col"] == "-1"  # 45 - (50 - 4)
