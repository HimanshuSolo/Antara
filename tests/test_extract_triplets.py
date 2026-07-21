from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.data.extract_triplets import (
    build_triplets,
    crop_patch,
    default_center,
    load_radiance,
    parse_scan_time,
    scan_to_patch,
    to_uint8,
)


def test_parse_scan_time_extracts_utc_start():
    name = "OR_ABI-L1b-RadF-M6C13_G16_s20241001210206_e20241001219526_c20241001219573.nc"
    dt = parse_scan_time(Path(name))
    # 2024, day-of-year 100 -> April 9; 12:10:20 UTC
    assert (dt.year, dt.month, dt.day) == (2024, 4, 9)
    assert (dt.hour, dt.minute, dt.second) == (12, 10, 20)


def test_parse_scan_time_rejects_unrecognized_filename():
    with pytest.raises(ValueError):
        parse_scan_time(Path("not_a_goes_file.nc"))


def test_default_center_of_even_shape():
    assert default_center((100, 200)) == (50, 100)


def test_default_center_of_odd_shape():
    # integer division, matching numpy/cv2 pixel-index conventions
    assert default_center((101, 201)) == (50, 100)


def test_to_uint8_maps_full_range_to_0_255():
    rad = np.linspace(0.0, 100.0, 100, dtype=np.float32).reshape(10, 10)
    img = to_uint8(rad, lo=0.0, hi=100.0)
    assert img.dtype == np.uint8
    assert img.min() == 0
    assert img.max() == 255


def test_to_uint8_clips_outliers_using_percentiles():
    rad = np.full((10, 10), 50.0, dtype=np.float32)
    rad[0, 0] = -1000.0  # a sensor outlier that shouldn't wash out contrast
    rad[0, 1] = 1000.0
    img = to_uint8(rad)
    # everything except the two outlier pixels is identical -> clipped to the
    # same normalized value, not spread across the full outlier-driven range
    assert img[1, 1] == img[5, 5]


def test_to_uint8_maps_nan_to_zero():
    rad = np.full((10, 10), 50.0, dtype=np.float32)
    rad[0, 0] = np.nan  # off-Earth-disk pixels are NaN in real radiance fields
    img = to_uint8(rad)
    assert img[0, 0] == 0


def test_crop_patch_centered_within_bounds():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = crop_patch(img, center=(5, 5), size=4)
    assert patch.shape == (4, 4)
    assert patch[0, 0] == img[3, 3]


def test_crop_patch_clips_at_image_boundary():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = crop_patch(img, center=(0, 0), size=4)
    # half=2, so y0/x0 clip to 0 instead of going negative -> a 2x2 patch
    assert patch.shape == (2, 2)


def _write_goes_scan(path: Path, radiance: np.ndarray) -> None:
    ds = xr.Dataset({"Rad": (("y", "x"), radiance)})
    ds.to_netcdf(path)


def test_load_radiance_reads_rad_variable_as_float32(tmp_path):
    nc_path = tmp_path / "scan.nc"
    _write_goes_scan(nc_path, np.array([[1, 2], [3, 4]], dtype=np.int16))

    rad = load_radiance(nc_path)

    assert rad.dtype == np.float32
    assert np.array_equal(rad, np.array([[1, 2], [3, 4]], dtype=np.float32))


def test_scan_to_patch_normalizes_and_crops(tmp_path):
    nc_path = tmp_path / "scan.nc"
    radiance = np.linspace(0, 100, 64, dtype=np.float32).reshape(8, 8)
    _write_goes_scan(nc_path, radiance)

    patch = scan_to_patch(nc_path, center=(4, 4), size=4)

    assert patch.shape == (4, 4)
    assert patch.dtype == np.uint8


def _scan_filename(minute_offset: int) -> str:
    # 2024, day-of-year 100 (April 9), 12:00 UTC + minute_offset minutes
    total_minutes = 12 * 60 + minute_offset
    hh, mm = divmod(total_minutes, 60)
    return f"OR_ABI-L1b-RadF-M6C13_G16_s2024100{hh:02d}{mm:02d}000_e2024100{hh:02d}{mm:02d}206_c2024100{hh:02d}{mm:02d}226.nc"


def test_build_triplets_writes_three_frames_per_triplet(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    nc_paths = []
    for i, minute in enumerate((0, 10, 20)):
        path = raw_dir / _scan_filename(minute)
        _write_goes_scan(path, np.full((8, 8), i * 10.0, dtype=np.float32))
        nc_paths.append(path)

    out_dir = tmp_path / "triplets"
    written = build_triplets(nc_paths, center=(4, 4), size=8, out_dir=out_dir)

    assert len(written) == 1
    triplet_dir = written[0]
    assert (triplet_dir / "t-1.png").exists()
    assert (triplet_dir / "t.png").exists()
    assert (triplet_dir / "t+1.png").exists()


def test_build_triplets_skips_triplets_spanning_an_abnormal_gap(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    nc_paths = []
    # gaps: 10, 10, 40, 10 minutes -- median is 10, so the 40-minute gap
    # (a missed/dropped scan) exceeds the default 1.5x threshold.
    for minute in (0, 10, 20, 60, 70):
        path = raw_dir / _scan_filename(minute)
        _write_goes_scan(path, np.zeros((8, 8), dtype=np.float32))
        nc_paths.append(path)

    out_dir = tmp_path / "triplets"
    written = build_triplets(nc_paths, center=(4, 4), size=8, out_dir=out_dir)

    # only the first triplet (scans at 0/10/20) doesn't straddle the gap
    assert len(written) == 1


def test_build_triplets_returns_empty_for_fewer_than_two_scans(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    path = raw_dir / _scan_filename(0)
    _write_goes_scan(path, np.zeros((8, 8), dtype=np.float32))

    written = build_triplets([path], center=(4, 4), size=8, out_dir=tmp_path / "triplets")

    assert written == []
