from pathlib import Path

from src.data.extract_triplets import default_center, parse_scan_time


def test_parse_scan_time_extracts_utc_start():
    name = "OR_ABI-L1b-RadF-M6C13_G16_s20241001210206_e20241001219526_c20241001219573.nc"
    dt = parse_scan_time(Path(name))
    # 2024, day-of-year 100 -> April 9; 12:10:20 UTC
    assert (dt.year, dt.month, dt.day) == (2024, 4, 9)
    assert (dt.hour, dt.minute, dt.second) == (12, 10, 20)


def test_parse_scan_time_rejects_unrecognized_filename():
    import pytest
    with pytest.raises(ValueError):
        parse_scan_time(Path("not_a_goes_file.nc"))


def test_default_center_of_even_shape():
    assert default_center((100, 200)) == (50, 100)


def test_default_center_of_odd_shape():
    # integer division, matching numpy/cv2 pixel-index conventions
    assert default_center((101, 201)) == (50, 100)
