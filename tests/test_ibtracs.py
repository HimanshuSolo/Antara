from datetime import datetime
from pathlib import Path

from src.data.ibtracs import download_ibtracs, interpolate_position, list_storms, load_track

FIXTURE = Path(__file__).parent / "fixtures" / "ibtracs_sample.csv"


def test_download_ibtracs_skips_when_dest_already_exists(tmp_path):
    dest = tmp_path / "ibtracs.csv"
    dest.write_bytes(b"already here")

    # No network access should be attempted -- if it were, this would
    # fail/hang against the bogus URL below.
    result = download_ibtracs(dest, url="http://example.invalid/not-a-real-file")

    assert result == dest
    assert dest.read_bytes() == b"already here"


def test_list_storms_finds_hurricane_strength_storms():
    storms = list_storms(FIXTURE, season=2024, min_wind_kt=64.0)
    assert storms == ["MILTON"]  # CALM never reaches 64kt (blank wind field)


def test_load_track_returns_sorted_fixes_for_named_storm():
    track = load_track(FIXTURE, "milton", 2024)  # case-insensitive
    assert len(track) == 3
    assert [fix.wind_kt for fix in track] == [30.0, 65.0, 155.0]
    assert track[0].time < track[-1].time


def test_load_track_ignores_other_storms_and_seasons():
    track = load_track(FIXTURE, "MILTON", season=2023)
    assert track == []


def test_interpolate_position_at_the_midpoint_between_two_fixes():
    track = load_track(FIXTURE, "MILTON", 2024)
    # second and third fixes: 2024-10-06 18:00 (22.5, -94.1) -> 2024-10-07
    # 20:00 (21.8, -90.9)
    midpoint = track[1].time + (track[2].time - track[1].time) / 2
    lat, lon = interpolate_position(track, midpoint)
    assert lat == (track[1].lat + track[2].lat) / 2
    assert lon == (track[1].lon + track[2].lon) / 2


def test_interpolate_position_exactly_on_a_fix_returns_that_fix():
    track = load_track(FIXTURE, "MILTON", 2024)
    lat, lon = interpolate_position(track, track[1].time)
    assert (lat, lon) == (track[1].lat, track[1].lon)


def test_interpolate_position_outside_track_range_returns_none():
    track = load_track(FIXTURE, "MILTON", 2024)
    before = track[0].time - (track[1].time - track[0].time)
    assert interpolate_position(track, before) is None


def test_interpolate_position_needs_at_least_two_fixes():
    assert interpolate_position([], datetime(2024, 10, 7)) is None
