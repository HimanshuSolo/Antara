from pathlib import Path

from src.data.ibtracs import download_ibtracs, list_storms, load_track

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
