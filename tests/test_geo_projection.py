import numpy as np
import pytest
import xarray as xr

from src.data.geo_projection import latlon_to_pixel, latlon_to_scan_angles, scan_angles_to_pixel

# real GOES-16 projection parameters, read from an actual downloaded scan
GOES16_PARAMS = dict(
    sat_lon_deg=-75.0,
    sat_height_m=35786023.0,
    semi_major_m=6378137.0,
    semi_minor_m=6356752.31414,
)


def test_sub_satellite_point_maps_to_origin():
    x, y = latlon_to_scan_angles(0.0, -75.0, **GOES16_PARAMS)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_real_storm_location_falls_within_full_disk_range():
    # Hurricane Milton, near peak intensity, Gulf of Mexico
    x, y = latlon_to_scan_angles(22.5, -94.1, **GOES16_PARAMS)
    full_disk_limit = 0.151844  # GOES-16 band 13 full-disk x/y extent, radians
    assert abs(x) < full_disk_limit
    assert abs(y) < full_disk_limit


def test_scan_angles_to_pixel_finds_nearest_index():
    x_coords = np.linspace(-1.0, 1.0, 11)  # 0.2 spacing
    y_coords = np.linspace(1.0, -1.0, 11)  # descending, like real GOES y
    row, col = scan_angles_to_pixel(x=0.21, y=-0.19, x_coords=x_coords, y_coords=y_coords)
    assert x_coords[col] == pytest.approx(0.2)
    assert y_coords[row] == pytest.approx(-0.2)


def _write_goes_projection_fixture(nc_path, n=21):
    proj_attrs = dict(
        longitude_of_projection_origin=GOES16_PARAMS["sat_lon_deg"],
        perspective_point_height=GOES16_PARAMS["sat_height_m"],
        semi_major_axis=GOES16_PARAMS["semi_major_m"],
        semi_minor_axis=GOES16_PARAMS["semi_minor_m"],
    )
    full_disk_limit = 0.151844  # GOES-16 band 13 full-disk x/y extent, radians
    x = np.linspace(-full_disk_limit, full_disk_limit, n)
    y = np.linspace(full_disk_limit, -full_disk_limit, n)
    ds = xr.Dataset(
        {
            "Rad": (("y", "x"), np.zeros((n, n), dtype=np.float32)),
            "goes_imager_projection": ((), 0, proj_attrs),
        },
        coords={"x": ("x", x), "y": ("y", y)},
    )
    ds.to_netcdf(nc_path)


def test_latlon_to_pixel_maps_sub_satellite_point_to_center(tmp_path):
    nc_path = tmp_path / "scan.nc"
    _write_goes_projection_fixture(nc_path, n=21)

    row, col = latlon_to_pixel(0.0, -75.0, nc_path)

    # sub-satellite point (lat=0, lon=sat_lon) is scan angle (0, 0), which
    # sits at the exact middle of a symmetric, odd-length coordinate array
    assert (row, col) == (10, 10)


def test_latlon_to_pixel_offset_point_moves_away_from_center(tmp_path):
    nc_path = tmp_path / "scan.nc"
    _write_goes_projection_fixture(nc_path, n=21)

    row, col = latlon_to_pixel(22.5, -94.1, nc_path)  # Hurricane Milton, near peak

    assert (row, col) != (10, 10)
