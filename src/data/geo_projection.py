"""Convert a real-world latitude/longitude (e.g. a cyclone's position from
IBTrACS) into a pixel row/col in a GOES ABI full-disk scan, so we can crop
a patch centered on a specific storm instead of the arbitrary image
center `extract_triplets.py` defaults to.

The forward geodetic-to-scan-angle transform is the standard formula
published in the GOES-R Product User's Guide (PUG) -- the same one every
GOES geolocation tool implements. Accuracy only needs to be good to within
a handful of pixels here (we crop a 256px = ~512km patch around the
result), not survey-grade, so this intentionally doesn't handle every
edge case (e.g. points behind the Earth from the satellite's view).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def latlon_to_scan_angles(
    lat_deg: float,
    lon_deg: float,
    sat_lon_deg: float,
    sat_height_m: float,
    semi_major_m: float,
    semi_minor_m: float,
) -> tuple[float, float]:
    """Forward geodetic -> GOES fixed-grid scan angle transform.

    Returns (x, y) in radians, matching the file's `x`/`y` coordinate
    variables. The sub-satellite point (lat=0, lon=sat_lon_deg) maps to
    exactly (0, 0).
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    lon0 = np.radians(sat_lon_deg)
    a, b = semi_major_m, semi_minor_m
    h = sat_height_m + a  # distance from Earth's center to the satellite

    e2 = 1 - (b**2 / a**2)
    phi_c = np.arctan((b**2 / a**2) * np.tan(lat))  # geocentric latitude
    rc = b / np.sqrt(1 - e2 * np.cos(phi_c) ** 2)  # Earth-center to surface point

    sx = h - rc * np.cos(phi_c) * np.cos(lon - lon0)
    sy = -rc * np.cos(phi_c) * np.sin(lon - lon0)
    sz = rc * np.sin(phi_c)

    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))
    return float(x), float(y)


def scan_angles_to_pixel(x: float, y: float, x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[int, int]:
    """Nearest-neighbor lookup of a scan angle into the file's actual
    per-pixel coordinate arrays. Returns (row, col).
    """
    col = int(np.argmin(np.abs(x_coords - x)))
    row = int(np.argmin(np.abs(y_coords - y)))
    return row, col


def latlon_to_pixel(lat_deg: float, lon_deg: float, sample_nc_path: Path) -> tuple[int, int]:
    """Convenience wrapper: read projection parameters and coordinate
    arrays straight from a real GOES scan, and return the (row, col)
    pixel closest to (lat_deg, lon_deg).
    """
    with xr.open_dataset(sample_nc_path) as ds:
        proj = ds["goes_imager_projection"].attrs
        x, y = latlon_to_scan_angles(
            lat_deg,
            lon_deg,
            sat_lon_deg=proj["longitude_of_projection_origin"],
            sat_height_m=proj["perspective_point_height"],
            semi_major_m=proj["semi_major_axis"],
            semi_minor_m=proj["semi_minor_axis"],
        )
        return scan_angles_to_pixel(x, y, ds["x"].values, ds["y"].values)
