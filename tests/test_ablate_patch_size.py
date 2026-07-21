from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.eval.ablate_patch_size import run_patch_size_ablation

MODEL_PATH = Path("models/film_net_fp32.pt")


def test_run_patch_size_ablation_requires_nc_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()  # no .nc files inside

    with pytest.raises(SystemExit, match="No .nc files found"):
        run_patch_size_ablation(raw_dir, [128], MODEL_PATH, tmp_path / "out")


def _scan_filename(minute_offset: int) -> str:
    # 2024, day-of-year 100 (April 9), 12:00 UTC + minute_offset minutes
    total_minutes = 12 * 60 + minute_offset
    hh, mm = divmod(total_minutes, 60)
    return f"OR_ABI-L1b-RadF-M6C13_G16_s2024100{hh:02d}{mm:02d}000_e2024100{hh:02d}{mm:02d}206_c2024100{hh:02d}{mm:02d}226.nc"


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)
def test_run_patch_size_ablation_evaluates_every_size(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for minute in (0, 10, 20):
        ds = xr.Dataset({"Rad": (("y", "x"), np.random.uniform(0, 100, (64, 64)).astype(np.float32))})
        ds.to_netcdf(raw_dir / _scan_filename(minute))

    out_dir = tmp_path / "out"
    results = run_patch_size_ablation(raw_dir, [32, 64], MODEL_PATH, out_dir)

    assert set(results) == {32, 64}
    for methods in results.values():
        assert set(methods) == {"farneback", "film"}
    assert (out_dir / "farneback_32.csv").exists()
    assert (out_dir / "film_32.csv").exists()
    assert (out_dir / "farneback_64.csv").exists()
    assert (out_dir / "film_64.csv").exists()
