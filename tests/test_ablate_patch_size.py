from pathlib import Path

import pytest

from src.eval.ablate_patch_size import run_patch_size_ablation

MODEL_PATH = Path("models/film_net_fp32.pt")


def test_run_patch_size_ablation_requires_nc_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()  # no .nc files inside

    with pytest.raises(SystemExit, match="No .nc files found"):
        run_patch_size_ablation(raw_dir, [128], MODEL_PATH, tmp_path / "out")
