from pathlib import Path

import cv2
import numpy as np
import pytest

from src.eval.evaluate_stratified import run_stratified

MODEL_PATH = Path("models/film_net_fp32.pt")


def _write_triplet(triplet_dir: Path) -> None:
    triplet_dir.mkdir(parents=True, exist_ok=True)
    for name in ("t-1.png", "t.png", "t+1.png"):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(triplet_dir / name), img)


def test_run_stratified_skips_missing_and_empty_subsets(tmp_path):
    calm_dir = tmp_path / "calm"  # missing entirely
    cyclone_dir = tmp_path / "cyclone"
    cyclone_dir.mkdir()  # exists but empty
    out_dir = tmp_path / "out"

    results = run_stratified(calm_dir, cyclone_dir, MODEL_PATH, out_dir)

    assert results == {}
    assert list(out_dir.iterdir()) == []


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)
def test_run_stratified_evaluates_both_subsets(tmp_path):
    calm_dir = tmp_path / "calm"
    cyclone_dir = tmp_path / "cyclone"
    _write_triplet(calm_dir / "triplet_0")
    _write_triplet(cyclone_dir / "triplet_0")
    out_dir = tmp_path / "out"

    results = run_stratified(calm_dir, cyclone_dir, MODEL_PATH, out_dir)

    assert set(results) == {"calm", "cyclone"}
    for methods in results.values():
        assert set(methods) == {"farneback", "film"}
        for mean_psnr, mean_ssim, mean_lpips in methods.values():
            assert isinstance(mean_psnr, float)
            assert isinstance(mean_ssim, float)
            assert isinstance(mean_lpips, float)

    assert (out_dir / "calm_farneback.csv").exists()
    assert (out_dir / "calm_film.csv").exists()
    assert (out_dir / "cyclone_farneback.csv").exists()
    assert (out_dir / "cyclone_film.csv").exists()
