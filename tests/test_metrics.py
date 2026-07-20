import math

import numpy as np
import pytest

from src.eval.metrics import lpips_distance, psnr, ssim, summarize


def test_psnr_identical_images_is_high():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    assert psnr(img, img) > 50


def test_ssim_identical_images_is_one():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    assert ssim(img, img) == 1.0


def test_psnr_and_ssim_drop_for_different_images():
    img_a = np.zeros((64, 64), dtype=np.uint8)
    img_b = np.full((64, 64), 255, dtype=np.uint8)
    assert psnr(img_a, img_b) < 5
    assert ssim(img_a, img_b) < 0.5


def _lpips(pred, target):
    try:
        return lpips_distance(pred, target)
    except Exception as exc:  # pragma: no cover -- network/backbone unavailable
        pytest.skip(f"LPIPS backbone unavailable: {exc}")


def test_lpips_identical_images_is_zero():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    assert _lpips(img, img) == pytest.approx(0.0, abs=1e-6)


def test_lpips_rises_for_different_images():
    img_a = np.zeros((64, 64), dtype=np.uint8)
    img_b = np.full((64, 64), 255, dtype=np.uint8)
    assert _lpips(img_a, img_b) > _lpips(img_a, img_a)


def test_summarize_averages_psnr_ssim_and_lpips():
    rows = [
        {"triplet": "a", "psnr": 20.0, "ssim": 0.5, "lpips": 0.3},
        {"triplet": "b", "psnr": 30.0, "ssim": 0.7, "lpips": 0.1},
    ]
    mean_psnr, mean_ssim, mean_lpips = summarize(rows)
    assert mean_psnr == 25.0
    assert mean_ssim == 0.6
    assert mean_lpips == 0.2


def test_summarize_handles_empty_rows():
    mean_psnr, mean_ssim, mean_lpips = summarize([])
    assert math.isnan(mean_psnr)
    assert math.isnan(mean_ssim)
    assert math.isnan(mean_lpips)
