import numpy as np

from src.eval.metrics import psnr, ssim


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
