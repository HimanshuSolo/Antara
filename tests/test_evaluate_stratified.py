import math

from src.eval.evaluate_stratified import summarize


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
