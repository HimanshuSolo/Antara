import math

from src.eval.evaluate_stratified import summarize


def test_summarize_averages_psnr_and_ssim():
    rows = [
        {"triplet": "a", "psnr": 20.0, "ssim": 0.5},
        {"triplet": "b", "psnr": 30.0, "ssim": 0.7},
    ]
    mean_psnr, mean_ssim = summarize(rows)
    assert mean_psnr == 25.0
    assert mean_ssim == 0.6


def test_summarize_handles_empty_rows():
    mean_psnr, mean_ssim = summarize([])
    assert math.isnan(mean_psnr)
    assert math.isnan(mean_ssim)
