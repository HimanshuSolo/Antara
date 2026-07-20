import csv
from pathlib import Path

import pytest

from src.eval.plot_stratified import build_stratified_figure, load_stratified_means, save_stratified_figure


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["triplet", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)


def _seed_results_dir(results_dir: Path) -> None:
    _write_csv(
        results_dir / "calm_farneback.csv",
        [
            {"triplet": "a", "psnr": 20.0, "ssim": 0.5, "lpips": 0.4},
            {"triplet": "b", "psnr": 24.0, "ssim": 0.7, "lpips": 0.2},
        ],
    )
    _write_csv(results_dir / "calm_film.csv", [{"triplet": "a", "psnr": 30.0, "ssim": 0.9, "lpips": 0.1}])
    _write_csv(results_dir / "cyclone_farneback.csv", [{"triplet": "c", "psnr": 10.0, "ssim": 0.2, "lpips": 0.6}])
    _write_csv(results_dir / "cyclone_film.csv", [{"triplet": "c", "psnr": 28.0, "ssim": 0.85, "lpips": 0.15}])


def test_load_stratified_means_averages_each_csv(tmp_path):
    _seed_results_dir(tmp_path)

    means = load_stratified_means(tmp_path)

    assert means["calm"]["farneback"] == pytest.approx((22.0, 0.6, 0.3))
    assert means["calm"]["film"] == pytest.approx((30.0, 0.9, 0.1))
    assert means["cyclone"]["farneback"] == pytest.approx((10.0, 0.2, 0.6))
    assert means["cyclone"]["film"] == pytest.approx((28.0, 0.85, 0.15))


def test_load_stratified_means_missing_csv_is_nan(tmp_path):
    means = load_stratified_means(tmp_path)
    import math

    assert math.isnan(means["calm"]["farneback"][0])


def test_build_stratified_figure_has_three_axes(tmp_path):
    _seed_results_dir(tmp_path)
    means = load_stratified_means(tmp_path)

    fig = build_stratified_figure(means)
    try:
        assert len(fig.axes) == 3
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_save_stratified_figure_writes_png(tmp_path):
    _seed_results_dir(tmp_path)
    out_path = tmp_path / "out" / "comparison.png"

    written = save_stratified_figure(tmp_path, out_path)

    assert written == out_path
    assert out_path.exists()
