from pathlib import Path

import cv2
import numpy as np
import pytest

from src.eval.plot_comparison import build_comparison_figure, save_comparisons

MODEL_PATH = Path("models/film_net_fp32.pt")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)


def _write_triplet(triplet_dir: Path) -> None:
    triplet_dir.mkdir(parents=True, exist_ok=True)
    for name in ("t-1.png", "t.png", "t+1.png"):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(triplet_dir / name), img)


def test_build_comparison_figure_has_five_panels(tmp_path):
    triplet_dir = tmp_path / "triplet_0"
    _write_triplet(triplet_dir)

    fig = build_comparison_figure(triplet_dir, MODEL_PATH)
    try:
        assert len(fig.axes) == 5
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_save_comparisons_writes_one_png_per_triplet(tmp_path):
    triplets_dir = tmp_path / "triplets"
    for i in range(2):
        _write_triplet(triplets_dir / f"triplet_{i}")
    out_dir = tmp_path / "out"

    written = save_comparisons(triplets_dir, MODEL_PATH, out_dir)

    assert len(written) == 2
    assert all(p.exists() for p in written)
