from pathlib import Path

import cv2
import numpy as np
import pytest

from src.eval.plot_multiframe import build_multiframe_figure, save_multiframe

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


def test_build_multiframe_figure_has_num_frames_plus_two_panels(tmp_path):
    triplet_dir = tmp_path / "triplet_0"
    _write_triplet(triplet_dir)

    fig = build_multiframe_figure(triplet_dir, MODEL_PATH, num_frames=3)
    try:
        assert len(fig.axes) == 5  # t-1, 3 synthesized, t+1
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_save_multiframe_writes_one_png_per_triplet(tmp_path):
    triplets_dir = tmp_path / "triplets"
    for i in range(2):
        _write_triplet(triplets_dir / f"triplet_{i}")
    out_dir = tmp_path / "out"

    written = save_multiframe(triplets_dir, MODEL_PATH, out_dir, num_frames=3)

    assert len(written) == 2
    assert all(p.exists() for p in written)
