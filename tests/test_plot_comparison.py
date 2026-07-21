from pathlib import Path

import cv2
import numpy as np
import pytest

from src.eval.plot_comparison import build_comparison_figure, list_triplet_dirs, load_triplet, save_comparisons

MODEL_PATH = Path("models/film_net_fp32.pt")

requires_film_checkpoint = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)


def _write_triplet(triplet_dir: Path) -> None:
    triplet_dir.mkdir(parents=True, exist_ok=True)
    for name in ("t-1.png", "t.png", "t+1.png"):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(triplet_dir / name), img)


@requires_film_checkpoint
def test_build_comparison_figure_has_five_panels(tmp_path):
    triplet_dir = tmp_path / "triplet_0"
    _write_triplet(triplet_dir)

    fig = build_comparison_figure(triplet_dir, MODEL_PATH)
    try:
        assert len(fig.axes) == 5
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


@requires_film_checkpoint
def test_save_comparisons_writes_one_png_per_triplet(tmp_path):
    triplets_dir = tmp_path / "triplets"
    for i in range(2):
        _write_triplet(triplets_dir / f"triplet_{i}")
    out_dir = tmp_path / "out"

    written = save_comparisons(triplets_dir, MODEL_PATH, out_dir)

    assert len(written) == 2
    assert all(p.exists() for p in written)


def test_list_triplet_dirs_sorts_and_skips_files(tmp_path):
    (tmp_path / "triplet_1").mkdir()
    (tmp_path / "triplet_0").mkdir()
    (tmp_path / "not_a_triplet.txt").write_text("")

    dirs = list_triplet_dirs(tmp_path)

    assert [d.name for d in dirs] == ["triplet_0", "triplet_1"]


def test_list_triplet_dirs_respects_limit(tmp_path):
    for i in range(3):
        (tmp_path / f"triplet_{i}").mkdir()

    dirs = list_triplet_dirs(tmp_path, limit=2)

    assert [d.name for d in dirs] == ["triplet_0", "triplet_1"]


def test_load_triplet_raises_when_a_frame_is_missing(tmp_path):
    triplet_dir = tmp_path / "triplet_0"
    triplet_dir.mkdir()
    cv2.imwrite(str(triplet_dir / "t-1.png"), np.zeros((8, 8), dtype=np.uint8))
    cv2.imwrite(str(triplet_dir / "t.png"), np.zeros((8, 8), dtype=np.uint8))
    # t+1.png deliberately missing

    with pytest.raises(FileNotFoundError, match="Missing t-1/t/t\\+1.png"):
        load_triplet(triplet_dir)
