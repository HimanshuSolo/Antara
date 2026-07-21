from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.deep.continual_finetune import continual_update

MODEL_PATH = Path("models/film_net_fp32.pt")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)


def _write_triplets(pool_dir: Path, count: int, start: int = 0) -> None:
    for i in range(start, start + count):
        d = pool_dir / f"triplet_{i:04d}"
        d.mkdir(parents=True)
        for name in ("t-1.png", "t.png", "t+1.png"):
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(d / name), img)


def test_continual_update_trains_only_on_the_most_recent_window(tmp_path):
    pool_dir = tmp_path / "pool"
    _write_triplets(pool_dir, count=6)
    out_path = tmp_path / "updated.pt"

    history = continual_update(
        MODEL_PATH, pool_dir, out_path, window_size=3, epochs=1, batch_size=2, val_fraction=0.0
    )

    assert len(history["train_loss"]) == 1
    window_dir = tmp_path / "updated_window"
    assert sorted(p.name for p in window_dir.iterdir()) == ["triplet_0003", "triplet_0004", "triplet_0005"]

    assert out_path.exists()
    reloaded = torch.jit.load(str(out_path), map_location="cpu")
    img1 = torch.rand(1, 3, 64, 64)
    img2 = torch.rand(1, 3, 64, 64)
    dt = torch.full((1, 1), 0.5)
    pred = reloaded(img1, img2, dt)
    assert pred.shape == img1.shape


def test_continual_update_uses_the_whole_pool_when_smaller_than_the_window(tmp_path):
    pool_dir = tmp_path / "pool"
    _write_triplets(pool_dir, count=2)
    out_path = tmp_path / "updated.pt"

    continual_update(MODEL_PATH, pool_dir, out_path, window_size=100, epochs=1, batch_size=2, val_fraction=0.0)

    window_dir = tmp_path / "updated_window"
    assert sorted(p.name for p in window_dir.iterdir()) == ["triplet_0000", "triplet_0001"]


def test_continual_update_window_stays_bounded_across_repeated_calls(tmp_path):
    # continual_update() is meant to be called again and again as new
    # triplets stream in -- the window must stay exactly window_size
    # entries, not accumulate stale ones from earlier calls.
    pool_dir = tmp_path / "pool"
    _write_triplets(pool_dir, count=3)
    out_path = tmp_path / "updated.pt"
    window_dir = tmp_path / "updated_window"

    continual_update(MODEL_PATH, pool_dir, out_path, window_size=2, epochs=1, batch_size=2, val_fraction=0.0)
    assert sorted(p.name for p in window_dir.iterdir()) == ["triplet_0001", "triplet_0002"]

    # a later "arrival" of more triplets -- the window should shift forward,
    # not grow to include the earlier call's now-stale entries
    _write_triplets(pool_dir, count=3, start=3)
    continual_update(MODEL_PATH, pool_dir, out_path, window_size=2, epochs=1, batch_size=2, val_fraction=0.0)

    assert sorted(p.name for p in window_dir.iterdir()) == ["triplet_0004", "triplet_0005"]


def test_continual_update_raises_when_pool_is_empty(tmp_path):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()

    with pytest.raises(ValueError, match="No triplets found"):
        continual_update(MODEL_PATH, pool_dir, tmp_path / "updated.pt")


def test_continual_update_reports_test_metrics_when_test_dir_given(tmp_path):
    pool_dir = tmp_path / "pool"
    _write_triplets(pool_dir, count=3)
    test_dir = tmp_path / "test"
    _write_triplets(test_dir, count=2)
    out_path = tmp_path / "updated.pt"
    test_csv = tmp_path / "results.csv"

    history = continual_update(
        MODEL_PATH, pool_dir, out_path, window_size=3, epochs=1, batch_size=2, val_fraction=0.0,
        test_dir=test_dir, test_csv=test_csv,
    )

    assert "test_metrics" in history
    mean_psnr, mean_ssim, mean_lpips = history["test_metrics"]
    assert mean_psnr == mean_psnr  # not NaN
    assert test_csv.exists()


def test_continual_update_omits_test_metrics_when_test_dir_not_given(tmp_path):
    pool_dir = tmp_path / "pool"
    _write_triplets(pool_dir, count=2)

    history = continual_update(MODEL_PATH, pool_dir, tmp_path / "updated.pt", window_size=2, epochs=1, batch_size=2)

    assert "test_metrics" not in history
