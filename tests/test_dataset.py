from pathlib import Path

import cv2
import numpy as np
import pytest

from src.deep.dataset import TripletDataset

TRIPLETS_DIR = Path("data/processed/triplets")


def _write_triplet(triplet_dir: Path) -> None:
    triplet_dir.mkdir(parents=True, exist_ok=True)
    for name in ("t-1.png", "t.png", "t+1.png"):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(triplet_dir / name), img)


def test_dataset_loads_synthetic_triplets(tmp_path):
    _write_triplet(tmp_path / "triplet_0000")

    ds = TripletDataset(tmp_path)
    assert len(ds) == 1

    frame_prev, frame_mid, frame_next = ds[0]
    assert frame_prev.shape == frame_mid.shape == frame_next.shape
    assert frame_prev.shape[0] == 3  # replicated to RGB
    assert frame_prev.dtype.is_floating_point
    assert 0.0 <= frame_prev.min() and frame_prev.max() <= 1.0


def test_dataset_ignores_dirs_missing_the_mid_frame(tmp_path):
    _write_triplet(tmp_path / "triplet_0000")
    incomplete = tmp_path / "triplet_0001"
    incomplete.mkdir()
    cv2.imwrite(str(incomplete / "t-1.png"), np.zeros((64, 64), dtype=np.uint8))
    # t.png deliberately missing -- shouldn't be counted as a triplet

    ds = TripletDataset(tmp_path)

    assert len(ds) == 1


def test_dataset_raises_when_no_triplets_found(tmp_path):
    with pytest.raises(ValueError, match="No triplets found"):
        TripletDataset(tmp_path)


@pytest.mark.skipif(
    not TRIPLETS_DIR.exists(),
    reason="no extracted triplets -- run fetch_goes.py + extract_triplets.py first",
)
def test_dataset_loads_real_triplets():
    ds = TripletDataset(TRIPLETS_DIR)
    assert len(ds) > 0

    frame_prev, frame_mid, frame_next = ds[0]
    assert frame_prev.shape == frame_mid.shape == frame_next.shape
    assert frame_prev.shape[0] == 3  # replicated to RGB
    assert frame_prev.dtype.is_floating_point
    assert 0.0 <= frame_prev.min() and frame_prev.max() <= 1.0
