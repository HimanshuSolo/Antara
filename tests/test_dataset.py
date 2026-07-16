from pathlib import Path

import pytest

from src.deep.dataset import TripletDataset

TRIPLETS_DIR = Path("data/processed/triplets")

pytestmark = pytest.mark.skipif(
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
