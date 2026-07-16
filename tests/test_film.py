from pathlib import Path

import numpy as np
import pytest

from src.deep.film_interpolate import interpolate_middle_frame

MODEL_PATH = Path("models/film_net_fp32.pt")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)


def test_interpolate_shape_matches_input():
    img1 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    pred = interpolate_middle_frame(img1, img2, MODEL_PATH)
    assert pred.shape == img1.shape
    assert pred.dtype == img1.dtype


def test_interpolate_handles_non_aligned_size():
    # 70x70 isn't a multiple of the network's 64-alignment requirement --
    # this exercises the pad/crop path in _pad_to_align.
    img1 = np.random.randint(0, 255, (70, 70), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (70, 70), dtype=np.uint8)
    pred = interpolate_middle_frame(img1, img2, MODEL_PATH)
    assert pred.shape == img1.shape
