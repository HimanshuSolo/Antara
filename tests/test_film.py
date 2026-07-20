from pathlib import Path

import numpy as np
import pytest

from src.deep.film_interpolate import interpolate_at, interpolate_middle_frame, interpolate_multi

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


def test_interpolate_at_half_matches_interpolate_middle_frame():
    img1 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    assert np.array_equal(interpolate_at(img1, img2, MODEL_PATH, 0.5), interpolate_middle_frame(img1, img2, MODEL_PATH))


def test_interpolate_at_rejects_t_outside_open_interval():
    img1 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    with pytest.raises(ValueError):
        interpolate_at(img1, img1, MODEL_PATH, 0.0)
    with pytest.raises(ValueError):
        interpolate_at(img1, img1, MODEL_PATH, 1.0)


def test_interpolate_multi_returns_evenly_spaced_frames():
    img1 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    frames = interpolate_multi(img1, img2, MODEL_PATH, num_frames=3)
    assert len(frames) == 3
    for frame in frames:
        assert frame.shape == img1.shape
        assert frame.dtype == img1.dtype
