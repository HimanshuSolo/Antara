import numpy as np

from src.utils.image import replicate_to_rgb01


def test_replicate_to_rgb01_adds_three_identical_channels():
    gray = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    rgb = replicate_to_rgb01(gray)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.float32
    assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
    assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])


def test_replicate_to_rgb01_scales_to_unit_range():
    gray = np.array([[0, 255]], dtype=np.uint8)
    rgb = replicate_to_rgb01(gray)

    assert rgb[0, 0, 0] == 0.0
    assert rgb[0, 1, 0] == 1.0
