import numpy as np

from src.baseline.farneback_interpolate import interpolate_middle_frame
from src.eval.metrics import psnr


def test_interpolate_shape_matches_input():
    img1 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    pred = interpolate_middle_frame(img1, img2)
    assert pred.shape == img1.shape
    assert pred.dtype == img1.dtype


def test_interpolate_of_static_scene_reconstructs_it():
    # a non-moving scene: flow is ~zero everywhere, so the "interpolated"
    # frame should closely match the (identical) input frames.
    img = np.zeros((64, 64), dtype=np.uint8)
    img[20:40, 20:40] = 200
    pred = interpolate_middle_frame(img, img)
    assert psnr(pred, img) > 30


def test_interpolate_of_translating_square_lands_near_midpoint():
    # a square moving 10px right between frames: the interpolated frame
    # should place it roughly 5px right of the starting position.
    frame_prev = np.zeros((64, 64), dtype=np.uint8)
    frame_prev[20:40, 10:30] = 200
    frame_next = np.zeros((64, 64), dtype=np.uint8)
    frame_next[20:40, 20:40] = 200

    expected_mid = np.zeros((64, 64), dtype=np.uint8)
    expected_mid[20:40, 15:35] = 200

    pred = interpolate_middle_frame(frame_prev, frame_next)
    assert psnr(pred, expected_mid) > psnr(frame_prev, expected_mid)
