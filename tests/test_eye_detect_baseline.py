import cv2
import numpy as np

from src.baseline.eye_detect import detect_eye


def _cold_shield_with_eye(shape=(128, 128), shield_center=(64, 64), shield_radius=40, eye=(64, 64)):
    """A synthetic frame: a large dark disk (the cold cloud shield) at
    mid-gray background, with a small bright dot at `eye`."""
    img = np.full(shape, 150, dtype=np.uint8)
    cv2.circle(img, (shield_center[1], shield_center[0]), shield_radius, 20, -1)
    cv2.circle(img, (eye[1], eye[0]), 3, 255, -1)
    return img


def test_detect_eye_finds_bright_spot_inside_the_cold_shield():
    img = _cold_shield_with_eye(eye=(60, 70))
    row, col = detect_eye(img)
    assert abs(row - 60) <= 3
    assert abs(col - 70) <= 3


def test_detect_eye_ignores_a_brighter_spot_outside_the_shield():
    # A brighter, larger spot far from the storm (e.g. warm clear-sky
    # ocean) that a naive global-brightest-pixel search would latch onto
    # instead of the dimmer-but-correctly-located eye.
    img = _cold_shield_with_eye(eye=(64, 64))
    cv2.circle(img, (10, 10), 5, 255, -1)

    row, col = detect_eye(img)

    assert abs(row - 64) <= 3
    assert abs(col - 64) <= 3


def test_detect_eye_falls_back_to_global_search_with_no_cold_shield():
    img = np.full((64, 64), 100, dtype=np.uint8)
    img[20, 30] = 255

    row, col = detect_eye(img, cold_percentile=0.0)

    assert (row, col) == (20, 30)
