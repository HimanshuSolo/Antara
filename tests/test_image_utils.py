import cv2
import numpy as np

from src.utils.image import load_triplet_frames, replicate_to_rgb01


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


def _write_triplet(triplet_dir):
    triplet_dir.mkdir(parents=True, exist_ok=True)
    for name in ("t-1.png", "t.png", "t+1.png"):
        cv2.imwrite(str(triplet_dir / name), np.zeros((8, 8), dtype=np.uint8))


def test_load_triplet_frames_returns_all_three_frames(tmp_path):
    _write_triplet(tmp_path)

    frames = load_triplet_frames(tmp_path)

    assert frames is not None
    frame_prev, frame_mid, frame_next = frames
    assert frame_prev.shape == frame_mid.shape == frame_next.shape == (8, 8)


def test_load_triplet_frames_returns_none_when_a_frame_is_missing(tmp_path):
    cv2.imwrite(str(tmp_path / "t-1.png"), np.zeros((8, 8), dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "t.png"), np.zeros((8, 8), dtype=np.uint8))
    # t+1.png deliberately missing

    assert load_triplet_frames(tmp_path) is None
