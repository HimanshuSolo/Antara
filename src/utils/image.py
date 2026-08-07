"""Small image-array helpers shared across modules that otherwise each
reimplement the same conversion.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def replicate_to_rgb01(gray: np.ndarray) -> np.ndarray:
    """Replicate a single-channel grayscale image to 3 identical channels,
    normalized to [0, 1] float32. Used wherever a network pretrained on RGB
    video (FILM, LPIPS' AlexNet backbone) needs to be fed a single-band
    satellite IR image.
    """
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 255.0


def load_triplet_frames(triplet_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load the (t-1, t, t+1) grayscale frames of a triplet directory, or
    None if any of the three PNGs is missing/unreadable. Callers decide
    whether a missing frame should be skipped or raised on.
    """
    p_prev, p_mid, p_next = triplet_dir / "t-1.png", triplet_dir / "t.png", triplet_dir / "t+1.png"
    if not (p_prev.exists() and p_mid.exists() and p_next.exists()):
        return None
    frame_prev = cv2.imread(str(p_prev), cv2.IMREAD_GRAYSCALE)
    frame_mid = cv2.imread(str(p_mid), cv2.IMREAD_GRAYSCALE)
    frame_next = cv2.imread(str(p_next), cv2.IMREAD_GRAYSCALE)
    if frame_prev is None or frame_mid is None or frame_next is None:
        return None
    return frame_prev, frame_mid, frame_next

