"""Small image-array helpers shared across modules that otherwise each
reimplement the same conversion.
"""
from __future__ import annotations

import numpy as np


def replicate_to_rgb01(gray: np.ndarray) -> np.ndarray:
    """Replicate a single-channel grayscale image to 3 identical channels,
    normalized to [0, 1] float32. Used wherever a network pretrained on RGB
    video (FILM, LPIPS' AlexNet backbone) needs to be fed a single-band
    satellite IR image.
    """
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 255.0
