"""Image-quality metrics for comparing a generated interpolated frame
against the real held-out ground-truth frame.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return peak_signal_noise_ratio(target, pred, data_range=255)


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return structural_similarity(target, pred, data_range=255)
