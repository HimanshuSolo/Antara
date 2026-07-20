"""Image-quality metrics for comparing a generated interpolated frame
against the real held-out ground-truth frame.
"""
from __future__ import annotations

import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return peak_signal_noise_ratio(target, pred, data_range=255)


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return structural_similarity(target, pred, data_range=255)


_lpips_model: lpips.LPIPS | None = None


def _get_lpips_model() -> lpips.LPIPS:
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex")
        _lpips_model.eval()
    return _lpips_model


def _to_lpips_tensor(gray: np.ndarray) -> torch.Tensor:
    # LPIPS' AlexNet backbone expects 3-channel input in [-1, 1] --
    # replicate the single IR channel, same trick as FILM's _to_tensor.
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2 - 1


def lpips_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Perceptual distance (lower = more similar), via an AlexNet-backed
    LPIPS network. Complements PSNR/SSIM's pixelwise comparison with one
    based on learned image features -- closer to human judgments of "does
    this look right," which matters most on the cyclone subset, where
    structurally-plausible-but-misaligned cloud detail can still score
    deceptively well on PSNR/SSIM.
    """
    model = _get_lpips_model()
    with torch.no_grad():
        dist = model(_to_lpips_tensor(pred), _to_lpips_tensor(target))
    return float(dist.item())


def summarize(rows: list[dict]) -> tuple[float, float, float]:
    """Average the psnr/ssim/lpips fields of a list of per-triplet result
    rows (as produced by evaluate_baseline.evaluate()/evaluate_film.evaluate())."""
    if not rows:
        return float("nan"), float("nan"), float("nan")
    mean_psnr = sum(r["psnr"] for r in rows) / len(rows)
    mean_ssim = sum(r["ssim"] for r in rows) / len(rows)
    mean_lpips = sum(r["lpips"] for r in rows) / len(rows)
    return mean_psnr, mean_ssim, mean_lpips
