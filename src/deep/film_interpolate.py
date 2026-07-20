"""Deep frame interpolation using Google's FILM model (Frame Interpolation
for Large Motion), via the TorchScript port at
https://github.com/dajes/frame-interpolation-pytorch.

Unlike the Farneback baseline, this network was pretrained (on ordinary
video) not just to estimate optical flow but to *learn* how to blend and
repair the warped result -- the exact "learned synthesis" piece the
classical baseline lacks. This module is used first with the pretrained
checkpoint as-is (no fine-tuning) to check the architecture transfers to
satellite imagery at all, before any satellite-specific training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.utils.image import replicate_to_rgb01

# the network downsamples internally; H/W must be a multiple of this,
# matching the padding convention used by the original FILM port.
_ALIGN = 64

_model_cache: dict[tuple[str, str], torch.jit.ScriptModule] = {}


def _load_model(model_path: Path, device: str) -> torch.jit.ScriptModule:
    key = (str(model_path), device)
    if key not in _model_cache:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def _pad_to_align(img: np.ndarray, align: int = _ALIGN) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    pad_h = (align - h % align) if h % align else 0
    pad_w = (align - w % align) if w % align else 0
    top, left = pad_h // 2, pad_w // 2
    crop = (top, left, top + h, left + w)
    padded = np.pad(img, ((top, pad_h - top), (left, pad_w - left), (0, 0)), mode="constant")
    return padded, crop


def _to_tensor(gray: np.ndarray) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    # FILM is pretrained on RGB video -- replicate the single IR channel
    # to 3 identical channels so the pretrained conv filters (which expect
    # 3 input channels) can be applied at all.
    rgb = replicate_to_rgb01(gray)
    padded, crop = _pad_to_align(rgb)
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
    return tensor, crop


def interpolate_at(
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    model_path: Path,
    t: float,
    device: str | None = None,
) -> np.ndarray:
    """Synthesize the frame at arbitrary normalized time `t` in (0, 1)
    between two grayscale satellite patches. FILM is trained as a
    continuous-time interpolator, not just a midpoint one, so `t` need not
    be 0.5 -- `interpolate_middle_frame` below is the t=0.5 special case
    every other script in this project calls.
    """
    if not 0.0 < t < 1.0:
        raise ValueError(f"t must be strictly between 0 and 1, got {t}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device)

    x0, crop = _to_tensor(frame_prev)
    x1, _ = _to_tensor(frame_next)
    x0, x1 = x0.to(device), x1.to(device)
    dt = x0.new_full((1, 1), t)

    with torch.no_grad():
        pred = model(x0, x1, dt).clamp(0, 1)

    # network output channels aren't guaranteed identical even though the
    # input channels were replicated -- average back down to grayscale
    # rather than just taking one channel.
    pred = pred[0].mean(dim=0).cpu().numpy()
    top, left, bottom, right = crop
    pred = pred[top:bottom, left:right]
    return (pred * 255).astype(np.uint8)


def interpolate_middle_frame(
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    model_path: Path,
    device: str | None = None,
) -> np.ndarray:
    """Synthesize the frame halfway between two grayscale satellite patches.

    Same (prev, next) -> mid contract as
    `src.baseline.farneback_interpolate.interpolate_middle_frame`, so it
    drops into the same evaluation flow.
    """
    return interpolate_at(frame_prev, frame_next, model_path, 0.5, device=device)


def interpolate_multi(
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    model_path: Path,
    num_frames: int = 3,
    device: str | None = None,
) -> list[np.ndarray]:
    """Synthesize `num_frames` evenly-spaced intermediate frames -- e.g.
    num_frames=3 gives frames at t=0.25/0.5/0.75, turning one real frame
    gap into 4x the temporal resolution instead of FILM's usual single
    midpoint. Per docs/PLAN.md's multi-frame stretch goal.
    """
    return [
        interpolate_at(frame_prev, frame_next, model_path, i / (num_frames + 1), device=device)
        for i in range(1, num_frames + 1)
    ]
