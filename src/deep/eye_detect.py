"""Inference for the trained CNN eye detector (`eye_detect_model.py`,
trained by `eye_detect_train.py`) -- the learned counterpart to
`src/baseline/eye_detect.py`'s classical heuristic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.deep.eye_detect_model import EyeDetectCNN

_model_cache: dict[str, EyeDetectCNN] = {}


def _load_model(model_path: Path) -> EyeDetectCNN:
    key = str(model_path)
    if key not in _model_cache:
        model = EyeDetectCNN()
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def detect_eye(frame: np.ndarray, model_path: Path) -> tuple[int, int]:
    """Return the (row, col) pixel location of the storm eye in `frame`, as
    predicted by the trained CNN."""
    model = _load_model(model_path)
    h, w = frame.shape
    tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor)[0]
    row_frac, col_frac = pred.tolist()
    return int(round(row_frac * h)), int(round(col_frac * w))
