"""PyTorch Dataset over extracted (t-1, t, t+1) satellite patch triplets.

Reuses the exact same grayscale-to-RGB replication and [0,1] normalization
convention as `film_interpolate.py`'s inference path, so a model trained
here sees the same input distribution it'll see at eval/inference time.
Patches are already a fixed size (a multiple of 64, see
`extract_triplets.py`), so no pad/crop-to-alignment is needed here --
that's only required for inference on arbitrary crop sizes.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _to_rgb_tensor(gray: np.ndarray) -> torch.Tensor:
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)


class TripletDataset(Dataset):
    def __init__(self, triplets_dir: Path):
        self.triplet_dirs = sorted(
            d for d in Path(triplets_dir).iterdir()
            if d.is_dir() and (d / "t.png").exists()
        )
        if not self.triplet_dirs:
            raise ValueError(f"No triplets found in {triplets_dir}")

    def __len__(self) -> int:
        return len(self.triplet_dirs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.triplet_dirs[idx]
        frame_prev = cv2.imread(str(d / "t-1.png"), cv2.IMREAD_GRAYSCALE)
        frame_mid = cv2.imread(str(d / "t.png"), cv2.IMREAD_GRAYSCALE)
        frame_next = cv2.imread(str(d / "t+1.png"), cv2.IMREAD_GRAYSCALE)
        return _to_rgb_tensor(frame_prev), _to_rgb_tensor(frame_mid), _to_rgb_tensor(frame_next)
