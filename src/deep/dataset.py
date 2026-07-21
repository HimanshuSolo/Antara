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

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.image import load_triplet_frames, replicate_to_rgb01


def _to_rgb_tensor(gray: np.ndarray) -> torch.Tensor:
    rgb = replicate_to_rgb01(gray)
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
        frames = load_triplet_frames(d)
        if frames is None:
            raise FileNotFoundError(f"Missing t-1/t/t+1.png under {d}")
        frame_prev, frame_mid, frame_next = frames
        return _to_rgb_tensor(frame_prev), _to_rgb_tensor(frame_mid), _to_rgb_tensor(frame_next)
