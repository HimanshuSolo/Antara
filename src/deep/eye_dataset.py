"""PyTorch Dataset over triplet frames labeled with storm-eye pixel
coordinates (`eye_labels.csv`, as written by `build_eye_labels.py`) --
supervision for the trained eye detector in `eye_detect_train.py`.

Unlike `dataset.py`'s TripletDataset (three frames per triplet feeding
FILM's two-frame-in-one-frame-out interpolation), this is one row per
labeled frame: each (image, eye position) pair is an independent training
example, not tied to its triplet siblings.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class EyeDataset(Dataset):
    def __init__(self, triplets_dir: Path, labels_csv: Path | None = None):
        self.triplets_dir = Path(triplets_dir)
        labels_csv = labels_csv or self.triplets_dir / "eye_labels.csv"
        with open(labels_csv, newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"No eye labels found in {labels_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        img_path = self.triplets_dir / row["triplet"] / f"{row['frame']}.png"
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Missing frame image at {img_path}")

        h, w = gray.shape
        image = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)
        # Normalized to [0, 1] rather than raw pixel coords, so the target
        # scale doesn't depend on this dataset's patch size (256px here,
        # but not guaranteed to match every triplets dir).
        target = torch.tensor([int(row["row"]) / h, int(row["col"]) / w], dtype=torch.float32)
        return image, target
