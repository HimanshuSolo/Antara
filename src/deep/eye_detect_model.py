"""A small CNN that regresses a tropical cyclone's eye pixel location
directly from a single band-13 IR patch -- the trained counterpart to
`src/baseline/eye_detect.py`'s classical heuristic, and the thing this
project's stratified evaluation approach exists to test: does a learned
model hold up where a hand-tuned heuristic starts to break down.

Deliberately small and trained from scratch (not a pretrained backbone,
unlike FILM) -- the input domain (single-channel IR patches) is a poor
match for ImageNet-style pretraining, and a handful of conv blocks is
already enough capacity for a single-object localization task.
"""
from __future__ import annotations

import torch
from torch import nn


class EyeDetectCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (N, 1, H, W) in [0, 1]. Output: (N, 2) -- (row, col)
        fraction of (H, W), each in [0, 1] via a final sigmoid."""
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return torch.sigmoid(self.head(pooled))
