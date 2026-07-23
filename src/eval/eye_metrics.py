"""Pixel-distance metric for cyclone eye-position detectors -- how far a
detected (row, col) is from another (row, col), whether that's ground
truth (detector accuracy) or the same detector's output on a different
frame (real-vs-synthesized position drift).
"""
from __future__ import annotations


def pixel_error(pred: tuple[float, float], target: tuple[float, float]) -> float:
    """Euclidean pixel distance between two (row, col) points."""
    return ((pred[0] - target[0]) ** 2 + (pred[1] - target[1]) ** 2) ** 0.5
