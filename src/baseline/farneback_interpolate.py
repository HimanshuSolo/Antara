"""Baseline: classical optical-flow-based frame interpolation using
Farneback dense optical flow (OpenCV).

This is the "traditional method" the ISRO problem statement calls out as
inadequate for fast, non-linear cloud motion -- it's the reference every
learned method in this project must beat, especially on the cyclone/storm
evaluation subset.
"""
from __future__ import annotations

import cv2
import numpy as np


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def interpolate_middle_frame(frame_prev: np.ndarray, frame_next: np.ndarray) -> np.ndarray:
    """Synthesize the frame halfway between `frame_prev` and `frame_next`.

    Computes bidirectional dense flow, scales each flow field by 0.5, and
    warps both frames toward the midpoint, then blends -- averaging fills
    in occlusion holes a single-direction warp would leave uncovered.
    """
    # cv2's stubs don't have an overload for flow=None (compute a fresh
    # field) even though it's valid and exactly what the docs recommend.
    flow_fwd = cv2.calcOpticalFlowFarneback(
        frame_prev, frame_next, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )  # type: ignore[call-overload]
    flow_bwd = cv2.calcOpticalFlowFarneback(
        frame_next, frame_prev, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )  # type: ignore[call-overload]

    warped_from_prev = _warp(frame_prev, flow_fwd * 0.5)
    warped_from_next = _warp(frame_next, flow_bwd * 0.5)

    return cv2.addWeighted(warped_from_prev, 0.5, warped_from_next, 0.5, 0)
