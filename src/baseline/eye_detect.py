"""Baseline: classical heuristic for locating a tropical cyclone's eye in a
band-13 IR patch.

The eye is a small, warm, clear-air spot at the storm's center, surrounded
by a much colder, higher cloud shield -- colder cloud tops emit less
radiance, so in this project's uint8 mapping (`extract_triplets.to_uint8`,
which does *not* invert radiance) the eye shows up as a locally *bright*
blob inside a dark spiral, visible by eye in the gallery's cyclone
triplets.

The naive version of this -- just take the single brightest pixel in the
whole patch -- fails badly in practice: warm clear-sky ocean far from the
storm is often brighter than the eye itself, so a global search latches
onto background instead. This two-stage version fixes that by first
locating the storm's cold cloud shield (the largest connected blob of cold
pixels), then searching for the brightest spot only within that shield's
vicinity -- the same "the eye is bright *and* inside the cold ring, not
just bright" constraint a human interpreting the image would apply.

Even so, this is the "no training" reference every learned eye detector in
this project must beat, mirroring `farneback_interpolate.py`'s role for
frame interpolation -- it's expected to degrade when the eye is
obscured by high cirrus or the cloud shield is irregular, which is exactly
the gap a trained detector (`src/deep/eye_detect.py`) is meant to close.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_eye(
    frame: np.ndarray,
    blur_ksize: int = 9,
    cold_percentile: float = 30.0,
    search_radius: int = 60,
) -> tuple[int, int]:
    """Return the (row, col) pixel location of the storm eye in `frame`.

    Stage 1: threshold the blurred frame at `cold_percentile` and take the
    centroid of the largest connected cold blob -- an estimate of the
    storm's cloud-shield center. Stage 2: within `search_radius` pixels of
    that centroid, return the brightest pixel -- the eye.

    Falls back to a plain brightest-pixel search over the whole frame if no
    cold blob is found at all (e.g. a mostly-clear scene with no storm).
    """
    blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)

    threshold = np.percentile(blurred, cold_percentile)
    mask = (blurred <= threshold).astype(np.uint8)
    n_components, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if n_components <= 1:
        row, col = np.unravel_index(np.argmax(blurred), blurred.shape)
        return int(row), int(col)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    center_col, center_row = centroids[largest]

    # A circular (not square) constraint: a square crop would let a decoy
    # bright spot at, say, (radius, radius) diagonally away sneak in even
    # though it's farther from the center than a spot at (radius, 0).
    rows, cols = np.indices(blurred.shape)
    within_radius = (rows - center_row) ** 2 + (cols - center_col) ** 2 <= search_radius**2
    candidate = np.where(within_radius, blurred, 0)

    row, col = np.unravel_index(np.argmax(candidate), candidate.shape)
    return int(row), int(col)
