"""Cyclone eye-tracking API: detects the storm eye's pixel location using
both a classical heuristic (`src/baseline/eye_detect.py`) and a trained CNN
(`src/deep/eye_detect.py`), on the real ground-truth middle frame *and* on
the FILM/Farneback-synthesized middle frame from the same gallery triplet.
Reports each detector's accuracy against the IBTrACS-derived ground truth,
and how far its detection *drifts* when run on a synthesized frame instead
of the real one -- does interpolation preserve where the storm actually
was, the question this feature exists to answer.

Surfaced on its own /track page (not /live's "Gallery" section), but reuses
`gallery.generate()`'s cache so this never reruns the Farneback/FILM
pipeline if a result for that triplet already exists.
"""
from __future__ import annotations

import base64
import csv
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.gallery import GALLERY_ITEMS, encode_png, generate
from src.baseline.eye_detect import detect_eye as classical_detect_eye
from src.deep.eye_detect import detect_eye as cnn_detect_eye
from src.eval.eye_metrics import pixel_error
from src.utils.image import load_triplet_frames

EYE_MODEL_CANDIDATES = [Path("models/eye_detect_cnn.pt")]

TRACK_ITEMS = [item for item in GALLERY_ITEMS if item["subset"] == "cyclone"]

router = APIRouter()

# keyed by f"{item_id}|{film_model_name}|{eye_model_name}" -- same
# never-changes-for-a-given-checkpoint-pair reasoning as gallery.py's cache.
_cache: dict[str, "TrackResult"] = {}


class TrackItem(BaseModel):
    id: str
    label: str
    frame_prev: str
    frame_next: str


class Detection(BaseModel):
    row: int
    col: int


class TrackResult(BaseModel):
    id: str
    ground_truth: Detection | None
    classical_real: Detection
    classical_farneback: Detection
    classical_film: Detection
    cnn_real: Detection
    cnn_farneback: Detection
    cnn_film: Detection
    classical_accuracy_px: float | None
    cnn_accuracy_px: float | None
    classical_farneback_drift_px: float
    classical_film_drift_px: float
    cnn_farneback_drift_px: float
    cnn_film_drift_px: float
    annotated_real: str
    annotated_farneback: str
    annotated_film: str


def resolve_eye_model_path() -> Path:
    for candidate in EYE_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise HTTPException(
        status_code=503,
        detail=(
            "No eye-detection checkpoint found in models/ -- run "
            "`.venv/bin/python -m src.deep.eye_detect_train` first."
        ),
    )


def _find_item(item_id: str) -> dict:
    for item in TRACK_ITEMS:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"Unknown trackable item '{item_id}'")


def _load_ground_truth(item: dict) -> tuple[int, int] | None:
    labels_path = item["dir"].parent / "eye_labels.csv"
    if not labels_path.exists():
        return None
    with labels_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["triplet"] == item["dir"].name and row["frame"] == "t":
                return int(row["row"]), int(row["col"])
    return None


def _decode_png(data_uri: str) -> np.ndarray:
    _header, b64data = data_uri.split(",", 1)
    buf = np.frombuffer(base64.b64decode(b64data), dtype=np.uint8)
    decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise ValueError("Failed to decode PNG data URI")
    return decoded


def _annotate(frame: np.ndarray, points: list[tuple[tuple[int, int], tuple[int, int, int]]]) -> str:
    """Draw a small crosshair marker at each (row, col) point in its given
    BGR color, on a 3-channel copy of `frame` (so markers are visible in
    color against the grayscale background)."""
    canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for (row, col), color in points:
        cv2.drawMarker(canvas, (col, row), color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
    return encode_png(canvas)


@router.get("/api/track", response_model=list[TrackItem])
def list_track_items() -> list[TrackItem]:
    items = []
    for raw_item in TRACK_ITEMS:
        item = cast(dict, raw_item)
        frames = load_triplet_frames(item["dir"])
        if frames is None:
            continue
        frame_prev, _frame_mid, frame_next = frames
        items.append(
            TrackItem(
                id=item["id"],
                label=item["label"],
                frame_prev=encode_png(frame_prev),
                frame_next=encode_png(frame_next),
            )
        )
    return items


@router.post("/api/track/{item_id}/detect", response_model=TrackResult)
def detect(item_id: str) -> TrackResult:
    item = _find_item(item_id)
    eye_model_path = resolve_eye_model_path()

    generate_result = generate(item_id)  # reuses gallery.py's own cache

    cache_key = f"{item_id}|{generate_result.model}|{eye_model_path.name}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    frames = load_triplet_frames(item["dir"])
    if frames is None:
        raise HTTPException(status_code=404, detail=f"Triplet data missing for '{item_id}'")
    _frame_prev, frame_mid, _frame_next = frames

    farneback_mid = _decode_png(generate_result.farneback_mid)
    film_mid = _decode_png(generate_result.film_mid)

    classical_real = classical_detect_eye(frame_mid)
    classical_farneback = classical_detect_eye(farneback_mid)
    classical_film = classical_detect_eye(film_mid)

    cnn_real = cnn_detect_eye(frame_mid, eye_model_path)
    cnn_farneback = cnn_detect_eye(farneback_mid, eye_model_path)
    cnn_film = cnn_detect_eye(film_mid, eye_model_path)

    ground_truth = _load_ground_truth(item)
    gt_markers = [(ground_truth, (0, 255, 0))] if ground_truth else []

    result = TrackResult(
        id=item_id,
        ground_truth=Detection(row=ground_truth[0], col=ground_truth[1]) if ground_truth else None,
        classical_real=Detection(row=classical_real[0], col=classical_real[1]),
        classical_farneback=Detection(row=classical_farneback[0], col=classical_farneback[1]),
        classical_film=Detection(row=classical_film[0], col=classical_film[1]),
        cnn_real=Detection(row=cnn_real[0], col=cnn_real[1]),
        cnn_farneback=Detection(row=cnn_farneback[0], col=cnn_farneback[1]),
        cnn_film=Detection(row=cnn_film[0], col=cnn_film[1]),
        classical_accuracy_px=round(pixel_error(classical_real, ground_truth), 2) if ground_truth else None,
        cnn_accuracy_px=round(pixel_error(cnn_real, ground_truth), 2) if ground_truth else None,
        classical_farneback_drift_px=round(pixel_error(classical_farneback, classical_real), 2),
        classical_film_drift_px=round(pixel_error(classical_film, classical_real), 2),
        cnn_farneback_drift_px=round(pixel_error(cnn_farneback, cnn_real), 2),
        cnn_film_drift_px=round(pixel_error(cnn_film, cnn_real), 2),
        annotated_real=_annotate(
            frame_mid, [(classical_real, (0, 165, 255)), (cnn_real, (255, 0, 0)), *gt_markers]
        ),
        annotated_farneback=_annotate(
            farneback_mid, [(classical_farneback, (0, 165, 255)), (cnn_farneback, (255, 0, 0))]
        ),
        annotated_film=_annotate(film_mid, [(classical_film, (0, 165, 255)), (cnn_film, (255, 0, 0))]),
    )
    _cache[cache_key] = result
    return result
