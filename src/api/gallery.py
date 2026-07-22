"""Gallery endpoint: a handful of curated real GOES-16 (t-1, t+1) pairs --
some from Hurricane Milton's 2024 cyclone-centered eval set, some from the
calm off-season set (see `src/data/build_cyclone_dataset.py` /
`build_calm_dataset.py`) -- browsable instantly with no model involved.
Synthesizing the middle frame is a separate, on-demand endpoint the
frontend calls when the user clicks "Generate", running the real
Farneback + FILM pipeline live rather than serving a pre-baked result, in
the same spirit as `src/api/live.py`. Each of these triplets also has a
real ground-truth middle frame on disk (unlike the live pipeline, where
the true midpoint doesn't exist yet), so this endpoint can report real
PSNR/SSIM/LPIPS for both methods alongside the images.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.baseline.farneback_interpolate import interpolate_middle_frame as farneback_interpolate
from src.deep.film_interpolate import interpolate_middle_frame as film_interpolate
from src.eval.metrics import lpips_distance, psnr, ssim
from src.utils.image import load_triplet_frames

GALLERY_ITEMS = [
    {
        "id": "milton-1",
        "label": "Hurricane Milton, eyewall (1)",
        "subset": "cyclone",
        "dir": Path("data/processed/triplets_cyclone/triplet_0000"),
    },
    {
        "id": "milton-2",
        "label": "Hurricane Milton, eyewall (2)",
        "subset": "cyclone",
        "dir": Path("data/processed/triplets_cyclone/triplet_0009"),
    },
    {
        "id": "milton-3",
        "label": "Hurricane Milton, eyewall (3)",
        "subset": "cyclone",
        "dir": Path("data/processed/triplets_cyclone/triplet_0020"),
    },
    {
        "id": "calm-1",
        "label": "Calm weather, off-season (1)",
        "subset": "calm",
        "dir": Path("data/processed/triplets_calm/triplet_0005"),
    },
    {
        "id": "calm-2",
        "label": "Calm weather, off-season (2)",
        "subset": "calm",
        "dir": Path("data/processed/triplets_calm/triplet_0030"),
    },
]

router = APIRouter()

# keyed by f"{item_id}|{model_name}" -- a generated result never changes
# for a given (triplet, checkpoint) pair, so there's no reason to rerun
# the pipeline once a result exists.
_cache: dict[str, "GenerateResult"] = {}


class GalleryItem(BaseModel):
    id: str
    label: str
    subset: str
    frame_prev: str
    frame_next: str


class GenerateResult(BaseModel):
    id: str
    farneback_mid: str
    film_mid: str
    ground_truth: str
    farneback_psnr: float
    farneback_ssim: float
    farneback_lpips: float
    film_psnr: float
    film_ssim: float
    film_lpips: float
    processing_seconds: float
    model: str


def encode_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG")
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def _find_item(item_id: str) -> dict:
    for item in GALLERY_ITEMS:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"Unknown gallery item '{item_id}'")


@router.get("/api/gallery", response_model=list[GalleryItem])
def list_gallery() -> list[GalleryItem]:
    items = []
    for item in GALLERY_ITEMS:
        frames = load_triplet_frames(item["dir"])
        if frames is None:
            continue
        frame_prev, _frame_mid, frame_next = frames
        items.append(
            GalleryItem(
                id=item["id"],
                label=item["label"],
                subset=item["subset"],
                frame_prev=encode_png(frame_prev),
                frame_next=encode_png(frame_next),
            )
        )
    return items


@router.post("/api/gallery/{item_id}/generate", response_model=GenerateResult)
def generate(item_id: str) -> GenerateResult:
    # deferred import -- avoids a circular import, since live.py mounts
    # this module's router and this is the only thing gallery.py needs
    # from it.
    from src.api.live import resolve_model_path

    item = _find_item(item_id)
    frames = load_triplet_frames(item["dir"])
    if frames is None:
        raise HTTPException(status_code=404, detail=f"Triplet data missing for '{item_id}'")
    frame_prev, frame_mid, frame_next = frames

    model_path = resolve_model_path()
    cache_key = f"{item_id}|{model_path.name}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    start = time.monotonic()
    farneback_pred = farneback_interpolate(frame_prev, frame_next)
    film_pred = film_interpolate(frame_prev, frame_next, model_path)

    result = GenerateResult(
        id=item_id,
        farneback_mid=encode_png(farneback_pred),
        film_mid=encode_png(film_pred),
        ground_truth=encode_png(frame_mid),
        farneback_psnr=round(psnr(farneback_pred, frame_mid), 2),
        farneback_ssim=round(ssim(farneback_pred, frame_mid), 4),
        farneback_lpips=round(lpips_distance(farneback_pred, frame_mid), 4),
        film_psnr=round(psnr(film_pred, frame_mid), 2),
        film_ssim=round(ssim(film_pred, frame_mid), 4),
        film_lpips=round(lpips_distance(film_pred, frame_mid), 4),
        processing_seconds=round(time.monotonic() - start, 1),
        model=model_path.name,
    )
    _cache[cache_key] = result
    return result
