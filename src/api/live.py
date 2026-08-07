"""Live pipeline API: runs the real Farneback/FILM interpolation methods
against the two most recently published GOES-16 scans -- there is no real
frame *between* them yet, so this is a genuinely live demo (not a canned
one) of the thing this whole project builds: synthesizing the missing
frame between two real satellite scans.

Run locally alongside the web frontend:
    .venv/bin/uvicorn src.api.live:app --reload --port 8000
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.gallery import router as gallery_router
from src.api.track import router as track_router
from src.baseline.farneback_interpolate import interpolate_middle_frame as farneback_interpolate
from src.data import extract_triplets, fetch_goes
from src.deep.film_interpolate import interpolate_middle_frame as film_interpolate

BAND = 13
PATCH_SIZE = 256
CACHE_DIR = Path("data/raw/live")

# preference order: best fine-tuned checkpoint first, falling back to the
# zero-shot pretrained one if fine-tuning hasn't been run locally.
# film_net_finetuned_colab.pt is the full-scale run (30 epochs / 207
# triplets on a Colab T4 GPU, 33.25 dB on a fully disjoint test day --
# see notebooks/finetune_on_colab.ipynb and the README's Status section),
# ranked above the earlier small-scale/CPU checkpoints.
MODEL_CANDIDATES = [
    Path("models/film_net_finetuned_colab.pt"),
    Path("models/film_net_finetuned_updated.pt"),
    Path("models/film_net_finetuned.pt"),
    Path("models/film_net_fp32.pt"),
]

app = FastAPI(title="Antara live pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(gallery_router)
app.include_router(track_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "ok", "service": "Antara live pipeline API", "docs": "/docs"}

_cache: dict[str, "LiveResult"] = {}


class LiveResult(BaseModel):
    prev_time: str
    next_time: str
    cadence_minutes: float
    frame_prev: str
    frame_next: str
    farneback_mid: str
    film_mid: str
    processing_seconds: float
    model: str


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise HTTPException(
        status_code=503,
        detail="No FILM checkpoint found in models/ -- run `.venv/bin/python -m src.deep.download_film` first.",
    )


def encode_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG")
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def load_patch(nc_path: Path, center: tuple[int, int], size: int) -> np.ndarray:
    rad = extract_triplets.load_radiance(nc_path)
    return extract_triplets.crop_patch(extract_triplets.to_uint8(rad), center, size)


def run_live_pipeline(model_path: Path) -> LiveResult:
    start = time.monotonic()

    try:
        prev_key, next_key = fetch_goes.latest_scan_pair(band=BAND)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    cache_key = f"{prev_key}|{next_key}|{model_path.name}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    prev_path = fetch_goes.download(prev_key, CACHE_DIR, bucket=fetch_goes.LIVE_BUCKET)
    next_path = fetch_goes.download(next_key, CACHE_DIR, bucket=fetch_goes.LIVE_BUCKET)

    prev_time = extract_triplets.parse_scan_time(prev_path)
    next_time = extract_triplets.parse_scan_time(next_path)

    center = extract_triplets.default_center(extract_triplets.load_radiance(prev_path).shape)
    patch_prev = load_patch(prev_path, center, PATCH_SIZE)
    patch_next = load_patch(next_path, center, PATCH_SIZE)

    farneback_pred = farneback_interpolate(patch_prev, patch_next)
    film_pred = film_interpolate(patch_prev, patch_next, model_path)

    result = LiveResult(
        prev_time=prev_time.isoformat() + "Z",
        next_time=next_time.isoformat() + "Z",
        cadence_minutes=round((next_time - prev_time).total_seconds() / 60, 1),
        frame_prev=encode_png(patch_prev),
        frame_next=encode_png(patch_next),
        farneback_mid=encode_png(farneback_pred),
        film_mid=encode_png(film_pred),
        processing_seconds=round(time.monotonic() - start, 1),
        model=model_path.name,
    )

    # only ever cache the single latest pair -- a new scan makes the old
    # entry stale and there's no reason to hold onto it.
    _cache.clear()
    _cache[cache_key] = result
    return result


@app.get("/api/live", response_model=LiveResult)
def get_live() -> LiveResult:
    return run_live_pipeline(resolve_model_path())


@app.get("/api/health")
def get_health() -> dict[str, bool]:
    return {"model_available": any(c.exists() for c in MODEL_CANDIDATES)}
