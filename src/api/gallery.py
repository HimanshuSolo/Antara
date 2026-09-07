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
import io
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

from src.api.report import build_report_pdf
from src.baseline.farneback_interpolate import interpolate_middle_frame as farneback_interpolate
from src.deep.film_interpolate import interpolate_middle_frame as film_interpolate
from src.eval.metrics import lpips_distance, psnr, ssim
from src.utils.image import load_triplet_frames

GALLERY_ITEMS = [
    {
        "id": "milton-1",
        "label": "Hurricane Milton, approaching peak intensity",
        "subset": "cyclone",
        "dir": Path("data/processed/triplets_cyclone/triplet_0013"),
    },
    {
        "id": "milton-2",
        "label": "Hurricane Milton, peak intensity",
        "subset": "cyclone",
        "dir": Path("data/processed/triplets_cyclone/triplet_0020"),
    },
    {
        "id": "milton-3",
        "label": "Hurricane Milton, eyewall rotating past peak",
        "subset": "cyclone",
        "dir": Path("data/processed/triplets_cyclone/triplet_0025"),
    },
    {
        "id": "calm-1",
        "label": "Calm weather, off-season (1)",
        "subset": "calm",
        "dir": Path("data/processed/triplets_calm/triplet_0044"),
    },
    {
        "id": "calm-2",
        "label": "Calm weather, off-season (2)",
        "subset": "calm",
        "dir": Path("data/processed/triplets_calm/triplet_0051"),
    },
]

router = APIRouter()

# keyed by f"{item_id}|{model_name}" -- a generated result never changes
# for a given (triplet, checkpoint) pair, so there's no reason to rerun
# the pipeline once a result exists.
_cache: dict[str, "GenerateResult"] = {}

# keyed by f"{item_id}|{model_name}|{num_frames}"
_loop_cache: dict[str, "LoopResult"] = {}


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


class LoopResult(BaseModel):
    id: str
    loop_gif: str
    loop_mp4: str | None
    num_frames: int
    processing_seconds: float
    model: str


class ReportResult(BaseModel):
    id: str
    report_pdf: str
    generated_at: str


def encode_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image as PNG")
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def encode_gif(frames: list[np.ndarray], duration_ms: int = 220) -> str:
    """Assemble grayscale frames into a looping animated GIF -- the
    real-world format a forecaster actually watches (a "satellite loop"),
    here made smoother than the raw scan cadence by the interpolated
    frames in between."""
    pil_frames = [Image.fromarray(frame).convert("L") for frame in frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF", save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0,
    )
    return "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def encode_mp4(frames: list[np.ndarray], fps: int = 5) -> str | None:
    """Assemble grayscale frames into an H.264 MP4 via ffmpeg -- unlike
    the GIF above (meant for an inline animated preview), this is the
    downloadable artifact: much smaller, and plays natively in any video
    player or a browser <video> element rather than just an <img>. Uses
    ffmpeg's libopenh264 encoder specifically, since it doesn't depend on
    GPU/hardware encode support the way h264_nvenc/vaapi/v4l2m2m do.
    Returns None (rather than raising) if ffmpeg isn't installed, so the
    GIF preview still works on a machine without it.
    """
    if shutil.which("ffmpeg") is None:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i, frame in enumerate(frames):
            cv2.imwrite(str(tmp / f"frame_{i:03d}.png"), frame)
        out_path = tmp / "loop.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-framerate", str(fps),
                    "-i", str(tmp / "frame_%03d.png"),
                    "-c:v", "libopenh264", "-pix_fmt", "yuv420p",
                    str(out_path),
                ],
                check=True, capture_output=True, timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        video_bytes = out_path.read_bytes()

    return "data:video/mp4;base64," + base64.b64encode(video_bytes).decode("ascii")


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


@router.post("/api/gallery/{item_id}/loop", response_model=LoopResult)
def generate_loop(item_id: str, num_frames: int = 5) -> LoopResult:
    """Real-world usage endpoint: assembles t-1, `num_frames` FILM-
    interpolated intermediate frames, and t+1 into a single looping
    sequence -- a higher-effective-frame-rate satellite motion loop, the
    actual format forecasters use to track storm motion, rather than a
    single static comparison image. Returned as both an inline-previewable
    GIF and a downloadable H.264 MP4 (smaller, shareable)."""
    from src.api.live import resolve_model_path
    from src.deep.film_interpolate import interpolate_multi

    if not 1 <= num_frames <= 9:
        raise HTTPException(status_code=422, detail="num_frames must be between 1 and 9")

    item = _find_item(item_id)
    frames = load_triplet_frames(item["dir"])
    if frames is None:
        raise HTTPException(status_code=404, detail=f"Triplet data missing for '{item_id}'")
    frame_prev, _frame_mid, frame_next = frames

    model_path = resolve_model_path()
    cache_key = f"{item_id}|{model_path.name}|{num_frames}"
    cached = _loop_cache.get(cache_key)
    if cached is not None:
        return cached

    start = time.monotonic()
    mid_frames = interpolate_multi(frame_prev, frame_next, model_path, num_frames=num_frames)
    sequence = [frame_prev, *mid_frames, frame_next]

    result = LoopResult(
        id=item_id,
        loop_gif=encode_gif(sequence),
        loop_mp4=encode_mp4(sequence),
        num_frames=num_frames,
        processing_seconds=round(time.monotonic() - start, 1),
        model=model_path.name,
    )
    _loop_cache[cache_key] = result
    return result


@router.post("/api/gallery/{item_id}/report", response_model=ReportResult)
def generate_report(item_id: str) -> ReportResult:
    """Real-world usage endpoint: a downloadable, one-page PDF summary of
    this triplet's Farneback vs. FILM comparison -- the kind of artifact
    an analyst would archive or attach to an incident report, not just
    view on a live web page. Reuses generate()'s cache, so this doesn't
    rerun the pipeline if the single-frame result already exists."""
    item = _find_item(item_id)
    result = generate(item_id)

    frames = load_triplet_frames(item["dir"])
    if frames is None:
        raise HTTPException(status_code=404, detail=f"Triplet data missing for '{item_id}'")
    frame_prev, _frame_mid, frame_next = frames

    pdf_bytes = build_report_pdf(
        label=item["label"],
        subset=item["subset"],
        model=result.model,
        frame_prev=encode_png(frame_prev),
        frame_next=encode_png(frame_next),
        ground_truth=result.ground_truth,
        farneback_mid=result.farneback_mid,
        film_mid=result.film_mid,
        farneback_psnr=result.farneback_psnr,
        farneback_ssim=result.farneback_ssim,
        farneback_lpips=result.farneback_lpips,
        film_psnr=result.film_psnr,
        film_ssim=result.film_ssim,
        film_lpips=result.film_lpips,
    )

    return ReportResult(
        id=item_id,
        report_pdf="data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("ascii"),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
