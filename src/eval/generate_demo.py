"""Self-contained, single-file interactive demo: a before/after slider
comparing the classical Farneback prediction against the learned FILM
prediction for one triplet -- per docs/PLAN.md's stretch goal, "a
before/after slider comparing classical vs. learned interpolation on a
cyclone event." Images are embedded as base64 data URIs, so the resulting
HTML file can be opened or shared on its own, with no dependency on
data/ (which is never committed) being present.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import cv2
import numpy as np

from src.baseline.farneback_interpolate import interpolate_middle_frame as farneback_interpolate
from src.deep.film_interpolate import interpolate_middle_frame as film_interpolate
from src.eval.metrics import psnr, ssim
from src.eval.plot_comparison import load_triplet


def encode_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return base64.b64encode(buf).decode("ascii")


def build_demo_html(
    triplet_name: str,
    farneback_b64: str,
    film_b64: str,
    farneback_metrics: tuple[float, float],
    film_metrics: tuple[float, float],
) -> str:
    farneback_psnr, farneback_ssim = farneback_metrics
    film_psnr, film_ssim = film_metrics
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Antara demo -- {triplet_name}</title>
<style>
  body {{ font-family: sans-serif; max-width: 700px; margin: 2rem auto; }}
  .slider-container {{ position: relative; width: 100%; user-select: none; }}
  .slider-container > img {{ display: block; width: 100%; }}
  .clip {{ position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden; }}
  .clip img {{ position: absolute; top: 0; left: 0; max-width: none; }}
  input[type="range"] {{ width: 100%; margin-top: 0.5rem; }}
  .caption {{ font-size: 0.9rem; color: #444; }}
</style>
</head>
<body>
  <h1>{triplet_name}: Farneback vs. FILM</h1>
  <div class="slider-container" id="container">
    <img id="film-img" src="data:image/png;base64,{film_b64}">
    <div class="clip" id="clip">
      <img id="farneback-img" src="data:image/png;base64,{farneback_b64}">
    </div>
  </div>
  <input type="range" min="0" max="100" value="50" id="slider">
  <p class="caption">
    Left of the slider: Farneback (PSNR {farneback_psnr:.1f} dB / SSIM {farneback_ssim:.2f}) --
    right: FILM (PSNR {film_psnr:.1f} dB / SSIM {film_ssim:.2f})
  </p>
  <script>
    const container = document.getElementById("container");
    const clip = document.getElementById("clip");
    const farnebackImg = document.getElementById("farneback-img");
    const slider = document.getElementById("slider");

    function syncWidth() {{
      farnebackImg.style.width = container.clientWidth + "px";
    }}
    window.addEventListener("resize", syncWidth);
    window.addEventListener("load", syncWidth);
    syncWidth();

    slider.addEventListener("input", () => {{
      clip.style.width = slider.value + "%";
    }});
  </script>
</body>
</html>
"""


def build_demo_for_triplet(triplet_dir: Path, film_model_path: Path, device: str | None = None) -> str:
    frame_prev, frame_mid, frame_next = load_triplet(triplet_dir)

    farneback_pred = farneback_interpolate(frame_prev, frame_next)
    film_pred = film_interpolate(frame_prev, frame_next, film_model_path, device=device)

    return build_demo_html(
        triplet_dir.name,
        encode_png(farneback_pred),
        encode_png(film_pred),
        (psnr(farneback_pred, frame_mid), ssim(farneback_pred, frame_mid)),
        (psnr(film_pred, frame_mid), ssim(film_pred, frame_mid)),
    )


def write_demo(triplet_dir: Path, film_model_path: Path, out_path: Path, device: str | None = None) -> Path:
    html = build_demo_for_triplet(triplet_dir, film_model_path, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


def save_demos(
    triplets_dir: Path,
    film_model_path: Path,
    out_dir: Path,
    device: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    triplet_dirs = sorted(p for p in triplets_dir.iterdir() if p.is_dir())
    if limit is not None:
        triplet_dirs = triplet_dirs[:limit]

    for triplet_dir in triplet_dirs:
        out_path = out_dir / f"{triplet_dir.name}.html"
        write_demo(triplet_dir, film_model_path, out_path, device=device)
        written.append(out_path)

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-dir", default="data/processed/triplets_cyclone")
    parser.add_argument("--film-model-path", default="models/film_net_fp32.pt")
    parser.add_argument("--out-dir", default="data/processed/demo")
    parser.add_argument("--device", default=None, help="cuda / cpu -- auto-detects if omitted")
    parser.add_argument("--limit", type=int, default=3, help="number of triplets to render (sorted order)")
    args = parser.parse_args()

    written = save_demos(
        Path(args.triplets_dir),
        Path(args.film_model_path),
        Path(args.out_dir),
        device=args.device,
        limit=args.limit,
    )
    print(f"Wrote {len(written)} demo pages to {args.out_dir}")
