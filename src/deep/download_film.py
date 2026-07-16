"""Download the pretrained FILM (Frame Interpolation for Large Motion)
checkpoint -- a PyTorch/TorchScript port of Google's FILM model, hosted as
a plain GitHub release asset by
https://github.com/dajes/frame-interpolation-pytorch (MIT-licensed).

This is a self-contained TorchScript file (architecture + weights in one
`.pt`), which is why FILM was chosen over RIFE: RIFE's practical/HD
checkpoints only ship as Python-3.7-only compiled bytecode with no
source, which blocks the fine-tuning this project needs to do later. See
docs/PLAN.md for the full rationale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

FILM_FP32_URL = (
    "https://github.com/dajes/frame-interpolation-pytorch/releases/"
    "download/v1.0.2/film_net_fp32.pt"
)


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models/film_net_fp32.pt")
    args = parser.parse_args()

    path = download(FILM_FP32_URL, Path(args.out))
    print(f"Saved to {path}")
