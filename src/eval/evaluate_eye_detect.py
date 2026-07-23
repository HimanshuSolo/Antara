"""Evaluate both cyclone eye detectors (classical + trained CNN) on a
cyclone triplets directory with `eye_labels.csv` ground truth:

1. Detector accuracy: pixel error between each detector's output on the
   real ground-truth middle frame and the true IBTrACS-derived label --
   how good is the detector, period.
2. Interpolation position drift: pixel distance between a detector's
   output on the real middle frame and its output on the FILM- or
   Farneback-*synthesized* middle frame for the same triplet -- does
   interpolation preserve where the storm's eye actually was, or does it
   introduce a position error a downstream tracker would see? This is the
   metric that ties eye detection back into this project's core question
   of interpolation quality.

Writes one row per triplet to a results CSV, in the same spirit as
`evaluate_film.py`/`evaluate_baseline.py`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.baseline.eye_detect import detect_eye as classical_detect_eye
from src.baseline.farneback_interpolate import interpolate_middle_frame as farneback_interpolate
from src.deep.eye_detect import detect_eye as cnn_detect_eye
from src.deep.film_interpolate import interpolate_middle_frame as film_interpolate
from src.eval.eye_metrics import pixel_error
from src.utils.image import load_triplet_frames

FIELDNAMES = [
    "triplet",
    "classical_accuracy_px",
    "cnn_accuracy_px",
    "classical_farneback_drift_px",
    "classical_film_drift_px",
    "cnn_farneback_drift_px",
    "cnn_film_drift_px",
]


def load_eye_labels(triplets_dir: Path) -> dict[tuple[str, str], tuple[int, int]]:
    labels_path = triplets_dir / "eye_labels.csv"
    with labels_path.open(newline="") as f:
        return {
            (row["triplet"], row["frame"]): (int(row["row"]), int(row["col"]))
            for row in csv.DictReader(f)
        }


def evaluate(triplets_dir: Path, film_model_path: Path, eye_model_path: Path) -> list[dict]:
    labels = load_eye_labels(triplets_dir)
    rows = []

    for triplet_dir in sorted(triplets_dir.glob("triplet_*")):
        true_mid = labels.get((triplet_dir.name, "t"))
        if true_mid is None:
            continue
        frames = load_triplet_frames(triplet_dir)
        if frames is None:
            continue
        frame_prev, frame_mid, frame_next = frames

        farneback_mid = farneback_interpolate(frame_prev, frame_next)
        film_mid = film_interpolate(frame_prev, frame_next, film_model_path)

        classical_real = classical_detect_eye(frame_mid)
        classical_farneback = classical_detect_eye(farneback_mid)
        classical_film = classical_detect_eye(film_mid)

        cnn_real = cnn_detect_eye(frame_mid, eye_model_path)
        cnn_farneback = cnn_detect_eye(farneback_mid, eye_model_path)
        cnn_film = cnn_detect_eye(film_mid, eye_model_path)

        rows.append({
            "triplet": triplet_dir.name,
            "classical_accuracy_px": round(pixel_error(classical_real, true_mid), 2),
            "cnn_accuracy_px": round(pixel_error(cnn_real, true_mid), 2),
            "classical_farneback_drift_px": round(pixel_error(classical_farneback, classical_real), 2),
            "classical_film_drift_px": round(pixel_error(classical_film, classical_real), 2),
            "cnn_farneback_drift_px": round(pixel_error(cnn_farneback, cnn_real), 2),
            "cnn_film_drift_px": round(pixel_error(cnn_film, cnn_real), 2),
        })

    return rows


def write_results(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-dir", default="data/processed/triplets_cyclone")
    parser.add_argument("--film-model-path", default="models/film_net_finetuned_colab.pt")
    parser.add_argument("--eye-model-path", default="models/eye_detect_cnn.pt")
    parser.add_argument("--out-path", default="data/processed/eye_detect_results.csv")
    args = parser.parse_args()

    results = evaluate(Path(args.triplets_dir), Path(args.film_model_path), Path(args.eye_model_path))
    write_results(results, Path(args.out_path))
    print(f"Wrote {len(results)} rows to {args.out_path}")
