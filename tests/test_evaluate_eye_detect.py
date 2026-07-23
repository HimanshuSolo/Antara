import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.deep.eye_detect_model import EyeDetectCNN
from src.eval.evaluate_eye_detect import evaluate, load_eye_labels, write_results

FILM_MODEL_PATH = Path("models/film_net_fp32.pt")

pytestmark = pytest.mark.skipif(
    not FILM_MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)


def _write_triplet(triplet_dir: Path, size: int = 64) -> None:
    triplet_dir.mkdir(parents=True, exist_ok=True)
    for name in ("t-1.png", "t.png", "t+1.png"):
        cv2.imwrite(str(triplet_dir / name), np.random.randint(0, 255, (size, size), dtype=np.uint8))


def _write_eye_labels(triplets_dir: Path, triplet_names: list[str]) -> None:
    with (triplets_dir / "eye_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["triplet", "frame", "row", "col"])
        writer.writeheader()
        for name in triplet_names:
            writer.writerow({"triplet": name, "frame": "t", "row": 32, "col": 32})


@pytest.fixture
def eye_model_path(tmp_path):
    path = tmp_path / "eye_detect_cnn.pt"
    torch.save(EyeDetectCNN().state_dict(), path)
    return path


def test_evaluate_writes_one_row_per_labeled_triplet(tmp_path, eye_model_path):
    triplets_dir = tmp_path / "triplets"
    for i in range(2):
        _write_triplet(triplets_dir / f"triplet_{i:04d}")
    _write_eye_labels(triplets_dir, ["triplet_0000", "triplet_0001"])

    rows = evaluate(triplets_dir, FILM_MODEL_PATH, eye_model_path)

    assert len(rows) == 2
    assert {row["triplet"] for row in rows} == {"triplet_0000", "triplet_0001"}
    for row in rows:
        assert set(row) == {
            "triplet", "classical_accuracy_px", "cnn_accuracy_px",
            "classical_farneback_drift_px", "classical_film_drift_px",
            "cnn_farneback_drift_px", "cnn_film_drift_px",
        }


def test_evaluate_skips_triplets_without_eye_labels(tmp_path, eye_model_path):
    triplets_dir = tmp_path / "triplets"
    _write_triplet(triplets_dir / "triplet_0000")
    _write_eye_labels(triplets_dir, [])  # no rows at all

    rows = evaluate(triplets_dir, FILM_MODEL_PATH, eye_model_path)

    assert rows == []


def test_load_eye_labels_keys_by_triplet_and_frame(tmp_path):
    triplets_dir = tmp_path / "triplets"
    triplets_dir.mkdir()
    _write_eye_labels(triplets_dir, ["triplet_0000"])

    labels = load_eye_labels(triplets_dir)

    assert labels[("triplet_0000", "t")] == (32, 32)


def test_write_results_writes_a_csv_with_a_header(tmp_path):
    out_path = tmp_path / "out" / "results.csv"
    rows = [{name: 0 for name in [
        "triplet", "classical_accuracy_px", "cnn_accuracy_px",
        "classical_farneback_drift_px", "classical_film_drift_px",
        "cnn_farneback_drift_px", "cnn_film_drift_px",
    ]}]

    write_results(rows, out_path)

    lines = out_path.read_text().strip().splitlines()
    assert lines[0].startswith("triplet,")
    assert len(lines) == 2
