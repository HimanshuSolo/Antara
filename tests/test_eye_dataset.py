import csv

import cv2
import numpy as np
import pytest
import torch

from src.deep.eye_dataset import EyeDataset


def _write_triplet(triplets_dir, name, size=16):
    d = triplets_dir / name
    d.mkdir(parents=True)
    for frame in ("t-1", "t", "t+1"):
        cv2.imwrite(str(d / f"{frame}.png"), np.zeros((size, size), dtype=np.uint8))
    return d


def _write_labels(triplets_dir, rows):
    with open(triplets_dir / "eye_labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["triplet", "frame", "row", "col"])
        writer.writeheader()
        writer.writerows(rows)


def test_eye_dataset_length_matches_label_rows(tmp_path):
    _write_triplet(tmp_path, "triplet_0000")
    _write_labels(tmp_path, [
        {"triplet": "triplet_0000", "frame": "t-1", "row": 4, "col": 8},
        {"triplet": "triplet_0000", "frame": "t", "row": 8, "col": 8},
    ])

    dataset = EyeDataset(tmp_path)

    assert len(dataset) == 2


def test_eye_dataset_returns_normalized_image_and_target(tmp_path):
    _write_triplet(tmp_path, "triplet_0000", size=16)
    _write_labels(tmp_path, [{"triplet": "triplet_0000", "frame": "t", "row": 4, "col": 12}])

    dataset = EyeDataset(tmp_path)
    image, target = dataset[0]

    assert image.shape == (1, 16, 16)
    assert image.dtype == torch.float32
    assert torch.allclose(target, torch.tensor([4 / 16, 12 / 16]))


def test_eye_dataset_raises_on_empty_labels_csv(tmp_path):
    (tmp_path / "eye_labels.csv").write_text("triplet,frame,row,col\n")

    with pytest.raises(ValueError, match="No eye labels"):
        EyeDataset(tmp_path)


def test_eye_dataset_raises_on_missing_frame_image(tmp_path):
    _write_labels(tmp_path, [{"triplet": "triplet_missing", "frame": "t", "row": 0, "col": 0}])

    dataset = EyeDataset(tmp_path)

    with pytest.raises(FileNotFoundError):
        dataset[0]
