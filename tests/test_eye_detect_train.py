import csv

import cv2
import numpy as np
import torch

from src.deep.eye_detect_train import train


def _write_dataset(triplets_dir, n_triplets=6, size=32):
    rows = []
    for i in range(n_triplets):
        d = triplets_dir / f"triplet_{i:04d}"
        d.mkdir(parents=True)
        for frame in ("t-1", "t", "t+1"):
            cv2.imwrite(str(d / f"{frame}.png"), np.random.randint(0, 255, (size, size), dtype=np.uint8))
            rows.append({"triplet": d.name, "frame": frame, "row": size // 2, "col": size // 2})
    with open(triplets_dir / "eye_labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["triplet", "frame", "row", "col"])
        writer.writeheader()
        writer.writerows(rows)


def test_train_runs_and_saves_a_checkpoint(tmp_path):
    _write_dataset(tmp_path / "triplets_cyclone")
    out_path = tmp_path / "eye_detect_cnn.pt"

    history = train(
        [tmp_path / "triplets_cyclone"], out_path, epochs=2, batch_size=4, device="cpu"
    )

    assert out_path.exists()
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2

    # the saved checkpoint loads back as a plain state dict
    state = torch.load(str(out_path), map_location="cpu")
    assert "head.weight" in state


def test_train_pools_multiple_triplets_dirs(tmp_path):
    _write_dataset(tmp_path / "storm_a", n_triplets=4)
    _write_dataset(tmp_path / "storm_b", n_triplets=4)
    out_path = tmp_path / "eye_detect_cnn.pt"

    history = train(
        [tmp_path / "storm_a", tmp_path / "storm_b"], out_path, epochs=1, batch_size=4, device="cpu"
    )

    assert out_path.exists()
    assert len(history["train_loss"]) == 1
