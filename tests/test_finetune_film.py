from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.deep.finetune_film import finetune

MODEL_PATH = Path("models/film_net_fp32.pt")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)


def _write_triplets(triplets_dir: Path, count: int) -> None:
    for i in range(count):
        d = triplets_dir / f"triplet_{i}"
        d.mkdir(parents=True)
        for name in ("t-1.png", "t.png", "t+1.png"):
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(d / name), img)


def test_finetune_runs_one_epoch_and_saves_a_loadable_checkpoint(tmp_path):
    triplets_dir = tmp_path / "triplets"
    _write_triplets(triplets_dir, count=5)
    out_path = tmp_path / "finetuned.pt"

    history = finetune(
        model_path=MODEL_PATH,
        triplets_dir=triplets_dir,
        out_path=out_path,
        epochs=1,
        batch_size=2,
        val_fraction=0.2,
    )

    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert all(loss == loss for loss in history["train_loss"])  # not NaN
    assert all(loss == loss for loss in history["val_loss"])

    assert out_path.exists()
    reloaded = torch.jit.load(str(out_path), map_location="cpu")
    img1 = torch.rand(1, 3, 64, 64)
    img2 = torch.rand(1, 3, 64, 64)
    dt = torch.full((1, 1), 0.5)
    pred = reloaded(img1, img2, dt)
    assert pred.shape == img1.shape


def test_finetune_skips_validation_split_for_a_single_triplet(tmp_path):
    triplets_dir = tmp_path / "triplets"
    _write_triplets(triplets_dir, count=1)
    out_path = tmp_path / "finetuned.pt"

    history = finetune(
        model_path=MODEL_PATH,
        triplets_dir=triplets_dir,
        out_path=out_path,
        epochs=1,
        batch_size=1,
        val_fraction=0.2,
    )

    assert len(history["train_loss"]) == 1
    assert history["val_loss"] == []
