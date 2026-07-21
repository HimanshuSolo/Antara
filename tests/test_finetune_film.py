from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import src.deep.finetune_film as finetune_film
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


def test_finetune_saves_the_best_val_loss_epoch_not_the_last(tmp_path, monkeypatch):
    """Force val_loss to dip at epoch 2 and rise again at epoch 3 (real
    l1_loss still drives training -- only the *validation* readout is
    faked, gated on torch.is_grad_enabled() since the val loop is the only
    place that runs under torch.no_grad()), then check the saved checkpoint
    matches the epoch-2 weights, not the epoch-3 ones training finished on."""
    triplets_dir = tmp_path / "triplets"
    _write_triplets(triplets_dir, count=5)
    out_path = tmp_path / "finetuned.pt"

    scripted_val_losses = [0.9, 0.1, 0.9]
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    real_l1_loss = torch.nn.functional.l1_loss
    state = {"epoch": 0, "captured_this_epoch": False}
    model_holder: dict[str, object] = {}

    def fake_l1_loss(pred, target):
        if not torch.is_grad_enabled():
            if not state["captured_this_epoch"]:
                state["epoch"] += 1
                state["captured_this_epoch"] = True
                snapshots[state["epoch"]] = {
                    k: v.detach().clone() for k, v in model_holder["model"].state_dict().items()
                }
            return torch.tensor(scripted_val_losses[state["epoch"] - 1])
        state["captured_this_epoch"] = False
        return real_l1_loss(pred, target)

    monkeypatch.setattr(torch.nn.functional, "l1_loss", fake_l1_loss)

    real_jit_load = torch.jit.load

    def spying_jit_load(*args, **kwargs):
        model = real_jit_load(*args, **kwargs)
        model_holder["model"] = model
        return model

    monkeypatch.setattr(finetune_film.torch.jit, "load", spying_jit_load)

    history = finetune(
        model_path=MODEL_PATH,
        triplets_dir=triplets_dir,
        out_path=out_path,
        epochs=3,
        batch_size=2,
        val_fraction=0.2,
    )

    assert history["val_loss"] == pytest.approx(scripted_val_losses)

    reloaded = torch.jit.load(str(out_path), map_location="cpu")
    saved_state = reloaded.state_dict()
    # snapshots[2] was taken right as epoch 2's validation began, i.e.
    # right after epoch 2's *training* step -- exactly the weights that
    # should have been restored as the best (lowest val_loss) epoch.
    for key, expected in snapshots[2].items():
        assert torch.allclose(saved_state[key], expected), f"{key} does not match the best (epoch 2) snapshot"
    mismatch = any(not torch.allclose(saved_state[key], snapshots[3][key]) for key in snapshots[3])
    assert mismatch, "saved checkpoint should differ from the final (epoch 3) weights"
