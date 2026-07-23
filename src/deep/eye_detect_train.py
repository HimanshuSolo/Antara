"""Train the small CNN eye detector (`eye_detect_model.py`) on IBTrACS-
derived eye-position labels (`build_eye_labels.py`).

Loss is plain MSE on normalized (row/H, col/W) coordinates. Mirrors
`finetune_film.py`'s conventions: a *chronological* (not random) train/val
split -- adjacent triplets share a real frame (triplet N's t+1 is triplet
N+1's t-1), so a random split would leak near-duplicate frames across the
boundary -- and the checkpoint saved is whichever epoch had the lowest
validation loss, not just the last one.

NOTE ON DATA: the only cyclone dataset in this repo right now is a single
storm's (Milton's) ~4.5-hour window (see build_cyclone_dataset.py) -- enough
to prove the training loop works, but not enough storm diversity for a
genuinely generalizing detector (train and val both come from one
continuous storm track). A real result needs pooling several storms
(`build_cyclone_dataset.py --storm-name ...` once per storm, then passing
every resulting directory to `--triplets-dirs`), the same "small-scale
local proof of concept now, full run separately" pattern this project
already used for FILM's fine-tuning (see notebooks/finetune_on_colab.ipynb).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.deep.eye_dataset import EyeDataset
from src.deep.eye_detect_model import EyeDetectCNN


def train(
    triplets_dirs: list[Path],
    out_path: Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
    device: str | None = None,
) -> dict:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = [EyeDataset(d) for d in triplets_dirs]
    dataset: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    total_size = sum(len(d) for d in datasets)

    val_size = max(1, int(total_size * val_fraction)) if total_size > 1 else 0
    train_size = total_size - val_size
    train_ds: Dataset
    val_ds: Dataset | None
    if val_size == 0:
        train_ds, val_ds = dataset, None
    else:
        train_ds = torch.utils.data.Subset(dataset, range(0, train_size))
        val_ds = torch.utils.data.Subset(dataset, range(train_size, total_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    model = EyeDetectCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            pred = model(images)
            loss = torch.nn.functional.mse_loss(pred, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        val_loss = None
        if val_loader:
            model.eval()
            val_running = 0.0
            val_batches = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    pred = model(images)
                    val_running += torch.nn.functional.mse_loss(pred, targets).item()
                    val_batches += 1
            val_loss = val_running / max(val_batches, 1)
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        msg = f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.5f}"
        if val_loss is not None:
            msg += f"  val_loss={val_loss:.5f}"
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restoring epoch {best_epoch}'s weights (best val_loss={best_val_loss:.5f})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    torch.save(model.state_dict(), out_path)
    print(f"Saved eye detector checkpoint to {out_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-dirs", nargs="+", default=["data/processed/triplets_cyclone"])
    parser.add_argument("--out-path", default="models/eye_detect_cnn.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default=None, help="cuda / cpu -- auto-detects if omitted")
    args = parser.parse_args()

    train(
        [Path(d) for d in args.triplets_dirs],
        Path(args.out_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        device=args.device,
    )
