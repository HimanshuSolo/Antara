"""Fine-tune the pretrained FILM checkpoint on satellite triplets.

This is the core contribution of the project: adapting a model pretrained
on ordinary video to the satellite/cloud visual domain, rather than
training a frame-interpolation model from scratch (which would need a
huge video dataset and far more compute than free-tier Colab/Kaggle
offers).

Loss is plain L1 reconstruction error between the predicted and real
held-out middle frame. The original FILM paper also uses a perceptual
(VGG) loss and a style loss; L1 alone is a deliberate scope cut for a
semester project -- worth calling out explicitly in the report rather
than silently deviating from the paper.

NOTE ON COMPUTE: auto-detects and uses a GPU if one is available (see
`--device`) -- a real training run over enough data to matter should
happen on a Colab/Kaggle GPU per docs/PLAN.md, since CPU fine-tuning of a
~34M-parameter video model does not scale to the epoch counts/data volume
a real result needs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.deep.dataset import TripletDataset


def finetune(
    model_path: Path,
    triplets_dir: Path,
    out_path: Path,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-5,
    val_fraction: float = 0.2,
    seed: int = 0,
    device: str | None = None,
) -> dict:
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torch.jit.load(str(model_path), map_location=device)
    model.to(device)

    dataset = TripletDataset(triplets_dir)
    # A *chronological* split, not torch's random_split: consecutive
    # triplets overlap in content (triplet N's "next" frame is triplet
    # N+1's "prev" frame), so a random split would leak near-duplicate
    # frames across the train/val boundary. Shuffling batches *within*
    # the training set (below) is fine -- it's only the split boundary
    # itself that must respect time order.
    val_size = max(1, int(len(dataset) * val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_ds: Dataset
    val_ds: Dataset | None
    if val_size == 0:
        train_ds, val_ds = dataset, None
    else:
        train_ds = torch.utils.data.Subset(dataset, range(0, train_size))
        val_ds = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for frame_prev, frame_mid, frame_next in train_loader:
            frame_prev = frame_prev.to(device)
            frame_mid = frame_mid.to(device)
            frame_next = frame_next.to(device)
            dt = frame_prev.new_full((frame_prev.shape[0], 1), 0.5)

            optimizer.zero_grad()
            pred = model(frame_prev, frame_next, dt)
            loss = torch.nn.functional.l1_loss(pred, frame_mid)
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
                for frame_prev, frame_mid, frame_next in val_loader:
                    frame_prev = frame_prev.to(device)
                    frame_mid = frame_mid.to(device)
                    frame_next = frame_next.to(device)
                    dt = frame_prev.new_full((frame_prev.shape[0], 1), 0.5)
                    pred = model(frame_prev, frame_next, dt)
                    val_running += torch.nn.functional.l1_loss(pred, frame_mid).item()
                    val_batches += 1
            val_loss = val_running / max(val_batches, 1)
            history["val_loss"].append(val_loss)

        msg = f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f"  val_loss={val_loss:.4f}"
        print(msg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.to("cpu")  # save a checkpoint that loads on any machine, GPU or not
    model.save(str(out_path))
    print(f"Saved fine-tuned checkpoint to {out_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/film_net_fp32.pt")
    parser.add_argument("--triplets-dir", default="data/processed/triplets")
    parser.add_argument("--out-path", default="models/film_net_finetuned.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default=None, help="cuda / cpu -- auto-detects if omitted")
    args = parser.parse_args()

    finetune(
        Path(args.model_path),
        Path(args.triplets_dir),
        Path(args.out_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        device=args.device,
    )
