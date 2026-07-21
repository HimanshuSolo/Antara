"""Incremental ("continual") fine-tuning: update an existing FILM checkpoint
on only the most recent triplets in a growing pool, instead of retraining
from scratch every time new satellite passes arrive -- docs/PLAN.md's
"online/continual fine-tuning ... adapting to seasonal cloud-pattern shift"
stretch goal.

Reuses finetune_film.finetune() and ablate_finetune_data.materialize_subset()
as-is: the only new idea here is windowing -- take just the `window_size`
most-recently-added triplets from the pool (triplet directories are sorted
lexicographically, which matches arrival order for triplet_NNNN-style
naming), so the model tracks recent conditions rather than being pulled
back toward stale ones as the pool grows over a season.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.deep.ablate_finetune_data import materialize_subset
from src.deep.finetune_film import finetune
from src.eval.plot_comparison import list_triplet_dirs


def continual_update(
    model_path: Path,
    pool_dir: Path,
    out_path: Path,
    window_size: int = 100,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 1e-5,
    val_fraction: float = 0.2,
    device: str | None = None,
) -> dict:
    """Fine-tune `model_path` on the most recent `window_size` triplets in
    `pool_dir` (the whole pool, if it's smaller), saving the updated
    checkpoint to `out_path`. Returns finetune()'s loss history.
    """
    all_dirs = list_triplet_dirs(pool_dir)
    if not all_dirs:
        raise ValueError(f"No triplets found in {pool_dir}")

    recent_dirs = all_dirs[-window_size:]
    window_dir = out_path.parent / f"{out_path.stem}_window"
    materialize_subset(recent_dirs, window_dir)

    print(f"Continual update: {len(recent_dirs)}/{len(all_dirs)} most recent triplets from {pool_dir}")
    return finetune(
        model_path,
        window_dir,
        out_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_fraction=val_fraction,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/film_net_finetuned.pt")
    parser.add_argument("--pool-dir", default="data/processed/triplets_stream")
    parser.add_argument("--out-path", default="models/film_net_finetuned_updated.pt")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default=None, help="cuda / cpu -- auto-detects if omitted")
    args = parser.parse_args()

    continual_update(
        Path(args.model_path),
        Path(args.pool_dir),
        Path(args.out_path),
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        device=args.device,
    )
