"""Fine-tuning data-volume ablation: fine-tune FILM on increasing numbers
of triplets drawn from the same pool and evaluate each resulting
checkpoint on the same held-out test set -- per docs/PLAN.md's Week 9-10
ablations. Answers whether more fine-tuning data keeps buying quality or
plateaus, holding everything else (epochs, lr, test set) fixed.

NOTE ON COMPUTE: like finetune_film.py, a real sweep over meaningful
triplet counts and epoch budgets should run on a Colab/Kaggle GPU --
each --counts value here is a full fine-tuning run, so this is
`finetune_film.py`'s CPU-scale caveat multiplied by len(counts).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.deep.finetune_film import finetune
from src.eval.evaluate_film import evaluate as evaluate_film
from src.eval.evaluate_stratified import summarize


def materialize_subset(triplet_dirs: list[Path], subset_dir: Path) -> Path:
    """Symlink the given triplet directories into a fresh subset directory,
    so TripletDataset can load exactly this subset without copying the
    (possibly large) source PNGs."""
    subset_dir.mkdir(parents=True, exist_ok=True)
    for d in triplet_dirs:
        link = subset_dir / d.name
        if not link.exists():
            link.symlink_to(d.resolve(), target_is_directory=True)
    return subset_dir


def run_finetune_data_ablation(
    model_path: Path,
    triplets_finetune_dir: Path,
    triplets_test_dir: Path,
    counts: list[int],
    out_dir: Path,
    epochs: int = 5,
    device: str | None = None,
) -> dict[int, tuple[float, float, float]]:
    all_dirs = sorted(d for d in triplets_finetune_dir.iterdir() if d.is_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, tuple[float, float, float]] = {}
    for count in counts:
        if count > len(all_dirs):
            print(f"Skipping count={count} -- only {len(all_dirs)} triplets available in {triplets_finetune_dir}")
            continue

        subset_dir = materialize_subset(all_dirs[:count], out_dir / f"triplets_{count}")
        checkpoint_path = out_dir / f"model_{count}.pt"
        finetune(model_path, subset_dir, checkpoint_path, epochs=epochs, device=device)

        film_rows = evaluate_film(triplets_test_dir, checkpoint_path, out_dir / f"test_{count}.csv", device=device)
        results[count] = summarize(film_rows)

    print(f"\n{'count':>6s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s}")
    print("-" * 32)
    for count, (mean_psnr, mean_ssim, mean_lpips) in results.items():
        print(f"{count:6d} {mean_psnr:8.2f} {mean_ssim:8.4f} {mean_lpips:8.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/film_net_fp32.pt")
    parser.add_argument("--triplets-finetune-dir", default="data/processed/triplets_finetune")
    parser.add_argument("--triplets-test-dir", default="data/processed/triplets_test")
    parser.add_argument("--counts", type=int, nargs="+", default=[8, 16, 24, 31])
    parser.add_argument("--out-dir", default="data/processed/ablation_finetune_data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default=None, help="cuda / cpu -- auto-detects if omitted")
    args = parser.parse_args()

    run_finetune_data_ablation(
        Path(args.model_path),
        Path(args.triplets_finetune_dir),
        Path(args.triplets_test_dir),
        args.counts,
        Path(args.out_dir),
        epochs=args.epochs,
        device=args.device,
    )
