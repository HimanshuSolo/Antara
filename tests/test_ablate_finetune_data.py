from pathlib import Path

import cv2
import numpy as np
import pytest

from src.deep.ablate_finetune_data import materialize_subset, run_finetune_data_ablation

MODEL_PATH = Path("models/film_net_fp32.pt")


def test_materialize_subset_symlinks_selected_dirs(tmp_path):
    pool_dir = tmp_path / "pool"
    triplet_a = pool_dir / "triplet_a"
    triplet_b = pool_dir / "triplet_b"
    triplet_a.mkdir(parents=True)
    triplet_b.mkdir(parents=True)
    (triplet_a / "t.png").write_bytes(b"fake-png-bytes")

    subset_dir = tmp_path / "subset"
    materialize_subset([triplet_a], subset_dir)

    assert (subset_dir / "triplet_a").is_dir()
    assert (subset_dir / "triplet_a" / "t.png").read_bytes() == b"fake-png-bytes"
    assert not (subset_dir / "triplet_b").exists()


def test_materialize_subset_is_idempotent(tmp_path):
    pool_dir = tmp_path / "pool"
    triplet_a = pool_dir / "triplet_a"
    triplet_a.mkdir(parents=True)

    subset_dir = tmp_path / "subset"
    materialize_subset([triplet_a], subset_dir)
    materialize_subset([triplet_a], subset_dir)  # must not raise on rerun

    assert (subset_dir / "triplet_a").is_dir()


def _write_triplets(triplets_dir: Path, count: int) -> None:
    for i in range(count):
        d = triplets_dir / f"triplet_{i}"
        d.mkdir(parents=True)
        for name in ("t-1.png", "t.png", "t+1.png"):
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(d / name), img)


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)
def test_run_finetune_data_ablation_evaluates_each_count(tmp_path):
    finetune_dir = tmp_path / "finetune_pool"
    _write_triplets(finetune_dir, count=4)
    test_dir = tmp_path / "test"
    _write_triplets(test_dir, count=2)
    out_dir = tmp_path / "out"

    results = run_finetune_data_ablation(
        MODEL_PATH, finetune_dir, test_dir, counts=[2, 4], out_dir=out_dir, epochs=1
    )

    assert set(results) == {2, 4}
    for mean_psnr, mean_ssim, mean_lpips in results.values():
        assert mean_psnr == mean_psnr  # not NaN
    assert (out_dir / "model_2.pt").exists()
    assert (out_dir / "model_4.pt").exists()
    assert (out_dir / "test_2.csv").exists()
    assert (out_dir / "test_4.csv").exists()


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)
def test_run_finetune_data_ablation_skips_counts_larger_than_the_pool(tmp_path):
    finetune_dir = tmp_path / "finetune_pool"
    _write_triplets(finetune_dir, count=2)
    test_dir = tmp_path / "test"
    _write_triplets(test_dir, count=1)
    out_dir = tmp_path / "out"

    results = run_finetune_data_ablation(
        MODEL_PATH, finetune_dir, test_dir, counts=[2, 8], out_dir=out_dir, epochs=1
    )

    assert set(results) == {2}
    assert not (out_dir / "model_8.pt").exists()
