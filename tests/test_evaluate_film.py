from pathlib import Path

import cv2
import numpy as np
import pytest

from src.eval.evaluate_film import evaluate

MODEL_PATH = Path("models/film_net_fp32.pt")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)


def _write_triplet(triplet_dir: Path) -> None:
    triplet_dir.mkdir(parents=True, exist_ok=True)
    for name in ("t-1.png", "t.png", "t+1.png"):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(triplet_dir / name), img)


def test_evaluate_writes_one_row_per_triplet_and_csv(tmp_path):
    triplets_dir = tmp_path / "triplets"
    for i in range(2):
        _write_triplet(triplets_dir / f"triplet_{i}")
    out_csv = tmp_path / "out" / "results.csv"

    rows = evaluate(triplets_dir, MODEL_PATH, out_csv)

    assert len(rows) == 2
    assert {row["triplet"] for row in rows} == {"triplet_0", "triplet_1"}
    for row in rows:
        assert set(row) == {"triplet", "psnr", "ssim", "lpips"}

    assert out_csv.exists()
    lines = out_csv.read_text().strip().splitlines()
    assert lines[0] == "triplet,psnr,ssim,lpips"
    assert len(lines) == 3


def test_evaluate_skips_incomplete_triplet(tmp_path):
    triplets_dir = tmp_path / "triplets"
    incomplete = triplets_dir / "triplet_0"
    incomplete.mkdir(parents=True)
    cv2.imwrite(str(incomplete / "t-1.png"), np.zeros((8, 8), dtype=np.uint8))
    # t.png and t+1.png deliberately missing.

    rows = evaluate(triplets_dir, MODEL_PATH, tmp_path / "out.csv")

    assert rows == []


def test_evaluate_returns_empty_list_for_empty_dir(tmp_path):
    triplets_dir = tmp_path / "triplets"
    triplets_dir.mkdir()

    rows = evaluate(triplets_dir, MODEL_PATH, tmp_path / "out.csv")

    assert rows == []
