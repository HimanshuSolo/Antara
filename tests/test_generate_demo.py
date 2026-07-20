from pathlib import Path

import cv2
import numpy as np
import pytest

from src.eval.generate_demo import build_demo_html, encode_png, save_demos, write_demo

MODEL_PATH = Path("models/film_net_fp32.pt")


def test_build_demo_html_embeds_images_metrics_and_slider():
    html = build_demo_html(
        triplet_name="triplet_0000",
        farneback_b64="AAAA",
        film_b64="BBBB",
        farneback_metrics=(20.0, 0.5),
        film_metrics=(30.0, 0.9),
    )

    assert "triplet_0000" in html
    assert "AAAA" in html
    assert "BBBB" in html
    assert "20.0" in html
    assert "30.0" in html
    assert '<input type="range"' in html


def test_encode_png_roundtrips_through_base64():
    import base64

    img = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
    b64 = encode_png(img)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(b64), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(decoded, img)


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)
def test_write_demo_writes_self_contained_html(tmp_path):
    triplet_dir = tmp_path / "triplet_0"
    triplet_dir.mkdir()
    for name in ("t-1.png", "t.png", "t+1.png"):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(triplet_dir / name), img)

    out_path = tmp_path / "out" / "demo.html"
    written = write_demo(triplet_dir, MODEL_PATH, out_path)

    assert written == out_path
    assert out_path.exists()
    assert "data:image/png;base64," in out_path.read_text()


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="FILM checkpoint not downloaded -- run `python -m src.deep.download_film` first",
)
def test_save_demos_writes_one_html_per_triplet(tmp_path):
    triplets_dir = tmp_path / "triplets"
    for i in range(2):
        triplet_dir = triplets_dir / f"triplet_{i}"
        triplet_dir.mkdir(parents=True)
        for name in ("t-1.png", "t.png", "t+1.png"):
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(triplet_dir / name), img)
    out_dir = tmp_path / "out"

    written = save_demos(triplets_dir, MODEL_PATH, out_dir)

    assert len(written) == 2
    assert all(p.exists() for p in written)
