import base64
import io

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.api import gallery


def _fake_frame(value: int = 0) -> np.ndarray:
    return np.full((64, 64), value, dtype=np.uint8)


def _write_triplet(dir_path):
    dir_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dir_path / "t-1.png"), _fake_frame(50))
    cv2.imwrite(str(dir_path / "t.png"), _fake_frame(100))
    cv2.imwrite(str(dir_path / "t+1.png"), _fake_frame(150))


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(gallery.router)
    return TestClient(app)


def test_list_gallery_skips_items_with_missing_data(tmp_path, monkeypatch):
    present_dir = tmp_path / "present"
    _write_triplet(present_dir)
    missing_dir = tmp_path / "missing"

    monkeypatch.setattr(
        gallery,
        "GALLERY_ITEMS",
        [
            {"id": "present", "label": "Present", "subset": "calm", "dir": present_dir},
            {"id": "missing", "label": "Missing", "subset": "cyclone", "dir": missing_dir},
        ],
    )

    resp = _client().get("/api/gallery")

    assert resp.status_code == 200
    body = resp.json()
    assert [item["id"] for item in body] == ["present"]
    assert body[0]["frame_prev"].startswith("data:image/png;base64,")
    assert body[0]["frame_next"].startswith("data:image/png;base64,")


def test_generate_runs_pipeline_and_reports_metrics(tmp_path, monkeypatch):
    item_dir = tmp_path / "item"
    _write_triplet(item_dir)
    monkeypatch.setattr(
        gallery, "GALLERY_ITEMS", [{"id": "item", "label": "Item", "subset": "calm", "dir": item_dir}]
    )
    gallery._cache.clear()

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    monkeypatch.setattr("src.api.live.resolve_model_path", lambda: model_path)
    monkeypatch.setattr(gallery, "farneback_interpolate", lambda a, b: _fake_frame(90))
    # ground truth is a constant 100; film_pred (99) is much closer to it
    # than farneback_pred (90), so film should score higher without
    # hitting an exact match (which drives PSNR to infinity).
    monkeypatch.setattr(gallery, "film_interpolate", lambda a, b, path: _fake_frame(99))

    resp = _client().post("/api/gallery/item/generate")

    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "model.pt"
    assert body["farneback_mid"].startswith("data:image/png;base64,")
    assert body["film_mid"].startswith("data:image/png;base64,")
    assert body["ground_truth"].startswith("data:image/png;base64,")
    assert body["film_psnr"] > body["farneback_psnr"]


def test_generate_caches_by_item_and_model(tmp_path, monkeypatch):
    item_dir = tmp_path / "item"
    _write_triplet(item_dir)
    monkeypatch.setattr(
        gallery, "GALLERY_ITEMS", [{"id": "item", "label": "Item", "subset": "calm", "dir": item_dir}]
    )
    gallery._cache.clear()

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    monkeypatch.setattr("src.api.live.resolve_model_path", lambda: model_path)

    calls = []
    monkeypatch.setattr(gallery, "farneback_interpolate", lambda a, b: (calls.append(1), _fake_frame(90))[1])
    monkeypatch.setattr(gallery, "film_interpolate", lambda a, b, path: _fake_frame(99))

    client = _client()
    client.post("/api/gallery/item/generate")
    client.post("/api/gallery/item/generate")

    assert len(calls) == 1


def test_generate_404s_for_unknown_item(monkeypatch):
    monkeypatch.setattr(gallery, "GALLERY_ITEMS", [])

    resp = _client().post("/api/gallery/nonexistent/generate")

    assert resp.status_code == 404


def test_generate_loop_returns_gif_with_correct_frame_count(tmp_path, monkeypatch):
    item_dir = tmp_path / "item"
    _write_triplet(item_dir)
    monkeypatch.setattr(
        gallery, "GALLERY_ITEMS", [{"id": "item", "label": "Item", "subset": "calm", "dir": item_dir}]
    )
    gallery._loop_cache.clear()

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    monkeypatch.setattr("src.api.live.resolve_model_path", lambda: model_path)
    monkeypatch.setattr(
        "src.deep.film_interpolate.interpolate_multi",
        lambda a, b, path, num_frames=3, device=None: [_fake_frame(90 + i) for i in range(num_frames)],
    )

    resp = _client().post("/api/gallery/item/loop?num_frames=4")

    assert resp.status_code == 200
    body = resp.json()
    assert body["num_frames"] == 4
    assert body["model"] == "model.pt"
    assert body["loop_gif"].startswith("data:image/gif;base64,")

    gif_bytes = base64.b64decode(body["loop_gif"].split(",", 1)[1])
    im = Image.open(io.BytesIO(gif_bytes))
    assert im.n_frames == 6  # t-1, 4 interpolated, t+1


def test_generate_loop_returns_a_playable_mp4(tmp_path, monkeypatch):
    item_dir = tmp_path / "item"
    _write_triplet(item_dir)
    monkeypatch.setattr(
        gallery, "GALLERY_ITEMS", [{"id": "item", "label": "Item", "subset": "calm", "dir": item_dir}]
    )
    gallery._loop_cache.clear()

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    monkeypatch.setattr("src.api.live.resolve_model_path", lambda: model_path)
    monkeypatch.setattr(
        "src.deep.film_interpolate.interpolate_multi",
        lambda a, b, path, num_frames=3, device=None: [_fake_frame(90 + i) for i in range(num_frames)],
    )

    resp = _client().post("/api/gallery/item/loop?num_frames=4")

    assert resp.status_code == 200
    body = resp.json()
    assert body["loop_mp4"] is not None
    assert body["loop_mp4"].startswith("data:video/mp4;base64,")

    mp4_bytes = base64.b64decode(body["loop_mp4"].split(",", 1)[1])
    mp4_path = tmp_path / "out.mp4"
    mp4_path.write_bytes(mp4_bytes)
    cap = cv2.VideoCapture(str(mp4_path))
    count = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    assert count == 6  # t-1, 4 interpolated, t+1


def test_generate_loop_mp4_is_none_when_ffmpeg_unavailable(tmp_path, monkeypatch):
    item_dir = tmp_path / "item"
    _write_triplet(item_dir)
    monkeypatch.setattr(
        gallery, "GALLERY_ITEMS", [{"id": "item", "label": "Item", "subset": "calm", "dir": item_dir}]
    )
    gallery._loop_cache.clear()

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    monkeypatch.setattr("src.api.live.resolve_model_path", lambda: model_path)
    monkeypatch.setattr(
        "src.deep.film_interpolate.interpolate_multi",
        lambda a, b, path, num_frames=3, device=None: [_fake_frame(90) for _ in range(num_frames)],
    )
    monkeypatch.setattr(gallery.shutil, "which", lambda cmd: None)

    resp = _client().post("/api/gallery/item/loop?num_frames=2")

    assert resp.status_code == 200
    assert resp.json()["loop_mp4"] is None


def test_generate_loop_caches_by_item_model_and_num_frames(tmp_path, monkeypatch):
    item_dir = tmp_path / "item"
    _write_triplet(item_dir)
    monkeypatch.setattr(
        gallery, "GALLERY_ITEMS", [{"id": "item", "label": "Item", "subset": "calm", "dir": item_dir}]
    )
    gallery._loop_cache.clear()

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    monkeypatch.setattr("src.api.live.resolve_model_path", lambda: model_path)

    calls = []

    def fake_interpolate_multi(a, b, path, num_frames=3, device=None):
        calls.append(num_frames)
        return [_fake_frame(90) for _ in range(num_frames)]

    monkeypatch.setattr("src.deep.film_interpolate.interpolate_multi", fake_interpolate_multi)

    client = _client()
    client.post("/api/gallery/item/loop?num_frames=3")
    client.post("/api/gallery/item/loop?num_frames=3")
    client.post("/api/gallery/item/loop?num_frames=5")

    assert calls == [3, 5]


def test_generate_loop_404s_for_unknown_item(monkeypatch):
    monkeypatch.setattr(gallery, "GALLERY_ITEMS", [])

    resp = _client().post("/api/gallery/nonexistent/loop")

    assert resp.status_code == 404


def test_generate_loop_422s_for_invalid_num_frames():
    resp = _client().post("/api/gallery/anything/loop?num_frames=15")

    assert resp.status_code == 422
