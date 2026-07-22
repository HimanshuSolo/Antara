import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import gallery


def _fake_frame(value: int = 0) -> np.ndarray:
    return np.full((64, 64), value, dtype=np.uint8)


def _write_triplet(dir_path):
    import cv2

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
