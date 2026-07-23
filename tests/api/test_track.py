from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import track


def _fake_frame(value: int = 0) -> np.ndarray:
    return np.full((64, 64), value, dtype=np.uint8)


def _write_triplet(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dir_path / "t-1.png"), _fake_frame(50))
    cv2.imwrite(str(dir_path / "t.png"), _fake_frame(100))
    cv2.imwrite(str(dir_path / "t+1.png"), _fake_frame(150))


def _write_eye_labels(triplets_dir: Path, triplet_name: str, row: int, col: int) -> None:
    triplets_dir.mkdir(parents=True, exist_ok=True)
    (triplets_dir / "eye_labels.csv").write_text(
        f"triplet,frame,row,col\n{triplet_name},t,{row},{col}\n"
    )


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(track.router)
    return TestClient(app)


class _FakeGenerateResult:
    def __init__(self, model: str, farneback_mid: str, film_mid: str):
        self.model = model
        self.farneback_mid = farneback_mid
        self.film_mid = film_mid


def _setup_item(tmp_path, monkeypatch, with_ground_truth: bool = True):
    triplets_dir = tmp_path / "triplets_cyclone"
    item_dir = triplets_dir / "triplet_0000"
    _write_triplet(item_dir)
    if with_ground_truth:
        _write_eye_labels(triplets_dir, "triplet_0000", row=32, col=32)

    monkeypatch.setattr(
        track, "TRACK_ITEMS", [{"id": "item", "label": "Item", "subset": "cyclone", "dir": item_dir}]
    )
    track._cache.clear()

    eye_model_path = tmp_path / "eye_detect_cnn.pt"
    eye_model_path.write_bytes(b"fake")
    monkeypatch.setattr(track, "EYE_MODEL_CANDIDATES", [eye_model_path])

    fake_result = _FakeGenerateResult(
        model="film.pt",
        farneback_mid=track.encode_png(_fake_frame(90)),
        film_mid=track.encode_png(_fake_frame(95)),
    )
    monkeypatch.setattr(track, "generate", lambda item_id: fake_result)


def test_list_track_items_returns_only_present_cyclone_items(tmp_path, monkeypatch):
    present_dir = tmp_path / "present"
    _write_triplet(present_dir)
    missing_dir = tmp_path / "missing"

    monkeypatch.setattr(
        track,
        "TRACK_ITEMS",
        [
            {"id": "present", "label": "Present", "subset": "cyclone", "dir": present_dir},
            {"id": "missing", "label": "Missing", "subset": "cyclone", "dir": missing_dir},
        ],
    )

    resp = _client().get("/api/track")

    assert resp.status_code == 200
    body = resp.json()
    assert [item["id"] for item in body] == ["present"]
    assert body[0]["frame_prev"].startswith("data:image/png;base64,")


def test_detect_returns_ground_truth_and_all_detections(tmp_path, monkeypatch):
    _setup_item(tmp_path, monkeypatch)
    monkeypatch.setattr(track, "classical_detect_eye", lambda frame: (30, 31))
    monkeypatch.setattr(track, "cnn_detect_eye", lambda frame, model_path: (33, 34))

    resp = _client().post("/api/track/item/detect")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ground_truth"] == {"row": 32, "col": 32}
    assert body["classical_real"] == {"row": 30, "col": 31}
    assert body["cnn_real"] == {"row": 33, "col": 34}
    assert body["classical_accuracy_px"] == round(((30 - 32) ** 2 + (31 - 32) ** 2) ** 0.5, 2)
    assert body["cnn_accuracy_px"] == round(((33 - 32) ** 2 + (34 - 32) ** 2) ** 0.5, 2)
    assert body["classical_farneback_drift_px"] == 0.0  # farneback/film dets are stubbed identically
    assert body["annotated_real"].startswith("data:image/png;base64,")
    assert body["annotated_farneback"].startswith("data:image/png;base64,")
    assert body["annotated_film"].startswith("data:image/png;base64,")


def test_detect_handles_missing_ground_truth(tmp_path, monkeypatch):
    _setup_item(tmp_path, monkeypatch, with_ground_truth=False)
    monkeypatch.setattr(track, "classical_detect_eye", lambda frame: (30, 31))
    monkeypatch.setattr(track, "cnn_detect_eye", lambda frame, model_path: (33, 34))

    resp = _client().post("/api/track/item/detect")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ground_truth"] is None
    assert body["classical_accuracy_px"] is None
    assert body["cnn_accuracy_px"] is None


def test_detect_caches_by_item_film_model_and_eye_model(tmp_path, monkeypatch):
    _setup_item(tmp_path, monkeypatch)
    calls = []

    def fake_classical(frame):
        calls.append(1)
        return (30, 31)

    monkeypatch.setattr(track, "classical_detect_eye", fake_classical)
    monkeypatch.setattr(track, "cnn_detect_eye", lambda frame, model_path: (33, 34))

    client = _client()
    client.post("/api/track/item/detect")
    client.post("/api/track/item/detect")

    assert len(calls) == 3  # real + farneback + film, only for the first (uncached) call


def test_detect_404s_for_unknown_item(monkeypatch):
    monkeypatch.setattr(track, "TRACK_ITEMS", [])

    resp = _client().post("/api/track/nonexistent/detect")

    assert resp.status_code == 404


def test_detect_503s_when_no_eye_model_checkpoint(tmp_path, monkeypatch):
    item_dir = tmp_path / "triplet_0000"
    _write_triplet(item_dir)
    monkeypatch.setattr(
        track, "TRACK_ITEMS", [{"id": "item", "label": "Item", "subset": "cyclone", "dir": item_dir}]
    )
    monkeypatch.setattr(track, "EYE_MODEL_CANDIDATES", [tmp_path / "nonexistent.pt"])

    resp = _client().post("/api/track/item/detect")

    assert resp.status_code == 503
