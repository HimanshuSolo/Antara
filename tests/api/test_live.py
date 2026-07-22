import numpy as np
from fastapi.testclient import TestClient

from src.api import live


def _fake_frame() -> np.ndarray:
    return np.zeros((64, 64), dtype=np.uint8)


def _setup_common(tmp_path, monkeypatch):
    live._cache.clear()

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake")
    monkeypatch.setattr(live, "MODEL_CANDIDATES", [model_path])

    prev_path = tmp_path / "OR_ABI-L1b-RadF-M6C13_G16_s20241001200206_e1_c1.nc"
    next_path = tmp_path / "OR_ABI-L1b-RadF-M6C13_G16_s20241001210206_e1_c1.nc"
    prev_path.write_bytes(b"")
    next_path.write_bytes(b"")

    monkeypatch.setattr(
        live.fetch_goes, "latest_scan_pair", lambda band=13: (str(prev_path), str(next_path))
    )

    downloaded: list[str] = []

    def fake_download(key, dest_dir, client=None, bucket=None):
        downloaded.append(key)
        return prev_path if key == str(prev_path) else next_path

    monkeypatch.setattr(live.fetch_goes, "download", fake_download)
    monkeypatch.setattr(
        live.extract_triplets, "load_radiance", lambda p: np.zeros((512, 512), dtype=np.float32)
    )
    monkeypatch.setattr(live, "farneback_interpolate", lambda a, b: _fake_frame())
    monkeypatch.setattr(live, "film_interpolate", lambda a, b, model_path: _fake_frame())

    return downloaded, model_path


def test_get_live_runs_pipeline_and_returns_pngs(tmp_path, monkeypatch):
    downloaded, model_path = _setup_common(tmp_path, monkeypatch)

    resp = TestClient(live.app).get("/api/live")

    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == model_path.name
    assert body["frame_prev"].startswith("data:image/png;base64,")
    assert body["frame_next"].startswith("data:image/png;base64,")
    assert body["farneback_mid"].startswith("data:image/png;base64,")
    assert body["film_mid"].startswith("data:image/png;base64,")
    assert body["cadence_minutes"] == 10.0
    assert downloaded == [str(tmp_path / "OR_ABI-L1b-RadF-M6C13_G16_s20241001200206_e1_c1.nc"),
                           str(tmp_path / "OR_ABI-L1b-RadF-M6C13_G16_s20241001210206_e1_c1.nc")]


def test_get_live_caches_the_latest_pair(tmp_path, monkeypatch):
    downloaded, _ = _setup_common(tmp_path, monkeypatch)
    client = TestClient(live.app)

    client.get("/api/live")
    client.get("/api/live")

    assert len(downloaded) == 2  # only downloaded once across both requests


def test_get_live_502s_when_no_recent_scans(monkeypatch):
    live._cache.clear()

    def raise_no_scans(band=13):
        raise RuntimeError("no scans")

    monkeypatch.setattr(live.fetch_goes, "latest_scan_pair", raise_no_scans)
    monkeypatch.setattr(live, "MODEL_CANDIDATES", [])
    # give resolve_model_path a real path so we reach the scan lookup at all
    import pathlib

    monkeypatch.setattr(live, "MODEL_CANDIDATES", [pathlib.Path(__file__)])

    resp = TestClient(live.app).get("/api/live")

    assert resp.status_code == 502


def test_get_live_503s_when_no_model_checkpoint_available(monkeypatch, tmp_path):
    live._cache.clear()
    monkeypatch.setattr(live, "MODEL_CANDIDATES", [tmp_path / "missing.pt"])

    resp = TestClient(live.app).get("/api/live")

    assert resp.status_code == 503


def test_health_reports_model_availability(monkeypatch, tmp_path):
    present = tmp_path / "model.pt"
    present.write_bytes(b"x")
    monkeypatch.setattr(live, "MODEL_CANDIDATES", [present])

    resp = TestClient(live.app).get("/api/health")

    assert resp.json() == {"model_available": True}
