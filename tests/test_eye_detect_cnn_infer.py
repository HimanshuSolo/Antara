import numpy as np
import torch

from src.deep.eye_detect import detect_eye
from src.deep.eye_detect_model import EyeDetectCNN


def test_detect_eye_returns_pixel_coords_within_frame_bounds(tmp_path):
    model = EyeDetectCNN()
    model_path = tmp_path / "eye_detect_cnn.pt"
    torch.save(model.state_dict(), model_path)

    frame = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    row, col = detect_eye(frame, model_path)

    assert 0 <= row < 64
    assert 0 <= col < 64


def test_detect_eye_caches_the_loaded_model(tmp_path, monkeypatch):
    import src.deep.eye_detect as eye_detect_module

    model = EyeDetectCNN()
    model_path = tmp_path / "eye_detect_cnn.pt"
    torch.save(model.state_dict(), model_path)
    eye_detect_module._model_cache.clear()

    load_calls = []
    real_torch_load = torch.load

    def counting_load(*args, **kwargs):
        load_calls.append(1)
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)

    frame = np.zeros((32, 32), dtype=np.uint8)
    detect_eye(frame, model_path)
    detect_eye(frame, model_path)

    assert len(load_calls) == 1
