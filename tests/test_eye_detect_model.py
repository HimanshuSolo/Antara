import torch

from src.deep.eye_detect_model import EyeDetectCNN


def test_forward_pass_output_shape():
    model = EyeDetectCNN()
    x = torch.rand(4, 1, 64, 64)

    out = model(x)

    assert out.shape == (4, 2)


def test_forward_pass_output_in_unit_range():
    model = EyeDetectCNN()
    x = torch.rand(2, 1, 64, 64)

    out = model(x)

    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_forward_pass_accepts_different_input_sizes():
    model = EyeDetectCNN()
    out_small = model(torch.rand(1, 1, 32, 32))
    out_large = model(torch.rand(1, 1, 256, 256))

    assert out_small.shape == (1, 2)
    assert out_large.shape == (1, 2)
