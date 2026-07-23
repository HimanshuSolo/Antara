from src.eval.eye_metrics import pixel_error


def test_pixel_error_of_identical_points_is_zero():
    assert pixel_error((10, 20), (10, 20)) == 0


def test_pixel_error_is_euclidean_distance():
    assert pixel_error((0, 0), (3, 4)) == 5.0


def test_pixel_error_is_symmetric():
    assert pixel_error((1, 2), (5, 8)) == pixel_error((5, 8), (1, 2))
