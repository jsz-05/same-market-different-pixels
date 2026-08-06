import numpy as np

from external_validation import fixed_coordinates, grayscale_background_contrast


def sample_window() -> np.ndarray:
    close = np.linspace(100.0, 104.0, 20)
    open_ = close - 0.3
    high = close + 0.8
    low = close - 0.9
    volume = np.linspace(10.0, 20.0, 20)
    return np.column_stack([open_, high, low, close, volume]).astype(np.float32)


def test_fixed_coordinates_preserve_order_and_scale() -> None:
    frame = fixed_coordinates(sample_window())
    assert frame.shape == (20, 5)
    assert frame["Close"].iloc[0] == 2.0
    assert np.all(frame["High"] >= np.maximum(frame["Open"], frame["Close"]))
    assert np.all(frame["Low"] <= np.minimum(frame["Open"], frame["Close"]))
    assert frame["Volume"].min() == 0.0
    assert frame["Volume"].max() == 1.0


def test_background_normalization_matches_exact_inversion() -> None:
    light = np.full((1, 16, 16, 3), 240, dtype=np.uint8)
    light[:, 4:12, 7:9, :] = 40
    dark = 255 - light
    normalized_light = grayscale_background_contrast(light)
    normalized_dark = grayscale_background_contrast(dark)
    assert np.array_equal(normalized_light, normalized_dark)
    assert normalized_light.dtype == np.uint8
