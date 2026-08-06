from __future__ import annotations

import numpy as np
import pandas as pd

from chart_core import (
    EVALUATION_STYLES,
    PairedChartDataset,
    build_examples,
    numeric_features,
    render_chart,
)


def sample_window() -> np.ndarray:
    base = np.linspace(100.0, 112.0, 20)
    opens = base + np.sin(np.arange(20))
    closes = base + np.cos(np.arange(20))
    highs = np.maximum(opens, closes) + 2.0
    lows = np.minimum(opens, closes) - 2.0
    volumes = np.linspace(10.0, 15.0, 20)
    return np.column_stack([opens, highs, lows, closes, volumes]).astype(np.float32)


def test_every_style_is_deterministic_and_nonempty() -> None:
    window = sample_window()
    for style in EVALUATION_STYLES:
        first = render_chart(window, style)
        second = render_chart(window, style)
        assert first.shape == (64, 64, 3)
        assert first.dtype == np.uint8
        assert np.array_equal(first, second)
        assert np.unique(first.reshape(-1, 3), axis=0).shape[0] >= 3


def test_renderer_is_price_unit_invariant_but_retains_return_scale() -> None:
    window = sample_window()
    rescaled = window.copy()
    rescaled[:, :4] *= 7.0
    assert np.array_equal(
        render_chart(window, "canonical"), render_chart(rescaled, "canonical")
    )

    amplified = window.copy()
    base = float(window[0, 3])
    amplified[:, :4] = base * np.square(np.maximum(window[:, :4], 1e-6) / base)
    assert not np.array_equal(
        render_chart(window, "canonical"), render_chart(amplified, "canonical")
    )


def test_build_examples_does_not_use_future_in_input() -> None:
    dates = pd.bdate_range("2020-01-01", periods=40)
    close = np.linspace(100, 120, len(dates))
    panel = pd.DataFrame(
        {
            "symbol": "TEST",
            "date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adj_close": close,
            "volume": 1000,
        }
    )
    examples = build_examples(panel, lookback=20, horizon=5, stride=5)
    assert examples
    first = examples[0]
    assert first.origin_date == str(dates[19].date())
    assert first.target_end_date == str(dates[24].date())
    assert first.ohlcv.shape == (20, 5)
    assert numeric_features(first).shape == (111,)


def test_paired_dataset_returns_matched_views() -> None:
    from chart_core import Example

    example = Example("TEST", "2020-01-01", "2020-01-08", sample_window(), -2.0)
    dataset = PairedChartDataset(
        [example],
        first_styles=("canonical",),
        second_styles=("dark",),
        randomize_first=False,
        randomize_second=False,
    )
    first, second, target, index = dataset[0]
    assert first.shape == second.shape == (3, 64, 64)
    assert not np.array_equal(first.numpy(), second.numpy())
    assert float(target) == -2.0
    assert index == 0
