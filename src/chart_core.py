"""Core data, rendering, models, and metrics for the chart-robustness study."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChartStyle:
    background: tuple[int, int, int]
    up: tuple[int, int, int]
    down: tuple[int, int, int]
    wick: tuple[int, int, int]
    volume_up: tuple[int, int, int]
    volume_down: tuple[int, int, int]
    hollow_up: bool = False
    body_scale: float = 1.0
    price_padding: float = 0.04


STYLES: dict[str, ChartStyle] = {
    "canonical": ChartStyle(
        (250, 250, 250), (32, 151, 93), (220, 68, 70), (45, 45, 45),
        (112, 196, 161), (235, 139, 140), price_padding=0.04,
    ),
    # Red-up/green-down is common in East Asian markets.
    "reversed": ChartStyle(
        (250, 250, 250), (220, 68, 70), (32, 151, 93), (45, 45, 45),
        (235, 139, 140), (112, 196, 161), price_padding=0.04,
    ),
    "dark": ChartStyle(
        (18, 22, 29), (38, 208, 124), (244, 83, 92), (215, 220, 226),
        (45, 126, 91), (139, 61, 69), price_padding=0.04,
    ),
    # Direction remains encoded by hollow-up versus filled-down candles.
    "monochrome": ChartStyle(
        (255, 255, 255), (35, 35, 35), (35, 35, 35), (35, 35, 35),
        (125, 125, 125), (125, 125, 125), hollow_up=True, price_padding=0.04,
    ),
    # Held out from finite style augmentation: a common colorblind-safe palette.
    "blue_orange": ChartStyle(
        (246, 247, 249), (0, 114, 178), (230, 123, 16), (50, 54, 59),
        (92, 156, 196), (235, 171, 94), body_scale=0.72, price_padding=0.12,
    ),
}

TRAIN_AUGMENTATION_STYLES = ("canonical", "reversed", "dark", "monochrome")
EVALUATION_STYLES = tuple(STYLES)


@dataclass(frozen=True)
class Example:
    symbol: str
    origin_date: str
    target_end_date: str
    ohlcv: np.ndarray
    target_log_rv: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def realized_volatility(log_returns: np.ndarray) -> float:
    return float(math.sqrt(252.0 / len(log_returns) * np.square(log_returns).sum()))


def build_examples(
    panel: pd.DataFrame,
    lookback: int = 20,
    horizon: int = 5,
    stride: int = 5,
) -> list[Example]:
    """Build weekly, non-overlapping volatility forecast examples.

    OHLC values are adjusted by the same daily factor as adjusted close, avoiding
    split discontinuities while retaining within-day candle geometry. Volume is
    transformed as log1p(volume); the renderer normalizes it within each window.
    """

    examples: list[Example] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        group = group.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        close = group["close"].to_numpy(dtype=np.float64)
        adjusted = group["adj_close"].to_numpy(dtype=np.float64)
        factor = np.divide(adjusted, close, out=np.ones_like(adjusted), where=close != 0)
        ohlc = group[["open", "high", "low", "close"]].to_numpy(dtype=np.float64, copy=True)
        ohlc *= factor[:, None]
        volume = np.log1p(group["volume"].fillna(0).to_numpy(dtype=np.float64))[:, None]
        values = np.concatenate([ohlc, volume], axis=1).astype(np.float32)
        dates = pd.to_datetime(group["date"]).dt.strftime("%Y-%m-%d").to_numpy()
        for end in range(lookback - 1, len(group) - horizon, stride):
            future = np.diff(np.log(adjusted[end : end + horizon + 1]))
            if not np.all(np.isfinite(future)):
                continue
            rv = realized_volatility(future)
            if not (rv > 0 and np.isfinite(rv)):
                continue
            window = values[end - lookback + 1 : end + 1].copy()
            if not np.all(np.isfinite(window)):
                continue
            examples.append(
                Example(
                    symbol=symbol,
                    origin_date=str(dates[end]),
                    target_end_date=str(dates[end + horizon]),
                    ohlcv=window,
                    target_log_rv=math.log(rv),
                )
            )
    return examples


def render_chart(
    window: np.ndarray,
    style_name: str,
    height: int = 64,
    width: int = 64,
    body_scale_override: float | None = None,
    price_padding_override: float | None = None,
) -> np.ndarray:
    """Render all OHLCV values without axes, text, or stochastic operations."""

    style = STYLES[style_name]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = style.background
    price_height = int(height * 0.78)
    volume_top = price_height + 2
    opens, highs, lows, closes, volumes = window.T
    # A fixed monotone map of log price relative to the first close preserves
    # percentage-move scale across windows. Per-window min/max autoscaling would
    # erase the most important signal for volatility forecasting.
    base_price = max(float(closes[0]), 1e-12)
    price_padding = style.price_padding if price_padding_override is None else price_padding_override
    return_scale = 0.10 * (1.0 + 2.0 * price_padding)

    def price_y(value: float) -> int:
        relative_log_price = math.log(max(float(value), 1e-12) / base_price)
        transformed = math.tanh(relative_log_price / return_scale)
        normalized = (1.0 - transformed) / 2.0
        return int(np.clip(round(2 + normalized * (price_height - 5)), 1, price_height - 2))

    centers = np.linspace(3, width - 4, len(window)).round().astype(int)
    body_scale = style.body_scale if body_scale_override is None else body_scale_override
    body_half = max(1, int((width / len(window)) * 0.30 * body_scale))
    min_volume = float(np.min(volumes))
    volume_span = max(float(np.max(volumes)) - min_volume, 1e-6)
    for index, x in enumerate(centers):
        up = closes[index] >= opens[index]
        color = style.up if up else style.down
        y_high, y_low = price_y(highs[index]), price_y(lows[index])
        image[min(y_high, y_low) : max(y_high, y_low) + 1, x, :] = style.wick
        y_open, y_close = price_y(opens[index]), price_y(closes[index])
        top, bottom = min(y_open, y_close), max(y_open, y_close)
        bottom = max(bottom, top + 1)
        left, right = max(0, x - body_half), min(width - 1, x + body_half)
        if up and style.hollow_up:
            image[top, left : right + 1, :] = color
            image[bottom, left : right + 1, :] = color
            image[top : bottom + 1, left, :] = color
            image[top : bottom + 1, right, :] = color
        else:
            image[top : bottom + 1, left : right + 1, :] = color
        relative_volume = (float(volumes[index]) - min_volume) / volume_span
        volume_height = int(round(relative_volume * (height - volume_top - 2)))
        if volume_height > 0:
            volume_color = style.volume_up if up else style.volume_down
            image[height - 2 - volume_height : height - 1, left : right + 1, :] = volume_color
    return image


class ChartDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        styles: Iterable[str],
        randomize: bool = False,
        procedural: bool = False,
    ):
        self.examples = examples
        self.styles = tuple(styles)
        self.randomize = randomize
        self.procedural = procedural
        if not self.styles:
            raise ValueError("at least one style is required")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        style = random.choice(self.styles) if self.randomize else self.styles[0]
        body_scale = random.uniform(0.65, 1.35) if self.procedural else None
        price_padding = random.uniform(0.0, 0.14) if self.procedural else None
        image = render_chart(
            example.ohlcv,
            style,
            body_scale_override=body_scale,
            price_padding_override=price_padding,
        )
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        return tensor, torch.tensor(example.target_log_rv, dtype=torch.float32), index


class PairedChartDataset(Dataset):
    """Two information-equivalent views of each OHLCV window.

    The first view supplies the supervised forecast loss. The second view is
    used for a paired prediction-consistency penalty. Random pairs are sampled
    independently on every epoch, with distinct styles whenever possible.
    """

    def __init__(
        self,
        examples: list[Example],
        first_styles: Iterable[str],
        second_styles: Iterable[str],
        randomize_first: bool,
        randomize_second: bool,
        procedural_first: bool = False,
        procedural_second: bool = False,
    ):
        self.examples = examples
        self.first_styles = tuple(first_styles)
        self.second_styles = tuple(second_styles)
        self.randomize_first = randomize_first
        self.randomize_second = randomize_second
        self.procedural_first = procedural_first
        self.procedural_second = procedural_second
        if not self.first_styles or not self.second_styles:
            raise ValueError("both paired style sets must be nonempty")

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _select(styles: tuple[str, ...], randomize: bool) -> str:
        return random.choice(styles) if randomize else styles[0]

    @staticmethod
    def _tensor(window: np.ndarray, style: str, procedural: bool) -> torch.Tensor:
        body_scale = random.uniform(0.65, 1.35) if procedural else None
        price_padding = random.uniform(0.0, 0.14) if procedural else None
        image = render_chart(
            window,
            style,
            body_scale_override=body_scale,
            price_padding_override=price_padding,
        )
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)

    def __getitem__(self, index: int):
        example = self.examples[index]
        first_style = self._select(self.first_styles, self.randomize_first)
        eligible_second = tuple(style for style in self.second_styles if style != first_style)
        second_pool = eligible_second or self.second_styles
        second_style = self._select(second_pool, self.randomize_second)
        return (
            self._tensor(example.ohlcv, first_style, self.procedural_first),
            self._tensor(example.ohlcv, second_style, self.procedural_second),
            torch.tensor(example.target_log_rv, dtype=torch.float32),
            index,
        )


class EdgeMap(nn.Module):
    """Fixed, per-image normalized Sobel magnitude to discard rendering color."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])[None, None],
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])[None, None],
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = (
            0.2126 * images[:, 0:1]
            + 0.7152 * images[:, 1:2]
            + 0.0722 * images[:, 2:3]
        )
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
        scale = magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return magnitude / scale


class TinyChartCNN(nn.Module):
    def __init__(self, use_edges: bool = False):
        super().__init__()
        self.edge_map = EdgeMap() if use_edges else nn.Identity()
        channels = 1 if use_edges else 3
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.edge_map(inputs))).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(4, out_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(inputs) + self.skip(inputs))


class SmallResNet(nn.Module):
    """A compact residual architecture used as a cross-architecture check."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1, bias=False), nn.GroupNorm(4, 24), nn.ReLU()
        )
        self.blocks = nn.Sequential(
            ResidualBlock(24, 24),
            ResidualBlock(24, 48, stride=2),
            ResidualBlock(48, 48),
            ResidualBlock(48, 96, stride=2),
            ResidualBlock(96, 96),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(96, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.stem(inputs)).flatten(1)
        return self.head(values).squeeze(-1)


def regression_metrics(
    targets_log_rv: np.ndarray,
    predictions_log_rv: np.ndarray,
    high_vol_threshold: float,
) -> dict[str, float]:
    labels = targets_log_rv >= high_vol_threshold
    spearman = float(spearmanr(targets_log_rv, predictions_log_rv).statistic)
    auc = float(roc_auc_score(labels, predictions_log_rv)) if len(np.unique(labels)) == 2 else float("nan")
    realized_var = np.exp(2.0 * targets_log_rv)
    predicted_var = np.exp(2.0 * np.clip(predictions_log_rv, -8.0, 3.0))
    ratio = realized_var / np.maximum(predicted_var, 1e-12)
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
    return {
        "mae_log_rv": float(mean_absolute_error(targets_log_rv, predictions_log_rv)),
        "spearman": spearman,
        "high_vol_auc": auc,
        "qlike": qlike,
    }


def numeric_features(example: Example) -> np.ndarray:
    """Causal numeric representation of the exact same 20-bar OHLCV window."""

    values = example.ohlcv.astype(np.float64)
    opens, highs, lows, closes, log_volume = values.T
    previous_close = np.r_[closes[0], closes[:-1]]
    returns = np.log(np.maximum(closes, 1e-12) / np.maximum(previous_close, 1e-12))
    overnight = np.log(np.maximum(opens, 1e-12) / np.maximum(previous_close, 1e-12))
    intraday = np.log(np.maximum(closes, 1e-12) / np.maximum(opens, 1e-12))
    ranges = np.log(np.maximum(highs, 1e-12) / np.maximum(lows, 1e-12))
    volume = (log_volume - log_volume.mean()) / max(log_volume.std(), 1e-6)
    sequence = np.column_stack([returns, overnight, intraday, ranges, volume]).ravel()
    summary = np.asarray(
        [
            np.sqrt(252.0 * np.mean(np.square(returns[-5:]))),
            np.sqrt(252.0 * np.mean(np.square(returns[-10:]))),
            np.sqrt(252.0 * np.mean(np.square(returns))),
            np.mean(ranges[-5:]),
            np.mean(ranges),
            np.max(ranges),
            np.mean(np.abs(returns)),
            np.sum(returns[-5:]),
            np.sum(returns),
            volume[-1],
            np.mean(volume[-5:]),
        ],
        dtype=np.float64,
    )
    return np.concatenate([sequence, summary]).astype(np.float32)
