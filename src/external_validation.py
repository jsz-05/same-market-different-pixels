#!/usr/bin/env python3
"""Post-protocol external-renderer and cheap-repair validation.

The frozen neural checkpoints are never updated here. External chart libraries
receive the same fixed price coordinates and normalized volume as the custom
renderer, isolating rasterizer, palette, line, candle, and panel-layout changes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import time
from io import BytesIO
from pathlib import Path

import kaleido
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from PIL import Image
from plotly.subplots import make_subplots

from chart_core import Example, build_examples, choose_device, render_chart
from run_study import make_model, predict_neural, split_examples


EXTERNAL_RENDERERS = {
    "mplfinance_yahoo_wide": {
        "library": "mplfinance",
        "style": "yahoo",
        "panel_ratios": (2, 1),
        "candle_width": 1.10,
        "candle_linewidth": 0.35,
        "volume_width": 0.95,
    },
    "mplfinance_nightclouds_thin": {
        "library": "mplfinance",
        "style": "nightclouds",
        "panel_ratios": (7, 1),
        "candle_width": 0.28,
        "candle_linewidth": 2.20,
        "volume_width": 0.45,
    },
    "plotly_white_narrow_volume": {
        "library": "plotly",
        "style": "plotly_white",
        "row_heights": (0.90, 0.10),
        "candle_linewidth": 3.0,
        "whiskerwidth": 1.0,
        "bar_gap": 0.55,
    },
}
MODEL_NAMES = ("rgb_canonical", "rgb_aug_consistency")
SEEDS = (2026, 2027, 2028)
RASTER_SIZE = 192
MODEL_SIZE = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_coordinates(window: np.ndarray) -> pd.DataFrame:
    values = window.astype(np.float64)
    opens, highs, lows, closes, volumes = values.T
    base = max(float(closes[0]), 1e-12)

    def price_map(prices: np.ndarray) -> np.ndarray:
        logs = np.log(np.maximum(prices, 1e-12) / base)
        return 2.0 + np.tanh(logs / 0.108)

    volume_span = max(float(np.max(volumes) - np.min(volumes)), 1e-9)
    normalized_volume = (volumes - np.min(volumes)) / volume_span
    return pd.DataFrame(
        {
            "Open": price_map(opens),
            "High": price_map(highs),
            "Low": price_map(lows),
            "Close": price_map(closes),
            "Volume": normalized_volume,
        },
        index=pd.date_range("2020-01-01", periods=len(window), freq="D"),
    )


def decode_and_resize(payload: bytes) -> np.ndarray:
    image = Image.open(BytesIO(payload)).convert("RGB")
    image = image.resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def render_mplfinance(example: Example, config: dict[str, object]) -> np.ndarray:
    frame = fixed_coordinates(example.ohlcv)
    figure, axes = mpf.plot(
        frame,
        type="candle",
        style=str(config["style"]),
        volume=True,
        axisoff=True,
        returnfig=True,
        figsize=(2, 2),
        panel_ratios=config["panel_ratios"],
        tight_layout=True,
        warn_too_much_data=1000,
        update_width_config={
            "candle_width": config["candle_width"],
            "candle_linewidth": config["candle_linewidth"],
            "volume_width": config["volume_width"],
        },
    )
    for axis in axes[:2]:
        axis.set_ylim(1.0, 3.0)
    for axis in axes[2:]:
        axis.set_ylim(0.0, 1.05)
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=RASTER_SIZE // 2,
        facecolor=figure.get_facecolor(),
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(figure)
    return decode_and_resize(buffer.getvalue())


def make_plotly_figure(example: Example, config: dict[str, object]) -> go.Figure:
    frame = fixed_coordinates(example.ohlcv)
    x_values = list(range(len(frame)))
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=list(config["row_heights"]),
        vertical_spacing=0.0,
    )
    figure.add_trace(
        go.Candlestick(
            x=x_values,
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            increasing_line_color="#00A676",
            decreasing_line_color="#E45756",
            increasing_fillcolor="#00A676",
            decreasing_fillcolor="#E45756",
            line_width=float(config["candle_linewidth"]),
            whiskerwidth=float(config["whiskerwidth"]),
        ),
        row=1,
        col=1,
    )
    colors = np.where(
        frame["Close"].to_numpy() >= frame["Open"].to_numpy(),
        "rgba(0,166,118,0.55)",
        "rgba(228,87,86,0.55)",
    )
    figure.add_trace(
        go.Bar(x=x_values, y=frame["Volume"], marker_color=colors),
        row=2,
        col=1,
    )
    figure.update_layout(
        template=str(config["style"]),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        bargap=float(config["bar_gap"]),
    )
    figure.update_xaxes(visible=False, fixedrange=True)
    figure.update_yaxes(visible=False, fixedrange=True)
    figure.update_yaxes(range=[1.0, 3.0], row=1, col=1)
    figure.update_yaxes(range=[0.0, 1.05], row=2, col=1)
    return figure


async def render_plotly(examples: list[Example], config: dict[str, object]) -> np.ndarray:
    images = np.empty((len(examples), MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
    async with kaleido.Kaleido(n=2, timeout=120) as renderer:
        for start in range(0, len(examples), 32):
            stop = min(start + 32, len(examples))
            figures = [make_plotly_figure(example, config) for example in examples[start:stop]]
            payloads = await asyncio.gather(
                *[
                    renderer.calc_fig(
                        figure,
                        opts={
                            "format": "png",
                            "width": RASTER_SIZE,
                            "height": RASTER_SIZE,
                            "scale": 1,
                        },
                    )
                    for figure in figures
                ]
            )
            for index, payload in enumerate(payloads, start=start):
                images[index] = decode_and_resize(payload)
            print(f"plotly rendered {stop}/{len(examples)}", flush=True)
    return images


def render_external_images(
    examples: list[Example], cache_dir: Path, force: bool
) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, np.ndarray] = {}
    for name, config in EXTERNAL_RENDERERS.items():
        path = cache_dir / f"{name}.npy"
        if path.exists() and not force:
            images = np.load(path)
            expected = (len(examples), MODEL_SIZE, MODEL_SIZE, 3)
            if images.shape != expected or images.dtype != np.uint8:
                raise RuntimeError(f"invalid renderer cache {path}: {images.shape}/{images.dtype}")
            print(f"loaded {name} from {path}", flush=True)
        elif config["library"] == "mplfinance":
            images = np.empty((len(examples), MODEL_SIZE, MODEL_SIZE, 3), dtype=np.uint8)
            for index, example in enumerate(examples):
                images[index] = render_mplfinance(example, config)
                if (index + 1) % 100 == 0 or index + 1 == len(examples):
                    print(f"{name} rendered {index + 1}/{len(examples)}", flush=True)
            np.save(path, images)
        else:
            images = asyncio.run(render_plotly(examples, config))
            np.save(path, images)
        output[name] = images
    return output


def built_in_images(examples: list[Example], style: str) -> np.ndarray:
    return np.stack([render_chart(example.ohlcv, style) for example in examples])


def grayscale_background_contrast(images: np.ndarray) -> np.ndarray:
    rgb = images.astype(np.float32) / 255.0
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    border = np.concatenate(
        [gray[:, 0, :], gray[:, -1, :], gray[:, 1:-1, 0], gray[:, 1:-1, -1]],
        axis=1,
    )
    background = np.median(border, axis=1)
    invert = background < 0.5
    gray[invert] = 1.0 - gray[invert]
    background = np.where(invert, 1.0 - background, background)
    deviations = np.abs(gray - background[:, None, None])
    scale = np.quantile(deviations.reshape(len(images), -1), 0.99, axis=1)
    scale = np.maximum(scale, 1e-3)
    normalized = 1.0 + (gray - background[:, None, None]) / scale[:, None, None]
    normalized = np.clip(normalized, 0.0, 1.0)
    repeated = np.repeat(normalized[..., None], 3, axis=-1)
    return np.rint(255.0 * repeated).astype(np.uint8)


@torch.no_grad()
def predict_images(
    model: torch.nn.Module,
    images: np.ndarray,
    target_mean: float,
    target_std: float,
    batch_size: int,
) -> np.ndarray:
    device = choose_device()
    model.to(device).eval()
    predictions = np.empty(len(images), dtype=np.float64)
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        tensor = torch.from_numpy(images[start:stop].copy()).permute(0, 3, 1, 2).float().div_(255.0)
        values = model(tensor.to(device)).detach().cpu().numpy()
        predictions[start:stop] = values * target_std + target_mean
    return predictions


def load_checkpoint(path: Path) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(str(checkpoint["spec"]["architecture"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def keyed_values(
    predictions: pd.DataFrame, model: str, seed: int, style: str, examples: list[Example]
) -> np.ndarray:
    subset = predictions[
        (predictions["model"] == model)
        & (predictions["seed"] == seed)
        & (predictions["style"] == style)
    ].copy()
    subset["origin_date"] = subset["origin_date"].astype(str)
    keyed = subset.set_index(["symbol", "origin_date"])["prediction_log_rv"]
    keys = [(example.symbol, example.origin_date) for example in examples]
    if len(keyed) != len(keys) or not set(keys).issubset(keyed.index):
        raise RuntimeError(f"missing source predictions for {model}/{seed}/{style}")
    return keyed.loc[keys].to_numpy(dtype=np.float64)


def append_prediction_rows(
    rows: list[pd.DataFrame],
    examples: list[Example],
    model: str,
    seed: int,
    style: str,
    values: np.ndarray,
) -> None:
    rows.append(
        pd.DataFrame(
            {
                "symbol": [example.symbol for example in examples],
                "origin_date": [example.origin_date for example in examples],
                "target_end_date": [example.target_end_date for example in examples],
                "target_log_rv": [example.target_log_rv for example in examples],
                "model": model,
                "seed": seed,
                "style": style,
                "prediction_log_rv": values,
            }
        )
    )


def metric_row(
    target: np.ndarray, canonical: np.ndarray, alternate: np.ndarray, threshold: float
) -> dict[str, float]:
    drift = np.abs(alternate - canonical)
    return {
        "mae_log_rv": float(np.mean(np.abs(target - alternate))),
        "mean_absolute_shift": float(np.mean(drift)),
        "decision_flip_rate": float(
            np.mean((canonical >= threshold) != (alternate >= threshold))
        ),
    }


def summarize(
    source: pd.DataFrame,
    extension: pd.DataFrame,
    examples: list[Example],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = np.asarray([example.target_log_rv for example in examples])
    comparisons = [
        ("synthetic_dark", "canonical_raw", "rgb_canonical", "canonical", "dark", source),
        (
            "synthetic_dark",
            "canonical_gray_norm",
            "rgb_canonical",
            "gray_canonical",
            "gray_dark",
            extension,
        ),
        (
            "synthetic_dark",
            "canonical_affine_2022_2025",
            "rgb_canonical",
            "affine_canonical",
            "affine_dark",
            extension,
        ),
        ("synthetic_dark", "rand_cons_raw", "rgb_aug_consistency", "canonical", "dark", source),
    ]
    for renderer in EXTERNAL_RENDERERS:
        for model, method, base_style, alt_style in [
            ("rgb_canonical", "canonical_raw", "canonical", renderer),
            ("rgb_canonical", "canonical_gray_norm", "gray_canonical", f"gray_{renderer}"),
            ("rgb_aug_consistency", "rand_cons_raw", "canonical", renderer),
            (
                "rgb_aug_consistency",
                "rand_cons_gray_norm",
                "gray_canonical",
                f"gray_{renderer}",
            ),
        ]:
            comparisons.append(
                ("external_renderer", method, model, base_style, alt_style, extension)
            )

    seed_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []
    for family, method, model, base_style, alt_style, frame in comparisons:
        canonical_seeds = []
        alternate_seeds = []
        for seed in SEEDS:
            base_frame = source if base_style == "canonical" else frame
            canonical = keyed_values(base_frame, model, seed, base_style, examples)
            alternate = keyed_values(frame, model, seed, alt_style, examples)
            canonical_seeds.append(canonical)
            alternate_seeds.append(alternate)
            seed_rows.append(
                {
                    "family": family,
                    "method": method,
                    "model": model,
                    "renderer": alt_style,
                    "seed": seed,
                    **metric_row(target, canonical, alternate, threshold),
                }
            )
        canonical_ensemble = np.mean(canonical_seeds, axis=0)
        alternate_ensemble = np.mean(alternate_seeds, axis=0)
        ensemble_rows.append(
            {
                "family": family,
                "method": method,
                "model": model,
                "renderer": alt_style,
                **metric_row(target, canonical_ensemble, alternate_ensemble, threshold),
            }
        )
    return pd.DataFrame(seed_rows), pd.DataFrame(ensemble_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/prices/etf_daily_final.parquet"))
    parser.add_argument("--checkpoints", type=Path, default=Path("results/final"))
    parser.add_argument("--source-predictions", type=Path, default=Path("results/final/predictions.parquet"))
    parser.add_argument("--source-manifest", type=Path, default=Path("results/final/manifest.json"))
    parser.add_argument("--protocol", type=Path, default=Path("protocol/extension_protocol.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/external_renderers"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_validation"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force-render", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    panel = pd.read_parquet(args.data)
    panel["date"] = pd.to_datetime(panel["date"])
    examples = build_examples(panel, lookback=20, horizon=5, stride=5)
    fit, final_test = split_examples(examples)
    calibration = [
        example
        for example in fit
        if "2022-01-01" <= example.origin_date <= "2025-12-31"
    ]
    source = pd.read_parquet(args.source_predictions)
    source_manifest = json.loads(args.source_manifest.read_text())
    threshold = float(source_manifest["high_vol_threshold_log_rv_fit_q80"])
    external_images = render_external_images(final_test, args.cache_dir, args.force_render)
    gray_external = {
        renderer: grayscale_background_contrast(images)
        for renderer, images in external_images.items()
    }
    canonical_images = built_in_images(final_test, "canonical")
    dark_images = built_in_images(final_test, "dark")
    gray_canonical = grayscale_background_contrast(canonical_images)
    gray_dark = grayscale_background_contrast(dark_images)

    rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        for seed in SEEDS:
            path = args.checkpoints / f"{model_name}_seed{seed}.pt"
            model, checkpoint = load_checkpoint(path)
            target_mean = float(checkpoint["target_mean"])
            target_std = float(checkpoint["target_std"])
            for renderer, images in external_images.items():
                values = predict_images(
                    model, images, target_mean, target_std, args.batch_size
                )
                append_prediction_rows(rows, final_test, model_name, seed, renderer, values)

            gray_base = predict_images(
                model, gray_canonical, target_mean, target_std, args.batch_size
            )
            append_prediction_rows(
                rows, final_test, model_name, seed, "gray_canonical", gray_base
            )
            for renderer, images in gray_external.items():
                values = predict_images(
                    model, images, target_mean, target_std, args.batch_size
                )
                append_prediction_rows(
                    rows,
                    final_test,
                    model_name,
                    seed,
                    f"gray_{renderer}",
                    values,
                )

            if model_name == "rgb_canonical":
                gray_alt = predict_images(model, gray_dark, target_mean, target_std, args.batch_size)
                append_prediction_rows(rows, final_test, model_name, seed, "gray_dark", gray_alt)

                calibration_targets = np.asarray(
                    [example.target_log_rv for example in calibration], dtype=np.float64
                )
                final_maps: dict[str, np.ndarray] = {}
                for style in ("canonical", "dark"):
                    dev_predictions = predict_neural(
                        model,
                        calibration,
                        style,
                        target_mean,
                        target_std,
                        args.batch_size,
                    )
                    slope, intercept = np.polyfit(dev_predictions, calibration_targets, deg=1)
                    calibration_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "style": style,
                            "calibration_start": "2022-01-01",
                            "calibration_end": "2025-12-31",
                            "n": len(calibration),
                            "slope": float(slope),
                            "intercept": float(intercept),
                        }
                    )
                    raw_final = keyed_values(source, model_name, seed, style, final_test)
                    final_maps[style] = intercept + slope * raw_final
                append_prediction_rows(
                    rows,
                    final_test,
                    model_name,
                    seed,
                    "affine_canonical",
                    final_maps["canonical"],
                )
                append_prediction_rows(
                    rows,
                    final_test,
                    model_name,
                    seed,
                    "affine_dark",
                    final_maps["dark"],
                )
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"scored model={model_name} seed={seed}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension = pd.concat(rows, ignore_index=True)
    by_seed, ensemble = summarize(source, extension, final_test, threshold)
    extension.to_parquet(args.output_dir / "predictions.parquet", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        args.output_dir / "affine_calibration_parameters.csv", index=False
    )
    by_seed.to_csv(args.output_dir / "metrics_by_seed.csv", index=False)
    ensemble.to_csv(args.output_dir / "ensemble_metrics.csv", index=False)
    cache_hashes = {
        path.stem: sha256(path) for path in sorted(args.cache_dir.glob("*.npy"))
    }
    manifest = {
        "status": "post_protocol_validation",
        "protocol_sha256": sha256(args.protocol),
        "source_data_sha256": sha256(args.data),
        "source_predictions_sha256": sha256(args.source_predictions),
        "final_test_examples": len(final_test),
        "calibration_examples_2022_2025": len(calibration),
        "external_renderers": EXTERNAL_RENDERERS,
        "renderer_cache_sha256": cache_hashes,
        "models": list(MODEL_NAMES),
        "seeds": list(SEEDS),
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ["mplfinance", "plotly", "kaleido", "matplotlib", "pillow"]
        },
        "device": str(choose_device()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(ensemble.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
