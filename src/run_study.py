#!/usr/bin/env python3
"""Run the frozen chart-rendering robustness study and save row-level predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from chart_core import (
    EVALUATION_STYLES,
    TRAIN_AUGMENTATION_STYLES,
    ChartDataset,
    Example,
    PairedChartDataset,
    SmallResNet,
    TinyChartCNN,
    build_examples,
    choose_device,
    numeric_features,
    regression_metrics,
    seed_everything,
)


MODEL_SPECS = {
    # Four primary training strategies use the identical TinyChartCNN. The two
    # consistency strategies differ in whether the supervised view is fixed or
    # itself renderer-randomized.
    "rgb_canonical": {
        "architecture": "tiny",
        "styles": ("canonical",),
        "randomize": False,
        "consistency_weight": 0.0,
    },
    "rgb_style_augmented": {
        "architecture": "tiny",
        "styles": TRAIN_AUGMENTATION_STYLES,
        "randomize": True,
        "consistency_weight": 0.0,
    },
    "rgb_paired_consistency": {
        "architecture": "tiny",
        "styles": ("canonical",),
        "randomize": False,
        "second_styles": TRAIN_AUGMENTATION_STYLES,
        "randomize_second": True,
        "consistency_weight": 0.5,
        "supervise_second": False,
    },
    "rgb_aug_consistency": {
        "architecture": "tiny",
        "styles": TRAIN_AUGMENTATION_STYLES,
        "randomize": True,
        "second_styles": TRAIN_AUGMENTATION_STYLES,
        "randomize_second": True,
        "procedural": True,
        "procedural_second": True,
        "consistency_weight": 0.5,
        "supervise_second": True,
    },
    "edge_canonical": {
        "architecture": "edge",
        "styles": ("canonical",),
        "randomize": False,
        "consistency_weight": 0.0,
    },
    "resnet_canonical": {
        "architecture": "resnet",
        "styles": ("canonical",),
        "randomize": False,
        "consistency_weight": 0.0,
    },
    "resnet_aug_consistency": {
        "architecture": "resnet",
        "styles": TRAIN_AUGMENTATION_STYLES,
        "randomize": True,
        "second_styles": TRAIN_AUGMENTATION_STYLES,
        "randomize_second": True,
        "consistency_weight": 0.5,
        "supervise_second": True,
    },
}


def split_examples(examples: list[Example], test_start: str = "2026-01-01"):
    # Development was inspected before the final protocol was frozen. The final
    # test begins in 2026 and no pre-2026 label is allowed to cross that boundary.
    fit = [e for e in examples if e.target_end_date < test_start]
    final_test = [e for e in examples if e.origin_date >= test_start]
    return fit, final_test


def make_model(architecture: str) -> nn.Module:
    if architecture == "tiny":
        return TinyChartCNN(use_edges=False)
    if architecture == "edge":
        return TinyChartCNN(use_edges=True)
    if architecture == "resnet":
        return SmallResNet()
    raise ValueError(f"unknown architecture: {architecture}")


def train_neural(
    fit: list[Example],
    spec: dict,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[nn.Module, float, float, list[float]]:
    seed_everything(seed)
    device = choose_device()
    model = make_model(spec["architecture"]).to(device)
    consistency_weight = float(spec.get("consistency_weight", 0.0))
    if consistency_weight > 0:
        dataset = PairedChartDataset(
            fit,
            first_styles=spec["styles"],
            second_styles=spec["second_styles"],
            randomize_first=spec["randomize"],
            randomize_second=spec["randomize_second"],
            procedural_first=spec.get("procedural", False),
            procedural_second=spec.get("procedural_second", False),
        )
    else:
        dataset = ChartDataset(
            fit,
            spec["styles"],
            randomize=spec["randomize"],
            procedural=spec.get("procedural", False),
        )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    target_mean = float(np.mean([e.target_log_rv for e in fit]))
    target_std = max(float(np.std([e.target_log_rv for e in fit])), 1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch in loader:
            if consistency_weight > 0:
                images, paired_images, targets, _ = batch
                paired_images = paired_images.to(device)
            else:
                images, targets, _ = batch
                paired_images = None
            images = images.to(device)
            standardized = ((targets - target_mean) / target_std).to(device)
            predictions = model(images)
            supervised = nn.functional.smooth_l1_loss(predictions, standardized)
            if paired_images is not None:
                paired_predictions = model(paired_images)
                if spec.get("supervise_second", False):
                    paired_supervised = nn.functional.smooth_l1_loss(
                        paired_predictions, standardized
                    )
                    supervised = 0.5 * (supervised + paired_supervised)
                consistency = nn.functional.mse_loss(predictions, paired_predictions)
                loss = supervised + consistency_weight * consistency
            else:
                loss = supervised
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))
        print(
            f"seed={seed} architecture={spec['architecture']} epoch={epoch + 1}/{epochs} "
            f"loss={losses[-1]:.4f}",
            flush=True,
        )
    return model, target_mean, target_std, losses


@torch.no_grad()
def predict_neural(
    model: nn.Module,
    examples: list[Example],
    style: str,
    target_mean: float,
    target_std: float,
    batch_size: int,
) -> np.ndarray:
    device = choose_device()
    model.to(device).eval()
    loader = DataLoader(ChartDataset(examples, (style,)), batch_size=batch_size, num_workers=0)
    predictions = np.empty(len(examples), dtype=np.float64)
    for images, _, indices in loader:
        values = model(images.to(device)).detach().cpu().numpy() * target_std + target_mean
        predictions[indices.numpy()] = values
    return predictions


def prediction_frame(
    examples: list[Example], model: str, seed: int, style: str, values: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [e.symbol for e in examples],
            "origin_date": [e.origin_date for e in examples],
            "target_end_date": [e.target_end_date for e in examples],
            "target_log_rv": [e.target_log_rv for e in examples],
            "model": model,
            "seed": seed,
            "style": style,
            "prediction_log_rv": values,
        }
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_numeric_models(fit: list[Example], test: list[Example]) -> list[pd.DataFrame]:
    x_fit = np.vstack([numeric_features(e) for e in fit])
    y_fit = np.asarray([e.target_log_rv for e in fit])
    x_test = np.vstack([numeric_features(e) for e in test])
    models = {
        "numeric_ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "numeric_histgb": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=2026,
        ),
    }
    frames = []
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_fit, y_fit)
        values = model.predict(x_test)
        print(f"{name} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        frames.append(prediction_frame(test, name, 2026, "numeric", values))
    # Econometric persistence baseline: current 5-day realized volatility.
    persistence = np.asarray(
        [np.log(max(float(numeric_features(e)[-11]), 1e-8)) for e in test], dtype=np.float64
    )
    frames.append(prediction_frame(test, "rv5_persistence", 2026, "numeric", persistence))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/prices/etf_daily.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final"))
    parser.add_argument("--protocol", type=Path, default=Path("protocol/final_protocol.json"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", nargs="*", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--models", nargs="*", default=list(MODEL_SPECS))
    parser.add_argument(
        "--test-start",
        default="2026-01-01",
        help="Forecast-origin boundary; the final protocol keeps the default.",
    )
    args = parser.parse_args()

    panel = pd.read_parquet(args.data)
    panel["date"] = pd.to_datetime(panel["date"])
    examples = build_examples(panel, lookback=20, horizon=5, stride=5)
    fit, final_test = split_examples(examples, args.test_start)
    if not final_test:
        raise RuntimeError(f"No examples on or after test boundary {args.test_start}")
    if any(e.target_end_date >= args.test_start for e in fit):
        raise RuntimeError("fit labels cross final-test boundary")
    high_vol_threshold = float(np.quantile([e.target_log_rv for e in fit], 0.8))
    print(
        f"device={choose_device()} fit={len(fit)} final_test={len(final_test)} "
        f"test_dates={final_test[0].origin_date}..{final_test[-1].origin_date}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = run_numeric_models(fit, final_test)
    training_log: dict[str, dict[str, list[float]]] = {}
    for model_name in args.models:
        spec = deepcopy(MODEL_SPECS[model_name])
        training_log[model_name] = {}
        for seed in args.seeds:
            started = time.perf_counter()
            model, target_mean, target_std, losses = train_neural(
                fit, spec, seed, args.epochs, args.batch_size
            )
            training_log[model_name][str(seed)] = losses
            for style in EVALUATION_STYLES:
                values = predict_neural(
                    model, final_test, style, target_mean, target_std, args.batch_size * 2
                )
                all_frames.append(prediction_frame(final_test, model_name, seed, style, values))
            checkpoint = {
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "target_mean": target_mean,
                "target_std": target_std,
                "model_name": model_name,
                "seed": seed,
                "epochs": args.epochs,
                "spec": spec,
            }
            torch.save(checkpoint, args.output_dir / f"{model_name}_seed{seed}.pt")
            print(
                f"completed model={model_name} seed={seed} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    predictions = pd.concat(all_frames, ignore_index=True)
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    targets = np.asarray([e.target_log_rv for e in final_test])
    summary = {}
    for (model, seed, style), group in predictions.groupby(["model", "seed", "style"]):
        ordered = group.sort_values(["symbol", "origin_date"])
        expected = pd.DataFrame(
            {"symbol": [e.symbol for e in final_test], "origin_date": [e.origin_date for e in final_test]}
        ).sort_values(["symbol", "origin_date"])
        if not np.array_equal(ordered[["symbol", "origin_date"]].to_numpy(), expected.to_numpy()):
            raise RuntimeError("prediction row order/key mismatch")
        summary[f"{model}|{seed}|{style}"] = regression_metrics(
            targets_log_rv=ordered["target_log_rv"].to_numpy(),
            predictions_log_rv=ordered["prediction_log_rv"].to_numpy(),
            high_vol_threshold=high_vol_threshold,
        )
    paired_summary = {}
    neural_predictions = predictions[predictions["style"].isin(EVALUATION_STYLES)]
    for (model, seed), group in neural_predictions.groupby(["model", "seed"]):
        pivot = group.pivot(
            index=["symbol", "origin_date"], columns="style", values="prediction_log_rv"
        )
        canonical = pivot["canonical"].to_numpy()
        for style in EVALUATION_STYLES:
            if style == "canonical":
                continue
            alternate = pivot[style].to_numpy()
            absolute_shift = np.abs(alternate - canonical)
            canonical_decision = canonical >= high_vol_threshold
            alternate_decision = alternate >= high_vol_threshold
            paired_summary[f"{model}|{seed}|{style}"] = {
                "mean_absolute_shift": float(np.mean(absolute_shift)),
                "median_absolute_shift": float(np.median(absolute_shift)),
                "p95_absolute_shift": float(np.quantile(absolute_shift, 0.95)),
                "prediction_correlation": float(np.corrcoef(canonical, alternate)[0, 1]),
                "decision_flip_rate": float(np.mean(canonical_decision != alternate_decision)),
            }
    manifest = {
        "fit_examples": len(fit),
        "final_test_examples": len(final_test),
        "final_test_min_origin": min(e.origin_date for e in final_test),
        "final_test_max_origin": max(e.origin_date for e in final_test),
        "test_start": args.test_start,
        "symbols": sorted(panel["symbol"].unique().tolist()),
        "lookback": 20,
        "horizon": 5,
        "stride": 5,
        "high_vol_threshold_log_rv_fit_q80": high_vol_threshold,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "models": args.models,
        "styles": list(EVALUATION_STYLES),
        "augmentation_styles": list(TRAIN_AUGMENTATION_STYLES),
        "device": str(choose_device()),
        "data_sha256": file_sha256(args.data),
        "protocol_sha256": file_sha256(args.protocol) if args.protocol.exists() else None,
        "model_specs": {name: MODEL_SPECS[name] for name in args.models},
        "training_log": training_log,
        "metrics": summary,
        "paired_metrics": paired_summary,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "training_log"}, indent=2))


if __name__ == "__main__":
    main()
