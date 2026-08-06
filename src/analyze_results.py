#!/usr/bin/env python3
"""Aggregate the frozen study, run paired block bootstrap, and make paper assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chart_core import regression_metrics


NEURAL_ORDER = [
    "rgb_canonical",
    "rgb_style_augmented",
    "rgb_paired_consistency",
    "rgb_aug_consistency",
    "edge_canonical",
    "resnet_canonical",
    "resnet_aug_consistency",
]
STYLE_ORDER = ["canonical", "reversed", "dark", "monochrome", "blue_orange"]
PRIMARY_MODELS = [
    "rgb_canonical",
    "rgb_style_augmented",
    "rgb_paired_consistency",
    "rgb_aug_consistency",
]
MODEL_LABELS = {
    "rgb_canonical": "Canonical",
    "rgb_style_augmented": "Style aug.",
    "rgb_paired_consistency": "Paired consistency",
    "rgb_aug_consistency": "Randomized + consistency",
    "edge_canonical": "Fixed edge map",
    "resnet_canonical": "ResNet canonical",
    "resnet_aug_consistency": "ResNet randomized + consistency",
    "numeric_ridge": "Numeric ridge",
    "numeric_histgb": "Numeric HistGB",
    "rv5_persistence": "RV(5) persistence",
}
STYLE_LABELS = {
    "canonical": "Canonical",
    "reversed": "Reversed",
    "dark": "Dark",
    "monochrome": "Monochrome",
    "blue_orange": "Held-out",
}


def metric_row(group: pd.DataFrame, threshold: float) -> dict[str, float]:
    return regression_metrics(
        group["target_log_rv"].to_numpy(),
        group["prediction_log_rv"].to_numpy(),
        threshold,
    )


def paired_row(pivot: pd.DataFrame, style: str, threshold: float) -> dict[str, float]:
    canonical = pivot["canonical"].to_numpy()
    alternate = pivot[style].to_numpy()
    shift = np.abs(alternate - canonical)
    return {
        "mean_absolute_shift": float(np.mean(shift)),
        "median_absolute_shift": float(np.median(shift)),
        "p95_absolute_shift": float(np.quantile(shift, 0.95)),
        "prediction_correlation": float(np.corrcoef(canonical, alternate)[0, 1]),
        "decision_flip_rate": float(
            np.mean((canonical >= threshold) != (alternate >= threshold))
        ),
    }


def cluster_samples(frame: pd.DataFrame, cluster: str, draws: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = {
        value: indices.to_numpy()
        for value, indices in frame.groupby(cluster, sort=True).groups.items()
    }
    names = np.asarray(list(groups), dtype=object)
    samples: list[np.ndarray] = []
    for _ in range(draws):
        selected = rng.choice(names, size=len(names), replace=True)
        samples.append(np.concatenate([groups[name] for name in selected]))
    return samples


def interval(values: list[float]) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_drift(
    predictions: pd.DataFrame,
    models: list[str],
    styles: list[str],
    threshold: float,
    draws: int,
) -> pd.DataFrame:
    keys = ["symbol", "origin_date"]
    neural = predictions[predictions["model"].isin(models)].copy()
    wide = neural.pivot(index=keys, columns=["model", "seed", "style"], values="prediction_log_rv")
    key_frame = wide.index.to_frame(index=False).reset_index(drop=True)
    date_samples = cluster_samples(key_frame, "origin_date", draws, 20260805)
    asset_samples = cluster_samples(key_frame, "symbol", draws, 20260806)
    rows: list[dict[str, object]] = []
    drift_arrays: dict[tuple[str, str], np.ndarray] = {}
    for model in models:
        seeds = sorted(neural.loc[neural["model"] == model, "seed"].unique())
        for style in styles:
            per_seed = []
            flip_seed = []
            for seed in seeds:
                canonical = wide[(model, seed, "canonical")].to_numpy()
                alternate = wide[(model, seed, style)].to_numpy()
                per_seed.append(np.abs(alternate - canonical))
                flip_seed.append((canonical >= threshold) != (alternate >= threshold))
            drift = np.mean(np.vstack(per_seed), axis=0)
            flips = np.mean(np.vstack(flip_seed), axis=0)
            drift_arrays[(model, style)] = drift
            date_values = [float(np.mean(drift[index])) for index in date_samples]
            asset_values = [float(np.mean(drift[index])) for index in asset_samples]
            date_low, date_high = interval(date_values)
            asset_low, asset_high = interval(asset_values)
            rows.append(
                {
                    "model": model,
                    "style": style,
                    "mean_absolute_shift": float(np.mean(drift)),
                    "date_ci_low": date_low,
                    "date_ci_high": date_high,
                    "asset_ci_low": asset_low,
                    "asset_ci_high": asset_high,
                    "decision_flip_rate": float(np.mean(flips)),
                }
            )
    baseline = "rgb_canonical"
    for row in rows:
        model, style = str(row["model"]), str(row["style"])
        if model == baseline:
            row["drift_reduction_vs_canonical"] = 0.0
            row["reduction_ci_low"] = 0.0
            row["reduction_ci_high"] = 0.0
            continue
        difference = drift_arrays[(baseline, style)] - drift_arrays[(model, style)]
        values = [float(np.mean(difference[index])) for index in date_samples]
        low, high = interval(values)
        row["drift_reduction_vs_canonical"] = float(np.mean(difference))
        row["reduction_ci_low"] = low
        row["reduction_ci_high"] = high
    return pd.DataFrame(rows)


def aggregate_metrics(predictions: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for (model, seed, style), group in predictions.groupby(["model", "seed", "style"]):
        seed_rows.append({"model": model, "seed": seed, "style": style, **metric_row(group, threshold)})
    by_seed = pd.DataFrame(seed_rows)

    keys = ["symbol", "origin_date", "target_end_date", "target_log_rv", "model", "style"]
    ensemble = predictions.groupby(keys, as_index=False)["prediction_log_rv"].mean()
    ensemble_rows = []
    for (model, style), group in ensemble.groupby(["model", "style"]):
        ensemble_rows.append({"model": model, "style": style, **metric_row(group, threshold)})
    return by_seed, pd.DataFrame(ensemble_rows)


def paired_ensemble(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    neural = predictions[predictions["style"].isin(STYLE_ORDER)]
    keys = ["symbol", "origin_date", "model", "style"]
    ensemble = neural.groupby(keys, as_index=False)["prediction_log_rv"].mean()
    rows = []
    for model, group in ensemble.groupby("model"):
        pivot = group.pivot(index=["symbol", "origin_date"], columns="style", values="prediction_log_rv")
        for style in STYLE_ORDER[1:]:
            rows.append({"model": model, "style": style, **paired_row(pivot, style, threshold)})
    return pd.DataFrame(rows)


def paired_by_seed(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    neural = predictions[predictions["style"].isin(STYLE_ORDER)]
    rows = []
    for (model, seed), group in neural.groupby(["model", "seed"], sort=True):
        pivot = group.pivot(
            index=["symbol", "origin_date"],
            columns="style",
            values="prediction_log_rv",
        )
        for style in STYLE_ORDER[1:]:
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "style": style,
                    **paired_row(pivot, style, threshold),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mae(
    predictions: pd.DataFrame,
    draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Date-cluster MAE intervals and paired renderer-induced MAE changes.

    Headline forecasting uses seed-ensemble predictions.  The contrast table is
    paired twice: alternate and canonical renders share the same example, and a
    bootstrap draw keeps all assets for each sampled calendar origin together.
    """

    keys = ["symbol", "origin_date", "target_log_rv", "model", "style"]
    ensemble = predictions.groupby(keys, as_index=False)["prediction_log_rv"].mean()
    rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    for (model, style), group in ensemble.groupby(["model", "style"], sort=True):
        group = group.sort_values(["origin_date", "symbol"]).reset_index(drop=True)
        losses = np.abs(group["target_log_rv"].to_numpy() - group["prediction_log_rv"].to_numpy())
        samples = cluster_samples(group, "origin_date", draws, 20260807)
        values = [float(np.mean(losses[index])) for index in samples]
        low, high = interval(values)
        rows.append(
            {
                "model": model,
                "style": style,
                "mae_log_rv": float(np.mean(losses)),
                "date_ci_low": low,
                "date_ci_high": high,
            }
        )

    neural = ensemble[ensemble["style"].isin(STYLE_ORDER)].copy()
    for model, group in neural.groupby("model", sort=True):
        wide = group.pivot(
            index=["symbol", "origin_date", "target_log_rv"],
            columns="style",
            values="prediction_log_rv",
        ).reset_index()
        wide = wide.sort_values(["origin_date", "symbol"]).reset_index(drop=True)
        canonical_loss = np.abs(wide["target_log_rv"].to_numpy() - wide["canonical"].to_numpy())
        samples = cluster_samples(wide, "origin_date", draws, 20260808)
        for style in STYLE_ORDER[1:]:
            alternate_loss = np.abs(
                wide["target_log_rv"].to_numpy() - wide[style].to_numpy()
            )
            difference = alternate_loss - canonical_loss
            values = [float(np.mean(difference[index])) for index in samples]
            low, high = interval(values)
            contrast_rows.append(
                {
                    "model": model,
                    "style": style,
                    "mae_change_vs_canonical": float(np.mean(difference)),
                    "date_ci_low": low,
                    "date_ci_high": high,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(contrast_rows)


def make_drift_figure(bootstrap: pd.DataFrame, output: Path) -> None:
    frame = bootstrap[
        bootstrap["model"].isin(PRIMARY_MODELS) & bootstrap["style"].isin(STYLE_ORDER[1:])
    ].copy()
    fig, ax = plt.subplots(figsize=(3.38, 2.35))
    x = np.arange(len(STYLE_ORDER) - 1)
    width = 0.19
    colors = ["#4355A5", "#2A9D8F", "#E9C46A", "#E76F51"]
    for offset, (model, color) in enumerate(zip(PRIMARY_MODELS, colors, strict=True)):
        part = frame.set_index(["model", "style"]).loc[model].reindex(STYLE_ORDER[1:])
        values = part["mean_absolute_shift"].to_numpy()
        low = values - part["date_ci_low"].to_numpy()
        high = part["date_ci_high"].to_numpy() - values
        ax.bar(
            x + (offset - 1.5) * width,
            values,
            width,
            label=MODEL_LABELS[model],
            color=color,
        )
        ax.errorbar(
            x + (offset - 1.5) * width,
            values,
            yerr=np.vstack([low, high]),
            fmt="none",
            ecolor="#303030",
            elinewidth=0.7,
            capsize=1.8,
        )
    ax.set_xticks(x, ["Reverse", "Dark", "Mono", "Held-out"], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Mean |prediction shift|", fontsize=7.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, frameon=False, fontsize=6.1, loc="upper left")
    fig.tight_layout(pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_accuracy_figure(metrics: pd.DataFrame, output: Path) -> None:
    models = PRIMARY_MODELS + [
        "edge_canonical",
        "resnet_canonical",
        "resnet_aug_consistency",
        "numeric_histgb",
        "rv5_persistence",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
    style_colors = {
        "canonical": "#4355A5",
        "dark": "#E76F51",
        "blue_orange": "#E9C46A",
        "numeric": "#666666",
    }
    plot_labels = {
        "rgb_canonical": "Canonical",
        "rgb_style_augmented": "Style aug.",
        "rgb_paired_consistency": "Paired cons.",
        "rgb_aug_consistency": "Rand.+cons.",
        "edge_canonical": "Edge",
        "resnet_canonical": "ResNet",
        "resnet_aug_consistency": "ResNet rand.+cons.",
        "numeric_histgb": "HistGB",
        "rv5_persistence": "RV(5)",
    }
    for model_index, model in enumerate(models):
        frame = metrics[metrics["model"] == model].set_index("style")
        if model.startswith("numeric") or model == "rv5_persistence":
            styles = ["numeric"]
        else:
            styles = ["canonical", "dark", "blue_orange"]
        for style in styles:
            if style not in frame.index:
                continue
            axes[0].scatter(
                model_index,
                frame.loc[style, "mae_log_rv"],
                color=style_colors[style],
                s=31,
                zorder=3,
            )
            axes[1].scatter(
                model_index,
                frame.loc[style, "spearman"],
                color=style_colors[style],
                s=31,
                zorder=3,
            )
    labels = [plot_labels[model] for model in models]
    for ax, ylabel in zip(axes, ["MAE (lower is better)", "Spearman (higher is better)"], strict=True):
        ax.set_xticks(np.arange(len(models)), labels, rotation=32, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=STYLE_LABELS.get(style, style.title()))
        for style, color in style_colors.items()
    ]
    legend = axes[0].legend(
        handles=handles,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#606060",
        framealpha=1.0,
        fontsize=7.5,
        loc="upper right",
    )
    legend.get_frame().set_linewidth(0.7)
    fig.tight_layout(pad=0.5, w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_invariance_scatter(
    predictions: pd.DataFrame,
    threshold: float,
    output: Path,
) -> None:
    """Show within-example prediction geometry for the central two contrasts."""

    models = ["rgb_canonical", "rgb_aug_consistency"]
    styles = ["dark", "blue_orange"]
    neural = predictions[predictions["model"].isin(models)].copy()
    ensemble = neural.groupby(
        ["symbol", "origin_date", "model", "style"], as_index=False
    )["prediction_log_rv"].mean()
    pivots = {}
    all_values = []
    for model in models:
        group = ensemble[ensemble["model"] == model]
        pivot = group.pivot(
            index=["symbol", "origin_date"],
            columns="style",
            values="prediction_log_rv",
        )
        pivots[model] = pivot
        all_values.extend(pivot[["canonical", *styles]].to_numpy().ravel().tolist())
    low, high = np.quantile(np.asarray(all_values), [0.005, 0.995])
    padding = 0.06 * (high - low)
    limits = (float(low - padding), float(high + padding))

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.35), sharex=True, sharey=True)
    for row, model in enumerate(models):
        pivot = pivots[model]
        canonical = pivot["canonical"].to_numpy()
        for col, style in enumerate(styles):
            alternate = pivot[style].to_numpy()
            flips = (canonical >= threshold) != (alternate >= threshold)
            ax = axes[row, col]
            ax.scatter(
                canonical[~flips],
                alternate[~flips],
                s=7,
                alpha=0.28,
                linewidths=0,
                color="#4355A5",
                rasterized=True,
            )
            ax.scatter(
                canonical[flips],
                alternate[flips],
                s=11,
                alpha=0.72,
                linewidths=0,
                color="#D1495B",
                label="Decision flip",
                rasterized=True,
            )
            ax.plot(limits, limits, color="#303030", linewidth=0.8, linestyle="--")
            ax.axvline(threshold, color="#888888", linewidth=0.55, linestyle=":")
            ax.axhline(threshold, color="#888888", linewidth=0.55, linestyle=":")
            drift = float(np.mean(np.abs(alternate - canonical)))
            ax.text(
                0.03,
                0.95,
                f"drift={drift:.3f}; flips={100 * np.mean(flips):.1f}%",
                transform=ax.transAxes,
                va="top",
                fontsize=7.1,
            )
            ax.set_title(
                f"{MODEL_LABELS[model]} / {STYLE_LABELS[style]}",
                fontsize=8.3,
            )
            ax.grid(alpha=0.15, linewidth=0.45)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
    for ax in axes[-1]:
        ax.set_xlabel("Canonical prediction (log RV)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Redrawn prediction (log RV)", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7.2, loc="lower right")
    fig.tight_layout(pad=0.55, h_pad=0.8, w_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("results/final/predictions.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("results/final/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Write all tabular artifacts without assuming the full figure model roster.",
    )
    args = parser.parse_args()

    predictions = pd.read_parquet(args.predictions)
    manifest = json.loads(args.manifest.read_text())
    threshold = float(manifest["high_vol_threshold_log_rv_fit_q80"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_seed, ensemble = aggregate_metrics(predictions, threshold)
    paired = paired_ensemble(predictions, threshold)
    paired_seed = paired_by_seed(predictions, threshold)
    mae_intervals, mae_contrasts = bootstrap_mae(predictions, args.bootstrap_draws)
    available_neural = [model for model in NEURAL_ORDER if model in predictions["model"].unique()]
    bootstrap = bootstrap_drift(
        predictions,
        available_neural,
        STYLE_ORDER[1:],
        threshold,
        args.bootstrap_draws,
    )
    by_seed.to_csv(args.output_dir / "metrics_by_seed.csv", index=False)
    ensemble.to_csv(args.output_dir / "ensemble_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_metrics.csv", index=False)
    paired_seed.to_csv(args.output_dir / "paired_metrics_by_seed.csv", index=False)
    mae_intervals.to_csv(args.output_dir / "bootstrap_mae.csv", index=False)
    mae_contrasts.to_csv(args.output_dir / "bootstrap_mae_contrasts.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_drift.csv", index=False)
    if not args.skip_figures:
        make_drift_figure(bootstrap, args.figure_dir / "renderer_drift.pdf")
        make_accuracy_figure(ensemble, args.figure_dir / "forecast_performance.pdf")
        make_invariance_scatter(
            predictions,
            threshold,
            args.figure_dir / "invariance_scatter.pdf",
        )

    compact = {
        "prediction_rows": int(len(predictions)),
        "examples": int(predictions[["symbol", "origin_date"]].drop_duplicates().shape[0]),
        "models": sorted(predictions["model"].unique().tolist()),
        "styles": STYLE_ORDER,
        "bootstrap_draws": args.bootstrap_draws,
        "figures_generated": not args.skip_figures,
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
