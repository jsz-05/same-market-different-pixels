#!/usr/bin/env python3
"""Combine extension metrics and create the single compact manuscript table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def one(frame: pd.DataFrame, method: str) -> pd.Series:
    match = frame[(frame["family"] == "synthetic_dark") & (frame["method"] == method)]
    if len(match) != 1:
        raise ValueError(f"expected one synthetic row for {method}, found {len(match)}")
    return match.iloc[0]


def stock_rows(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    keys = ["symbol", "origin_date", "target_log_rv", "model", "style"]
    ensemble = predictions.groupby(keys, as_index=False)["prediction_log_rv"].mean()
    rows = []
    for model, method in [
        ("rgb_canonical", "canonical_raw"),
        ("rgb_aug_consistency", "rand_cons_raw"),
    ]:
        group = ensemble[ensemble["model"] == model]
        pivot = group.pivot(
            index=["symbol", "origin_date", "target_log_rv"],
            columns="style",
            values="prediction_log_rv",
        ).reset_index()
        target = pivot["target_log_rv"].to_numpy()
        canonical = pivot["canonical"].to_numpy()
        dark = pivot["dark"].to_numpy()
        rows.append(
            {
                "family": "stock_replication",
                "method": method,
                "model": model,
                "renderer": "dark",
                "mae_log_rv": float(np.mean(np.abs(target - dark))),
                "mean_absolute_shift": float(np.mean(np.abs(dark - canonical))),
                "decision_flip_rate": float(
                    np.mean((canonical >= threshold) != (dark >= threshold))
                ),
            }
        )
    return pd.DataFrame(rows)


def tex_row(panel: str, training: str, renderer: str, row: pd.Series) -> str:
    return (
        f"{panel} & {training} & {renderer} & {row['mae_log_rv']:.3f} & "
        f"{row['mean_absolute_shift']:.3f} & "
        f"{100.0 * row['decision_flip_rate']:.1f}\\%" + r" \\"
    )


def command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def ensemble_prediction(frame: pd.DataFrame, model: str, style: str) -> pd.Series:
    subset = frame[(frame["model"] == model) & (frame["style"] == style)].copy()
    subset["origin_date"] = subset["origin_date"].astype(str)
    if subset.empty:
        raise ValueError(f"missing predictions for {model}/{style}")
    return subset.groupby(["symbol", "origin_date"])["prediction_log_rv"].mean().sort_index()


def paired_drift(
    base_frame: pd.DataFrame,
    alternate_frame: pd.DataFrame,
    model: str,
    base_style: str,
    alternate_style: str,
) -> pd.Series:
    base = ensemble_prediction(base_frame, model, base_style).rename("base")
    alternate = ensemble_prediction(alternate_frame, model, alternate_style).rename("alternate")
    aligned = pd.concat([base, alternate], axis=1, join="inner").dropna()
    if len(aligned) != len(base) or len(aligned) != len(alternate):
        raise ValueError(f"unaligned paired predictions for {model}/{alternate_style}")
    return (aligned["alternate"] - aligned["base"]).abs()


def clustered_reduction(
    canonical_drift: pd.Series,
    repair_drift: pd.Series,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    paired = pd.concat(
        [canonical_drift.rename("canonical"), repair_drift.rename("repair")],
        axis=1,
        join="inner",
    ).dropna()
    if len(paired) != len(canonical_drift) or len(paired) != len(repair_drift):
        raise ValueError("reduction drift arrays are not aligned")
    frame = paired.reset_index()
    difference = frame["canonical"].to_numpy() - frame["repair"].to_numpy()
    frame["calendar_cluster"] = (
        pd.to_datetime(frame["origin_date"]).dt.to_period("W-FRI").astype(str)
    )
    groups = {
        week: indices.to_numpy()
        for week, indices in frame.groupby("calendar_cluster", sort=True).groups.items()
    }
    weeks = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        selected = rng.choice(weeks, size=len(weeks), replace=True)
        indices = np.concatenate([groups[week] for week in selected])
        values.append(float(np.mean(difference[indices])))
    return {
        "absolute_reduction": float(np.mean(difference)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "calendar_clusters": int(len(weeks)),
        "cluster_unit": "week_ending_friday",
        "bootstrap_draws": int(draws),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external", type=Path, default=Path("results/external_validation/ensemble_metrics.csv")
    )
    parser.add_argument(
        "--external-predictions",
        type=Path,
        default=Path("results/external_validation/predictions.parquet"),
    )
    parser.add_argument(
        "--source-predictions",
        type=Path,
        default=Path("results/final/predictions.parquet"),
    )
    parser.add_argument(
        "--stock-predictions",
        type=Path,
        default=Path("results/stock_replication/predictions.parquet"),
    )
    parser.add_argument(
        "--stock-manifest", type=Path, default=Path("results/stock_replication/manifest.json")
    )
    parser.add_argument(
        "--stock-download-manifest",
        type=Path,
        default=Path("data/raw/prices/sp100_manifest_final.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/extension_analysis"))
    parser.add_argument("--latex-dir", type=Path, default=Path("paper/generated"))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()

    external = pd.read_csv(args.external)
    external_predictions = pd.read_parquet(args.external_predictions)
    source_predictions = pd.read_parquet(args.source_predictions)
    stock_predictions = pd.read_parquet(args.stock_predictions)
    stock_manifest = json.loads(args.stock_manifest.read_text())
    stock_download = json.loads(args.stock_download_manifest.read_text())
    stock = stock_rows(
        stock_predictions, float(stock_manifest["high_vol_threshold_log_rv_fit_q80"])
    )

    cheap = pd.DataFrame(
        [
            one(external, "canonical_raw"),
            one(external, "canonical_gray_norm"),
            one(external, "canonical_affine_2022_2025"),
            one(external, "rand_cons_raw"),
        ]
    )
    real = external[external["family"] == "external_renderer"].copy()
    worst = (
        real.sort_values("mean_absolute_shift", ascending=False)
        .groupby("method", as_index=False, sort=False)
        .first()
    )
    canonical_worst = worst[worst["method"] == "canonical_raw"].iloc[0]
    canonical_gray_worst = worst[worst["method"] == "canonical_gray_norm"].iloc[0]
    repair_worst = worst[worst["method"] == "rand_cons_raw"].iloc[0]
    repair_gray_worst = worst[worst["method"] == "rand_cons_gray_norm"].iloc[0]
    stock_canonical = stock[stock["method"] == "canonical_raw"].iloc[0]
    stock_repair = stock[stock["method"] == "rand_cons_raw"].iloc[0]

    external_interval = clustered_reduction(
        paired_drift(
            source_predictions,
            external_predictions,
            "rgb_canonical",
            "canonical",
            str(canonical_worst["renderer"]),
        ),
        paired_drift(
            source_predictions,
            external_predictions,
            "rgb_aug_consistency",
            "canonical",
            str(repair_worst["renderer"]),
        ),
        args.bootstrap_draws,
        20260811,
    )
    stock_interval = clustered_reduction(
        paired_drift(
            stock_predictions,
            stock_predictions,
            "rgb_canonical",
            "canonical",
            "dark",
        ),
        paired_drift(
            stock_predictions,
            stock_predictions,
            "rgb_aug_consistency",
            "canonical",
            "dark",
        ),
        args.bootstrap_draws,
        20260812,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.latex_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([cheap, real, stock], ignore_index=True)
    summary.to_csv(args.output_dir / "extension_summary.csv", index=False)
    bootstrap_rows = [
        {
            "panel": "external_renderer",
            "canonical_renderer": canonical_worst["renderer"],
            "repair_renderer": repair_worst["renderer"],
            **external_interval,
        },
        {
            "panel": "stock_replication",
            "canonical_renderer": "dark",
            "repair_renderer": "dark",
            **stock_interval,
        },
    ]
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output_dir / "bootstrap_reductions.csv", index=False
    )

    rows = [
        tex_row("ETF cheap repair", "Canonical, raw", "Dark", cheap.iloc[0]),
        tex_row("", "+ gray normalization", "Dark", cheap.iloc[1]),
        tex_row("", "+ affine calibration", "Dark", cheap.iloc[2]),
        tex_row("", "Rand.+cons., raw", "Dark", cheap.iloc[3]),
        r"\midrule",
        tex_row("ETF external", "Canonical, raw", "Worst of 3", canonical_worst),
        tex_row("", "Canonical + gray", "Worst of 3", canonical_gray_worst),
        tex_row("", "Rand.+cons., raw", "Worst of 3", repair_worst),
        tex_row("", "Rand.+cons. + gray", "Worst of 3", repair_gray_worst),
        r"\midrule",
        tex_row("100 stocks", "Canonical, raw", "Dark", stock_canonical),
        tex_row("", "Rand.+cons., raw", "Dark", stock_repair),
    ]
    table = """\\begin{table*}[t]
  \\centering
  \\caption{Post-protocol stress tests using seed-ensemble predictions. External
  rows select each method's worst paired drift separately across three fixed
  mplfinance and Plotly compositions. MAE is for the named alternate view;
  drift and flips are paired to its corresponding canonical view.}
  \\label{tab:extension}
  \\footnotesize
  \\setlength{\\tabcolsep}{5.0pt}
  \\begin{tabular}{lllrrr}
    \\toprule
    Panel & Training/repair & Renderer & MAE & Paired drift & Flip \\\\
    \\midrule
""" + "\n".join(rows) + """
    \\bottomrule
  \\end{tabular}
\\end{table*}
"""
    (args.latex_dir / "extension_table.tex").write_text(table)

    external_reduction = 100.0 * (
        1.0
        - float(repair_worst["mean_absolute_shift"])
        / float(canonical_worst["mean_absolute_shift"])
    )
    stock_reduction = 100.0 * (
        1.0
        - float(stock_repair["mean_absolute_shift"])
        / float(stock_canonical["mean_absolute_shift"])
    )
    macros = [
        command("GrayDrift", f"{cheap.iloc[1]['mean_absolute_shift']:.3f}"),
        command("AffineDrift", f"{cheap.iloc[2]['mean_absolute_shift']:.3f}"),
        command("ExternalCanonDrift", f"{canonical_worst['mean_absolute_shift']:.3f}"),
        command(
            "ExternalCanonGrayDrift",
            f"{canonical_gray_worst['mean_absolute_shift']:.3f}",
        ),
        command("ExternalRepairDrift", f"{repair_worst['mean_absolute_shift']:.3f}"),
        command(
            "ExternalRepairGrayDrift",
            f"{repair_gray_worst['mean_absolute_shift']:.3f}",
        ),
        command(
            "ExternalCanonFlip",
            f"{100.0 * canonical_worst['decision_flip_rate']:.1f}\\%",
        ),
        command(
            "ExternalRepairFlip",
            f"{100.0 * repair_worst['decision_flip_rate']:.1f}\\%",
        ),
        command("ExternalReduction", f"{external_reduction:.1f}\\%"),
        command(
            "ExternalReductionCI",
            f"[{external_interval['ci_low']:.3f}, {external_interval['ci_high']:.3f}]",
        ),
        command("StockCanonDrift", f"{stock_canonical['mean_absolute_shift']:.3f}"),
        command("StockRepairDrift", f"{stock_repair['mean_absolute_shift']:.3f}"),
        command("StockReduction", f"{stock_reduction:.1f}\\%"),
        command(
            "StockReductionCI",
            f"[{stock_interval['ci_low']:.3f}, {stock_interval['ci_high']:.3f}]",
        ),
        command("StockN", f"{len(stock_download['symbols_complete']):,}"),
    ]
    (args.latex_dir / "extension_macros.tex").write_text("\n".join(macros) + "\n")
    print(summary.to_string(index=False))
    print(f"wrote {args.latex_dir / 'extension_table.tex'}")


if __name__ == "__main__":
    main()
