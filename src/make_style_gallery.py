#!/usr/bin/env python3
"""Render one pre-test OHLCV window under every evaluation style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chart_core import EVALUATION_STYLES, build_examples, render_chart


LABELS = {
    "canonical": "Canonical",
    "reversed": "Reversed colors",
    "dark": "Dark mode",
    "monochrome": "Monochrome / hollow",
    "blue_orange": "Held-out composition",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/prices/etf_daily_final.parquet"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/style_gallery.pdf"))
    args = parser.parse_args()

    panel = pd.read_parquet(args.data)
    panel["date"] = pd.to_datetime(panel["date"])
    development = panel[(panel["symbol"] == "SPY") & (panel["date"] < "2021-01-01")]
    examples = build_examples(development, lookback=20, horizon=5, stride=1)
    example = min(examples, key=lambda item: abs(pd.Timestamp(item.origin_date) - pd.Timestamp("2020-03-20")))

    fig, axes = plt.subplots(1, len(EVALUATION_STYLES), figsize=(7.1, 1.72))
    for axis, style in zip(axes, EVALUATION_STYLES, strict=True):
        axis.imshow(render_chart(example.ohlcv, style), interpolation="nearest")
        axis.set_title(LABELS[style], fontsize=7.5, pad=3)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#999999")
            spine.set_linewidth(0.45)
    fig.tight_layout(pad=0.25, w_pad=0.35)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"rendered {example.symbol} origin={example.origin_date} to {args.output}")


if __name__ == "__main__":
    main()
