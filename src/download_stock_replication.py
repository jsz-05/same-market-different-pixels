#!/usr/bin/env python3
"""Download the frozen post-protocol S&P 100 replication panel."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from download_prices import download_symbol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("protocol/extension_protocol.json"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/prices/sp100_daily_final.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/raw/prices/sp100_manifest_final.json")
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    replication = protocol["stock_replication"]
    symbols = list(replication["symbols_requested"])
    minimum_rows = int(replication["minimum_valid_daily_rows"])
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for index, symbol in enumerate(symbols):
        try:
            frame = download_symbol(symbol, args.start, args.end)
            if len(frame) < minimum_rows:
                raise RuntimeError(f"only {len(frame)} valid rows")
            frames.append(frame)
            print(f"{symbol}: {len(frame)} rows", flush=True)
        except Exception as exc:
            failures[symbol] = repr(exc)
            print(f"{symbol}: FAILED {exc}", flush=True)
        if index + 1 < len(symbols):
            time.sleep(0.15)

    if not frames:
        raise RuntimeError("No stock symbols downloaded")
    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    manifest = {
        "status": "post_protocol_replication",
        "source": "Yahoo Finance chart endpoint",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "extension_protocol": str(args.protocol),
        "requested_start": args.start,
        "requested_end_exclusive": args.end,
        "minimum_valid_rows": minimum_rows,
        "symbols_requested": symbols,
        "symbols_complete": sorted(panel["symbol"].unique().tolist()),
        "failures": failures,
        "rows": int(len(panel)),
        "min_date": str(panel["date"].min().date()),
        "max_date": str(panel["date"].max().date()),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
