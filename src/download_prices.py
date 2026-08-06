#!/usr/bin/env python3
"""Download a credential-free diversified ETF OHLCV panel from Yahoo chart data."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


DEFAULT_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "VEA",
    "VWO",
    "EWJ",
    "EWG",
    "EWU",
    "EWZ",
    "FXI",
    "TLT",
    "IEF",
    "SHY",
    "TIP",
    "LQD",
    "HYG",
    "EMB",
    "AGG",
    "GLD",
    "SLV",
    "USO",
    "DBA",
    "GDX",
    "VNQ",
    "XLF",
    "XLK",
    "XLE",
    "XLI",
    "XLP",
    "XLY",
    "XLV",
    "XLU",
    "XLB",
    "UUP",
    "FXE",
]


def unix_seconds(date: str) -> int:
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp())


def download_symbol(symbol: str, start: str, end: str, timeout: int = 30) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": unix_seconds(start),
        "period2": unix_seconds(end),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "Mozilla/5.0 ICAIF academic research"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    error = payload.get("chart", {}).get("error")
    if error:
        raise RuntimeError(f"Yahoo error for {symbol}: {error}")
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose", [])
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adjusted if adjusted else quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    frame.insert(0, "symbol", symbol)
    frame = frame.dropna(subset=["open", "high", "low", "close", "adj_close"])
    frame = frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--output", type=Path, default=Path("data/raw/prices/etf_daily.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/prices/manifest.json"))
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for index, symbol in enumerate(args.symbols):
        try:
            frame = download_symbol(symbol, args.start, args.end)
            if len(frame) < 500:
                raise RuntimeError(f"only {len(frame)} valid rows")
            frames.append(frame)
            print(f"{symbol}: {len(frame)} rows", flush=True)
        except Exception as exc:  # preserve partial progress and report every failure
            failures[symbol] = repr(exc)
            print(f"{symbol}: FAILED {exc}", flush=True)
        if index + 1 < len(args.symbols):
            time.sleep(0.15)

    if not frames:
        raise RuntimeError("No symbols downloaded")
    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    manifest = {
        "source": "Yahoo Finance chart endpoint",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": args.start,
        "requested_end_exclusive": args.end,
        "symbols_requested": args.symbols,
        "symbols_complete": sorted(panel["symbol"].unique().tolist()),
        "failures": failures,
        "rows": int(len(panel)),
        "min_date": str(panel["date"].min().date()),
        "max_date": str(panel["date"].max().date()),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

