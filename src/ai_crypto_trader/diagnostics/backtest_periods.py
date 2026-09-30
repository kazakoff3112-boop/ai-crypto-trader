"""Offline comparison of explicit evaluation windows; never runs a strategy."""

from __future__ import annotations

import argparse
import sqlite3

from ..backtesting.periods import BacktestPeriod, compare_periods, parse_utc
from ..config import load_settings
from ..database.historical_cache import HistoricalCache
from ..market_data.candle import HistoricalDataUnavailable, INTERVAL_MINUTES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two UTC backtest periods without trading")
    for name in ("first-start", "first-end", "second-start", "second-end"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"), default="BTCUSDT")
    parser.add_argument("--interval", choices=tuple(INTERVAL_MINUTES), default="15m")
    args = parser.parse_args(argv)
    try:
        first = BacktestPeriod.create(parse_utc(args.first_start), parse_utc(args.first_end), args.interval)
        second = BacktestPeriod.create(parse_utc(args.second_start), parse_utc(args.second_end), args.interval)
        comparison = compare_periods(first, second)
        print(f"First evaluation: [{first.evaluation_start.isoformat()}, {first.evaluation_end.isoformat()})")
        print(f"Second evaluation: [{second.evaluation_start.isoformat()}, {second.evaluation_end.isoformat()})")
        print(f"Order: {comparison.order}")
        print(f"Overlap: {'yes' if comparison.overlap else 'no'}")
        print(f"Strictly non-overlapping: {'yes' if comparison.strictly_non_overlapping else 'no'}")
        print(f"Gap: {comparison.gap}")
        print(f"First duration: {first.evaluation_end - first.evaluation_start}; "
              f"expected candles: {first.evaluation_candles}")
        print(f"Second duration: {second.evaluation_end - second.evaluation_start}; "
              f"expected candles: {second.evaluation_candles}")
        settings = load_settings()
        if settings.historical_cache_path.is_file():
            cache = HistoricalCache.open_readonly(settings.historical_cache_path)
            for label, period in (("First", first), ("Second", second)):
                count = cache.count_range(args.symbol, args.interval,
                                          period.evaluation_start, period.evaluation_end)
                print(f"{label} cached candles: {count}/{period.evaluation_candles} "
                      f"({'complete by count' if count == period.evaluation_candles else 'incomplete'})")
        else:
            print("Cached candle counts: unavailable (no local cache)")
        return 0
    except (ValueError, HistoricalDataUnavailable, sqlite3.Error, OSError):
        print("Period comparison unavailable: check UTC dates, alignment and local cache.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
