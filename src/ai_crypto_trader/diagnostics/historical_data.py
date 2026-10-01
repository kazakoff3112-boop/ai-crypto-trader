"""Explicit public-history download, or strictly offline cache summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import sqlite3

from ..bybit.client import BybitReadOnlyClient
from ..config import Settings, load_settings
from ..database.historical_cache import HistoricalCache
from ..market_data.historical import (
    HistoricalDataUnavailable, HistoricalMarketDataClient, INTERVAL_MINUTES, utc_from_ms,
)

MAX_DOWNLOAD_DAYS = 950


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bybit Demo public spot candle history")
    parser.add_argument("--offline-summary", action="store_true", help="read local cache only; no network")
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--interval", choices=tuple(INTERVAL_MINUTES))
    parser.add_argument("--days", type=int)
    args = parser.parse_args(argv)
    if not args.offline_summary and (args.symbol is None or args.interval is None or args.days is None):
        parser.error("download requires --symbol, --interval and --days")
    if args.days is not None and not 1 <= args.days <= MAX_DOWNLOAD_DAYS:
        parser.error(f"--days must be between 1 and {MAX_DOWNLOAD_DAYS}")
    try:
        settings = load_settings()
        cache = HistoricalCache(settings.historical_cache_path)
        if args.offline_summary:
            if args.days is not None:
                parser.error("--days is not used with --offline-summary")
            entries = [row for row in cache.summary() if (args.symbol is None or row[0] == args.symbol)
                       and (args.interval is None or row[1] == args.interval)]
            if not entries:
                print("No cached historical candles.")
            for symbol, interval, count, first, last, fetched in entries:
                print(f"{symbol} {interval}: {count} candles; {utc_from_ms(first).isoformat()}"
                      f" to {utc_from_ms(last).isoformat()} UTC; source Bybit V5 spot; fetched {fetched}")
            return 0
        step = INTERVAL_MINUTES[args.interval] * 60
        now = datetime.now(timezone.utc)
        end = datetime.fromtimestamp(int(now.timestamp()) // step * step, tz=timezone.utc)
        start = end - timedelta(days=args.days)
        candles = HistoricalMarketDataClient(BybitReadOnlyClient(Settings())).fetch_candles(
            args.symbol, args.interval, start, end
        )
        cache.save(candles)
        print(f"Cached {len(candles)} completed {args.symbol} {args.interval} candles "
              f"from {start.isoformat()} to {end.isoformat()} (exclusive), source Bybit V5 spot.")
        return 0
    except (HistoricalDataUnavailable, OSError, sqlite3.Error, ValueError) as exc:
        # Never show raw transport/SQLite exceptions or credential-bearing values.
        print(f"Historical data unavailable: {type(exc).__name__}. Check the requested range and local cache.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
