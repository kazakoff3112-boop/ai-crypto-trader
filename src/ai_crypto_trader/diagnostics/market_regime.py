"""Offline, read-only hourly BTC market-regime report from local cache."""

from __future__ import annotations

import argparse
from decimal import Decimal
import sqlite3

from ..market_data.market_regime import (
    MarketRegimeAnalyzer, REGIMES, RegimeUnavailable, WARMUP_CANDLES,
)
from ..backtesting.periods import BacktestPeriod, parse_utc
from ..config import load_settings
from ..database.historical_cache import HistoricalCache
from ..market_data.candle import HistoricalDataUnavailable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline BTCUSDT 1h market-regime diagnostics")
    parser.add_argument("--symbol", required=True, choices=("BTCUSDT",))
    parser.add_argument("--interval", required=True, choices=("1h",))
    parser.add_argument("--start", required=True, help="inclusive UTC hourly candle open")
    parser.add_argument("--end", required=True, help="exclusive UTC hourly candle open")
    parser.add_argument("--tail", type=int, default=0, help="show last N classified hours")
    args = parser.parse_args(argv)
    if args.tail < 0:
        parser.error("--tail must be non-negative")
    try:
        period = BacktestPeriod.create(parse_utc(args.start), parse_utc(args.end), "1h",
                                       warmup_candles=WARMUP_CANDLES)
        settings = load_settings()
        cache = HistoricalCache.open_readonly(settings.historical_cache_path)
        candles = cache.load(args.symbol, "1h", period.warmup_start, period.evaluation_end)
        report = MarketRegimeAnalyzer().analyze(
            candles, evaluation_start=period.evaluation_start,
            evaluation_end=period.evaluation_end,
        )
        print("MARKET REGIME REPORT")
        print(f"Evaluation period: [{report.evaluation_start.isoformat()}, "
              f"{report.evaluation_end.isoformat()}) UTC")
        print(f"Valid classified hours: {report.valid_classified_hours}")
        for regime in REGIMES:
            stats = report.stats_for(regime)
            print(f"{regime.value}: {stats.hours} hours ({stats.percentage}%)")
        print("Transitions:")
        print(f"  Total transitions: {report.total_transitions}")
        for transition in report.transitions:
            print(f"  {transition.source.value} -> {transition.destination.value}: {transition.count}")
        print(f"First valid regime timestamp: {report.first_valid_regime_timestamp.isoformat()}")
        print(f"Last regime timestamp: {report.last_regime_timestamp.isoformat()}")
        for regime in REGIMES:
            print(f"Longest continuous {regime.value} run: "
                  f"{report.stats_for(regime).longest_run_hours} hours")
        print("NEXT-HOUR BTC RETURNS BY CURRENT REGIME (diagnostic only)")
        for regime in REGIMES:
            stats = report.stats_for(regime)
            average = ("N/A" if stats.average_next_hour_return is None else
                       f"{stats.average_next_hour_return * Decimal('100')}%")
            print(f"  {regime.value}: average {average}; samples {stats.next_hour_sample_count}")
        if args.tail:
            print("LAST REGIME POINTS")
            print("Effective UTC | Close | SMA50 | SMA200 | Slope | Regime")
            for point in report.points[-args.tail:]:
                print(f"{point.effective_time_utc.isoformat()} | {point.close} | {point.sma50} | "
                      f"{point.sma200} | {point.slope_direction} | {point.regime.value}")
        return 0
    except (ValueError, HistoricalDataUnavailable, RegimeUnavailable, sqlite3.Error, OSError):
        print("Market regime unavailable: check local 1h cache, UTC period and candle integrity.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
