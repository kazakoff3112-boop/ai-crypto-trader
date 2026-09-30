"""Explicit offline backtest of cached candles; never places an order."""

from __future__ import annotations

import argparse
from datetime import timedelta
from decimal import Decimal
import sqlite3

from ..backtesting import BacktestConfig, BacktestEngine, BacktestError
from ..backtesting.benchmark import run_buy_and_hold
from ..backtesting.test_strategies import BuyOnceStrategy, BuyThenSellStrategy, NoTradeStrategy
from ..config import load_settings
from ..database.historical_cache import HistoricalCache
from ..market_data.candle import HistoricalDataUnavailable, INTERVAL_MINUTES, utc_from_ms
from ..strategies.sma_cross import SMACrossoverStrategy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline backtest; never places trades")
    parser.add_argument("--symbol", required=True, choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--interval", required=True, choices=tuple(INTERVAL_MINUTES))
    parser.add_argument("--strategy", required=True,
                        choices=("no-trade", "buy-once", "buy-then-sell", "sma-cross"))
    parser.add_argument("--fast", type=int)
    parser.add_argument("--slow", type=int)
    parser.add_argument("--position-size-pct", type=Decimal)
    args = parser.parse_args(argv)
    if args.strategy != "sma-cross" and any(
        value is not None for value in (args.fast, args.slow, args.position_size_pct)
    ):
        parser.error("SMA parameters require --strategy sma-cross")
    fixtures = {
        "no-trade": NoTradeStrategy,
        "buy-once": BuyOnceStrategy,
        "buy-then-sell": BuyThenSellStrategy,
    }
    try:
        settings = load_settings()
        cache = HistoricalCache.open_readonly(settings.historical_cache_path)
        matches = [row for row in cache.summary() if row[0] == args.symbol and row[1] == args.interval]
        if len(matches) != 1:
            raise HistoricalDataUnavailable("No cached candles for symbol and interval")
        _, _, _, first_ms, last_ms, _ = matches[0]
        step = timedelta(minutes=INTERVAL_MINUTES[args.interval])
        candles = cache.load(args.symbol, args.interval, utc_from_ms(first_ms),
                             utc_from_ms(last_ms) + step)
        strategy = (
            SMACrossoverStrategy(
                fast=20 if args.fast is None else args.fast,
                slow=50 if args.slow is None else args.slow,
                position_size_pct=(settings.backtest_position_size_pct
                                   if args.position_size_pct is None else args.position_size_pct),
            ) if args.strategy == "sma-cross" else fixtures[args.strategy]()
        )
        execution_config = BacktestConfig.from_settings(settings)
        result = BacktestEngine().run(
            candles, strategy, symbol=args.symbol, interval=args.interval,
            config=execution_config,
        )
        print(f"OFFLINE BACKTEST — {result.strategy_name}")
        if result.strategy_parameters:
            print("Parameters: " + ", ".join(f"{name}={value}" for name, value in result.strategy_parameters))
        print(f"Symbol/interval: {args.symbol} {args.interval}; candles: {result.candles_processed}")
        print(f"Initial capital: {result.initial_capital} USDT")
        print(f"Final equity: {result.final_equity} USDT")
        print(f"Total P&L: {result.total_pnl} USDT; return: {result.total_return_percent}%")
        print(f"Realized/unrealized P&L: {result.realized_pnl} / {result.unrealized_pnl} USDT")
        print(f"Commission/slippage: {result.total_commissions} / {result.total_slippage} USDT")
        print(f"BUY/SELL: {result.buy_operations} / {result.sell_operations}")
        print(f"Max/min equity: {result.max_equity} / {result.min_equity} USDT")
        print(f"Max drawdown: {result.max_drawdown_usdt} USDT ({result.max_drawdown_percent}%)")
        print(f"Exposure: {result.exposure_percent}% of closed candles")
        print(f"Backtest ledger entries: {result.ledger_entries}; no PAPER trades or Bybit calls")
        if args.strategy == "sma-cross":
            benchmark = run_buy_and_hold(candles, symbol=args.symbol, interval=args.interval,
                                         config=execution_config, initial_capital=result.initial_capital)
            print(f"Buy & Hold return: {benchmark.total_return_percent}%")
            print(f"SMA strategy return: {result.total_return_percent}%")
            print(f"Difference: {result.total_return_percent - benchmark.total_return_percent} percentage points")
        return 0
    except (HistoricalDataUnavailable, BacktestError, sqlite3.Error, OSError, ValueError):
        print("Backtest unavailable: check local cache completeness, configuration and candle integrity.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
