"""Explicit offline backtest of cached candles with technical fixtures only."""

from __future__ import annotations

import argparse
from datetime import timedelta
from decimal import Decimal
import sqlite3

from ..backtesting import BacktestConfig, BacktestEngine, BacktestError
from ..backtesting.test_strategies import BuyOnceStrategy, BuyThenSellStrategy, NoTradeStrategy
from ..config import load_settings
from ..database.historical_cache import HistoricalCache
from ..market_data.candle import HistoricalDataUnavailable, INTERVAL_MINUTES, utc_from_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline technical backtest; never places trades")
    parser.add_argument("--symbol", required=True, choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--interval", required=True, choices=tuple(INTERVAL_MINUTES))
    parser.add_argument("--strategy", required=True, choices=("no-trade", "buy-once", "buy-then-sell"))
    args = parser.parse_args(argv)
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
        result = BacktestEngine().run(
            candles, fixtures[args.strategy](), symbol=args.symbol, interval=args.interval,
            config=BacktestConfig.from_settings(settings),
        )
        print(f"OFFLINE BACKTEST — technical fixture: {args.strategy}")
        print(f"Symbol/interval: {args.symbol} {args.interval}; candles: {result.candles_processed}")
        print(f"Initial capital: {result.initial_capital} USDT")
        print(f"Final equity: {result.final_equity} USDT")
        print(f"Total P&L: {result.total_pnl} USDT; return: {result.total_return_percent}%")
        print(f"Realized/unrealized P&L: {result.realized_pnl} / {result.unrealized_pnl} USDT")
        print(f"Commission/slippage: {result.total_commissions} / {result.total_slippage} USDT")
        print(f"BUY/SELL: {result.buy_operations} / {result.sell_operations}")
        print(f"Max/min equity: {result.max_equity} / {result.min_equity} USDT")
        print(f"Max drawdown: {result.max_drawdown_usdt} USDT ({result.max_drawdown_percent}%)")
        print(f"Backtest ledger entries: {result.ledger_entries}; no PAPER trades or Bybit calls")
        return 0
    except (HistoricalDataUnavailable, BacktestError, sqlite3.Error, OSError, ValueError):
        print("Backtest unavailable: check local cache completeness, configuration and candle integrity.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
