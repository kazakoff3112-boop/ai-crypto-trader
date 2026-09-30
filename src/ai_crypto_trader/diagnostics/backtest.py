"""Explicit offline backtest of cached candles; never places an order."""

from __future__ import annotations

import argparse
from datetime import timedelta
from decimal import Decimal
import sqlite3

from ..backtesting import BacktestConfig, BacktestEngine, BacktestError
from ..backtesting.analytics import analyze_backtest
from ..backtesting.benchmark import run_buy_and_hold
from ..backtesting.test_strategies import BuyOnceStrategy, BuyThenSellStrategy, NoTradeStrategy
from ..config import load_settings
from ..database.historical_cache import HistoricalCache
from ..market_data.candle import HistoricalDataUnavailable, INTERVAL_MINUTES, utc_from_ms
from ..strategies.sma_cross import SMACrossoverStrategy


def _usdt(value: Decimal | None) -> str:
    return "N/A" if value is None else f"{value} USDT"


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
            quality = analyze_backtest(result)
            print("TRADE QUALITY")
            print(f"Completed trades: {quality.completed_trade_count}")
            print(f"Wins: {quality.winning_trades}")
            print(f"Losses: {quality.losing_trades}")
            print(f"Breakeven: {quality.breakeven_trades}")
            print(f"Win rate: {quality.win_rate_percent}%")
            print(f"Loss rate: {quality.loss_rate_percent}%")
            print(f"Average / median net P&L: {_usdt(quality.average_net_pnl)} / {_usdt(quality.median_net_pnl)}")
            print(f"Average winner: {_usdt(quality.average_winner)}")
            print(f"Average loser: {_usdt(quality.average_loser)}")
            print(f"Largest winner / loser: {_usdt(quality.largest_winner)} / {_usdt(quality.largest_loser)}")
            print(f"Gross profit / loss: {quality.gross_profit} / {quality.gross_loss} USDT")
            print(f"Profit factor: {quality.profit_factor if quality.profit_factor is not None else 'N/A (no losses)'}")
            print(f"Average holding time: {quality.average_holding_time if quality.average_holding_time is not None else 'N/A'}")
            print(f"Min / max holding time: {quality.minimum_holding_time if quality.minimum_holding_time is not None else 'N/A'} / "
                  f"{quality.maximum_holding_time if quality.maximum_holding_time is not None else 'N/A'}")
            print(f"Turnover: {quality.total_traded_notional} USDT")
            print(f"Total commission / slippage: {quality.total_commissions} / {quality.total_slippage} USDT")
            print(f"P&L before costs: {quality.strategy_pnl_before_costs} USDT")
            print(f"Trading costs: {quality.total_trading_costs} USDT")
            print(f"P&L after costs: {quality.strategy_pnl_after_costs} USDT")
            print(f"Commission/slippage drag: {quality.commission_drag} / {quality.slippage_drag} USDT")
            print(f"Approx. return before costs: {quality.approximate_return_before_costs_percent}%")
            print(f"Return after costs: {quality.return_after_costs_percent}%")
            if quality.open_position is not None:
                print(f"Open position (not a completed trade): {quality.open_position.quantity} {args.symbol}; "
                      f"unrealized P&L {quality.open_position.unrealized_pnl} USDT")
            benchmark = run_buy_and_hold(candles, symbol=args.symbol, interval=args.interval,
                                         config=execution_config, initial_capital=result.initial_capital)
            benchmark_quality = analyze_backtest(benchmark)
            print(f"Buy & Hold final equity: {benchmark.final_equity} USDT")
            print(f"Buy & Hold return: {benchmark.total_return_percent}%")
            print(f"Buy & Hold max drawdown: {benchmark.max_drawdown_usdt} USDT "
                  f"({benchmark.max_drawdown_percent}%)")
            print(f"Buy & Hold trading costs: {benchmark_quality.total_trading_costs} USDT")
            print(f"SMA strategy return: {result.total_return_percent}%")
            print(f"Difference: {result.total_return_percent - benchmark.total_return_percent} percentage points")
        return 0
    except (HistoricalDataUnavailable, BacktestError, sqlite3.Error, OSError, ValueError):
        print("Backtest unavailable: check local cache completeness, configuration and candle integrity.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
