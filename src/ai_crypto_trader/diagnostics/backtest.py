"""Explicit offline backtest of cached candles; never places an order."""

from __future__ import annotations

import argparse
from datetime import timedelta
from decimal import Decimal
import sqlite3

from ..backtesting import BacktestConfig, BacktestEngine, BacktestError, BacktestResult
from ..backtesting.analytics import BacktestAnalyticsError, analyze_backtest
from ..backtesting.benchmark import run_buy_and_hold, run_cash_benchmark
from ..backtesting.periods import BacktestPeriod, parse_utc
from ..backtesting.test_strategies import BuyOnceStrategy, BuyThenSellStrategy, NoTradeStrategy
from ..config import load_settings
from ..database.historical_cache import HistoricalCache
from ..market_data.candle import HistoricalDataUnavailable, INTERVAL_MINUTES, utc_from_ms
from ..strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from ..strategies.donchian_breakout import DonchianBreakoutStrategy
from ..strategies.sma_cross import SMACrossoverStrategy


def _usdt(value: Decimal | None) -> str:
    return "N/A" if value is None else f"{value} USDT"


def _print_benchmark(label: str, allocation_pct: Decimal, result: BacktestResult,
                     costs: Decimal) -> None:
    print(f"{label} — {allocation_pct}% allocation")
    print(f"  Allocation: {allocation_pct}%")
    print(f"  Initial capital: {result.initial_capital} USDT")
    print(f"  Final equity: {result.final_equity} USDT")
    print(f"  Total P&L: {result.total_pnl} USDT")
    print(f"  Return: {result.total_return_percent}%")
    print(f"  Max drawdown: {result.max_drawdown_usdt} USDT ({result.max_drawdown_percent}%)")
    print(f"  Commission: {result.total_commissions} USDT")
    print(f"  Slippage: {result.total_slippage} USDT")
    print(f"  Trading costs: {costs} USDT")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline backtest; never places trades")
    parser.add_argument("--symbol", required=True, choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--interval", required=True, choices=tuple(INTERVAL_MINUTES))
    parser.add_argument("--strategy", required=True,
                        choices=("no-trade", "buy-once", "buy-then-sell", "sma-cross",
                                 "donchian-breakout", "bollinger-mean-reversion"))
    parser.add_argument("--fast", type=int)
    parser.add_argument("--slow", type=int)
    parser.add_argument("--entry-lookback", type=int)
    parser.add_argument("--exit-lookback", type=int)
    parser.add_argument("--bollinger-period", type=int)
    parser.add_argument("--stddev-multiplier", type=Decimal)
    parser.add_argument("--position-size-pct", type=Decimal)
    parser.add_argument("--start", help="inclusive ISO 8601 UTC candle open")
    parser.add_argument("--end", help="exclusive ISO 8601 UTC candle open")
    args = parser.parse_args(argv)
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be supplied together")
    if args.strategy != "sma-cross" and any(value is not None for value in (args.fast, args.slow)):
        parser.error("--fast and --slow require --strategy sma-cross")
    if args.strategy != "donchian-breakout" and any(
        value is not None for value in (args.entry_lookback, args.exit_lookback)
    ):
        parser.error("Donchian lookbacks require --strategy donchian-breakout")
    if args.strategy != "bollinger-mean-reversion" and any(
        value is not None for value in (args.bollinger_period, args.stddev_multiplier)
    ):
        parser.error("Bollinger parameters require --strategy bollinger-mean-reversion")
    if args.strategy not in {"sma-cross", "donchian-breakout", "bollinger-mean-reversion"} and args.position_size_pct is not None:
        parser.error("Position size requires a baseline strategy")
    fixtures = {
        "no-trade": NoTradeStrategy,
        "buy-once": BuyOnceStrategy,
        "buy-then-sell": BuyThenSellStrategy,
    }
    try:
        settings = load_settings()
        if args.strategy == "sma-cross":
            strategy = SMACrossoverStrategy(
                fast=20 if args.fast is None else args.fast,
                slow=50 if args.slow is None else args.slow,
                position_size_pct=(settings.backtest_position_size_pct
                                   if args.position_size_pct is None else args.position_size_pct),
            )
        elif args.strategy == "donchian-breakout":
            strategy = DonchianBreakoutStrategy(
                entry_lookback=96 if args.entry_lookback is None else args.entry_lookback,
                exit_lookback=48 if args.exit_lookback is None else args.exit_lookback,
                position_size_pct=(Decimal("25") if args.position_size_pct is None
                                   else args.position_size_pct),
            )
        elif args.strategy == "bollinger-mean-reversion":
            strategy = BollingerMeanReversionStrategy(
                bollinger_period=20 if args.bollinger_period is None else args.bollinger_period,
                stddev_multiplier=(Decimal("2") if args.stddev_multiplier is None
                                   else args.stddev_multiplier),
                position_size_pct=(Decimal("25") if args.position_size_pct is None
                                   else args.position_size_pct),
            )
        else:
            strategy = fixtures[args.strategy]()
        cache = HistoricalCache.open_readonly(settings.historical_cache_path)
        period = None
        if args.start is not None:
            period = BacktestPeriod.create(
                parse_utc(args.start), parse_utc(args.end), args.interval,
                warmup_candles=strategy.minimum_candles - 2 if hasattr(strategy, "minimum_candles") else 0,
            )
            candles = cache.load(args.symbol, args.interval,
                                 period.warmup_start or period.evaluation_start,
                                 period.evaluation_end)
        else:
            matches = [row for row in cache.summary() if row[0] == args.symbol and row[1] == args.interval]
            if len(matches) != 1:
                raise HistoricalDataUnavailable("No cached candles for symbol and interval")
            _, _, _, first_ms, last_ms, _ = matches[0]
            step = timedelta(minutes=INTERVAL_MINUTES[args.interval])
            candles = cache.load(args.symbol, args.interval, utc_from_ms(first_ms),
                                 utc_from_ms(last_ms) + step)
        execution_config = BacktestConfig.from_settings(settings)
        result = BacktestEngine().run(
            candles, strategy, symbol=args.symbol, interval=args.interval,
            config=execution_config, period=period,
        )
        print(f"OFFLINE BACKTEST — {result.strategy_name}")
        print(f"Evaluation period: [{result.evaluation_start.isoformat()}, {result.evaluation_end.isoformat()}) UTC")
        print(f"Warm-up period: [{result.warmup_start.isoformat()}, {result.evaluation_start.isoformat()}) UTC"
              if result.warmup_start is not None else "Warm-up period: none")
        print(f"Evaluation candles: {result.evaluation_candles}")
        print(f"Warm-up candles: {result.warmup_candles}")
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
        print(f"Time exposure: {result.time_exposure_percent}% of evaluation candles")
        print(f"Average capital exposure: {result.average_capital_exposure_percent}%")
        print(f"Maximum capital exposure: {result.maximum_capital_exposure_percent}%")
        print(f"Backtest ledger entries: {result.ledger_entries}; no PAPER trades or Bybit calls")
        if args.strategy in {"sma-cross", "donchian-breakout", "bollinger-mean-reversion"}:
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
            evaluation_candles = candles[result.warmup_candles:]
            allocation_pct = strategy.position_size_pct
            matched = run_buy_and_hold(evaluation_candles, symbol=args.symbol, interval=args.interval,
                                       config=execution_config, initial_capital=result.initial_capital,
                                       allocation_pct=allocation_pct)
            full = run_buy_and_hold(evaluation_candles, symbol=args.symbol, interval=args.interval,
                                    config=execution_config, initial_capital=result.initial_capital)
            cash = run_cash_benchmark(evaluation_candles, symbol=args.symbol, interval=args.interval,
                                      config=execution_config, initial_capital=result.initial_capital)
            for benchmark in (matched, full, cash):
                if (benchmark.evaluation_start != result.evaluation_start
                        or benchmark.evaluation_end != result.evaluation_end
                        or benchmark.initial_capital != result.initial_capital):
                    raise BacktestError("Benchmark evaluation period or capital differs from strategy")
            matched_costs = analyze_backtest(matched).total_trading_costs
            full_costs = analyze_backtest(full).total_trading_costs
            cash_costs = analyze_backtest(cash).total_trading_costs
            print("BENCHMARKS")
            _print_benchmark("CASH", Decimal("0"), cash, cash_costs)
            _print_benchmark("MATCHED BUY & HOLD", allocation_pct, matched, matched_costs)
            _print_benchmark("FULL BUY & HOLD", Decimal("100"), full, full_costs)
            print("FAIR COMPARISON — strategy vs matched allocation")
            print(f"Strategy return: {result.total_return_percent}%")
            print(f"Matched B&H return: {matched.total_return_percent}%")
            print(f"Difference: {result.total_return_percent - matched.total_return_percent} percentage points")
            print(f"Strategy max drawdown: {result.max_drawdown_usdt} USDT ({result.max_drawdown_percent}%)")
            print(f"Matched B&H max drawdown: {matched.max_drawdown_usdt} USDT "
                  f"({matched.max_drawdown_percent}%)")
            print(f"Time exposure: {result.time_exposure_percent}%")
            print(f"Average capital exposure: {result.average_capital_exposure_percent}%")
            print(f"Maximum capital exposure: {result.maximum_capital_exposure_percent}%")
        return 0
    except (HistoricalDataUnavailable, BacktestError, BacktestAnalyticsError,
            sqlite3.Error, OSError, ValueError):
        print("Backtest unavailable: check local cache completeness, configuration and candle integrity.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
