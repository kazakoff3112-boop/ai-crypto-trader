"""Bollinger baseline: closed-candle math, causality, execution and isolation."""

from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ai_crypto_trader.backtesting import BacktestConfig, BacktestEngine, BacktestError, PortfolioSnapshot, Signal
from ai_crypto_trader.backtesting.analytics import analyze_backtest
from ai_crypto_trader.backtesting.benchmark import run_buy_and_hold
from ai_crypto_trader.backtesting.periods import BacktestPeriod
from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.backtest import main
from ai_crypto_trader.market_data.candle import Candle
from ai_crypto_trader.strategies.bollinger_mean_reversion import (
    BollingerMeanReversionStrategy, calculate_bollinger_bands,
)

D = Decimal
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
STEP = timedelta(minutes=1)


def candles():
    rows = [Candle("BTCUSDT", "1m", START + index * STEP,
                   D("100"), D("100"), D("100"), D("100"), D("1"), D("1"))
            for index in range(22)]
    rows[19] = replace(rows[19], low=D("70"), close=D("70"))
    rows[20] = replace(rows[20], open=D("80"), low=D("80"))
    rows[21] = replace(rows[21], open=D("120"), high=D("120"), close=D("120"))
    return tuple(rows)


def snapshot(quantity="0", cash="1000", equity="1000"):
    return PortfolioSnapshot("BTCUSDT", D(cash), D(quantity), D("0"), D("0"), D(equity))


class BollingerTests(unittest.TestCase):
    def setUp(self):
        self.data = candles()
        self.strategy = BollingerMeanReversionStrategy()
        self.clock = START + timedelta(days=1)

    def run_engine(self, data=None, strategy=None, config=None, period=None):
        return BacktestEngine().run(self.data if data is None else data,
                                    self.strategy if strategy is None else strategy,
                                    symbol="BTCUSDT", interval="1m", as_of_utc=self.clock,
                                    config=BacktestConfig() if config is None else config,
                                    period=period)

    def test_population_mean_variance_stddev_and_bands(self):
        bands = calculate_bollinger_bands(tuple(D(index) for index in range(1, 21)))
        self.assertEqual(bands.middle, D("10.5"))
        self.assertEqual(bands.variance, D("33.25"))
        self.assertEqual(bands.stddev, D("33.25").sqrt())
        self.assertEqual(bands.upper, bands.middle + D("2") * bands.stddev)
        self.assertEqual(bands.lower, bands.middle - D("2") * bands.stddev)
        exact = calculate_bollinger_bands((D("1"), D("3")), period=2)
        self.assertEqual((exact.middle, exact.variance, exact.stddev,
                          exact.upper, exact.lower), (D("2"), D("1"), D("1"), D("4"), D("0")))
        self.assertTrue(all(isinstance(value, Decimal) for value in
                            (bands.middle, bands.variance, bands.stddev, bands.upper, bands.lower)))

    def test_zero_stddev_flat_window_never_buys(self):
        flat = tuple(D("100") for _ in range(20))
        bands = calculate_bollinger_bands(flat)
        self.assertEqual((bands.middle, bands.variance, bands.stddev,
                          bands.upper, bands.lower), (D("100"), D("0"), D("0"), D("100"), D("100")))
        flat_history = tuple(replace(item, low=D("100"), close=D("100")) for item in self.data[:20])
        self.assertEqual(self.strategy.on_candle(flat_history, snapshot()), Signal())

    def test_buy_only_strictly_below_lower_with_no_position(self):
        self.assertEqual(self.strategy.on_candle(self.data[:19], snapshot()), Signal())
        self.assertEqual(self.strategy.on_candle(self.data[:20], snapshot()),
                         Signal("BUY", budget_usdt=D("250")))
        self.assertEqual(self.strategy.on_candle(self.data[:20], snapshot(quantity="1", cash="750")),
                         Signal())
        no_breakout = replace(self.data[19], low=D("100"), close=D("100"))
        self.assertEqual(self.strategy.on_candle((*self.data[:19], no_breakout), snapshot()), Signal())
        above_lower = replace(self.data[19], high=D("105"), low=D("100"), close=D("105"))
        self.assertEqual(self.strategy.on_candle((*self.data[:19], above_lower), snapshot()), Signal())

    def test_sell_at_or_above_middle_only_with_position(self):
        held = snapshot(quantity="1", cash="750")
        self.assertEqual(self.strategy.on_candle(self.data[:21], held),
                         Signal("SELL", quantity=D("1")))
        higher = replace(self.data[20], high=D("110"), close=D("110"))
        self.assertEqual(self.strategy.on_candle((*self.data[:20], higher), held),
                         Signal("SELL", quantity=D("1")))
        below = replace(self.data[20], close=D("90"))
        self.assertEqual(self.strategy.on_candle((*self.data[:20], below), held), Signal())
        self.assertEqual(self.strategy.on_candle(self.data[:21], snapshot()), Signal())
        flat_history = tuple(replace(item, low=D("100"), close=D("100")) for item in self.data[:20])
        self.assertEqual(self.strategy.on_candle(flat_history, held), Signal("SELL", quantity=D("1")))

    def test_future_candles_do_not_change_signal_and_snapshot_is_immutable(self):
        changed = list(self.data)
        changed[20] = replace(changed[20], open=D("1000"), high=D("1000"), close=D("1000"))
        changed[21] = replace(changed[21], open=D("1000"), high=D("1000"), close=D("1000"))
        self.assertEqual(self.strategy.on_candle(self.data[:20], snapshot()),
                         self.strategy.on_candle(tuple(changed[:20]), snapshot()))

        class RecordingBollinger:
            minimum_candles = 21

            def __init__(self):
                self.signals = []

            def on_candle(self, history, portfolio):
                self.signals.append((len(history), BollingerMeanReversionStrategy().on_candle(history, portfolio)))
                return self.signals[-1][1]

        original, future_changed = RecordingBollinger(), RecordingBollinger()
        self.run_engine(strategy=original)
        self.run_engine(data=changed, strategy=future_changed)
        self.assertEqual(original.signals[19], future_changed.signals[19])
        with self.assertRaises(FrozenInstanceError):
            snapshot().cash = D("0")

    def test_next_open_position_sizing_costs_long_only_and_analytics(self):
        config = BacktestConfig(D("0.001"), D("5"))
        result = self.run_engine(config=config)
        self.assertEqual(result.strategy_name, "Bollinger Mean Reversion")
        self.assertEqual(dict(result.strategy_parameters), {
            "bollinger_period": "20", "stddev_multiplier": "2", "position_size_pct": "25",
        })
        self.assertEqual([trade.side for trade in result.ledger], ["BUY", "SELL"])
        buy, sell = result.ledger
        self.assertEqual(buy.signal_candle_timestamp_utc, self.data[19].open_time_utc)
        self.assertEqual(buy.execution_candle_timestamp_utc, self.data[20].open_time_utc)
        self.assertEqual(sell.signal_candle_timestamp_utc, self.data[20].open_time_utc)
        self.assertEqual(sell.execution_candle_timestamp_utc, self.data[21].open_time_utc)
        self.assertEqual(buy.execution_price, D("80") * (D("1") + D("5") / D("10000")))
        self.assertEqual(sell.execution_price, D("120") * (D("1") - D("5") / D("10000")))
        self.assertLessEqual(buy.gross_value + buy.commission, D("250"))
        self.assertEqual(sell.quantity, buy.quantity)
        self.assertEqual(result.final_quantity, D("0"))
        self.assertGreaterEqual(result.final_cash, D("0"))
        self.assertEqual(buy.commission, buy.gross_value * config.fee_rate)
        self.assertEqual(sell.commission, sell.gross_value * config.fee_rate)
        self.assertEqual(buy.slippage_cost, buy.quantity * (buy.execution_price - self.data[20].open))
        self.assertEqual(sell.slippage_cost, sell.quantity * (self.data[21].open - sell.execution_price))
        report = analyze_backtest(result)
        self.assertEqual(report.completed_trade_count, 1)
        self.assertEqual(report.total_trading_costs, result.total_commissions + result.total_slippage)
        self.assertEqual(report.strategy_pnl_after_costs, result.total_pnl)
        self.assertAlmostEqual(report.strategy_pnl_before_costs - report.total_trading_costs,
                               result.total_pnl, delta=D("1e-20"))
        self.assertGreater(result.time_exposure_percent, D("0"))
        self.assertGreater(result.average_capital_exposure_percent, D("0"))
        self.assertGreater(result.maximum_capital_exposure_percent, D("0"))

    def test_explicit_period_uses_19_warmup_candles_without_trades(self):
        period = BacktestPeriod.create(self.data[19].open_time_utc, self.data[-1].close_time_utc,
                                       "1m", warmup_candles=19)
        result = self.run_engine(period=period)
        self.assertEqual((result.warmup_candles, result.evaluation_candles), (19, 3))
        self.assertEqual(result.ledger[0].signal_candle_timestamp_utc, period.evaluation_start)
        self.assertTrue(all(trade.execution_candle_timestamp_utc >= period.evaluation_start
                            for trade in result.ledger))
        self.assertEqual(result.equity_curve[0].equity, D("1000"))
        self.assertEqual(len(result.equity_curve), 3)
        matched = run_buy_and_hold(self.data[19:], symbol="BTCUSDT", interval="1m",
                                   config=BacktestConfig(), allocation_pct=D("25"))
        self.assertEqual(matched.evaluation_start, result.evaluation_start)
        self.assertEqual(matched.evaluation_end, result.evaluation_end)
        self.assertEqual(matched.ledger[0].execution_candle_timestamp_utc,
                         self.data[20].open_time_utc)

    def test_invalid_parameters_prices_and_history_fail_closed(self):
        for period, multiplier, size in ((1, D("2"), D("25")), (20, D("0"), D("25")),
                                         (20, D("-1"), D("25")), (20, D("NaN"), D("25")),
                                         (20, 2.0, D("25")),
                                         (20, D("2"), D("0")), (20, D("2"), D("101")),
                                         (20, D("2"), D("NaN"))):
            with self.subTest(period=period, multiplier=multiplier, size=size):
                with self.assertRaises(ValueError):
                    BollingerMeanReversionStrategy(period, multiplier, size)
        for bad in (D("NaN"), D("Infinity"), D("0"), D("-1"), 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                calculate_bollinger_bands((D("100"),) * 19 + (bad,))
        with self.assertRaises(ValueError):
            calculate_bollinger_bands((D("1"), D("1e999999")), period=2)
        with self.assertRaises(BacktestError):
            self.run_engine(data=self.data[:20])
        for changed in (
            [*self.data[:10], self.data[9], *self.data[11:]],
            [*self.data[:10], self.data[11], *self.data[12:]],
            [replace(self.data[0], symbol="ETHUSDT"), *self.data[1:]],
            [replace(self.data[0], interval="5m"), *self.data[1:]],
            [replace(self.data[0], high=D("0")), *self.data[1:]],
        ):
            with self.subTest(changed=changed), self.assertRaises(BacktestError):
                self.run_engine(data=changed)

    def test_engine_rejects_short_and_unknown_signal(self):
        class RogueStrategy:
            minimum_candles = 21

            def on_candle(self, history, portfolio):
                return Signal("SHORT", quantity=D("1")) if len(history) == 20 else Signal()

        with self.assertRaises(BacktestError):
            self.run_engine(strategy=RogueStrategy())

    def test_cli_offline_analytics_benchmarks_and_local_files_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            HistoricalCache(history_path).save(list(self.data))
            paper_path.write_bytes(b"paper-ledger-sentinel")
            before_history, before_paper = history_path.read_bytes(), paper_path.read_bytes()
            args = ["--symbol", "BTCUSDT", "--interval", "1m", "--strategy", "bollinger-mean-reversion",
                    "--bollinger-period", "20", "--stddev-multiplier", "2", "--position-size-pct", "25",
                    "--start", self.data[19].open_time_utc.isoformat(),
                    "--end", self.data[-1].close_time_utc.isoformat()]
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(history_path),
                                        "DATABASE_PATH": str(paper_path),
                                        "BYBIT_API_KEY": "unused", "BYBIT_API_SECRET": "unused"}):
                for flags in (args, args[:6] + args[12:]):
                    output = StringIO()
                    with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                        with redirect_stdout(output):
                            code = main(flags)
                    self.assertEqual(code, 0, output.getvalue())
                    for phrase in ("bollinger_period=20", "stddev_multiplier=2",
                                   "position_size_pct=25", "Warm-up candles: 19", "TRADE QUALITY",
                                   "Completed trades:", "Win rate:", "Profit factor:", "Turnover:",
                                   "P&L before costs:", "P&L after costs:",
                                   "CASH — 0% allocation", "MATCHED BUY & HOLD — 25% allocation",
                                   "FULL BUY & HOLD — 100% allocation", "FAIR COMPARISON"):
                        self.assertIn(phrase, output.getvalue())
            self.assertEqual(history_path.read_bytes(), before_history)
            self.assertEqual(paper_path.read_bytes(), before_paper)

    def test_no_network_dependency_or_bybit_order_methods(self):
        probe = subprocess.run([sys.executable, "-c",
                                "import sys; import ai_crypto_trader.strategies.bollinger_mean_reversion; "
                                "assert 'ai_crypto_trader.bybit.client' not in sys.modules"],
                               capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        client = BybitReadOnlyClient(Settings())
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(client, name))


if __name__ == "__main__":
    unittest.main()
