from contextlib import redirect_stdout
from dataclasses import replace
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
from ai_crypto_trader.backtesting.benchmark import run_buy_and_hold
from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings, load_settings
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.backtest import main
from ai_crypto_trader.market_data.candle import Candle
from ai_crypto_trader.strategies.sma_cross import SMACrossoverStrategy, simple_moving_average

D = Decimal
START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def candles(closes, *, opens=None):
    opens = closes if opens is None else opens
    return tuple(Candle("BTCUSDT", "1m", START + timedelta(minutes=i), D(str(op)),
                        max(D(str(op)), D(str(close))), min(D(str(op)), D(str(close))),
                        D(str(close)), D("1"), D("1"))
                 for i, (op, close) in enumerate(zip(opens, closes)))


def snapshot(*, cash="1000", quantity="0", equity="1000"):
    return PortfolioSnapshot("BTCUSDT", D(cash), D(quantity), D("0"), D("0"), D(equity))


class SMAEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.strategy = SMACrossoverStrategy(2, 3, D("25"))
        self.data = candles((10, 10, 10, 12, 12, 8, 8))
        self.clock = START + timedelta(days=1)

    def run_engine(self, data=None, config=None, strategy=None):
        return BacktestEngine().run(data or self.data, strategy or self.strategy,
                                    symbol="BTCUSDT", interval="1m", as_of_utc=self.clock,
                                    config=config or BacktestConfig(D("0"), D("0")))

    def test_decimal_sma_uses_last_closed_values(self):
        self.assertEqual(simple_moving_average((D("1"), D("2"), D("4")), 2), D("3"))
        self.assertEqual(simple_moving_average((D("1"), D("2"), D("4")), 3), D("7") / D("3"))
        with self.assertRaises(ValueError):
            simple_moving_average((D("1"),), 2)
        with self.assertRaises(ValueError):
            simple_moving_average((D("NaN"), D("2")), 2)

    def test_buy_and_sell_crossovers_and_next_open(self):
        buy = self.strategy.on_candle(self.data[:4], snapshot())
        self.assertEqual(buy, Signal("BUY", budget_usdt=D("250")))
        self.assertEqual(self.strategy.on_candle(self.data[:5], snapshot()), Signal())
        sell = self.strategy.on_candle(self.data[:6], snapshot(cash="750", quantity="1", equity="760"))
        self.assertEqual(sell, Signal("SELL", quantity=D("1")))
        result = self.run_engine()
        self.assertEqual(result.final_equity, D("916.6666666666666666666666666"))
        self.assertEqual(result.total_pnl, D("-83.3333333333333333333333334"))
        self.assertEqual([t.side for t in result.ledger], ["BUY", "SELL"])
        self.assertEqual(result.ledger[0].signal_candle_timestamp_utc, START + timedelta(minutes=3))
        self.assertEqual(result.ledger[0].execution_candle_timestamp_utc, START + timedelta(minutes=4))
        self.assertEqual(result.ledger[1].signal_candle_timestamp_utc, START + timedelta(minutes=5))
        self.assertEqual(result.ledger[1].execution_candle_timestamp_utc, START + timedelta(minutes=6))
        self.assertEqual(result.strategy_name, "SMA Crossover")
        self.assertEqual(dict(result.strategy_parameters)["position_size_pct"], "25")
        self.assertEqual(result.exposure_percent, D("2") / D("7") * D("100"))

    def test_original_costed_strategy_equity_and_pnl_unchanged(self):
        result = self.run_engine(config=BacktestConfig())
        self.assertEqual(result.final_equity, D("916.1674157926031989000504743"))
        self.assertEqual(result.total_pnl, D("-83.8325842073968010999495257"))

    def test_no_repeat_buy_or_sell_without_position(self):
        self.assertEqual(self.strategy.on_candle(self.data[:4], snapshot(quantity="1")), Signal())
        self.assertEqual(self.strategy.on_candle(self.data[:6], snapshot()), Signal())

    def test_equal_means_only_true_crossing(self):
        flat = candles((10, 10, 10, 10, 10, 10))
        self.assertTrue(all(self.strategy.on_candle(flat[:i], snapshot()) == Signal()
                            for i in range(1, len(flat) + 1)))

    def test_future_candle_change_does_not_change_signal(self):
        original = self.data
        changed = (*original[:4], replace(original[4], open=D("1000"), high=D("1000"),
                                          close=D("1000")), *original[5:])
        signal_at_four = self.strategy.on_candle(original[:4], snapshot())
        self.assertEqual(signal_at_four, self.strategy.on_candle(changed[:4], snapshot()))
        self.assertEqual(signal_at_four, Signal("BUY", budget_usdt=D("250")))

        class RecordingSMA:
            minimum_candles = 5

            def __init__(self):
                self.signals = []

            def on_candle(self, history, portfolio):
                signal = SMACrossoverStrategy(2, 3, D("25")).on_candle(history, portfolio)
                self.signals.append(signal)
                return signal

        first, second = RecordingSMA(), RecordingSMA()
        self.run_engine(data=original, strategy=first)
        self.run_engine(data=changed, strategy=second)
        self.assertEqual(first.signals[3], second.signals[3])

    def test_25_percent_budget_survives_next_open_price_gap(self):
        data = candles((10, 10, 10, 12, 1000, 1000),
                       opens=(10, 10, 10, 12, 1000, 1000))
        result = self.run_engine(data=data, config=BacktestConfig(D("0.001"), D("5")))
        trade = result.ledger[0]
        self.assertLessEqual(trade.gross_value + trade.commission, D("250"))
        self.assertGreaterEqual(result.equity_curve[4].equity, D("0"))
        self.assertEqual(result.buy_operations, 1)

    def test_invalid_parameters_and_insufficient_candles_fail(self):
        for fast, slow, pct in ((0, 3, D("25")), (3, 3, D("25")),
                                (4, 3, D("25")), (2, 3, D("0")),
                                (2, 3, D("101")), (2, 3, D("NaN"))):
            with self.subTest(fast=fast, slow=slow, pct=pct):
                with self.assertRaises(ValueError):
                    SMACrossoverStrategy(fast, slow, pct)
        with self.assertRaises(BacktestError):
            self.run_engine(data=self.data[:4])
        with self.assertRaises(BacktestError):
            self.run_engine(data=[*self.data[:3], self.data[4], *self.data[5:]])
        with self.assertRaises(BacktestError):
            self.run_engine(data=[replace(self.data[0], close=D("0")), *self.data[1:]])

    def test_settings_default_env_override_and_rejection(self):
        self.assertEqual(Settings().backtest_position_size_pct, D("25"))
        self.assertEqual(load_settings(environ={"BACKTEST_POSITION_SIZE_PCT": "30"}).backtest_position_size_pct,
                         D("30"))
        for value in ("0", "101", "NaN", "-1"):
            with self.assertRaises(ValueError):
                load_settings(environ={"BACKTEST_POSITION_SIZE_PCT": value})

    def test_buy_hold_benchmark_uses_same_cost_model(self):
        config = BacktestConfig(D("0.01"), D("100"))
        benchmark = run_buy_and_hold(self.data, symbol="BTCUSDT", interval="1m", config=config)
        trade = benchmark.ledger[0]
        self.assertEqual(benchmark.buy_operations, 1)
        self.assertEqual(benchmark.sell_operations, 0)
        self.assertEqual(trade.signal_candle_timestamp_utc, START)
        self.assertEqual(trade.execution_candle_timestamp_utc, START + timedelta(minutes=1))
        self.assertEqual(trade.execution_price, D("10.1"))
        self.assertEqual(trade.commission, trade.gross_value * config.fee_rate)
        self.assertEqual(trade.slippage_cost, trade.quantity * D("0.1"))
        self.assertEqual(benchmark.total_commissions, trade.commission)
        self.assertEqual(benchmark.total_slippage, trade.slippage_cost)
        self.assertEqual(benchmark.final_equity,
                         D("1000") - trade.gross_value - trade.commission + trade.quantity * self.data[-1].close)
        self.assertEqual(benchmark.strategy_name, "Buy & Hold benchmark")

    def test_sma_and_benchmark_share_fee_and_slippage_rules(self):
        data = candles((10, 10, 10, 12, 12, 8, 8), opens=(10, 10, 10, 10, 10, 10, 10))
        config = BacktestConfig(D("0.01"), D("100"))
        sma = self.run_engine(data=data, config=config)
        hold = run_buy_and_hold(data, symbol="BTCUSDT", interval="1m", config=config)
        for trade in (sma.ledger[0], hold.ledger[0]):
            self.assertEqual(trade.execution_price, D("10.1"))
            self.assertEqual(trade.commission, trade.gross_value * config.fee_rate)
            self.assertEqual(trade.slippage_cost, trade.quantity * D("0.1"))

    def test_cli_offline_and_databases_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            historical = Path(directory) / "history.sqlite3"
            paper = Path(directory) / "paper.sqlite3"
            HistoricalCache(historical).save(list(self.data))
            paper.write_bytes(b"paper-ledger-sentinel")
            before_history, before_paper = historical.read_bytes(), paper.read_bytes()
            output = StringIO()
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(historical),
                                        "DATABASE_PATH": str(paper),
                                        "BYBIT_API_KEY": "unused", "BYBIT_API_SECRET": "unused"}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                    with redirect_stdout(output):
                        code = main(["--symbol", "BTCUSDT", "--interval", "1m",
                                     "--strategy", "sma-cross", "--fast", "2", "--slow", "3",
                                     "--position-size-pct", "25"])
            self.assertEqual(code, 0, output.getvalue())
            self.assertIn("Matched B&H return:", output.getvalue())
            self.assertIn("Difference:", output.getvalue())
            self.assertEqual(historical.read_bytes(), before_history)
            self.assertEqual(paper.read_bytes(), before_paper)

    def test_no_bybit_dependency_or_order_methods(self):
        probe = subprocess.run([sys.executable, "-c",
                                "import sys; import ai_crypto_trader.strategies.sma_cross; "
                                "import ai_crypto_trader.backtesting.benchmark; "
                                "assert 'ai_crypto_trader.bybit.client' not in sys.modules"],
                               capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        client = BybitReadOnlyClient(Settings())
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(client, name))


if __name__ == "__main__":
    unittest.main()
