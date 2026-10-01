"""Offline benchmark allocation and exposure regression tests."""

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from ai_crypto_trader.backtesting import BacktestConfig, BacktestEngine, BacktestError
from ai_crypto_trader.backtesting.analytics import analyze_backtest
from ai_crypto_trader.backtesting.benchmark import run_buy_and_hold, run_cash_benchmark
from ai_crypto_trader.backtesting.test_strategies import BuyOnceStrategy, NoTradeStrategy
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.backtest import main
from ai_crypto_trader.market_data.candle import Candle
from ai_crypto_trader.strategies.sma_cross import SMACrossoverStrategy

D = Decimal
START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def candles(closes=("100", "110", "120", "80"), opens=None):
    opens = closes if opens is None else opens
    return tuple(Candle(
        "BTCUSDT", "1m", START + timedelta(minutes=index), D(op),
        max(D(op), D(close)), min(D(op), D(close)), D(close), D("1"), D("1"),
    ) for index, (op, close) in enumerate(zip(opens, closes)))


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.data = candles()
        self.zero_cost = BacktestConfig(D("0"), D("0"))

    def test_full_buy_and_hold_allocates_available_capital(self):
        result = run_buy_and_hold(self.data, symbol="BTCUSDT", interval="1m",
                                  config=self.zero_cost)
        self.assertEqual(dict(result.strategy_parameters)["allocation_pct"], "100")
        self.assertEqual(result.ledger[0].gross_value, D("1000"))
        self.assertEqual(result.final_cash, D("0"))
        self.assertEqual(result.buy_operations, 1)
        self.assertEqual(result.sell_operations, 0)

    def test_matched_buy_and_hold_allocates_only_25_percent_and_keeps_cash(self):
        result = run_buy_and_hold(self.data, symbol="BTCUSDT", interval="1m",
                                  config=self.zero_cost, allocation_pct=D("25"))
        self.assertEqual(dict(result.strategy_parameters)["allocation_pct"], "25")
        self.assertEqual(result.ledger[0].gross_value, D("250"))
        self.assertEqual(result.final_cash, D("750"))
        self.assertEqual(result.final_quantity, D("250") / self.data[1].open)
        self.assertEqual(result.ledger_entries, 1)
        self.assertEqual(result.final_equity, D("750") + result.final_quantity * self.data[-1].close)

    def test_matched_buy_and_hold_never_rebalances(self):
        result = run_buy_and_hold(self.data, symbol="BTCUSDT", interval="1m",
                                  config=self.zero_cost, allocation_pct=D("25"))
        self.assertEqual([trade.side for trade in result.ledger], ["BUY"])
        self.assertEqual(result.ledger[0].signal_candle_timestamp_utc, START)
        self.assertEqual(result.ledger[0].execution_candle_timestamp_utc,
                         START + timedelta(minutes=1))
        self.assertEqual(result.final_cash, D("750"))

    def test_commission_slippage_and_cost_identity_once(self):
        config = BacktestConfig(D("0.01"), D("100"))
        result = run_buy_and_hold(self.data, symbol="BTCUSDT", interval="1m",
                                  config=config, allocation_pct=D("25"))
        fill = result.ledger[0]
        self.assertEqual(fill.execution_price, self.data[1].open * D("1.01"))
        self.assertEqual(fill.commission, fill.gross_value * config.fee_rate)
        self.assertEqual(fill.slippage_cost, fill.quantity * (fill.execution_price - self.data[1].open))
        self.assertLessEqual(fill.gross_value + fill.commission, D("250"))
        self.assertGreaterEqual(result.final_cash, D("750"))
        report = analyze_backtest(result)
        self.assertEqual(report.total_trading_costs, result.total_commissions + result.total_slippage)
        self.assertAlmostEqual(report.strategy_pnl_before_costs - report.total_trading_costs,
                               result.total_pnl, delta=D("1e-20"))
        self.assertAlmostEqual(result.final_equity,
                               result.final_cash + result.final_quantity * self.data[-1].close,
                               delta=D("1e-20"))

    def test_benchmark_starts_only_with_evaluation_candles(self):
        evaluation = self.data[1:]
        result = run_buy_and_hold(evaluation, symbol="BTCUSDT", interval="1m",
                                  config=self.zero_cost, allocation_pct=D("25"))
        self.assertEqual(result.evaluation_start, evaluation[0].open_time_utc)
        self.assertEqual(result.ledger[0].signal_candle_timestamp_utc,
                         evaluation[0].open_time_utc)
        self.assertEqual(result.ledger[0].execution_candle_timestamp_utc,
                         evaluation[1].open_time_utc)
        self.assertEqual(result.warmup_candles, 0)
        self.assertEqual(result.equity_curve[0].equity, D("1000"))

    def test_cash_benchmark_is_flat_without_costs(self):
        result = run_cash_benchmark(self.data, symbol="BTCUSDT", interval="1m",
                                    config=BacktestConfig(D("0.01"), D("100")))
        self.assertEqual((result.initial_capital, result.final_equity), (D("1000"), D("1000")))
        self.assertEqual((result.total_pnl, result.total_return_percent), (D("0"), D("0")))
        self.assertEqual((result.max_drawdown_usdt, result.max_drawdown_percent), (D("0"), D("0")))
        self.assertEqual((result.total_commissions, result.total_slippage), (D("0"), D("0")))
        self.assertEqual(result.ledger_entries, 0)
        self.assertEqual(result.average_capital_exposure_percent, D("0"))

    def test_time_and_capital_exposure_are_distinct_and_decimal(self):
        result = BacktestEngine().run(self.data, BuyOnceStrategy(D("1")), symbol="BTCUSDT",
                                      interval="1m", config=self.zero_cost)
        expected = [D("0"), D("110") / D("1000") * D("100"),
                    D("120") / D("1010") * D("100"), D("80") / D("970") * D("100")]
        self.assertEqual(result.time_exposure_percent, D("75"))
        self.assertEqual(result.exposure_percent, result.time_exposure_percent)
        self.assertEqual([point.capital_exposure_percent for point in result.equity_curve], expected)
        self.assertEqual(result.average_capital_exposure_percent, sum(expected) / D("4"))
        self.assertEqual(result.maximum_capital_exposure_percent, max(expected))
        self.assertTrue(all(isinstance(value, Decimal) for value in expected))

    def test_invalid_allocation_fails_closed(self):
        for value in (D("0"), D("-1"), D("101"), D("NaN"), 25.0):
            with self.subTest(value=value), self.assertRaises(BacktestError):
                run_buy_and_hold(self.data, symbol="BTCUSDT", interval="1m",
                                 allocation_pct=value)

    def test_cli_compares_to_matched_not_full_and_preserves_local_files(self):
        # Seven synthetic candles give SMA(2/3) exactly one round trip.
        data = candles(("10", "10", "10", "12", "12", "8", "8"))
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "history.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            HistoricalCache(cache_path).save(list(data))
            paper_path.write_bytes(b"paper-ledger-sentinel")
            before_cache, before_paper = cache_path.read_bytes(), paper_path.read_bytes()
            output = StringIO()
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(cache_path),
                                        "DATABASE_PATH": str(paper_path),
                                        "BYBIT_API_KEY": "unused", "BYBIT_API_SECRET": "unused"}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                    with redirect_stdout(output):
                        code = main(["--symbol", "BTCUSDT", "--interval", "1m",
                                     "--strategy", "sma-cross", "--fast", "2", "--slow", "3",
                                     "--position-size-pct", "25"])
            self.assertEqual(code, 0, output.getvalue())
            expected_strategy = BacktestEngine().run(data, SMACrossoverStrategy(2, 3, D("25")),
                symbol="BTCUSDT", interval="1m",
                config=BacktestConfig(), as_of_utc=START + timedelta(days=1))
            expected_matched = run_buy_and_hold(data, symbol="BTCUSDT", interval="1m",
                                                 config=BacktestConfig(), allocation_pct=D("25"))
            difference = expected_strategy.total_return_percent - expected_matched.total_return_percent
            self.assertIn(f"Difference: {difference} percentage points", output.getvalue())
            self.assertIn("MATCHED BUY & HOLD — 25% allocation", output.getvalue())
            self.assertIn("FULL BUY & HOLD — 100% allocation", output.getvalue())
            self.assertIn("CASH — 0% allocation", output.getvalue())
            self.assertIn("Time exposure:", output.getvalue())
            self.assertIn("Average capital exposure:", output.getvalue())
            self.assertIn("Maximum capital exposure:", output.getvalue())
            self.assertEqual(cache_path.read_bytes(), before_cache)
            self.assertEqual(paper_path.read_bytes(), before_paper)

    def test_no_trade_has_zero_time_and_capital_exposure(self):
        result = BacktestEngine().run(self.data, NoTradeStrategy(), symbol="BTCUSDT",
                                      interval="1m", config=self.zero_cost)
        self.assertEqual((result.time_exposure_percent,
                          result.average_capital_exposure_percent,
                          result.maximum_capital_exposure_percent), (D("0"), D("0"), D("0")))


if __name__ == "__main__":
    unittest.main()
