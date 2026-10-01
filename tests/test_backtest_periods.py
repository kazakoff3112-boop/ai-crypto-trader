from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ai_crypto_trader.backtesting import BacktestConfig, BacktestEngine, BacktestError, Signal
from ai_crypto_trader.backtesting.benchmark import run_buy_and_hold
from ai_crypto_trader.backtesting.periods import BacktestPeriod, compare_periods, parse_utc
from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.backtest import main as backtest_main
from ai_crypto_trader.diagnostics.backtest_periods import main as compare_main
from ai_crypto_trader.market_data.candle import Candle
from ai_crypto_trader.strategies.donchian_breakout import DonchianBreakoutStrategy
from ai_crypto_trader.strategies.sma_cross import SMACrossoverStrategy

D = Decimal
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
STEP = timedelta(minutes=1)


def candles(count=120):
    rows = [Candle("BTCUSDT", "1m", START + STEP * index, D("10"), D("10"), D("9"),
                   D("10"), D("1"), D("1")) for index in range(count)]
    if count > 50:
        rows[50] = replace(rows[50], high=D("12"), close=D("12"))
    if count > 51:
        rows[51] = replace(rows[51], open=D("20"), high=D("20"), low=D("19"), close=D("20"))
    if count > 96:
        rows[96] = replace(rows[96], high=D("21"), close=D("21"))
    if count > 97:
        rows[97] = replace(rows[97], open=D("22"), high=D("22"), low=D("21"), close=D("22"))
    return tuple(rows)


class PeriodTests(unittest.TestCase):
    def setUp(self):
        self.data = candles()
        self.clock = START + timedelta(days=1)

    def run_period(self, strategy, period, data=None):
        source = self.data if data is None else data
        subset = tuple(candle for candle in source
                       if (period.warmup_start or period.evaluation_start)
                       <= candle.open_time_utc < period.evaluation_end)
        return BacktestEngine().run(subset, strategy, symbol="BTCUSDT", interval="1m",
                                    config=BacktestConfig(D("0"), D("0")),
                                    as_of_utc=self.clock, period=period)

    def test_start_inclusive_end_exclusive_and_result_metadata(self):
        period = BacktestPeriod.create(START + STEP * 60, START + STEP * 63, "1m",
                                       warmup_candles=50)
        result = self.run_period(SMACrossoverStrategy(), period)
        self.assertEqual((result.evaluation_start, result.evaluation_end),
                         (START + STEP * 60, START + STEP * 63))
        self.assertEqual((result.warmup_start, result.warmup_candles,
                          result.evaluation_candles, result.candles_processed),
                         (START + STEP * 10, 50, 3, 3))
        self.assertEqual(result.equity_curve[0].timestamp_utc, START + STEP * 61)
        self.assertEqual(result.equity_curve[-1].timestamp_utc, START + STEP * 63)
        self.assertEqual(len(result.equity_curve), 3)

    def test_warmup_never_trades_or_enters_pnl(self):
        class WarmupProbe:
            minimum_candles = 4  # two warm-up candles, two evaluation candles

            def __init__(self):
                self.seen = []

            def on_candle(self, history, portfolio):
                self.seen.append((len(history), portfolio.cash))
                return Signal("BUY", quantity=D("1")) if len(history) == 1 else Signal()

        period = BacktestPeriod.create(START + STEP * 2, START + STEP * 4, "1m",
                                       warmup_candles=2)
        probe = WarmupProbe()
        result = self.run_period(probe, period)
        self.assertEqual(probe.seen, [(3, D("1000")), (4, D("1000"))])
        self.assertEqual(result.ledger_entries, 0)
        self.assertEqual(result.final_equity, D("1000"))
        self.assertEqual(len(result.equity_curve), 2)

    def test_first_evaluation_signal_uses_warmup_but_executes_next_open(self):
        period = BacktestPeriod.create(START + STEP * 50, START + STEP * 53, "1m",
                                       warmup_candles=50)
        result = self.run_period(SMACrossoverStrategy(), period)
        self.assertEqual(result.buy_operations, 1)
        self.assertEqual(result.ledger[0].signal_candle_timestamp_utc, START + STEP * 50)
        self.assertEqual(result.ledger[0].execution_candle_timestamp_utc, START + STEP * 51)
        self.assertTrue(all(trade.execution_candle_timestamp_utc >= period.evaluation_start
                            for trade in result.ledger))
        self.assertEqual(result.equity_curve[0].equity, D("1000"))

    def test_donchian_default_uses_96_warmup_candles(self):
        # Previous 96 highs are at most 20; evaluation candle 96 closes at 21.
        period = BacktestPeriod.create(START + STEP * 96, START + STEP * 99, "1m",
                                       warmup_candles=96)
        result = self.run_period(DonchianBreakoutStrategy(), period)
        self.assertEqual(result.warmup_candles, 96)
        self.assertEqual(result.ledger[0].signal_candle_timestamp_utc, START + STEP * 96)
        self.assertEqual(result.ledger[0].execution_candle_timestamp_utc, START + STEP * 97)

    def test_benchmark_starts_only_in_evaluation(self):
        period = BacktestPeriod.create(START + STEP * 50, START + STEP * 53, "1m",
                                       warmup_candles=50)
        evaluation = self.data[50:53]
        benchmark = run_buy_and_hold(evaluation, symbol="BTCUSDT", interval="1m",
                                     config=BacktestConfig(D("0"), D("0")))
        self.assertEqual(benchmark.evaluation_start, period.evaluation_start)
        self.assertEqual(benchmark.evaluation_end, period.evaluation_end)
        self.assertEqual(benchmark.ledger[0].signal_candle_timestamp_utc, START + STEP * 50)
        self.assertEqual(benchmark.ledger[0].execution_candle_timestamp_utc, START + STEP * 51)
        self.assertEqual(benchmark.warmup_candles, 0)

    def test_overlap_adjacency_reverse_order_and_gap(self):
        first = BacktestPeriod.create(START, START + STEP * 10, "1m")
        adjacent = BacktestPeriod.create(START + STEP * 10, START + STEP * 20, "1m")
        overlapping = BacktestPeriod.create(START + STEP * 9, START + STEP * 20, "1m")
        later = BacktestPeriod.create(START + STEP * 12, START + STEP * 20, "1m")
        self.assertTrue(compare_periods(first, adjacent).strictly_non_overlapping)
        self.assertEqual(compare_periods(first, adjacent).gap, timedelta(0))
        self.assertTrue(compare_periods(first, overlapping).overlap)
        self.assertFalse(compare_periods(first, overlapping).strictly_non_overlapping)
        self.assertEqual(compare_periods(first, later).gap, STEP * 2)
        self.assertEqual(compare_periods(later, first).order, "second-before-first")

    def test_invalid_dates_alignment_and_start_end_rejected(self):
        for value in ("2024-01-01T00:00:00", "2024-01-01T03:00:00+03:00",
                      "2024-01-01T00:00:00-00:00", "bad", "2024-01-01"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_utc(value)
        self.assertEqual(parse_utc("2024-01-01T00:00:00Z"), START)
        with self.assertRaises(ValueError):
            BacktestPeriod.create(START, START, "1m")
        with self.assertRaises(ValueError):
            BacktestPeriod.create(START + STEP, START, "1m")
        with self.assertRaises(ValueError):
            BacktestPeriod.create(START + timedelta(seconds=1), START + STEP * 2, "1m")

    def test_insufficient_warmup_gaps_duplicates_and_mixed_candles_fail_closed(self):
        period = BacktestPeriod.create(START + STEP * 50, START + STEP * 53, "1m",
                                       warmup_candles=50)
        for data in (self.data[1:],
                     [*self.data[:20], *self.data[21:]],
                     [*self.data[:51], *self.data[52:]],
                     [*self.data[:20], self.data[19], *self.data[21:]],
                     [replace(self.data[20], symbol="ETHUSDT") if index == 20 else item
                      for index, item in enumerate(self.data)],
                     [replace(self.data[50], interval="5m") if index == 50 else item
                      for index, item in enumerate(self.data)]):
            with self.subTest(data=data):
                with self.assertRaises(BacktestError):
                    self.run_period(SMACrossoverStrategy(), period, data=data)
        short_period = BacktestPeriod.create(START + STEP * 50, START + STEP * 53, "1m",
                                             warmup_candles=49)
        with self.assertRaises(BacktestError):
            self.run_period(SMACrossoverStrategy(), short_period)

    def test_cli_loads_only_requested_range_and_preserves_databases(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            cache = HistoricalCache(history_path)
            cache.save(list(self.data))
            paper_path.write_bytes(b"paper-ledger-sentinel")
            before_history, before_paper = history_path.read_bytes(), paper_path.read_bytes()
            start, end = START + STEP * 50, START + STEP * 53
            output = StringIO()
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(history_path),
                                        "DATABASE_PATH": str(paper_path),
                                        "BYBIT_API_KEY": "unused", "BYBIT_API_SECRET": "unused"}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                    with redirect_stdout(output):
                        code = backtest_main(["--symbol", "BTCUSDT", "--interval", "1m",
                                              "--strategy", "sma-cross", "--start", start.isoformat(),
                                              "--end", end.isoformat()])
            self.assertEqual(code, 0, output.getvalue())
            for phrase in ("Evaluation period:", "Warm-up period:", "Evaluation candles: 3",
                           "Warm-up candles: 50", "candles: 3", "Matched B&H return:"):
                self.assertIn(phrase, output.getvalue())
            matched = run_buy_and_hold(self.data[50:53], symbol="BTCUSDT", interval="1m",
                                       config=BacktestConfig(), allocation_pct=D("25"))
            self.assertIn(f"  Final equity: {matched.final_equity} USDT", output.getvalue())
            self.assertEqual(matched.ledger[0].execution_candle_timestamp_utc,
                             START + STEP * 51)
            self.assertEqual(history_path.read_bytes(), before_history)
            self.assertEqual(paper_path.read_bytes(), before_paper)

    def test_cli_missing_cache_missing_ranges_and_gaps_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            args = ["--symbol", "BTCUSDT", "--interval", "1m", "--strategy", "sma-cross",
                    "--start", (START + STEP * 50).isoformat(),
                    "--end", (START + STEP * 53).isoformat()]
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(path)}):
                with redirect_stdout(StringIO()):
                    self.assertEqual(backtest_main(args), 1)
                self.assertFalse(path.exists())
                HistoricalCache(path).save(list(self.data))
                with redirect_stdout(StringIO()):
                    self.assertEqual(backtest_main(args[:-1] + [(START + STEP * 121).isoformat()]), 1)
                for missing_index in (20, 51):
                    with sqlite3.connect(path) as db:
                        row = db.execute("SELECT * FROM candles WHERE open_time_ms=?",
                                         (int((START + STEP * missing_index).timestamp() * 1000),)).fetchone()
                        db.execute("DELETE FROM candles WHERE open_time_ms=?",
                                   (int((START + STEP * missing_index).timestamp() * 1000),))
                    with redirect_stdout(StringIO()):
                        self.assertEqual(backtest_main(args), 1)
                    with sqlite3.connect(path) as db:
                        db.execute("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)

    def test_compare_cli_reports_relation_and_counts_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            paper = Path(directory) / "paper.sqlite3"
            HistoricalCache(path).save(list(self.data))
            paper.write_bytes(b"paper-ledger-sentinel")
            before_cache, before_paper = path.read_bytes(), paper.read_bytes()
            output = StringIO()
            args = ["--first-start", START.isoformat(), "--first-end", (START + STEP * 10).isoformat(),
                    "--second-start", (START + STEP * 12).isoformat(),
                    "--second-end", (START + STEP * 20).isoformat(),
                    "--symbol", "BTCUSDT", "--interval", "1m"]
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(path),
                                        "DATABASE_PATH": str(paper)}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                    with redirect_stdout(output):
                        self.assertEqual(compare_main(args), 0)
            self.assertIn("Overlap: no", output.getvalue())
            self.assertIn("Gap: 0:02:00", output.getvalue())
            self.assertIn("First cached candles: 10/10", output.getvalue())
            self.assertIn("Second cached candles: 8/8", output.getvalue())
            self.assertEqual(path.read_bytes(), before_cache)
            self.assertEqual(paper.read_bytes(), before_paper)

    def test_compare_cli_without_cache_does_not_create_one(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.sqlite3"
            output = StringIO()
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(missing)}):
                with redirect_stdout(output):
                    self.assertEqual(compare_main([
                        "--first-start", START.isoformat(),
                        "--first-end", (START + STEP * 2).isoformat(),
                        "--second-start", (START + STEP * 2).isoformat(),
                        "--second-end", (START + STEP * 4).isoformat(),
                        "--interval", "1m",
                    ]), 0)
            self.assertIn("Overlap: no", output.getvalue())
            self.assertIn("Cached candle counts: unavailable", output.getvalue())
            self.assertFalse(missing.exists())

    def test_no_network_dependency_and_no_bybit_order_methods(self):
        probe = subprocess.run([sys.executable, "-c",
                                "import sys; import ai_crypto_trader.backtesting.periods; "
                                "import ai_crypto_trader.diagnostics.backtest_periods; "
                                "assert 'ai_crypto_trader.bybit.client' not in sys.modules"],
                               capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        client = BybitReadOnlyClient(Settings())
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(client, name))


if __name__ == "__main__":
    unittest.main()
