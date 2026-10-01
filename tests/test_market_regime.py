"""Offline hourly regime classification, aggregation and time-availability tests."""

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

from ai_crypto_trader.market_data.market_regime import (
    MarketRegime, MarketRegimeAnalyzer, RegimePoint, RegimeUnavailable,
    REQUIRED_CANDLES, WARMUP_CANDLES,
)
from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.market_regime import main
from ai_crypto_trader.market_data.candle import Candle

D = Decimal
START = datetime(2024, 1, 1, 2, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def candles(closes):
    return tuple(Candle("BTCUSDT", "1h", START + index * HOUR, D(str(price)),
                        D(str(price)), D(str(price)), D(str(price)), D("1"), D("1"))
                 for index, price in enumerate(closes))


def rising(count=229):
    return candles([100 + index for index in range(count)])


def report_for(data):
    return MarketRegimeAnalyzer().analyze(
        data, evaluation_start=data[WARMUP_CANDLES].open_time_utc,
        evaluation_end=data[-1].close_time_utc,
        as_of_utc=START + timedelta(days=30),
    )


class MarketRegimeTests(unittest.TestCase):
    def setUp(self):
        self.data = rising()
        self.analyzer = MarketRegimeAnalyzer()
        self.clock = START + timedelta(days=30)

    def test_first_valid_point_needs_224_closed_candles(self):
        self.assertEqual((REQUIRED_CANDLES, WARMUP_CANDLES), (224, 223))
        self.assertIsNone(self.analyzer.classify_latest(self.data[:223], as_of_utc=self.clock))
        point = self.analyzer.classify_latest(self.data[:224], as_of_utc=self.clock)
        self.assertEqual(point.effective_time_utc, self.data[223].close_time_utc)
        with self.assertRaises(RegimeUnavailable):
            self.analyzer.analyze(self.data[1:], evaluation_start=self.data[223].open_time_utc,
                                  evaluation_end=self.data[-1].close_time_utc,
                                  as_of_utc=self.clock)

    def test_sma50_sma200_and_24_hour_previous_slope(self):
        point = self.analyzer.classify_latest(self.data[:224], as_of_utc=self.clock)
        self.assertEqual(point.sma50, sum((item.close for item in self.data[174:224]), D("0")) / D("50"))
        self.assertEqual(point.sma200, sum((item.close for item in self.data[24:224]), D("0")) / D("200"))
        self.assertEqual(point.previous_sma200,
                         sum((item.close for item in self.data[:200]), D("0")) / D("200"))
        self.assertEqual(point.sma200 - point.previous_sma200, D("24"))
        self.assertEqual(point.slope_direction, "RISING")

    def test_bull_bear_and_sideways_rules(self):
        bull = self.analyzer.classify_latest(self.data[:224], as_of_utc=self.clock)
        bear_data = candles([1000 - index for index in range(224)])
        bear = self.analyzer.classify_latest(bear_data, as_of_utc=self.clock)
        flat_data = candles([100] * 224)
        sideways = self.analyzer.classify_latest(flat_data, as_of_utc=self.clock)
        self.assertEqual(bull.regime, MarketRegime.BULL)
        self.assertTrue(bull.close > bull.sma200 and bull.sma50 > bull.sma200
                        and bull.sma200 > bull.previous_sma200)
        self.assertEqual(bear.regime, MarketRegime.BEAR)
        self.assertTrue(bear.close < bear.sma200 and bear.sma50 < bear.sma200
                        and bear.sma200 < bear.previous_sma200)
        self.assertEqual(sideways.regime, MarketRegime.SIDEWAYS)
        self.assertEqual((sideways.close, sideways.sma50, sideways.sma200,
                          sideways.previous_sma200), (D("100"),) * 4)
        self.assertEqual(sideways.slope_direction, "FLAT")

    def test_future_candles_do_not_change_regime_at_n(self):
        original = report_for(self.data)
        changed = list(self.data)
        changed[-1] = replace(changed[-1], open=D("10000"), high=D("10000"),
                              low=D("10000"), close=D("10000"))
        altered = report_for(tuple(changed))
        self.assertEqual(original.points[0], altered.points[0])
        self.assertEqual(original.points[-2], altered.points[-2])
        self.assertEqual(self.analyzer.classify_latest(self.data[:224], as_of_utc=self.clock),
                         self.analyzer.classify_latest(tuple(changed[:224]), as_of_utc=self.clock))

    def test_regime_at_15m_uses_last_fully_closed_hour_only(self):
        report = report_for(self.data)
        first, second = report.points[:2]
        self.assertEqual(first.effective_time_utc.hour, 10)
        self.assertEqual(second.effective_time_utc.hour, 11)
        at_1045 = first.effective_time_utc + timedelta(minutes=45)
        self.assertIsNone(report.regime_at(first.effective_time_utc - timedelta(minutes=15)))
        self.assertEqual(report.point_at(first.effective_time_utc), first)
        self.assertEqual(report.point_at(at_1045), first)
        self.assertEqual(report.point_at(second.effective_time_utc - timedelta(minutes=15)), first)
        self.assertEqual(report.point_at(second.effective_time_utc), second)
        self.assertEqual(report.point_at(report.evaluation_end + timedelta(minutes=45)), report.points[-1])
        self.assertEqual(report.regime_at(at_1045), first.regime)
        self.assertEqual(report.regime_at(second.effective_time_utc), second.regime)
        self.assertIsNone(report.regime_at(report.evaluation_end + HOUR))
        with self.assertRaises(RegimeUnavailable):
            report.regime_at(datetime(2024, 1, 10, 10, 45))

    def test_transitions_runs_percentages_and_next_hour_returns(self):
        values = [D("100")] * 223 + [D("100"), D("110"), D("121"), D("100"), D("90"), D("99")]
        data = candles(values)
        ordered_regimes = (MarketRegime.BULL, MarketRegime.BULL, MarketRegime.SIDEWAYS,
                           MarketRegime.BEAR, MarketRegime.BEAR, MarketRegime.SIDEWAYS)

        def chosen_point(history, index):
            offset = index - WARMUP_CANDLES
            candle = history[index]
            return RegimePoint(candle.close_time_utc, "BTCUSDT", "1h", candle.close,
                               D("100"), D("100"), D("100"), ordered_regimes[offset])

        with patch("ai_crypto_trader.market_data.market_regime._classify", side_effect=chosen_point):
            report = report_for(data)
        self.assertEqual(report.valid_classified_hours, 6)
        self.assertEqual([report.stats_for(regime).hours for regime in
                          (MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS)], [2, 2, 2])
        self.assertEqual(report.total_transitions, 3)
        self.assertEqual(report.transition_count(MarketRegime.BULL, MarketRegime.SIDEWAYS), 1)
        self.assertEqual(report.transition_count(MarketRegime.SIDEWAYS, MarketRegime.BEAR), 1)
        self.assertEqual(report.transition_count(MarketRegime.BEAR, MarketRegime.SIDEWAYS), 1)
        self.assertEqual(report.regime_at(report.points[2].effective_time_utc - timedelta(minutes=15)),
                         MarketRegime.BULL)
        self.assertEqual(report.regime_at(report.points[2].effective_time_utc),
                         MarketRegime.SIDEWAYS)
        self.assertEqual(sum(item.count for item in report.transitions), report.total_transitions)
        self.assertEqual(report.stats_for(MarketRegime.BULL).longest_run_hours, 2)
        self.assertEqual(report.stats_for(MarketRegime.BEAR).longest_run_hours, 2)
        self.assertEqual(report.stats_for(MarketRegime.SIDEWAYS).longest_run_hours, 1)
        self.assertAlmostEqual(sum((item.percentage for item in report.stats), D("0")),
                               D("100"), delta=D("1e-20"))
        self.assertEqual(report.stats_for(MarketRegime.BULL).average_next_hour_return, D("0.1"))
        self.assertEqual(report.stats_for(MarketRegime.BEAR).average_next_hour_return, D("0"))
        self.assertEqual(report.stats_for(MarketRegime.SIDEWAYS).average_next_hour_return,
                         D("100") / D("121") - D("1"))
        self.assertEqual([item.next_hour_sample_count for item in report.stats], [2, 2, 1])
        self.assertEqual(sum(item.next_hour_sample_count for item in report.stats), 5)

    def test_last_candle_is_excluded_even_if_only_one_evaluation_hour(self):
        report = report_for(self.data[:224])
        self.assertEqual(report.valid_classified_hours, 1)
        self.assertEqual(sum(item.next_hour_sample_count for item in report.stats), 0)
        self.assertTrue(all(item.average_next_hour_return is None for item in report.stats))

    def test_invalid_data_time_symbol_interval_and_period_fail_closed(self):
        variants = (
            [*self.data[:100], self.data[99], *self.data[101:]],
            [*self.data[:100], self.data[101], *self.data[102:]],
            [replace(self.data[0], symbol="ETHUSDT"), *self.data[1:]],
            [replace(self.data[0], interval="15m"), *self.data[1:]],
            [replace(self.data[0], high=D("0")), *self.data[1:]],
        )
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(RegimeUnavailable):
                report_for(variant)
        with self.assertRaises(RegimeUnavailable):
            self.analyzer.analyze(self.data, evaluation_start=self.data[223].open_time_utc,
                                  evaluation_end=self.data[223].open_time_utc, as_of_utc=self.clock)
        with self.assertRaises(RegimeUnavailable):
            self.analyzer.analyze(self.data, evaluation_start=datetime(2024, 1, 10, 9),
                                  evaluation_end=self.data[-1].close_time_utc, as_of_utc=self.clock)
        with self.assertRaises(RegimeUnavailable):
            self.analyzer.classify_latest(self.data, as_of_utc=self.data[-1].open_time_utc)
        with self.assertRaises(RegimeUnavailable):
            self.analyzer.classify_latest(candles([D("1e999999")] * 224),
                                          as_of_utc=self.clock)

    def test_cli_reads_cache_only_and_preserves_paper_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "history.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            HistoricalCache(cache_path).save(list(self.data))
            paper_path.write_bytes(b"paper-ledger-sentinel")
            before_cache, before_paper = cache_path.read_bytes(), paper_path.read_bytes()
            output = StringIO()
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(cache_path),
                                        "DATABASE_PATH": str(paper_path),
                                        "BYBIT_API_KEY": "unused", "BYBIT_API_SECRET": "unused"}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                    with redirect_stdout(output):
                        code = main(["--symbol", "BTCUSDT", "--interval", "1h",
                                     "--start", self.data[223].open_time_utc.isoformat(),
                                     "--end", self.data[-1].close_time_utc.isoformat(),
                                     "--tail", "2"])
            self.assertEqual(code, 0, output.getvalue())
            for phrase in ("MARKET REGIME REPORT", "Valid classified hours: 6", "BULL:",
                           "BEAR:", "SIDEWAYS:", "Transitions:", "BULL -> BEAR:",
                           "First valid regime timestamp:", "Longest continuous BULL run:",
                           "NEXT-HOUR BTC RETURNS", "LAST REGIME POINTS", "Slope | Regime"):
                self.assertIn(phrase, output.getvalue())
            self.assertEqual(cache_path.read_bytes(), before_cache)
            self.assertEqual(paper_path.read_bytes(), before_paper)
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(cache_path),
                                        "DATABASE_PATH": str(paper_path)}):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--symbol", "BTCUSDT", "--interval", "1h",
                                           "--start", self.data[222].open_time_utc.isoformat(),
                                           "--end", self.data[-1].close_time_utc.isoformat()]), 1)
            self.assertEqual(cache_path.read_bytes(), before_cache)
            self.assertEqual(paper_path.read_bytes(), before_paper)

    def test_no_network_import_or_bybit_order_methods(self):
        probe = subprocess.run([sys.executable, "-c",
                                "import sys; import ai_crypto_trader.market_data.market_regime; "
                                "import ai_crypto_trader.diagnostics.market_regime; "
                                "assert 'ai_crypto_trader.bybit.client' not in sys.modules"],
                               capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        client = BybitReadOnlyClient(Settings())
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(client, name))


if __name__ == "__main__":
    unittest.main()
