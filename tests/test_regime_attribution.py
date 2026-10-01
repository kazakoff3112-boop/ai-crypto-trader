"""Read-only attribution of existing 15m fills to last closed 1h regimes."""

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

from ai_crypto_trader.backtesting import BacktestConfig, BacktestEngine, Signal
from ai_crypto_trader.backtesting.analytics import analyze_backtest
from ai_crypto_trader.backtesting.regime_attribution import (
    RegimeAttributionUnavailable, attribute_backtest_by_regime,
)
from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.backtest import main
from ai_crypto_trader.market_data.candle import Candle
from ai_crypto_trader.market_data.market_regime import MarketRegime, MarketRegimeReport, RegimePoint
from ai_crypto_trader.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from ai_crypto_trader.strategies.donchian_breakout import DonchianBreakoutStrategy
from ai_crypto_trader.strategies.sma_cross import SMACrossoverStrategy

D = Decimal
START = datetime(2024, 1, 10, 10, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
QUARTER_HOUR = timedelta(minutes=15)


def strategy_candles():
    execution_opens = {3: "100", 7: "120", 11: "100", 15: "80",
                       19: "100", 23: "110", 27: "100"}
    return tuple(Candle(
        "BTCUSDT", "15m", START + index * QUARTER_HOUR,
        D(execution_opens.get(index, "100")),
        max(D(execution_opens.get(index, "100")), D("100")),
        min(D(execution_opens.get(index, "100")), D("100")),
        D("100"), D("1"), D("1"),
    ) for index in range(29))


def regime_report():
    regimes = (MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS,
               MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS,
               MarketRegime.BULL, MarketRegime.BULL)
    points = tuple(RegimePoint(START + index * HOUR, "BTCUSDT", "1h", D("100"),
                               D("100"), D("100"), D("100"), regime)
                   for index, regime in enumerate(regimes))
    return MarketRegimeReport(START - HOUR, START + 7 * HOUR, points, (), (), 0)


def constant_regime_report(hours=27):
    points = tuple(RegimePoint(START + index * HOUR, "BTCUSDT", "1h", D("100"),
                               D("100"), D("100"), D("100"), MarketRegime.BULL)
                   for index in range(hours))
    return MarketRegimeReport(START - HOUR, START + (hours - 1) * HOUR,
                              points, (), (), 0)


class ScheduledStrategy:
    actions = {3: Signal("BUY", quantity=D("1")), 7: Signal("SELL", quantity=D("1")),
               11: Signal("BUY", quantity=D("1")), 15: Signal("SELL", quantity=D("1")),
               19: Signal("BUY", quantity=D("1")), 23: Signal("SELL", quantity=D("1")),
               27: Signal("BUY", quantity=D("1"))}

    def on_candle(self, history, portfolio):
        return self.actions.get(len(history), Signal())


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.candles = strategy_candles()
        self.report = regime_report()
        self.result = BacktestEngine().run(
            self.candles, ScheduledStrategy(), symbol="BTCUSDT", interval="15m",
            as_of_utc=START + timedelta(days=2), config=BacktestConfig(),
        )

    def attribution(self, report=None, result=None):
        return attribute_backtest_by_regime(self.result if result is None else result,
                                            self.report if report is None else report,
                                            interval="15m")

    def test_four_regimes_per_trade_use_signal_close_not_candle_open(self):
        attribution = self.attribution()
        first, second, third = attribution.completed_trades
        self.assertEqual((first.entry_signal_regime, first.entry_execution_regime,
                          first.exit_signal_regime, first.exit_execution_regime),
                         (MarketRegime.BULL, MarketRegime.BULL,
                          MarketRegime.BEAR, MarketRegime.BEAR))
        self.assertEqual(second.entry_signal_regime, MarketRegime.SIDEWAYS)
        self.assertEqual(second.exit_signal_regime, MarketRegime.BULL)
        self.assertEqual(third.entry_signal_regime, MarketRegime.BEAR)
        self.assertEqual(third.exit_signal_regime, MarketRegime.SIDEWAYS)
        self.assertEqual(first.entry_signal_time_utc, START + timedelta(minutes=45))
        self.assertEqual(first.entry_execution_time_utc, first.entry_signal_time_utc)
        self.assertEqual(first.exit_signal_time_utc, START + timedelta(hours=1, minutes=45))
        self.assertEqual(first.trade.entry_timestamp_utc, first.entry_execution_time_utc)
        self.assertEqual(self.report.regime_at(START + timedelta(minutes=45)), MarketRegime.BULL)
        self.assertEqual(self.report.regime_at(START + HOUR), MarketRegime.BEAR)

    def test_entry_regime_stats_shares_and_costs_reconcile(self):
        attribution = self.attribution()
        quality = analyze_backtest(self.result)
        bull = attribution.group_for(MarketRegime.BULL)
        bear = attribution.group_for(MarketRegime.BEAR)
        sideways = attribution.group_for(MarketRegime.SIDEWAYS)
        self.assertEqual([item.completed_trades for item in attribution.groups], [1, 1, 1])
        self.assertEqual((bull.winning_trades, bear.winning_trades, sideways.losing_trades), (1, 1, 1))
        self.assertEqual((bull.win_rate_percent, bear.win_rate_percent,
                          sideways.win_rate_percent), (D("100"), D("100"), D("0")))
        self.assertAlmostEqual(sum((item.entry_share_percent for item in attribution.groups), D("0")),
                               D("100"), delta=D("1e-20"))
        self.assertEqual(bull.total_net_pnl, attribution.completed_trades[0].trade.net_pnl_after_costs)
        self.assertEqual(bear.total_net_pnl, attribution.completed_trades[2].trade.net_pnl_after_costs)
        self.assertEqual(sideways.total_net_pnl, attribution.completed_trades[1].trade.net_pnl_after_costs)
        for item in attribution.groups:
            self.assertAlmostEqual(item.pnl_before_costs - item.total_costs,
                                   item.total_net_pnl, delta=D("1e-20"))
            self.assertEqual(item.total_costs, item.commissions + item.slippage)
            self.assertIsNotNone(item.average_holding_time)
            self.assertGreater(item.total_turnover, D("0"))
            self.assertEqual(item.average_net_pnl, item.median_net_pnl)
        self.assertAlmostEqual(sum((item.total_net_pnl for item in attribution.groups), D("0")),
                               sum((trade.net_pnl_after_costs for trade in quality.completed_trades), D("0")),
                               delta=D("1e-20"))
        self.assertEqual(bull.gross_loss, D("0"))
        self.assertIsNone(bull.profit_factor)
        self.assertGreater(sideways.gross_loss, D("0"))
        self.assertEqual(sideways.profit_factor, D("0"))

    def test_matrix_contains_only_completed_trade_pnl(self):
        attribution = self.attribution()
        expected = ((MarketRegime.BULL, MarketRegime.BEAR),
                    (MarketRegime.SIDEWAYS, MarketRegime.BULL),
                    (MarketRegime.BEAR, MarketRegime.SIDEWAYS))
        self.assertEqual(len(attribution.transitions), 9)
        for index, (entry, exit_) in enumerate(expected):
            cell = attribution.transition_for(entry, exit_)
            self.assertEqual(cell.trade_count, 1)
            self.assertEqual(cell.total_net_pnl,
                             attribution.completed_trades[index].trade.net_pnl_after_costs)
        self.assertEqual(sum(item.trade_count for item in attribution.transitions), 3)
        self.assertAlmostEqual(sum((item.total_net_pnl for item in attribution.transitions), D("0")),
                               attribution.total_completed_net_pnl, delta=D("1e-20"))

    def test_multiple_trades_in_one_regime_use_median_and_profit_factor(self):
        all_bull = replace(self.report, points=tuple(
            replace(point, regime=MarketRegime.BULL) for point in self.report.points
        ))
        attribution = self.attribution(report=all_bull)
        bull = attribution.group_for(MarketRegime.BULL)
        values = sorted(item.trade.net_pnl_after_costs for item in attribution.completed_trades)
        self.assertEqual((bull.completed_trades, bull.winning_trades, bull.losing_trades), (3, 2, 1))
        self.assertEqual(bull.entry_share_percent, D("100"))
        self.assertEqual(bull.median_net_pnl, values[1])
        self.assertEqual(bull.average_net_pnl, sum(values, D("0")) / D("3"))
        self.assertEqual(bull.profit_factor, bull.gross_profit / bull.gross_loss)
        self.assertEqual(bull.average_winner, bull.gross_profit / D("2"))
        self.assertEqual(bull.average_loser, -bull.gross_loss)

    def test_open_position_shown_separately_and_excluded_from_win_rate(self):
        attribution = self.attribution()
        self.assertIsNotNone(attribution.open_position)
        self.assertEqual(attribution.open_position.entry_signal_regime, MarketRegime.BULL)
        self.assertEqual(attribution.open_position.entry_execution_regime, MarketRegime.BULL)
        self.assertEqual(sum(group.completed_trades for group in attribution.groups), 3)
        self.assertEqual(self.result.buy_operations, 4)
        self.assertEqual(self.result.sell_operations, 3)
        self.assertNotEqual(sum((group.commissions for group in attribution.groups), D("0")),
                            self.result.total_commissions)

    def test_future_hourly_points_cannot_change_past_attribution(self):
        original = self.attribution()
        changed = list(self.report.points)
        changed[2] = replace(changed[2], regime=MarketRegime.BEAR)
        future_report = replace(self.report, points=tuple(changed))
        later = self.attribution(report=future_report)
        self.assertEqual(original.completed_trades[0], later.completed_trades[0])
        self.assertNotEqual(original.completed_trades[1].entry_signal_regime,
                            later.completed_trades[1].entry_signal_regime)
        self.assertEqual(original.open_position, later.open_position)

    def test_missing_hourly_regime_or_inconsistent_signal_time_fails_closed(self):
        missing = replace(self.report, points=self.report.points[1:])
        with self.assertRaises(RegimeAttributionUnavailable):
            self.attribution(report=missing)
        shifted_fill = replace(self.result.ledger[0],
                               signal_candle_timestamp_utc=self.result.ledger[0].signal_candle_timestamp_utc
                               - QUARTER_HOUR)
        bad_result = replace(self.result, ledger=(shifted_fill, *self.result.ledger[1:]))
        with self.assertRaises(RegimeAttributionUnavailable):
            self.attribution(result=bad_result)

    def test_attribution_does_not_change_original_strategy_results(self):
        before = self.result
        self.attribution()
        self.assertEqual(self.result, before)
        self.assertEqual((self.result.final_equity, self.result.total_pnl,
                          self.result.buy_operations, self.result.sell_operations,
                          self.result.total_commissions, self.result.total_slippage,
                          self.result.max_drawdown_usdt),
                         (before.final_equity, before.total_pnl, before.buy_operations,
                          before.sell_operations, before.total_commissions,
                          before.total_slippage, before.max_drawdown_usdt))

    def test_sma_donchian_and_bollinger_results_remain_unchanged(self):
        sma_closes = ("10", "10", "10", "12", "12", "8", "8")
        sma_data = tuple(Candle("BTCUSDT", "15m", START + index * QUARTER_HOUR,
                                D(close), D(close), D(close), D(close), D("1"), D("1"))
                         for index, close in enumerate(sma_closes))
        donchian = [Candle("BTCUSDT", "15m", START + index * QUARTER_HOUR,
                           D("10"), D("10"), D("9"), D("10"), D("1"), D("1"))
                    for index in range(100)]
        donchian[96] = replace(donchian[96], high=D("11"), close=D("11"))
        donchian[97] = replace(donchian[97], open=D("20"), high=D("20"),
                               low=D("19"), close=D("20"))
        donchian[98] = replace(donchian[98], low=D("8"), close=D("8"))
        donchian[99] = replace(donchian[99], open=D("7"), high=D("7"),
                               low=D("6"), close=D("7"))
        bollinger = [Candle("BTCUSDT", "15m", START + index * QUARTER_HOUR,
                            D("100"), D("100"), D("100"), D("100"), D("1"), D("1"))
                     for index in range(22)]
        bollinger[19] = replace(bollinger[19], low=D("70"), close=D("70"))
        bollinger[20] = replace(bollinger[20], open=D("80"), low=D("80"))
        bollinger[21] = replace(bollinger[21], open=D("120"), high=D("120"), close=D("120"))
        scenarios = (
            (SMACrossoverStrategy(2, 3, D("25")), sma_data, D("916.1674157926031989000504743")),
            (DonchianBreakoutStrategy(), tuple(donchian), D("837.2378932911166794225264991")),
            (BollingerMeanReversionStrategy(), tuple(bollinger), None),
        )
        report = constant_regime_report()
        for strategy, data, pinned_equity in scenarios:
            with self.subTest(strategy=strategy):
                result = BacktestEngine().run(data, strategy, symbol="BTCUSDT", interval="15m",
                                              as_of_utc=START + timedelta(days=2), config=BacktestConfig())
                before = (result.final_equity, result.total_pnl, result.buy_operations,
                          result.sell_operations, result.total_commissions,
                          result.total_slippage, result.max_drawdown_usdt)
                attribution = attribute_backtest_by_regime(result, report, interval="15m")
                self.assertEqual(sum(group.completed_trades for group in attribution.groups), 1)
                self.assertEqual(before, (result.final_equity, result.total_pnl, result.buy_operations,
                                          result.sell_operations, result.total_commissions,
                                          result.total_slippage, result.max_drawdown_usdt))
                if pinned_equity is not None:
                    self.assertEqual(result.final_equity, pinned_equity)

    def test_cli_flag_uses_only_local_hourly_cache_and_is_optional(self):
        base = datetime(2024, 1, 1, 2, tzinfo=timezone.utc)
        hourly = [Candle("BTCUSDT", "1h", base + index * HOUR, D(100 + index),
                         D(100 + index), D(100 + index), D(100 + index), D("1"), D("1"))
                  for index in range(227)]
        closes = ("10", "10", "10", "12", "12", "8", "8")
        quarter = [Candle("BTCUSDT", "15m", START + index * QUARTER_HOUR, D(close),
                          D(close), D(close), D(close), D("1"), D("1"))
                   for index, close in enumerate(closes)]
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "history.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            cache = HistoricalCache(cache_path)
            cache.save(hourly)
            cache.save(quarter)
            paper_path.write_bytes(b"paper-ledger-sentinel")
            before_cache, before_paper = cache_path.read_bytes(), paper_path.read_bytes()
            args = ["--symbol", "BTCUSDT", "--interval", "15m", "--strategy", "sma-cross",
                    "--fast", "2", "--slow", "3", "--position-size-pct", "25"]
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(cache_path),
                                        "DATABASE_PATH": str(paper_path),
                                        "BYBIT_API_KEY": "unused", "BYBIT_API_SECRET": "unused"}):
                explicit_period = ["--start", quarter[3].open_time_utc.isoformat(),
                                   "--end", quarter[-1].close_time_utc.isoformat(),
                                   "--regime-analysis"]
                for flags in ([], ["--regime-analysis"], explicit_period):
                    output = StringIO()
                    with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                        with redirect_stdout(output):
                            code = main([*args, *flags])
                    self.assertEqual(code, 0, output.getvalue())
                    self.assertEqual("REGIME TRADE ATTRIBUTION" in output.getvalue(), bool(flags))
                    if flags:
                        for phrase in ("ENTRY REGIME: BULL", "ENTRY REGIME: BEAR",
                                       "ENTRY REGIME: SIDEWAYS", "ENTRY -> EXIT REGIME MATRIX",
                                       "Trading costs:"):
                            self.assertIn(phrase, output.getvalue())
                    if flags == explicit_period:
                        self.assertIn("Warm-up candles: 3", output.getvalue())
            self.assertEqual(cache_path.read_bytes(), before_cache)
            self.assertEqual(paper_path.read_bytes(), before_paper)

    def test_cli_missing_hourly_warmup_fails_without_writing(self):
        data = [Candle("BTCUSDT", "15m", START + index * QUARTER_HOUR,
                       D("10"), D("10"), D("10"), D("10"), D("1"), D("1"))
                for index in range(7)]
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "history.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            HistoricalCache(cache_path).save(data)
            paper_path.write_bytes(b"paper-ledger-sentinel")
            before_cache, before_paper = cache_path.read_bytes(), paper_path.read_bytes()
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(cache_path),
                                        "DATABASE_PATH": str(paper_path)}):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--symbol", "BTCUSDT", "--interval", "15m",
                                           "--strategy", "sma-cross", "--fast", "2", "--slow", "3",
                                           "--regime-analysis"]), 1)
            self.assertEqual(cache_path.read_bytes(), before_cache)
            self.assertEqual(paper_path.read_bytes(), before_paper)

    def test_no_bybit_import_or_order_methods(self):
        probe = subprocess.run([sys.executable, "-c",
                                "import sys; import ai_crypto_trader.backtesting.regime_attribution; "
                                "assert 'ai_crypto_trader.bybit.client' not in sys.modules"],
                               capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        client = BybitReadOnlyClient(Settings())
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(client, name))


if __name__ == "__main__":
    unittest.main()
