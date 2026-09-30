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
from ai_crypto_trader.backtesting.analytics import BacktestAnalyticsError, analyze_backtest
from ai_crypto_trader.backtesting.benchmark import run_buy_and_hold
from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.backtest import main
from ai_crypto_trader.market_data.candle import Candle

D = Decimal
START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def candles(opens, closes=None):
    closes = opens if closes is None else closes
    return tuple(Candle("BTCUSDT", "1m", START + timedelta(minutes=index),
                        D(str(op)), max(D(str(op)), D(str(close))),
                        min(D(str(op)), D(str(close))), D(str(close)), D("1"), D("1"))
                 for index, (op, close) in enumerate(zip(opens, closes)))


class ScheduledStrategy:
    def __init__(self, schedule):
        self.schedule = schedule

    def on_candle(self, history, portfolio):
        return self.schedule.get(len(history), Signal())


def run(data, schedule, *, fee="0", slip="0"):
    return BacktestEngine().run(
        data, ScheduledStrategy(schedule), symbol="BTCUSDT", interval="1m",
        config=BacktestConfig(D(fee), D(slip)), as_of_utc=START + timedelta(days=1),
    )


class BacktestAnalyticsTests(unittest.TestCase):
    def test_one_winner_full_round_trip_and_exact_cost_identity(self):
        result = run(candles((100, 100, 120, 120)),
                     {1: Signal("BUY", D("1")), 2: Signal("SELL", D("1"))},
                     fee="0.01", slip="100")
        report = analyze_backtest(result)
        self.assertEqual(report.completed_trade_count, 1)
        self.assertEqual((report.winning_trades, report.losing_trades, report.breakeven_trades), (1, 0, 0))
        self.assertEqual(report.win_rate_percent, D("100"))
        self.assertEqual(report.loss_rate_percent, D("0"))
        trade = report.completed_trades[0]
        self.assertEqual((trade.entry_timestamp_utc, trade.exit_timestamp_utc),
                         (START + timedelta(minutes=1), START + timedelta(minutes=2)))
        self.assertEqual(trade.holding_time, timedelta(minutes=1))
        self.assertEqual((trade.entry_quantity, trade.entry_execution_price,
                          trade.exit_execution_price), (D("1"), D("101"), D("118.8")))
        self.assertEqual((trade.gross_entry_notional, trade.gross_exit_notional),
                         (D("101"), D("118.8")))
        self.assertEqual((trade.entry_commission, trade.exit_commission),
                         (D("1.01"), D("1.188")))
        self.assertEqual((trade.entry_slippage, trade.exit_slippage), (D("1"), D("1.2")))
        self.assertEqual(trade.gross_pnl_before_costs, D("20"))
        self.assertEqual(trade.total_trading_costs, D("4.398"))
        self.assertEqual(trade.net_pnl_after_costs, D("15.602"))
        self.assertEqual(trade.return_on_allocated_capital_percent,
                         D("15.602") / D("102.01") * D("100"))
        self.assertEqual(report.strategy_pnl_before_costs, D("20"))
        self.assertEqual(report.strategy_pnl_after_costs, D("15.602"))
        self.assertEqual(report.commission_drag, D("2.198"))
        self.assertEqual(report.slippage_drag, D("2.2"))
        self.assertEqual(report.combined_cost_drag, D("4.398"))
        self.assertEqual(report.strategy_pnl_before_costs - report.combined_cost_drag,
                         report.strategy_pnl_after_costs)
        self.assertEqual(report.return_after_costs_percent, result.total_return_percent)
        self.assertEqual(report.approximate_return_before_costs_percent, D("2"))
        self.assertEqual(report.total_traded_notional, D("219.8"))
        self.assertIsNone(report.profit_factor)

    def test_one_loser(self):
        report = analyze_backtest(run(candles((100, 100, 80, 80)),
                                      {1: Signal("BUY", D("1")), 2: Signal("SELL", D("1"))},
                                      fee="0.01", slip="100"))
        self.assertEqual((report.winning_trades, report.losing_trades), (0, 1))
        self.assertEqual(report.gross_profit, D("0"))
        self.assertEqual(report.gross_loss, D("23.602"))
        self.assertEqual(report.average_loser, D("-23.602"))
        self.assertEqual(report.largest_loser, D("-23.602"))
        self.assertEqual(report.completed_trades[0].gross_pnl_before_costs, D("-20"))
        self.assertEqual(report.completed_trades[0].net_pnl_after_costs, D("-23.602"))
        self.assertEqual(report.loss_rate_percent, D("100"))

    def test_multiple_trades_win_rate_median_profit_factor_turnover_durations(self):
        data = candles((100, 100, 120, 100, 80, 100, 100))
        schedule = {1: Signal("BUY", D("1")), 2: Signal("SELL", D("1")),
                    3: Signal("BUY", D("1")), 4: Signal("SELL", D("1"))}
        report = analyze_backtest(run(data, schedule))
        self.assertEqual(report.completed_trade_count, 2)
        self.assertEqual((report.winning_trades, report.losing_trades, report.breakeven_trades),
                         (1, 1, 0))
        self.assertEqual((report.win_rate_percent, report.loss_rate_percent), (D("50"), D("50")))
        self.assertEqual(report.average_net_pnl, D("0"))
        self.assertEqual(report.median_net_pnl, D("0"))
        self.assertEqual((report.average_winner, report.average_loser), (D("20"), D("-20")))
        self.assertEqual((report.largest_winner, report.largest_loser), (D("20"), D("-20")))
        self.assertEqual((report.gross_profit, report.gross_loss, report.profit_factor),
                         (D("20"), D("20"), D("1")))
        self.assertEqual(report.total_traded_notional, D("400"))
        self.assertEqual((report.average_holding_time, report.minimum_holding_time,
                          report.maximum_holding_time), (timedelta(minutes=1),) * 3)

    def test_open_position_excluded_from_completed_and_marked_to_market(self):
        report = analyze_backtest(run(candles((100, 100, 110)),
                                      {1: Signal("BUY", D("1"))},
                                      fee="0.01", slip="100"))
        self.assertEqual(report.completed_trade_count, 0)
        self.assertEqual(report.win_rate_percent, D("0"))
        self.assertIsNone(report.average_net_pnl)
        self.assertIsNone(report.profit_factor)
        self.assertIsNotNone(report.open_position)
        self.assertEqual(report.open_position.quantity, D("1"))
        self.assertEqual(report.open_position.unrealized_pnl, D("7.99"))
        self.assertEqual(report.open_position.gross_unrealized_pnl_before_costs, D("10"))
        self.assertEqual(report.strategy_pnl_before_costs, D("10"))
        self.assertEqual(report.total_trading_costs, D("2.01"))
        self.assertEqual(report.strategy_pnl_after_costs, D("7.99"))
        self.assertEqual(report.total_traded_notional, D("101"))

    def test_breakeven_and_empty_no_trade(self):
        flat = candles((100, 100, 100))
        breakeven = analyze_backtest(run(flat, {1: Signal("BUY", D("1")),
                                               2: Signal("SELL", D("1"))}))
        self.assertEqual(breakeven.breakeven_trades, 1)
        self.assertEqual(breakeven.win_rate_percent, D("0"))
        empty = analyze_backtest(run(flat, {}))
        self.assertEqual(empty.completed_trade_count, 0)
        self.assertEqual(empty.total_traded_notional, D("0"))
        self.assertEqual(empty.strategy_pnl_before_costs, D("0"))

    def test_completed_trade_plus_open_position_reconciles(self):
        data = candles((100, 100, 120, 110, 115))
        result = run(data, {1: Signal("BUY", D("1")), 2: Signal("SELL", D("1")),
                            3: Signal("BUY", D("1"))}, fee="0.001", slip="5")
        report = analyze_backtest(result)
        self.assertEqual(report.completed_trade_count, 1)
        self.assertIsNotNone(report.open_position)
        self.assertEqual(report.total_traded_notional,
                         sum((fill.gross_value for fill in result.ledger), D("0")))
        self.assertAlmostEqual(report.strategy_pnl_before_costs - report.total_trading_costs,
                               report.strategy_pnl_after_costs, places=18)

    def test_accounting_tampering_is_rejected(self):
        result = run(candles((100, 100, 120, 120)),
                     {1: Signal("BUY", D("1")), 2: Signal("SELL", D("1"))},
                     fee="0.01", slip="100")
        broken = (
            replace(result, total_pnl=result.total_pnl + D("1")),
            replace(result, total_commissions=D("0")),
            replace(result, final_equity=result.final_equity + D("1")),
            replace(result, realized_pnl=result.realized_pnl + D("1")),
            replace(result, ledger=(result.ledger[0],
                                    replace(result.ledger[1], slippage_cost=D("0")))),
        )
        for altered in broken:
            with self.subTest(altered=altered):
                with self.assertRaises(BacktestAnalyticsError):
                    analyze_backtest(altered)

    def test_partial_close_and_pyramiding_are_rejected(self):
        data = candles((100, 100, 100, 100))
        pyramided = run(data, {1: Signal("BUY", D("1")), 2: Signal("BUY", D("1"))})
        partial = run(data, {1: Signal("BUY", D("1")), 2: Signal("SELL", D("0.5"))})
        with self.assertRaises(BacktestAnalyticsError):
            analyze_backtest(pyramided)
        with self.assertRaises(BacktestAnalyticsError):
            analyze_backtest(partial)

    def test_many_decimal_round_trips_reconcile_with_bounded_tolerance(self):
        prices = ["100.123456789123456789" if index % 2 == 0 else
                  "101.987654321987654321" for index in range(401)]
        schedule = {index: Signal("BUY" if index % 2 else "SELL", D("0.001"))
                    for index in range(1, 401)}
        result = run(candles(prices), schedule, fee="0.001", slip="5")
        report = analyze_backtest(result)
        self.assertEqual(report.completed_trade_count, 200)
        self.assertIsNone(report.open_position)
        self.assertAlmostEqual(report.strategy_pnl_before_costs - report.total_trading_costs,
                               report.strategy_pnl_after_costs, places=18)

    def test_benchmark_drawdown_and_costs(self):
        result = run_buy_and_hold(candles((100, 100, 80)), symbol="BTCUSDT", interval="1m",
                                  config=BacktestConfig(D("0"), D("0")))
        report = analyze_backtest(result)
        self.assertEqual(result.max_drawdown_usdt, D("200"))
        self.assertEqual(result.max_drawdown_percent, D("20"))
        self.assertEqual(report.completed_trade_count, 0)
        self.assertEqual(report.strategy_pnl_after_costs, D("-200"))
        self.assertEqual(report.total_trading_costs, D("0"))

    def test_cli_report_is_offline_and_does_not_change_databases(self):
        data = candles((10, 10, 10, 12, 12, 8, 8))
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            HistoricalCache(history_path).save(list(data))
            paper_path.write_bytes(b"paper-ledger-sentinel")
            history_before, paper_before = history_path.read_bytes(), paper_path.read_bytes()
            output = StringIO()
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(history_path),
                                        "DATABASE_PATH": str(paper_path),
                                        "BYBIT_API_KEY": "unused", "BYBIT_API_SECRET": "unused"}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                    with redirect_stdout(output):
                        code = main(["--symbol", "BTCUSDT", "--interval", "1m",
                                     "--strategy", "sma-cross", "--fast", "2", "--slow", "3",
                                     "--position-size-pct", "25"])
            self.assertEqual(code, 0, output.getvalue())
            for phrase in ("TRADE QUALITY", "Completed trades:", "Wins:", "Losses:",
                           "Win rate:", "Profit factor:",
                           "P&L before costs:", "Trading costs:", "P&L after costs:",
                           "Buy & Hold final equity:", "Buy & Hold max drawdown:",
                           "Buy & Hold trading costs:"):
                self.assertIn(phrase, output.getvalue())
            self.assertEqual(history_path.read_bytes(), history_before)
            self.assertEqual(paper_path.read_bytes(), paper_before)

    def test_no_network_dependency_or_order_methods(self):
        probe = subprocess.run([sys.executable, "-c",
                                "import sys; import ai_crypto_trader.backtesting.analytics; "
                                "assert 'ai_crypto_trader.bybit.client' not in sys.modules"],
                               capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        client = BybitReadOnlyClient(Settings())
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(client, name))


if __name__ == "__main__":
    unittest.main()
