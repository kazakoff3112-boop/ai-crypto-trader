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
from ai_crypto_trader.backtesting.analytics import analyze_backtest
from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.backtest import main
from ai_crypto_trader.market_data.candle import Candle
from ai_crypto_trader.strategies.donchian_breakout import DonchianBreakoutStrategy

D = Decimal
START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def base_candles():
    rows = [Candle("BTCUSDT", "1m", START + timedelta(minutes=index),
                   D("10"), D("10"), D("9"), D("10"), D("1"), D("1"))
            for index in range(100)]
    rows[96] = replace(rows[96], close=D("11"), high=D("11"))
    rows[97] = replace(rows[97], open=D("20"), high=D("20"), low=D("19"), close=D("20"))
    rows[98] = replace(rows[98], close=D("8"), low=D("8"))
    rows[99] = replace(rows[99], open=D("7"), high=D("7"), low=D("6"), close=D("7"))
    return tuple(rows)


def view(quantity="0", cash="1000", equity="1000"):
    return PortfolioSnapshot("BTCUSDT", D(cash), D(quantity), D("0"), D("0"), D(equity))


class DonchianTests(unittest.TestCase):
    def setUp(self):
        self.data = base_candles()
        self.strategy = DonchianBreakoutStrategy()

    def run_engine(self, data=None, strategy=None, config=None):
        return BacktestEngine().run(
            self.data if data is None else data,
            self.strategy if strategy is None else strategy,
            symbol="BTCUSDT", interval="1m",
            config=config or BacktestConfig(D("0"), D("0")),
            as_of_utc=START + timedelta(days=1),
        )

    def test_defaults_parameters_and_buy_above_previous_96_high(self):
        self.assertEqual((self.strategy.entry_lookback, self.strategy.exit_lookback,
                          self.strategy.position_size_pct), (96, 48, D("25")))
        self.assertEqual(self.strategy.minimum_candles, 98)
        self.assertEqual(dict(self.strategy.parameters), {
            "entry_lookback": "96", "exit_lookback": "48", "position_size_pct": "25",
        })
        self.assertEqual(self.strategy.on_candle(self.data[:96], view()), Signal())
        self.assertEqual(self.strategy.on_candle(self.data[:97], view()),
                         Signal("BUY", budget_usdt=D("250")))

    def test_strict_buy_no_breakout_and_current_high_excluded(self):
        equal = replace(self.data[96], high=D("10"), close=D("10"))
        self.assertEqual(self.strategy.on_candle((*self.data[:96], equal), view()), Signal())
        below = replace(self.data[96], high=D("10"), close=D("9"))
        self.assertEqual(self.strategy.on_candle((*self.data[:96], below), view()), Signal())
        current_spike = replace(self.data[96], high=D("100"), close=D("11"))
        self.assertEqual(self.strategy.on_candle((*self.data[:96], current_spike), view()),
                         Signal("BUY", budget_usdt=D("250")))

    def test_sell_below_previous_48_low_and_no_sell_at_equality(self):
        self.assertEqual(self.strategy.on_candle(self.data[:99], view(quantity="1", cash="750")),
                         Signal("SELL", quantity=D("1")))
        equal = replace(self.data[98], low=D("9"), close=D("9"))
        self.assertEqual(self.strategy.on_candle((*self.data[:98], equal),
                                                  view(quantity="1", cash="750")), Signal())
        current_low_spike = replace(self.data[98], low=D("1"), close=D("8"))
        self.assertEqual(self.strategy.on_candle((*self.data[:98], current_low_spike),
                                                  view(quantity="1", cash="750")),
                         Signal("SELL", quantity=D("1")))

    def test_no_repeat_buy_or_sell_without_position(self):
        self.assertEqual(self.strategy.on_candle(self.data[:97], view(quantity="1")), Signal())
        self.assertEqual(self.strategy.on_candle(self.data[:99], view()), Signal())

    def test_future_candle_does_not_change_signal(self):
        changed = list(self.data)
        changed[97] = replace(changed[97], high=D("1000"), close=D("1000"))
        self.assertEqual(self.strategy.on_candle(self.data[:97], view()),
                         self.strategy.on_candle(tuple(changed[:97]), view()))

        class RecordingDonchian:
            minimum_candles = 98

            def __init__(self):
                self.seen = []

            def on_candle(self, history, portfolio):
                signal = DonchianBreakoutStrategy().on_candle(history, portfolio)
                self.seen.append((len(history), signal))
                return signal

        original_strategy, changed_strategy = RecordingDonchian(), RecordingDonchian()
        self.run_engine(strategy=original_strategy)
        self.run_engine(data=changed, strategy=changed_strategy)
        self.assertEqual(original_strategy.seen[96], changed_strategy.seen[96])
        self.assertEqual(original_strategy.seen[96][1], Signal("BUY", budget_usdt=D("250")))

    def test_next_open_execution_sizing_costs_long_only_and_analytics(self):
        config = BacktestConfig(D("0.001"), D("5"))
        result = self.run_engine(config=config)
        self.assertEqual(result.strategy_name, "Donchian Breakout")
        self.assertEqual(dict(result.strategy_parameters), dict(self.strategy.parameters))
        self.assertEqual([trade.side for trade in result.ledger], ["BUY", "SELL"])
        buy, sell = result.ledger
        self.assertEqual(buy.signal_candle_timestamp_utc, self.data[96].open_time_utc)
        self.assertEqual(buy.execution_candle_timestamp_utc, self.data[97].open_time_utc)
        self.assertEqual(sell.signal_candle_timestamp_utc, self.data[98].open_time_utc)
        self.assertEqual(sell.execution_candle_timestamp_utc, self.data[99].open_time_utc)
        self.assertEqual(buy.execution_price, D("20") * (D("1") + D("5") / D("10000")))
        self.assertEqual(sell.execution_price, D("7") * (D("1") - D("5") / D("10000")))
        self.assertLessEqual(buy.gross_value + buy.commission, D("250"))
        self.assertEqual(sell.quantity, buy.quantity)
        self.assertEqual(result.final_quantity, D("0"))
        self.assertGreaterEqual(result.final_cash, D("0"))
        self.assertEqual(buy.commission, buy.gross_value * config.fee_rate)
        self.assertEqual(sell.commission, sell.gross_value * config.fee_rate)
        self.assertEqual(buy.slippage_cost, buy.quantity * (buy.execution_price - self.data[97].open))
        self.assertEqual(sell.slippage_cost, sell.quantity * (self.data[99].open - sell.execution_price))
        report = analyze_backtest(result)
        self.assertEqual(report.completed_trade_count, 1)
        self.assertEqual(report.total_commissions, result.total_commissions)
        self.assertEqual(report.total_slippage, result.total_slippage)
        self.assertEqual(report.total_traded_notional, buy.gross_value + sell.gross_value)

    def test_invalid_lookbacks_size_and_history_fail_closed(self):
        for entry, exit_, pct in ((0, 48, D("25")), (-1, 48, D("25")),
                                  (96, 0, D("25")), (96, -1, D("25")),
                                  (96, 48, D("0")), (96, 48, D("101")),
                                  (96, 48, D("NaN"))):
            with self.subTest(entry=entry, exit=exit_, pct=pct):
                with self.assertRaises(ValueError):
                    DonchianBreakoutStrategy(entry, exit_, pct)
        with self.assertRaises(BacktestError):
            self.run_engine(data=self.data[:97])
        variants = (
            [*self.data[:50], self.data[49], *self.data[51:]],
            [*self.data[:50], self.data[51], *self.data[52:]],
            [replace(self.data[0], symbol="ETHUSDT"), *self.data[1:]],
            [replace(self.data[0], interval="5m"), *self.data[1:]],
            [replace(self.data[0], high=D("0")), *self.data[1:]],
        )
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(BacktestError):
                    self.run_engine(data=variant)

    def test_engine_still_rejects_unknown_action_overspend_and_short(self):
        class RogueStrategy:
            minimum_candles = 98

            def __init__(self, signal):
                self.signal = signal

            def on_candle(self, history, portfolio):
                return self.signal if len(history) == 1 else Signal()

        for signal in (Signal("SHORT", quantity=D("1")),
                       Signal("BUY", quantity=D("1000")),
                       Signal("SELL", quantity=D("1"))):
            with self.subTest(signal=signal):
                with self.assertRaises(BacktestError):
                    self.run_engine(strategy=RogueStrategy(signal))

    def test_cli_reuses_trade_quality_offline_and_preserves_databases(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            HistoricalCache(history_path).save(list(self.data))
            paper_path.write_bytes(b"paper-ledger-sentinel")
            history_before, paper_before = history_path.read_bytes(), paper_path.read_bytes()
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(history_path),
                                        "DATABASE_PATH": str(paper_path),
                                        "BYBIT_API_KEY": "unused", "BYBIT_API_SECRET": "unused"}):
                for flags in (("--entry-lookback", "96", "--exit-lookback", "48",
                               "--position-size-pct", "25"), ()):
                    output = StringIO()
                    with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                        with redirect_stdout(output):
                            code = main(["--symbol", "BTCUSDT", "--interval", "1m",
                                         "--strategy", "donchian-breakout", *flags])
                    self.assertEqual(code, 0, output.getvalue())
                    for phrase in ("entry_lookback=96", "exit_lookback=48", "position_size_pct=25",
                                   "TRADE QUALITY", "Completed trades:", "Profit factor:",
                                   "Turnover:", "P&L before costs:", "P&L after costs:",
                                   "Buy & Hold return:", "Buy & Hold max drawdown:"):
                        self.assertIn(phrase, output.getvalue())
            self.assertEqual(history_path.read_bytes(), history_before)
            self.assertEqual(paper_path.read_bytes(), paper_before)

    def test_no_network_dependency_or_bybit_order_methods(self):
        probe = subprocess.run([sys.executable, "-c",
                                "import sys; import ai_crypto_trader.strategies.donchian_breakout; "
                                "assert 'ai_crypto_trader.bybit.client' not in sys.modules"],
                               capture_output=True, text=True)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        client = BybitReadOnlyClient(Settings())
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(client, name))


if __name__ == "__main__":
    unittest.main()
