from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ai_crypto_trader.backtesting import BacktestConfig, BacktestEngine, BacktestError, Signal
from ai_crypto_trader.backtesting.test_strategies import (
    BuyOnceStrategy, BuyThenSellStrategy, NoTradeStrategy,
)
from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.backtest import main
from ai_crypto_trader.market_data.candle import Candle, HistoricalDataUnavailable
from ai_crypto_trader.safety.guard import SafetyViolation


START = datetime(2024, 1, 1, tzinfo=timezone.utc)
D = Decimal


def candles(opens=("100", "100", "120", "100"), closes=("100", "110", "120", "100"),
            *, symbol="BTCUSDT", interval="1m"):
    return tuple(Candle(symbol, interval, START + timedelta(minutes=i), D(op),
                        max(D(op), D(close)), min(D(op), D(close)), D(close),
                        D("1"), D("1")) for i, (op, close) in enumerate(zip(opens, closes)))


class StaticStrategy:
    def __init__(self, actions):
        self.actions = actions
        self.seen = []

    def on_candle(self, history, portfolio):
        self.seen.append((len(history), history[-1].open_time_utc, portfolio))
        return self.actions.get(len(history), Signal())


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.engine = BacktestEngine()
        self.data = candles()
        self.as_of = START + timedelta(days=1)

    def run_backtest(self, strategy=None, data=None, config=None, **kwargs):
        as_of = kwargs.pop("as_of_utc", self.as_of)
        return self.engine.run(self.data if data is None else data,
                               strategy or NoTradeStrategy(), symbol="BTCUSDT",
                               interval="1m", as_of_utc=as_of, config=config, **kwargs)

    def test_no_trade_starts_and_ends_at_1000(self):
        result = self.run_backtest()
        self.assertEqual(result, self.run_backtest())
        self.assertEqual(result.initial_capital, D("1000"))
        self.assertEqual(result.final_equity, D("1000"))
        self.assertEqual(result.total_pnl, D("0"))
        self.assertEqual(result.ledger_entries, 0)
        self.assertEqual(result.candles_processed, 4)
        self.assertEqual(len(result.equity_curve), 4)
        self.assertTrue(all(p.equity == D("1000") for p in result.equity_curve))
        self.assertEqual(self.run_backtest(initial_capital=D("250")).final_equity, D("250"))

    def test_signal_after_n_executes_next_open_and_no_future_passed(self):
        strategy = StaticStrategy({1: Signal("BUY", D("1"))})
        result = self.run_backtest(strategy, config=BacktestConfig(D("0"), D("0")))
        self.assertEqual([x[0] for x in strategy.seen], [1, 2, 3, 4])
        self.assertEqual(strategy.seen[0][2].quantity, D("0"))
        self.assertEqual(strategy.seen[1][2].quantity, D("1"))
        with self.assertRaises(FrozenInstanceError):
            strategy.seen[0][2].cash = D("0")
        trade = result.ledger[0]
        self.assertEqual(trade.signal_candle_timestamp_utc, START)
        self.assertEqual(trade.execution_candle_timestamp_utc, START + timedelta(minutes=1))
        self.assertEqual(trade.execution_price, D("100"))
        self.assertEqual(trade.portfolio_equity_after_trade, D("1000"))
        self.assertEqual(result.equity_curve[0].equity, D("1000"))
        self.assertEqual(result.equity_curve[1].equity, D("1010"))

    def test_buy_slippage_commission_once_and_open_position_mark_to_market(self):
        result = self.run_backtest(BuyOnceStrategy(D("1")),
                                   config=BacktestConfig(D("0.01"), D("100")))
        trade = result.ledger[0]
        self.assertEqual(trade.execution_price, D("101"))
        self.assertEqual(trade.gross_value, D("101"))
        self.assertEqual(trade.commission, D("1.01"))
        self.assertEqual(trade.slippage_cost, D("1"))
        self.assertEqual(result.total_commissions, D("1.01"))
        self.assertEqual(result.total_slippage, D("1"))
        self.assertEqual(result.final_equity, D("997.99"))
        self.assertEqual(result.unrealized_pnl, D("-2.01"))
        self.assertEqual(result.realized_pnl, D("0"))
        self.assertEqual(result.buy_operations, 1)
        self.assertEqual(result.sell_operations, 0)

    def test_sell_slippage_profitable_realized_and_no_double_count(self):
        result = self.run_backtest(BuyThenSellStrategy(D("1")),
                                   config=BacktestConfig(D("0.01"), D("100")))
        self.assertEqual(result.ledger[1].execution_candle_timestamp_utc,
                         START + timedelta(minutes=2))
        self.assertEqual(result.ledger[1].execution_price, D("118.8"))
        self.assertEqual(result.ledger[1].commission, D("1.188"))
        self.assertEqual(result.ledger[1].slippage_cost, D("1.20"))
        self.assertEqual(result.total_commissions, D("2.198"))
        self.assertEqual(result.total_slippage, D("2.20"))
        self.assertEqual(result.realized_pnl, D("15.602"))
        self.assertEqual(result.final_equity, D("1015.602"))
        self.assertEqual(result.total_pnl, result.realized_pnl + result.unrealized_pnl)
        self.assertEqual(result.ledger_entries, 2)

    def test_losing_scenario_and_drawdown(self):
        data = candles(("100", "100", "80", "90"), ("100", "120", "80", "90"))
        holding = self.run_backtest(BuyOnceStrategy(D("1")), data=data,
                                    config=BacktestConfig(D("0"), D("0")))
        self.assertEqual([p.equity for p in holding.equity_curve],
                         [D("1000"), D("1020"), D("980"), D("990")])
        self.assertEqual(holding.max_equity, D("1020"))
        self.assertEqual(holding.min_equity, D("980"))
        self.assertEqual(holding.max_drawdown_usdt, D("40"))
        self.assertEqual(holding.max_drawdown_percent, D("40") / D("1020") * D("100"))
        sold = self.run_backtest(BuyThenSellStrategy(D("1")), data=data,
                                 config=BacktestConfig(D("0"), D("0")))
        self.assertEqual(sold.realized_pnl, D("-20"))
        self.assertEqual(sold.final_equity, D("980"))

    def test_last_candle_signal_is_not_executed(self):
        strategy = StaticStrategy({4: Signal("BUY", D("1"))})
        result = self.run_backtest(strategy)
        self.assertEqual(result.ledger_entries, 0)

    def test_insufficient_cash_asset_and_short_fail_closed(self):
        for action, qty in (("BUY", "100"), ("SELL", "1")):
            with self.subTest(action=action):
                with self.assertRaises(BacktestError):
                    self.run_backtest(StaticStrategy({1: Signal(action, D(qty))}))
        self.assertEqual(self.run_backtest().final_equity, D("1000"))

    def test_unknown_action_bad_quantities_and_configuration(self):
        for signal in (Signal("SHORT", D("1")), Signal("BUY", D("0")),
                       Signal("BUY", D("-1")), Signal("BUY", D("NaN")),
                       Signal("BUY", 1.0), Signal("HOLD", D("1")), None):
            with self.subTest(signal=signal):
                with self.assertRaises(BacktestError):
                    self.run_backtest(StaticStrategy({1: signal}))
        for fee, slip in ((D("-1"), D("0")), (D("1"), D("0")),
                          (D("0"), D("-1")), (D("0"), D("10000")),
                          (D("NaN"), D("0"))):
            with self.assertRaises(BacktestError):
                BacktestConfig(fee, slip)
        for capital in (D("0"), D("-1"), D("NaN"), 1000.0):
            with self.assertRaises(BacktestError):
                self.run_backtest(initial_capital=capital)

    def test_candle_integrity_rejections(self):
        data = list(self.data)
        variants = (
            data[:1],
            [data[1], data[0], data[2], data[3]],
            [data[0], data[0], data[2], data[3]],
            [data[0], data[2], data[3]],
            [replace(data[0], symbol="ETHUSDT"), *data[1:]],
            [replace(data[0], interval="5m"), *data[1:]],
            [replace(data[0], high=D("0")), *data[1:]],
            [replace(data[0], volume=D("-1")), *data[1:]],
        )
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(BacktestError):
                    self.run_backtest(data=variant)
        with self.assertRaises(BacktestError):
            self.run_backtest(as_of_utc=START + timedelta(minutes=3, seconds=30))
        with self.assertRaises(SafetyViolation):
            self.engine.run(self.data, NoTradeStrategy(), symbol="SOLUSDT", interval="1m")
        with self.assertRaises(SafetyViolation):
            self.engine.run(self.data, NoTradeStrategy(), symbol="BTCUSDT", interval="1m", category="linear")

    def test_offline_cli_does_not_modify_cache_or_trade_database(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "history.sqlite3"
            trade_path = Path(temp) / "paper.sqlite3"
            cache = HistoricalCache(cache_path)
            cache.save(list(self.data))
            trade_path.write_bytes(b"paper-ledger-unchanged")
            cache_before = cache_path.read_bytes()
            paper_before = trade_path.read_bytes()
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(cache_path),
                                        "DATABASE_PATH": str(trade_path),
                                        "BYBIT_API_KEY": "not-used", "BYBIT_API_SECRET": "not-used"}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network attempt")):
                    self.assertEqual(main(["--symbol", "BTCUSDT", "--interval", "1m",
                                           "--strategy", "no-trade"]), 0)
            self.assertEqual(cache_path.read_bytes(), cache_before)
            self.assertEqual(trade_path.read_bytes(), paper_before)
            readonly = HistoricalCache.open_readonly(cache_path)
            with self.assertRaises(HistoricalDataUnavailable):
                readonly.save(list(self.data))

    def test_no_network_dependency_and_no_order_methods(self):
        script = ("import sys; import ai_crypto_trader.backtesting; "
                  "assert 'ai_crypto_trader.bybit.client' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        client = BybitReadOnlyClient(Settings())
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(client, name))


if __name__ == "__main__":
    unittest.main()
