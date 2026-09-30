from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
import tempfile
import unittest

from ai_crypto_trader.bybit import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.diagnostics.paper_trade import PriceChangedBeyondTolerance, _refresh_plan
from ai_crypto_trader.market_data import MarketDataUnavailable, SpotQuote
from ai_crypto_trader.portfolio import InsufficientAssets, InsufficientFunds, Portfolio, PortfolioError
from ai_crypto_trader.safety import SafetyViolation
from ai_crypto_trader.trading_engine import PaperExecutionConfig, PaperTradingEngine


NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


class FakeMarket:
    def __init__(self) -> None:
        self.quotes = {
            symbol: SpotQuote(symbol, Decimal("999999"), Decimal("99"), Decimal("101"), NOW)
            for symbol in ("BTCUSDT", "ETHUSDT")
        }
        self.calls: list[str] = []
        self.error = False

    def get_spot_quote(self, symbol: str, *, category: str = "spot") -> SpotQuote:
        self.calls.append(symbol)
        if self.error:
            raise MarketDataUnavailable("network unavailable")
        return self.quotes[symbol]


class PaperEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "paper.sqlite3"
        self.settings = Settings(paper_slippage_bps=Decimal("100"))
        self.portfolio = Portfolio.open(self.settings, database_path=str(self.path))
        self.market = FakeMarket()
        self.engine = PaperTradingEngine(
            self.market, self.portfolio, PaperExecutionConfig.from_settings(self.settings),
            clock=lambda: NOW,
        )

    def tearDown(self) -> None:
        self.portfolio.close()
        self.directory.cleanup()

    def state(self) -> tuple:
        return self.portfolio.cash, self.portfolio.btc, self.portfolio.eth, self.portfolio.ledger.entries

    def test_buy_uses_ask_and_worse_slippage_not_last(self) -> None:
        before = self.state()
        plan = self.engine.preview_buy("BTCUSDT", "100")
        self.assertEqual(self.state(), before)
        self.assertEqual(plan.execution_price, Decimal("102.01"))
        self.assertGreater(plan.execution_price, plan.ask)
        self.assertNotEqual(plan.execution_price, plan.last_price)
        self.assertEqual(plan.slippage_cost, plan.quantity * Decimal("1.01"))
        self.assertEqual(plan.commission, plan.gross_value * Decimal("0.001"))
        self.assertLessEqual(plan.gross_value + plan.commission, Decimal("100"))

    def test_sell_uses_bid_and_worse_slippage_not_last(self) -> None:
        self.engine.paper_buy("BTCUSDT", "100")
        plan = self.engine.preview_sell("BTCUSDT", "0.2")
        self.assertEqual(plan.execution_price, Decimal("98.01"))
        self.assertLess(plan.execution_price, plan.bid)
        self.assertNotEqual(plan.execution_price, plan.last_price)
        self.assertEqual(plan.slippage_cost, plan.quantity * Decimal("0.99"))

    def test_buy_and_sell_persist_and_restart_with_separate_costs(self) -> None:
        buy = self.engine.paper_buy("BTCUSDT", "100")
        self.assertEqual(buy.usdt_balance_after, self.portfolio.cash)
        self.assertEqual(buy.asset_quantity_after, self.portfolio.btc)
        self.assertEqual(buy.gross_value + buy.commission, Decimal("100"))
        self.assertEqual(self.portfolio.ledger.entries[0].market_ask, Decimal("101"))
        sell = self.engine.paper_sell("BTCUSDT", "0.2")
        self.assertEqual(sell.realized_pnl, self.portfolio.ledger.entries[1].realized_pnl)
        self.assertEqual(self.portfolio.total_slippage, buy.slippage_cost + sell.slippage_cost)
        expected = (self.portfolio.cash, self.portfolio.btc, self.portfolio.realized_pnl)
        self.portfolio.close()
        self.portfolio = Portfolio.open(self.settings, database_path=str(self.path))
        self.assertEqual((self.portfolio.cash, self.portfolio.btc, self.portfolio.realized_pnl), expected)
        self.assertEqual(self.portfolio.ledger.entries[0].market_bid, Decimal("99"))
        self.assertEqual(self.portfolio.ledger.entries[1].market_ask, Decimal("101"))

    def test_btc_and_eth_independent_and_demo_balance_unused(self) -> None:
        self.engine.paper_buy("BTCUSDT", "100")
        self.engine.paper_buy("ETHUSDT", "200")
        self.assertGreater(self.portfolio.btc, 0)
        self.assertGreater(self.portfolio.eth, 0)
        self.assertEqual(self.portfolio.cash, Decimal("700"))
        self.assertIsNone(self.engine.preview_sell("BTCUSDT", "0.1").equity)

    def test_insufficient_cash_and_assets_are_atomic(self) -> None:
        before = self.state()
        with self.assertRaises(InsufficientFunds):
            self.engine.paper_buy("BTCUSDT", "1001")
        with self.assertRaises(InsufficientAssets):
            self.engine.paper_sell("ETHUSDT", "1")
        self.assertEqual(self.state(), before)

    def test_invalid_quantity_symbol_category_and_config(self) -> None:
        before = self.state()
        for amount in ("0", "-1", "NaN"):
            with self.subTest(amount=amount), self.assertRaises(PortfolioError):
                self.engine.paper_buy("BTCUSDT", amount)
        with self.assertRaises(SafetyViolation):
            self.engine.paper_buy("SOLUSDT", "100")
        with self.assertRaises(SafetyViolation):
            self.engine.paper_buy("BTCUSDT", "100", category="linear")
        for bps in (Decimal("-1"), Decimal("10000"), Decimal("NaN")):
            with self.assertRaises(ValueError):
                PaperExecutionConfig(slippage_bps=bps)
        with self.assertRaises(ValueError):
            PaperExecutionConfig(fee_rate=Decimal("NaN"))
        self.assertEqual(self.state(), before)

    def test_stale_invalid_spread_and_market_error_are_atomic(self) -> None:
        before = self.state()
        self.market.quotes["BTCUSDT"] = replace(
            self.market.quotes["BTCUSDT"], timestamp_utc=NOW - timedelta(seconds=61)
        )
        with self.assertRaises(PortfolioError):
            self.engine.paper_buy("BTCUSDT", "100")
        self.market.quotes["BTCUSDT"] = replace(
            self.market.quotes["BTCUSDT"], timestamp_utc=NOW,
            ask=Decimal("98"),
        )
        with self.assertRaises(PortfolioError):
            self.engine.paper_buy("BTCUSDT", "100")
        self.market.error = True
        with self.assertRaises(MarketDataUnavailable):
            self.engine.paper_buy("BTCUSDT", "100")
        self.assertEqual(self.state(), before)

    def test_database_error_is_atomic(self) -> None:
        connection = sqlite3.connect(str(self.path))
        connection.execute(
            "CREATE TRIGGER test_fail BEFORE INSERT ON ledger_entries "
            "BEGIN SELECT RAISE(ABORT, 'storage failure'); END"
        )
        connection.commit()
        before = self.state()
        with self.assertRaises(sqlite3.IntegrityError):
            self.engine.paper_buy("BTCUSDT", "100")
        self.assertEqual(self.state(), before)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM ledger_entries").fetchone()[0], 0)
        connection.close()

    def test_sell_all_and_no_persistent_execution(self) -> None:
        self.engine.paper_buy("BTCUSDT", "100")
        result = self.engine.paper_sell("BTCUSDT", sell_all=True)
        self.assertEqual(result.asset_quantity_after, Decimal("0"))
        ephemeral = PaperTradingEngine(self.market, Portfolio(self.settings),
                                       PaperExecutionConfig.from_settings(self.settings), clock=lambda: NOW)
        with self.assertRaises(PortfolioError):
            ephemeral.paper_buy("BTCUSDT", "100")

    def test_no_bybit_order_methods(self) -> None:
        client = BybitReadOnlyClient(Settings())
        for name in ("create_order", "amend_order", "cancel_order", "place_order"):
            self.assertFalse(hasattr(client, name))

    def test_confirmation_delay_refreshes_expired_preview_before_execution(self) -> None:
        current_time = [NOW]

        class RefreshingMarket:
            def get_spot_quote(self, symbol: str, *, category: str = "spot") -> SpotQuote:
                return SpotQuote(
                    symbol,
                    Decimal("100"),
                    Decimal("99"),
                    Decimal("101"),
                    current_time[0],
                )

        engine = PaperTradingEngine(
            RefreshingMarket(),
            self.portfolio,
            PaperExecutionConfig.from_settings(self.settings),
            clock=lambda: current_time[0],
        )
        original = engine.preview_buy("BTCUSDT", "100")
        current_time[0] += timedelta(seconds=61)
        with self.assertRaisesRegex(PortfolioError, "stale"):
            engine.execute_preview(original)

        refreshed = _refresh_plan(
            engine,
            original,
            buy_usdt="100",
            maximum=Decimal("250"),
            price_tolerance_bps=Decimal("10"),
        )
        result = engine.execute_preview(refreshed)
        self.assertEqual(result.usdt_balance_after, Decimal("900"))
        self.assertEqual(len(self.portfolio.ledger), 1)

    def test_refreshed_price_beyond_tolerance_is_not_executed(self) -> None:
        ask = [Decimal("101")]

        class MovingMarket:
            def get_spot_quote(self, symbol: str, *, category: str = "spot") -> SpotQuote:
                return SpotQuote(symbol, ask[0], Decimal("99"), ask[0], NOW)

        engine = PaperTradingEngine(
            MovingMarket(), self.portfolio,
            PaperExecutionConfig.from_settings(self.settings), clock=lambda: NOW,
        )
        original = engine.preview_buy("BTCUSDT", "100")
        ask[0] = Decimal("102")
        with self.assertRaises(PriceChangedBeyondTolerance):
            _refresh_plan(
                engine,
                original,
                buy_usdt="100",
                maximum=Decimal("250"),
                price_tolerance_bps=Decimal("10"),
            )
        self.assertEqual(self.portfolio.cash, Decimal("1000"))
        self.assertEqual(len(self.portfolio.ledger), 0)


if __name__ == "__main__":
    unittest.main()
