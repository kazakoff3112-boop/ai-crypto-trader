from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from io import StringIO
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from ai_crypto_trader.analytics import AnalyticsUnavailable, build_portfolio_report, build_trade_history
from ai_crypto_trader.config import Settings
from ai_crypto_trader.diagnostics import paper_history, paper_report
from ai_crypto_trader.market_data import SpotQuote
from ai_crypto_trader.portfolio import Portfolio, PortfolioError


NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


class FakeMarket:
    def __init__(self, btc_bid: str = "110", eth_bid: str = "210") -> None:
        self.quotes = {
            "BTCUSDT": SpotQuote("BTCUSDT", Decimal("999"), Decimal(btc_bid), Decimal(btc_bid) + 1, NOW),
            "ETHUSDT": SpotQuote("ETHUSDT", Decimal("999"), Decimal(eth_bid), Decimal(eth_bid) + 1, NOW),
        }
        self.calls: list[str] = []

    def get_spot_quote(self, symbol: str, *, category: str = "spot") -> SpotQuote:
        self.calls.append(symbol)
        return self.quotes[symbol]


class AnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "analytics.sqlite3"
        self.settings = Settings(database_path=self.path)
        self.portfolio = Portfolio.open(self.settings)
        self.market = FakeMarket()

    def tearDown(self) -> None:
        self.portfolio.close()
        self.directory.cleanup()

    def report(self):
        return build_portfolio_report(self.portfolio, self.market, clock=lambda: NOW)

    def test_cash_only_needs_no_quote_and_has_zero_return(self) -> None:
        report = self.report()
        self.assertEqual(self.market.calls, [])
        self.assertEqual(report.initial_capital, Decimal("1000"))
        self.assertEqual(report.total_equity, Decimal("1000"))
        self.assertEqual(report.total_pnl, Decimal("0"))
        self.assertEqual(report.total_return_percent, Decimal("0"))
        self.assertIsNone(report.btc.current_bid)

    def test_open_btc_uses_bid_and_counts_costs_once(self) -> None:
        self.portfolio.buy(
            "BTCUSDT", "1", "101", market_bid="99", market_ask="100", slippage_cost="1"
        )
        report = self.report()
        self.assertEqual(self.market.calls, ["BTCUSDT"])
        self.assertEqual(report.btc.average_entry_price, Decimal("101.101"))
        self.assertEqual(report.btc.market_value, Decimal("110"))
        self.assertEqual(report.btc.unrealized_pnl, Decimal("8.899"))
        self.assertEqual(report.total_equity, Decimal("1008.899"))
        self.assertEqual(report.total_pnl, Decimal("8.899"))
        self.assertEqual(report.total_return_percent, Decimal("0.889900"))
        self.assertEqual(report.total_commissions, Decimal("0.101"))
        self.assertEqual(report.total_slippage, Decimal("1"))

    def test_long_fractional_quantity_reconciles_with_decimal_rounding(self) -> None:
        self.portfolio.buy(
            "BTCUSDT",
            Decimal("0.001181980596052103570477626817"),
            Decimal("84519.2385"),
            market_bid=Decimal("84476"),
            market_ask=Decimal("84477"),
            slippage_cost=Decimal("0.04992508740634677666161924031"),
        )
        self.market.quotes["BTCUSDT"] = SpotQuote(
            "BTCUSDT", Decimal("85000"), Decimal("85000"), Decimal("85001"), NOW
        )
        report = self.report()
        self.assertGreater(report.total_equity, Decimal("1000"))
        self.assertEqual(report.total_pnl, report.total_equity - report.initial_capital)

    def test_open_eth_uses_its_bid(self) -> None:
        self.portfolio.buy("ETHUSDT", "1", "200")
        report = self.report()
        self.assertEqual(self.market.calls, ["ETHUSDT"])
        self.assertEqual(report.eth.market_value, Decimal("210"))
        self.assertEqual(report.eth.unrealized_pnl, Decimal("9.800"))

    def test_both_assets_have_complete_equity(self) -> None:
        self.portfolio.buy("BTCUSDT", "1", "100")
        self.portfolio.buy("ETHUSDT", "1", "200")
        report = self.report()
        self.assertEqual(self.market.calls, ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(report.total_market_value, Decimal("320"))
        self.assertEqual(report.total_equity, Decimal("1019.700"))
        self.assertEqual(report.total_pnl, Decimal("19.700"))

    def test_losing_position(self) -> None:
        self.portfolio.buy("BTCUSDT", "1", "100")
        self.market.quotes["BTCUSDT"] = replace(
            self.market.quotes["BTCUSDT"], bid=Decimal("90"), ask=Decimal("91")
        )
        report = self.report()
        self.assertEqual(report.unrealized_pnl, Decimal("-10.100"))
        self.assertEqual(report.total_equity, Decimal("989.900"))

    def test_realized_and_unrealized_after_partial_sell(self) -> None:
        self.portfolio.buy(
            "BTCUSDT", "1", "101", market_bid="99", market_ask="100", slippage_cost="1"
        )
        self.portfolio.sell(
            "BTCUSDT", "0.4", "109", market_bid="110", market_ask="111", slippage_cost="0.4"
        )
        report = self.report()
        self.assertEqual(report.realized_pnl, Decimal("3.1160"))
        self.assertEqual(report.unrealized_pnl, Decimal("5.3394"))
        self.assertEqual(report.total_pnl, Decimal("8.4554"))
        self.assertEqual(report.total_equity, Decimal("1008.4554"))
        self.assertEqual(report.total_return_percent, Decimal("0.8455400"))
        self.assertEqual(report.total_commissions, Decimal("0.1446"))
        self.assertEqual(report.total_slippage, Decimal("1.4"))

    def test_missing_stale_and_invalid_quotes_fail_closed(self) -> None:
        self.portfolio.buy("BTCUSDT", "1", "100")
        original = self.market.quotes["BTCUSDT"]
        for bad in (
            None,
            replace(original, timestamp_utc=NOW - timedelta(seconds=61)),
            replace(original, bid=Decimal("0")),
            replace(original, bid=Decimal("112")),
            replace(original, symbol="ETHUSDT"),
        ):
            with self.subTest(bad=bad):
                if bad is None:
                    del self.market.quotes["BTCUSDT"]
                else:
                    self.market.quotes["BTCUSDT"] = bad
                with self.assertRaises(AnalyticsUnavailable):
                    self.report()
                self.market.quotes["BTCUSDT"] = original

    def test_history_is_chronological_with_separate_buy_sell_counts(self) -> None:
        buy = self.portfolio.buy("BTCUSDT", "1", "100")
        sell = self.portfolio.sell("BTCUSDT", "0.5", "110")
        history = build_trade_history(self.portfolio.ledger)
        self.assertEqual([entry.id for entry in history.entries], [buy.id, sell.id])
        self.assertEqual(history.summary.buy_operations, 1)
        self.assertEqual(history.summary.sell_operations, 1)
        self.assertEqual(history.summary.total_commissions, buy.commission + sell.commission)
        self.assertEqual(history.summary.cumulative_realized_pnl, sell.realized_pnl)

    def test_read_only_portfolio_blocks_trades_and_preserves_database(self) -> None:
        self.portfolio.buy("BTCUSDT", "1", "100")
        before = sha256(self.path.read_bytes()).digest()
        readonly = Portfolio.open(self.settings, read_only=True)
        with self.assertRaises(PortfolioError):
            readonly.buy("ETHUSDT", "1", "100")
        with self.assertRaises(PortfolioError):
            readonly.sell("BTCUSDT", "1", "100")
        market = FakeMarket()
        build_portfolio_report(readonly, market, clock=lambda: NOW)
        build_trade_history(readonly.ledger)
        readonly.close()
        self.assertEqual(sha256(self.path.read_bytes()).digest(), before)

    def test_read_only_open_fails_for_missing_or_corrupt_database(self) -> None:
        with self.assertRaises(sqlite3.OperationalError):
            Portfolio.open(Settings(database_path=Path(self.directory.name) / "missing.sqlite3"), read_only=True)
        self.portfolio.buy("BTCUSDT", "1", "100")
        self.portfolio.close()
        connection = sqlite3.connect(str(self.path))
        connection.execute("DROP TRIGGER ledger_entries_prevent_update")
        connection.execute("UPDATE ledger_entries SET symbol = 'SOLUSDT'")
        connection.commit()
        connection.close()
        self.portfolio = Portfolio()
        with self.assertRaises(Exception):
            Portfolio.open(self.settings, read_only=True)

    def test_report_cli_rejects_stale_quote_without_writing(self) -> None:
        self.portfolio.buy("BTCUSDT", "1", "100")
        self.market.quotes["BTCUSDT"] = replace(
            self.market.quotes["BTCUSDT"], timestamp_utc=NOW - timedelta(seconds=61)
        )
        before = sha256(self.path.read_bytes()).digest()
        with (
            patch.object(paper_report, "load_settings", return_value=self.settings),
            patch.object(paper_report, "BybitReadOnlyClient", return_value=object()),
            patch.object(paper_report, "MarketDataClient", return_value=self.market),
            patch.object(
                paper_report,
                "build_portfolio_report",
                side_effect=lambda p, m: build_portfolio_report(p, m, clock=lambda: NOW),
            ),
            redirect_stdout(StringIO()) as output,
        ):
            self.assertEqual(paper_report.main(), 1)
        self.assertIn("unavailable", output.getvalue())
        self.assertNotIn("Total equity:", output.getvalue())
        self.assertEqual(sha256(self.path.read_bytes()).digest(), before)

    def test_both_clis_reject_corrupt_ledger(self) -> None:
        self.portfolio.buy("BTCUSDT", "1", "100")
        self.portfolio.close()
        connection = sqlite3.connect(str(self.path))
        connection.execute("DROP TRIGGER ledger_entries_prevent_update")
        connection.execute("UPDATE ledger_entries SET symbol = 'SOLUSDT'")
        connection.commit()
        connection.close()
        self.portfolio = Portfolio()
        before = sha256(self.path.read_bytes()).digest()
        with (
            patch.object(paper_report, "load_settings", return_value=self.settings),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(paper_report.main(), 1)
        with (
            patch.object(paper_history, "load_settings", return_value=self.settings),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(paper_history.main(), 1)
        self.assertEqual(sha256(self.path.read_bytes()).digest(), before)

    def test_report_and_history_cli_do_not_write_or_send_orders(self) -> None:
        self.portfolio.buy("BTCUSDT", "1", "100")
        before = sha256(self.path.read_bytes()).digest()
        with (
            patch.object(paper_report, "load_settings", return_value=self.settings),
            patch.object(paper_report, "BybitReadOnlyClient", return_value=object()),
            patch.object(paper_report, "MarketDataClient", return_value=self.market),
            patch.object(
                paper_report,
                "build_portfolio_report",
                side_effect=lambda p, m: build_portfolio_report(p, m, clock=lambda: NOW),
            ),
            redirect_stdout(StringIO()) as report_output,
        ):
            self.assertEqual(paper_report.main(), 0)
        self.assertIn("PAPER PORTFOLIO REPORT", report_output.getvalue())
        with (
            patch.object(paper_history, "load_settings", return_value=self.settings),
            redirect_stdout(StringIO()) as history_output,
        ):
            self.assertEqual(paper_history.main(), 0)
        self.assertIn("BUY operations: 1", history_output.getvalue())
        self.assertEqual(sha256(self.path.read_bytes()).digest(), before)


if __name__ == "__main__":
    unittest.main()
