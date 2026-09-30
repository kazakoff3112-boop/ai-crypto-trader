from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from ai_crypto_trader.config import Settings
from ai_crypto_trader.diagnostics import paper_trade
from ai_crypto_trader.market_data import SpotQuote
from ai_crypto_trader.portfolio import Portfolio


class FakeMarketData:
    stale = False
    invalid = False

    def __init__(self, unused_client) -> None:
        pass

    def get_spot_quote(self, symbol: str, *, category: str = "spot") -> SpotQuote:
        timestamp = datetime.now(timezone.utc)
        if self.stale:
            timestamp -= timedelta(seconds=61)
        bid = Decimal("100")
        ask = Decimal("99") if self.invalid else Decimal("101")
        return SpotQuote(symbol, Decimal("100.5"), bid, ask, timestamp)


class PaperTradeDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "manual.sqlite3"
        self.settings = Settings(
            database_path=self.path,
            paper_slippage_bps=Decimal("100"),
            max_manual_paper_trade_usdt=Decimal("250"),
        )
        self.output: list[str] = []
        FakeMarketData.stale = False
        FakeMarketData.invalid = False

    def tearDown(self) -> None:
        self.directory.cleanup()

    def run_command(self, arguments: list[str], confirmation: str = "") -> int:
        with (
            patch.object(paper_trade, "load_settings", return_value=self.settings),
            patch.object(paper_trade, "BybitReadOnlyClient", return_value=object()),
            patch.object(paper_trade, "MarketDataClient", FakeMarketData),
        ):
            return paper_trade.main(
                arguments,
                input_fn=lambda prompt: confirmation,
                output=self.output.append,
            )

    def portfolio_state(self) -> tuple:
        portfolio = Portfolio.open(self.settings)
        state = (portfolio.cash, portfolio.btc, portfolio.eth, len(portfolio.ledger))
        portfolio.close()
        return state

    def test_execute_without_or_with_wrong_confirmation_is_cancelled(self) -> None:
        args = ["--execute-paper", "--side", "buy", "--symbol", "BTCUSDT", "--usdt", "100"]
        self.assertEqual(self.run_command(args, ""), 2)
        self.assertEqual(self.portfolio_state(), (Decimal("1000"), Decimal("0"), Decimal("0"), 0))
        self.assertEqual(self.run_command(args, "yes"), 2)
        self.assertEqual(self.portfolio_state(), (Decimal("1000"), Decimal("0"), Decimal("0"), 0))

    def test_exact_confirmation_executes_buy_and_restart_restores_it(self) -> None:
        result = self.run_command(
            ["--execute-paper", "--side", "buy", "--symbol", "BTCUSDT", "--usdt", "100"],
            "YES",
        )
        self.assertEqual(result, 0)
        self.assertTrue(any(paper_trade.PAPER_WARNING in line for line in self.output))
        portfolio = Portfolio.open(self.settings)
        self.assertEqual(portfolio.cash, Decimal("900"))
        self.assertGreater(portfolio.btc, Decimal("0"))
        self.assertEqual(len(portfolio.ledger), 1)
        portfolio.close()
        restored = Portfolio.open(self.settings)
        self.assertEqual(restored.cash, Decimal("900"))
        self.assertEqual(len(restored.ledger), 1)
        restored.close()

    def test_sell_quantity_and_sell_all_work(self) -> None:
        buy = ["--execute-paper", "--side", "buy", "--symbol", "BTCUSDT", "--usdt", "100"]
        self.assertEqual(self.run_command(buy, "YES"), 0)
        portfolio = Portfolio.open(self.settings)
        half = portfolio.btc / 2
        portfolio.close()
        self.assertEqual(
            self.run_command(
                ["--execute-paper", "--side", "sell", "--symbol", "BTCUSDT", "--quantity", str(half)],
                "YES",
            ),
            0,
        )
        self.assertEqual(
            self.run_command(
                ["--execute-paper", "--side", "sell", "--symbol", "BTCUSDT", "--all"],
                "YES",
            ),
            0,
        )
        self.assertEqual(self.portfolio_state()[1], Decimal("0"))

    def test_buy_limit_allows_250_and_rejects_above(self) -> None:
        self.assertEqual(
            self.run_command(
                ["--execute-paper", "--side", "buy", "--symbol", "BTCUSDT", "--usdt", "250"],
                "YES",
            ),
            0,
        )
        before = self.portfolio_state()
        self.assertEqual(
            self.run_command(
                ["--execute-paper", "--side", "buy", "--symbol", "ETHUSDT", "--usdt", "250.01"],
                "YES",
            ),
            1,
        )
        self.assertEqual(self.portfolio_state(), before)

    def test_sell_notional_above_limit_is_rejected(self) -> None:
        portfolio = Portfolio.open(self.settings)
        portfolio.buy("BTCUSDT", "3", "100")
        portfolio.close()
        before = self.portfolio_state()
        self.assertEqual(
            self.run_command(
                ["--execute-paper", "--side", "sell", "--symbol", "BTCUSDT", "--quantity", "3"],
                "YES",
            ),
            1,
        )
        self.assertEqual(self.portfolio_state(), before)

    def test_preview_and_execute_conflict_and_missing_mode_are_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_command(["--preview", "--execute-paper"])
        with self.assertRaises(SystemExit):
            self.run_command([])

    def test_status_changes_nothing_and_uses_no_market_data(self) -> None:
        before = self.portfolio_state()
        with (
            patch.object(paper_trade, "load_settings", return_value=self.settings),
            patch.object(paper_trade, "MarketDataClient", side_effect=AssertionError("network path used")),
        ):
            self.assertEqual(paper_trade.main(["--status"], output=self.output.append), 0)
        self.assertEqual(self.portfolio_state(), before)
        self.assertTrue(any("Ledger entries: 0" in line for line in self.output))

    def test_insufficient_cash_and_asset_are_rejected(self) -> None:
        portfolio = Portfolio.open(self.settings)
        portfolio.buy("BTCUSDT", "9", "100")
        portfolio.close()
        before = self.portfolio_state()
        self.assertEqual(
            self.run_command(
                ["--execute-paper", "--side", "buy", "--symbol", "ETHUSDT", "--usdt", "250"],
                "YES",
            ),
            1,
        )
        self.assertEqual(
            self.run_command(
                ["--execute-paper", "--side", "sell", "--symbol", "ETHUSDT", "--quantity", "1"],
                "YES",
            ),
            1,
        )
        self.assertEqual(self.portfolio_state(), before)

    def test_stale_and_invalid_quote_change_nothing(self) -> None:
        before = self.portfolio_state()
        args = ["--execute-paper", "--side", "buy", "--symbol", "BTCUSDT", "--usdt", "100"]
        FakeMarketData.stale = True
        self.assertEqual(self.run_command(args, "YES"), 1)
        FakeMarketData.stale = False
        FakeMarketData.invalid = True
        self.assertEqual(self.run_command(args, "YES"), 1)
        self.assertEqual(self.portfolio_state(), before)

    def test_database_failure_leaves_no_partial_trade(self) -> None:
        self.assertEqual(self.run_command(["--status"]), 0)
        connection = sqlite3.connect(str(self.path))
        connection.execute(
            "CREATE TRIGGER test_fail BEFORE INSERT ON ledger_entries "
            "BEGIN SELECT RAISE(ABORT, 'storage failure'); END"
        )
        connection.commit()
        before = self.portfolio_state()
        self.assertEqual(
            self.run_command(
                ["--execute-paper", "--side", "buy", "--symbol", "BTCUSDT", "--usdt", "100"],
                "YES",
            ),
            1,
        )
        self.assertEqual(self.portfolio_state(), before)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM ledger_entries").fetchone()[0], 0)
        connection.close()


if __name__ == "__main__":
    unittest.main()
