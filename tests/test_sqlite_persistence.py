from decimal import Decimal
from pathlib import Path
import sqlite3
import tempfile
import unittest

from ai_crypto_trader.database import (
    CURRENT_SCHEMA_VERSION,
    DataIntegrityError,
    apply_migrations,
    current_schema_version,
)
from ai_crypto_trader.database.migrations import MIGRATIONS
from ai_crypto_trader.database.sqlite_store import DECIMAL_FIELDS
from ai_crypto_trader.portfolio import Portfolio, PortfolioError


class SQLitePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "portfolio.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_new_database_is_created_and_starts_with_1000_usdt(self) -> None:
        portfolio = Portfolio.open(database_path=str(self.database_path))
        self.assertTrue(self.database_path.is_file())
        self.assertEqual(portfolio.cash, Decimal("1000"))
        self.assertEqual(len(portfolio.ledger), 0)
        portfolio.close()

    def test_migrations_apply_and_are_idempotent(self) -> None:
        connection = sqlite3.connect(str(self.database_path))
        self.assertEqual(apply_migrations(connection), CURRENT_SCHEMA_VERSION)
        first_rows = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        self.assertEqual(apply_migrations(connection), CURRENT_SCHEMA_VERSION)
        second_rows = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(current_schema_version(connection), CURRENT_SCHEMA_VERSION)
        connection.close()

    def test_v1_database_with_existing_ledger_upgrades_without_loss(self) -> None:
        old_trade = Portfolio().buy("BTCUSDT", "0.01", "10000")
        connection = sqlite3.connect(str(self.database_path))
        connection.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at_utc TEXT NOT NULL)"
        )
        for statement in MIGRATIONS[0]:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO schema_migrations VALUES (1, '2026-01-01T00:00:00+00:00')"
        )
        connection.execute("INSERT INTO portfolio_metadata VALUES ('initial_cash', '1000')")
        columns = "id, timestamp_utc, operation_type, symbol, side, " + ", ".join(DECIMAL_FIELDS)
        values = (
            old_trade.id, old_trade.timestamp_utc.isoformat(), old_trade.operation_type,
            old_trade.symbol, old_trade.side,
            *(str(getattr(old_trade, field)) for field in DECIMAL_FIELDS),
        )
        connection.execute(
            f"INSERT INTO ledger_entries ({columns}) VALUES ({', '.join('?' for _ in values)})",
            values,
        )
        connection.commit()
        connection.close()

        restored = Portfolio.open(database_path=str(self.database_path))
        self.assertEqual(restored.btc, Decimal("0.01"))
        self.assertEqual(restored.cash, Decimal("899.90000"))
        self.assertEqual(restored.ledger.entries[0], old_trade)
        restored.close()
        connection = sqlite3.connect(str(self.database_path))
        self.assertEqual(current_schema_version(connection), CURRENT_SCHEMA_VERSION)
        self.assertEqual(apply_migrations(connection), CURRENT_SCHEMA_VERSION)
        connection.close()

    def test_buy_survives_restart(self) -> None:
        portfolio = Portfolio.open(database_path=str(self.database_path))
        original = portfolio.buy("BTCUSDT", "0.01", "10000")
        portfolio.close()

        restored = Portfolio.open(database_path=str(self.database_path))
        self.assertEqual(restored.btc, Decimal("0.01"))
        self.assertEqual(restored.cash, Decimal("899.90000"))
        self.assertEqual(restored.ledger.entries[0], original)
        restored.close()

    def test_sell_realized_pnl_commission_and_slippage_survive_restart(self) -> None:
        portfolio = Portfolio.open(database_path=str(self.database_path))
        portfolio.buy("BTCUSDT", "0.01", "10000", slippage_cost="0.25")
        portfolio.sell("BTCUSDT", "0.004", "12000", slippage_cost="0.10")
        expected = (
            portfolio.cash,
            portfolio.btc,
            portfolio.realized_pnl,
            portfolio.total_commission,
            portfolio.total_slippage,
        )
        portfolio.close()

        restored = Portfolio.open(database_path=str(self.database_path))
        actual = (
            restored.cash,
            restored.btc,
            restored.realized_pnl,
            restored.total_commission,
            restored.total_slippage,
        )
        self.assertEqual(actual, expected)
        self.assertEqual(len(restored.ledger), 2)
        restored.close()

    def test_btc_eth_and_ledger_order_survive_restart(self) -> None:
        portfolio = Portfolio.open(database_path=str(self.database_path))
        first = portfolio.buy("ETHUSDT", "0.1", "1000")
        second = portfolio.buy("BTCUSDT", "0.01", "10000")
        third = portfolio.sell("ETHUSDT", "0.04", "1100")
        portfolio.close()

        restored = Portfolio.open(database_path=str(self.database_path))
        self.assertEqual(restored.btc, Decimal("0.01"))
        self.assertEqual(restored.eth, Decimal("0.06"))
        self.assertEqual([entry.id for entry in restored.ledger], [first.id, second.id, third.id])
        restored.close()

    def test_decimal_round_trip_is_exact(self) -> None:
        portfolio = Portfolio.open(database_path=str(self.database_path))
        original = portfolio.buy(
            "BTCUSDT",
            Decimal("0.000000000000000001"),
            Decimal("123.123456789123456789"),
            slippage_cost=Decimal("0.0000000000000000007"),
        )
        portfolio.close()

        restored = Portfolio.open(database_path=str(self.database_path))
        loaded = restored.ledger.entries[0]
        self.assertEqual(loaded.quantity, original.quantity)
        self.assertEqual(loaded.execution_price, original.execution_price)
        self.assertEqual(loaded.gross_value, original.gross_value)
        self.assertEqual(loaded.slippage_cost, original.slippage_cost)
        restored.close()

    def test_sqlite_rejects_update_and_delete(self) -> None:
        portfolio = Portfolio.open(database_path=str(self.database_path))
        portfolio.buy("BTCUSDT", "0.01", "10000")
        portfolio.close()

        connection = sqlite3.connect(str(self.database_path))
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute("UPDATE ledger_entries SET symbol = 'ETHUSDT' WHERE sequence_no = 1")
        connection.rollback()
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM ledger_entries WHERE sequence_no = 1")
        connection.rollback()
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM ledger_entries").fetchone()[0], 1)
        connection.close()

    def test_failed_transaction_changes_neither_portfolio_nor_database(self) -> None:
        portfolio = Portfolio.open(database_path=str(self.database_path))
        connection = sqlite3.connect(str(self.database_path))
        connection.execute(
            """
            CREATE TRIGGER test_reject_insert BEFORE INSERT ON ledger_entries
            BEGIN SELECT RAISE(ABORT, 'simulated storage failure'); END
            """
        )
        connection.commit()
        before = (portfolio.cash, portfolio.btc, portfolio.ledger.entries)

        with self.assertRaises(sqlite3.IntegrityError):
            portfolio.buy("BTCUSDT", "0.01", "10000")

        self.assertEqual((portfolio.cash, portfolio.btc, portfolio.ledger.entries), before)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM ledger_entries").fetchone()[0], 0)
        connection.close()
        portfolio.close()

    def test_corrupted_impossible_state_is_detected_on_restart(self) -> None:
        portfolio = Portfolio.open(database_path=str(self.database_path))
        portfolio.buy("BTCUSDT", "0.01", "10000")
        portfolio.close()

        connection = sqlite3.connect(str(self.database_path))
        connection.execute("DROP TRIGGER ledger_entries_prevent_update")
        connection.execute("UPDATE ledger_entries SET symbol = 'SOLUSDT' WHERE sequence_no = 1")
        connection.commit()
        connection.close()

        with self.assertRaises((DataIntegrityError, PortfolioError)):
            Portfolio.open(database_path=str(self.database_path))


if __name__ == "__main__":
    unittest.main()
