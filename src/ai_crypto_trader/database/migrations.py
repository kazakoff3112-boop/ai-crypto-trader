"""Small ordered SQLite migration system with no external dependencies."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3


MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        """
        CREATE TABLE portfolio_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE ledger_entries (
            sequence_no INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT NOT NULL UNIQUE,
            timestamp_utc TEXT NOT NULL,
            operation_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity TEXT NOT NULL CHECK(typeof(quantity) = 'text'),
            execution_price TEXT NOT NULL CHECK(typeof(execution_price) = 'text'),
            gross_value TEXT NOT NULL CHECK(typeof(gross_value) = 'text'),
            commission TEXT NOT NULL CHECK(typeof(commission) = 'text'),
            slippage_cost TEXT NOT NULL CHECK(typeof(slippage_cost) = 'text'),
            usdt_change TEXT NOT NULL CHECK(typeof(usdt_change) = 'text'),
            base_asset_change TEXT NOT NULL CHECK(typeof(base_asset_change) = 'text'),
            realized_pnl TEXT NOT NULL CHECK(typeof(realized_pnl) = 'text'),
            usdt_balance_after TEXT NOT NULL CHECK(typeof(usdt_balance_after) = 'text'),
            position_after TEXT NOT NULL CHECK(typeof(position_after) = 'text'),
            average_entry_price_after TEXT NOT NULL CHECK(typeof(average_entry_price_after) = 'text')
        )
        """,
        """
        CREATE TRIGGER ledger_entries_prevent_update
        BEFORE UPDATE ON ledger_entries
        BEGIN
            SELECT RAISE(ABORT, 'ledger_entries are append-only');
        END
        """,
        """
        CREATE TRIGGER ledger_entries_prevent_delete
        BEFORE DELETE ON ledger_entries
        BEGIN
            SELECT RAISE(ABORT, 'ledger_entries are append-only');
        END
        """,
    ),
    (
        "ALTER TABLE ledger_entries ADD COLUMN market_bid TEXT CHECK(market_bid IS NULL OR typeof(market_bid) = 'text')",
        "ALTER TABLE ledger_entries ADD COLUMN market_ask TEXT CHECK(market_ask IS NULL OR typeof(market_ask) = 'text')",
    ),
)

CURRENT_SCHEMA_VERSION = len(MIGRATIONS)


def _ensure_version_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at_utc TEXT NOT NULL
        )
        """
    )
    connection.commit()


def current_schema_version(connection: sqlite3.Connection) -> int:
    _ensure_version_table(connection)
    row = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
    return int(row[0])


def apply_migrations(connection: sqlite3.Connection) -> int:
    """Apply every missing migration exactly once, each in its own transaction."""
    _ensure_version_table(connection)
    applied = {int(row[0]) for row in connection.execute("SELECT version FROM schema_migrations")}
    unknown = [version for version in applied if version > CURRENT_SCHEMA_VERSION]
    if unknown:
        raise RuntimeError(f"Database schema is newer than this application: {max(unknown)}")

    for version, statements in enumerate(MIGRATIONS, start=1):
        if version in applied:
            continue
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at_utc) VALUES (?, ?)",
                (version, datetime.now(timezone.utc).isoformat()),
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
    return current_schema_version(connection)
