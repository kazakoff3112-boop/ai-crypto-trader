"""Exact, transactional SQLite storage for the append-only ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sqlite3

from ..portfolio.ledger import LedgerEntry
from .migrations import apply_migrations


class DataIntegrityError(RuntimeError):
    """Raised when persisted financial data cannot be trusted."""


DECIMAL_FIELDS = (
    "quantity",
    "execution_price",
    "gross_value",
    "commission",
    "slippage_cost",
    "usdt_change",
    "base_asset_change",
    "realized_pnl",
    "usdt_balance_after",
    "position_after",
    "average_entry_price_after",
)


class SQLiteLedgerStore:
    def __init__(self, database_path: str | Path, initial_cash: Decimal) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        apply_migrations(self._connection)
        self._initialize_metadata(initial_cash)

    @property
    def initial_cash(self) -> Decimal:
        row = self._connection.execute(
            "SELECT value FROM portfolio_metadata WHERE key = 'initial_cash'"
        ).fetchone()
        if row is None:
            raise DataIntegrityError("Missing initial_cash metadata")
        return self._parse_decimal(row[0], "initial_cash")

    def append(self, entry: LedgerEntry) -> None:
        values = (
            entry.id,
            entry.timestamp_utc.isoformat(),
            entry.operation_type,
            entry.symbol,
            entry.side,
            *(str(getattr(entry, field)) for field in DECIMAL_FIELDS),
            str(entry.market_bid) if entry.market_bid is not None else None,
            str(entry.market_ask) if entry.market_ask is not None else None,
        )
        columns = (
            "id, timestamp_utc, operation_type, symbol, side, "
            + ", ".join(DECIMAL_FIELDS)
            + ", market_bid, market_ask"
        )
        placeholders = ", ".join("?" for _ in values)
        try:
            with self._connection:
                self._connection.execute(
                    f"INSERT INTO ledger_entries ({columns}) VALUES ({placeholders})", values
                )
        except sqlite3.Error:
            self._connection.rollback()
            raise

    def load_entries(self) -> tuple[LedgerEntry, ...]:
        self._check_database_integrity()
        rows = self._connection.execute(
            "SELECT * FROM ledger_entries ORDER BY sequence_no ASC"
        ).fetchall()
        entries: list[LedgerEntry] = []
        seen_ids: set[str] = set()
        for expected_sequence, row in enumerate(rows, start=1):
            if row["sequence_no"] != expected_sequence:
                raise DataIntegrityError("Ledger sequence is not contiguous")
            if not row["id"] or row["id"] in seen_ids:
                raise DataIntegrityError("Ledger entry IDs must be non-empty and unique")
            seen_ids.add(row["id"])
            timestamp = self._parse_timestamp(row["timestamp_utc"])
            decimals = {field: self._parse_decimal(row[field], field) for field in DECIMAL_FIELDS}
            for field in ("market_bid", "market_ask"):
                decimals[field] = None if row[field] is None else self._parse_decimal(row[field], field)
            entries.append(
                LedgerEntry(
                    id=row["id"],
                    timestamp_utc=timestamp,
                    operation_type=row["operation_type"],
                    symbol=row["symbol"],
                    side=row["side"],
                    **decimals,
                )
            )

        sequence_row = self._connection.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'ledger_entries'"
        ).fetchone()
        recorded_sequence = 0 if sequence_row is None else int(sequence_row[0])
        if recorded_sequence != len(rows):
            raise DataIntegrityError("Ledger sequence counter does not match stored history")
        return tuple(entries)

    def close(self) -> None:
        self._connection.close()

    def _initialize_metadata(self, initial_cash: Decimal) -> None:
        if not initial_cash.is_finite() or initial_cash < 0:
            raise DataIntegrityError("Initial cash must be a finite non-negative Decimal")
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO portfolio_metadata(key, value) VALUES ('initial_cash', ?)",
                (str(initial_cash),),
            )
        if self.initial_cash != initial_cash:
            raise DataIntegrityError("Configured initial cash conflicts with existing database")

    def _check_database_integrity(self) -> None:
        result = self._connection.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise DataIntegrityError(f"SQLite integrity check failed: {result[0] if result else 'unknown'}")

    @staticmethod
    def _parse_decimal(value: object, field: str) -> Decimal:
        if not isinstance(value, str):
            raise DataIntegrityError(f"{field} is not stored as exact TEXT")
        try:
            number = Decimal(value)
        except (InvalidOperation, ValueError) as exc:
            raise DataIntegrityError(f"Invalid Decimal in {field}") from exc
        if not number.is_finite():
            raise DataIntegrityError(f"Non-finite Decimal in {field}")
        return number

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if not isinstance(value, str):
            raise DataIntegrityError("Invalid ledger timestamp")
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError as exc:
            raise DataIntegrityError("Invalid ledger timestamp") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise DataIntegrityError("Ledger timestamp must be UTC")
        return timestamp
