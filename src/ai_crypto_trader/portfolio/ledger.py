"""Immutable financial events and their append-only in-memory journal."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Iterable, Iterator


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    timestamp_utc: datetime
    operation_type: str
    symbol: str
    side: str
    quantity: Decimal
    execution_price: Decimal
    gross_value: Decimal
    commission: Decimal
    slippage_cost: Decimal
    usdt_change: Decimal
    base_asset_change: Decimal
    realized_pnl: Decimal
    usdt_balance_after: Decimal
    position_after: Decimal
    average_entry_price_after: Decimal
    market_bid: Decimal | None = None
    market_ask: Decimal | None = None


class Ledger:
    """Append-only journal: entries can be inspected but not replaced or deleted."""

    def __init__(
        self,
        entries: Iterable[LedgerEntry] = (),
        *,
        persist_entry: Callable[[LedgerEntry], None] | None = None,
        close_storage: Callable[[], None] | None = None,
    ) -> None:
        self._entries = list(entries)
        self._persist_entry = persist_entry
        self._close_storage = close_storage

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def is_persistent(self) -> bool:
        return self._persist_entry is not None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self.entries)

    def _append(self, entry: LedgerEntry) -> None:
        """Persist first, then expose in memory; no update/delete API exists."""
        if self._persist_entry is not None:
            self._persist_entry(entry)
        self._entries.append(entry)

    def close(self) -> None:
        if self._close_storage is not None:
            self._close_storage()
            self._close_storage = None
