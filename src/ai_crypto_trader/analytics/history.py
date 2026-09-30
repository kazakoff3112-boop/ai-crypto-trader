"""Read-only chronological view of the immutable paper ledger."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..portfolio import Ledger, LedgerEntry


@dataclass(frozen=True)
class TradeHistorySummary:
    buy_operations: int
    sell_operations: int
    total_commissions: Decimal
    total_slippage: Decimal
    cumulative_realized_pnl: Decimal


@dataclass(frozen=True)
class TradeHistory:
    entries: tuple[LedgerEntry, ...]
    summary: TradeHistorySummary


def build_trade_history(ledger: Ledger) -> TradeHistory:
    entries = tuple(sorted(ledger.entries, key=lambda entry: entry.timestamp_utc))
    summary = TradeHistorySummary(
        buy_operations=sum(entry.side == "BUY" for entry in entries),
        sell_operations=sum(entry.side == "SELL" for entry in entries),
        total_commissions=sum((entry.commission for entry in entries), Decimal("0")),
        total_slippage=sum((entry.slippage_cost for entry in entries), Decimal("0")),
        cumulative_realized_pnl=sum((entry.realized_pnl for entry in entries), Decimal("0")),
    )
    return TradeHistory(entries, summary)
