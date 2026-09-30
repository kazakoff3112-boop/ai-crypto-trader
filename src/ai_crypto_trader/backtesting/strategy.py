"""Minimal causal strategy contract; no future candles or mutable Portfolio exposed."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from ..market_data.candle import Candle


@dataclass(frozen=True)
class PortfolioSnapshot:
    symbol: str
    cash: Decimal
    quantity: Decimal
    average_entry_price: Decimal
    realized_pnl: Decimal


@dataclass(frozen=True)
class Signal:
    action: str = "HOLD"
    quantity: Decimal | None = None


class Strategy(Protocol):
    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        """Called after history[-1] closes; history contains no later candles."""
        ...
