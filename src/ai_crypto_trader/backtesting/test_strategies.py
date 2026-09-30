"""Technical fixtures only. These are not investment strategies."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .strategy import PortfolioSnapshot, Signal
from ..market_data.candle import Candle


class NoTradeStrategy:
    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        return Signal()


@dataclass(frozen=True)
class BuyOnceStrategy:
    quantity: Decimal = Decimal("0.001")

    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        return Signal("BUY", self.quantity) if len(history) == 1 else Signal()


@dataclass(frozen=True)
class BuyThenSellStrategy:
    quantity: Decimal = Decimal("0.001")

    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        if len(history) == 1:
            return Signal("BUY", self.quantity)
        if len(history) == 2:
            return Signal("SELL", self.quantity)
        return Signal()
