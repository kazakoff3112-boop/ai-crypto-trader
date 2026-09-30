"""Transparent long-only SMA crossover baseline for offline backtests only."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from ..backtesting.strategy import PortfolioSnapshot, Signal
from ..market_data.candle import Candle

ZERO = Decimal("0")


def simple_moving_average(closes: Sequence[Decimal], period: int) -> Decimal:
    if type(period) is not int or period <= 0 or len(closes) < period:
        raise ValueError("SMA requires a positive period and enough closed prices")
    window = closes[-period:]
    if any(not isinstance(price, Decimal) or not price.is_finite() or price <= ZERO
           for price in window):
        raise ValueError("SMA requires positive finite Decimal prices")
    return sum(window, ZERO) / Decimal(period)


@dataclass(frozen=True)
class SMACrossoverStrategy:
    fast: int = 20
    slow: int = 50
    position_size_pct: Decimal = Decimal("25")

    def __post_init__(self) -> None:
        if (type(self.fast) is not int or type(self.slow) is not int
                or self.fast <= 0 or self.slow <= 0 or self.fast >= self.slow):
            raise ValueError("SMA periods require 0 < fast < slow")
        if (not isinstance(self.position_size_pct, Decimal)
                or not self.position_size_pct.is_finite()
                or not ZERO < self.position_size_pct <= Decimal("100")):
            raise ValueError("Position size must be a Decimal percentage in (0, 100]")

    @property
    def minimum_candles(self) -> int:
        # slow closed prices, one previous comparison, one next-open execution.
        return self.slow + 2

    @property
    def name(self) -> str:
        return "SMA Crossover"

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return (("fast", str(self.fast)), ("slow", str(self.slow)),
                ("position_size_pct", str(self.position_size_pct)))

    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        if len(history) < self.slow + 1:
            return Signal()
        closes = tuple(candle.close for candle in history)
        fast_prev = simple_moving_average(closes[:-1], self.fast)
        slow_prev = simple_moving_average(closes[:-1], self.slow)
        fast_now = simple_moving_average(closes, self.fast)
        slow_now = simple_moving_average(closes, self.slow)
        if fast_prev <= slow_prev and fast_now > slow_now and portfolio.quantity == ZERO:
            budget = min(portfolio.equity, portfolio.cash) * self.position_size_pct / Decimal("100")
            if budget > ZERO:
                return Signal("BUY", budget_usdt=budget)
        if fast_prev >= slow_prev and fast_now < slow_now and portfolio.quantity > ZERO:
            return Signal("SELL", quantity=portfolio.quantity)
        return Signal()
