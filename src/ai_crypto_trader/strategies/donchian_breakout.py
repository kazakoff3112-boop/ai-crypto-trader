"""Fixed-parameter long-only Donchian baseline for offline backtests only."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..backtesting.strategy import PortfolioSnapshot, Signal
from ..market_data.candle import Candle

ZERO = Decimal("0")


@dataclass(frozen=True)
class DonchianBreakoutStrategy:
    entry_lookback: int = 96
    exit_lookback: int = 48
    position_size_pct: Decimal = Decimal("25")

    def __post_init__(self) -> None:
        if (type(self.entry_lookback) is not int or self.entry_lookback <= 0
                or type(self.exit_lookback) is not int or self.exit_lookback <= 0):
            raise ValueError("Donchian lookbacks must be positive integers")
        if (not isinstance(self.position_size_pct, Decimal)
                or not self.position_size_pct.is_finite()
                or not ZERO < self.position_size_pct <= Decimal("100")):
            raise ValueError("Position size must be a Decimal percentage in (0, 100]")

    @property
    def minimum_candles(self) -> int:
        # Previous window, current closed signal candle, next execution candle.
        return max(self.entry_lookback, self.exit_lookback) + 2

    @property
    def name(self) -> str:
        return "Donchian Breakout"

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return (("entry_lookback", str(self.entry_lookback)),
                ("exit_lookback", str(self.exit_lookback)),
                ("position_size_pct", str(self.position_size_pct)))

    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        """The current closed candle is history[-1]; windows end at history[-2]."""
        if portfolio.quantity == ZERO:
            if len(history) < self.entry_lookback + 1:
                return Signal()
            previous_high = max(candle.high for candle in history[-self.entry_lookback - 1:-1])
            if history[-1].close > previous_high:
                budget = min(portfolio.equity, portfolio.cash) * self.position_size_pct / Decimal("100")
                if budget > ZERO:
                    return Signal("BUY", budget_usdt=budget)
        else:
            if len(history) < self.exit_lookback + 1:
                return Signal()
            previous_low = min(candle.low for candle in history[-self.exit_lookback - 1:-1])
            if history[-1].close < previous_low:
                return Signal("SELL", quantity=portfolio.quantity)
        return Signal()
