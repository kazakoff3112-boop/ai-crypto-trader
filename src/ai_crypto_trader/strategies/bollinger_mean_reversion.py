"""Fixed-parameter long-only Bollinger mean-reversion baseline, offline only."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException
from typing import Sequence

from ..backtesting.strategy import PortfolioSnapshot, Signal
from ..market_data.candle import Candle

ZERO = Decimal("0")
HUNDRED = Decimal("100")


@dataclass(frozen=True)
class BollingerBands:
    middle: Decimal
    variance: Decimal
    stddev: Decimal
    upper: Decimal
    lower: Decimal


def calculate_bollinger_bands(closes: Sequence[Decimal], period: int = 20,
                              multiplier: Decimal = Decimal("2")) -> BollingerBands:
    """Use the last N closed prices and population variance; never use float."""
    if type(period) is not int or period < 2 or len(closes) < period:
        raise ValueError("Bollinger bands need at least two closed prices and a full window")
    if not isinstance(multiplier, Decimal) or not multiplier.is_finite() or multiplier <= ZERO:
        raise ValueError("Bollinger multiplier must be a positive finite Decimal")
    window = closes[-period:]
    if any(not isinstance(price, Decimal) or not price.is_finite() or price <= ZERO
           for price in window):
        raise ValueError("Bollinger closes must be positive finite Decimal prices")
    try:
        count = Decimal(period)
        middle = sum(window, ZERO) / count
        variance = sum(((price - middle) ** 2 for price in window), ZERO) / count
        stddev = variance.sqrt()
        upper = middle + multiplier * stddev
        lower = middle - multiplier * stddev
    except DecimalException:
        raise ValueError("Bollinger bands cannot be calculated") from None
    if any(not value.is_finite() for value in (middle, variance, stddev, upper, lower)):
        raise ValueError("Bollinger bands must be finite")
    if variance < ZERO or stddev < ZERO or upper < middle or lower > middle:
        raise ValueError("Bollinger bands are inconsistent")
    return BollingerBands(middle, variance, stddev, upper, lower)


@dataclass(frozen=True)
class BollingerMeanReversionStrategy:
    bollinger_period: int = 20
    stddev_multiplier: Decimal = Decimal("2")
    position_size_pct: Decimal = Decimal("25")

    def __post_init__(self) -> None:
        if type(self.bollinger_period) is not int or self.bollinger_period < 2:
            raise ValueError("Bollinger period must be an integer >= 2")
        if (not isinstance(self.stddev_multiplier, Decimal)
                or not self.stddev_multiplier.is_finite() or self.stddev_multiplier <= ZERO):
            raise ValueError("Bollinger multiplier must be a positive finite Decimal")
        if (not isinstance(self.position_size_pct, Decimal)
                or not self.position_size_pct.is_finite()
                or not ZERO < self.position_size_pct <= HUNDRED):
            raise ValueError("Position size must be a Decimal percentage in (0, 100]")

    @property
    def minimum_candles(self) -> int:
        # One complete signal window followed by the next execution candle.
        return self.bollinger_period + 1

    @property
    def name(self) -> str:
        return "Bollinger Mean Reversion"

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return (("bollinger_period", str(self.bollinger_period)),
                ("stddev_multiplier", str(self.stddev_multiplier)),
                ("position_size_pct", str(self.position_size_pct)))

    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        if len(history) < self.bollinger_period:
            return Signal()
        bands = calculate_bollinger_bands(
            tuple(candle.close for candle in history[-self.bollinger_period:]),
            self.bollinger_period, self.stddev_multiplier,
        )
        close = history[-1].close
        if portfolio.quantity == ZERO and close < bands.lower:
            budget = min(portfolio.equity, portfolio.cash) * self.position_size_pct / HUNDRED
            if budget > ZERO:
                return Signal("BUY", budget_usdt=budget)
        elif portfolio.quantity > ZERO and close >= bands.middle:
            return Signal("SELL", quantity=portfolio.quantity)
        return Signal()
