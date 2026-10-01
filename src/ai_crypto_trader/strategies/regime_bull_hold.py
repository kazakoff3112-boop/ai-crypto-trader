"""Pre-registered BTCUSDT 1h regime-transition strategy for offline backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable

from ..backtesting.strategy import PortfolioSnapshot, Signal
from ..market_data.candle import Candle
from ..market_data.market_regime import MarketRegime, RegimeUnavailable, WARMUP_CANDLES

ZERO = Decimal("0")
POSITION_SIZE_PCT = Decimal("25")


@dataclass(frozen=True)
class RegimeTransitionStats:
    total_valid_transitions: int
    transitions_into_bull: int
    buy_signals: int
    transitions_out_of_bull: int
    sell_signals: int
    bull_hours_while_position_open: int
    skipped_unavailable: int


@dataclass
class RegimeBullHoldStrategy:
    """Trade only transitions between two completed hourly regimes.

    The caller provides the existing MarketRegimeReport.regime_at lookup. The
    strategy sees a timestamp lookup, not future regime points or candles.
    """

    regime_at: Callable[[datetime], MarketRegime | None] = field(repr=False)
    minimum_candles = WARMUP_CANDLES + 3  # 224 prior hours + signal + execution.
    _transitions: int = field(default=0, init=False, repr=False)
    _into_bull: int = field(default=0, init=False, repr=False)
    _buys: int = field(default=0, init=False, repr=False)
    _out_of_bull: int = field(default=0, init=False, repr=False)
    _sells: int = field(default=0, init=False, repr=False)
    _bull_held_hours: int = field(default=0, init=False, repr=False)
    _skipped: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.regime_at):
            raise ValueError("A regime lookup is required")

    @property
    def name(self) -> str:
        return "1h BULL Regime Hold"

    @property
    def position_size_pct(self) -> Decimal:
        return POSITION_SIZE_PCT

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return (("position_size_pct", "25"),)

    @property
    def transition_stats(self) -> RegimeTransitionStats:
        return RegimeTransitionStats(self._transitions, self._into_bull,
                                     self._buys, self._out_of_bull, self._sells,
                                     self._bull_held_hours, self._skipped)

    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        if len(history) < 2:
            return Signal()
        previous_candle, current_candle = history[-2:]
        if (previous_candle.symbol != "BTCUSDT" or current_candle.symbol != "BTCUSDT"
                or previous_candle.interval != "1h" or current_candle.interval != "1h"
                or current_candle.open_time_utc - previous_candle.open_time_utc != timedelta(hours=1)
                or portfolio.symbol != "BTCUSDT" or not portfolio.cash.is_finite()
                or not portfolio.quantity.is_finite() or portfolio.cash < ZERO
                or portfolio.quantity < ZERO or not portfolio.equity.is_finite()
                or portfolio.equity <= ZERO):
            raise RegimeUnavailable("Invalid regime strategy candle or portfolio state")
        try:
            previous = self.regime_at(previous_candle.close_time_utc)
            current = self.regime_at(current_candle.close_time_utc)
        except RegimeUnavailable:
            self._skipped += 1
            return Signal()
        if previous is None or current is None:
            self._skipped += 1
            return Signal()
        if type(previous) is not MarketRegime or type(current) is not MarketRegime:
            raise RegimeUnavailable("Unknown hourly regime")
        if current is MarketRegime.BULL and portfolio.quantity > ZERO:
            self._bull_held_hours += 1
        if previous is current:
            return Signal()
        self._transitions += 1
        if current is MarketRegime.BULL:
            self._into_bull += 1
            if portfolio.quantity == ZERO:
                budget = min(portfolio.cash, portfolio.equity) * POSITION_SIZE_PCT / Decimal("100")
                if budget <= ZERO:
                    raise RegimeUnavailable("No virtual capital for regime entry")
                self._buys += 1
                return Signal("BUY", budget_usdt=budget)
        elif previous is MarketRegime.BULL:
            self._out_of_bull += 1
            if portfolio.quantity > ZERO:
                self._sells += 1
                return Signal("SELL", quantity=portfolio.quantity)
        return Signal()
