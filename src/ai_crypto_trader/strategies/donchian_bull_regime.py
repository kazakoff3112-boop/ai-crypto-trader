"""Fixed Donchian hypothesis: permit new entries only in a known BULL regime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Callable

from ..backtesting.strategy import PortfolioSnapshot, Signal
from ..market_data.candle import Candle
from ..market_data.market_regime import MarketRegime, RegimeUnavailable
from .donchian_breakout import DonchianBreakoutStrategy


@dataclass(frozen=True)
class RegimeFilterStats:
    raw_buy_signals: int
    allowed_bull: int
    blocked_bear: int
    blocked_sideways: int
    unavailable: int


@dataclass
class DonchianBullRegimeStrategy:
    """Only a causal regime lookup is exposed; future regime points are not passed in."""

    regime_at: Callable[[datetime], MarketRegime | None] = field(repr=False)
    _donchian: DonchianBreakoutStrategy = field(default_factory=DonchianBreakoutStrategy,
                                                  init=False, repr=False)
    _raw: int = field(default=0, init=False, repr=False)
    _bull: int = field(default=0, init=False, repr=False)
    _bear: int = field(default=0, init=False, repr=False)
    _sideways: int = field(default=0, init=False, repr=False)
    _unavailable: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.regime_at):
            raise ValueError("A causal regime lookup is required")

    @property
    def minimum_candles(self) -> int:
        return self._donchian.minimum_candles

    @property
    def position_size_pct(self) -> Decimal:
        return self._donchian.position_size_pct

    @property
    def name(self) -> str:
        return "Donchian BULL Regime"

    @property
    def parameters(self) -> tuple[tuple[str, str], ...]:
        return (*self._donchian.parameters, ("entry_regime", "BULL"))

    @property
    def filter_stats(self) -> RegimeFilterStats:
        return RegimeFilterStats(self._raw, self._bull, self._bear,
                                 self._sideways, self._unavailable)

    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        signal = self._donchian.on_candle(history, portfolio)
        if signal.action != "BUY":
            return signal  # SELL never depends on the regime.
        self._raw += 1
        try:
            # The signal exists only when the current 15m candle is fully closed.
            regime = self.regime_at(history[-1].close_time_utc)
        except RegimeUnavailable:
            regime = None
        if type(regime) is MarketRegime and regime is MarketRegime.BULL:
            self._bull += 1
            return signal
        if type(regime) is MarketRegime and regime is MarketRegime.BEAR:
            self._bear += 1
        elif type(regime) is MarketRegime and regime is MarketRegime.SIDEWAYS:
            self._sideways += 1
        else:
            self._unavailable += 1
        return Signal()
