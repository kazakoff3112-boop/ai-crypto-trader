"""Buy & Hold comparison baseline; never used by live or PAPER execution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from ..market_data.candle import Candle
from .engine import BacktestConfig, BacktestEngine, BacktestResult
from .strategy import PortfolioSnapshot, Signal


@dataclass(frozen=True)
class _BuyAndHoldBenchmark:
    name: str = "Buy & Hold benchmark"
    parameters: tuple[tuple[str, str], ...] = (("entry", "first next-open"),)

    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        if len(history) == 1:
            return Signal("BUY", budget_usdt=min(portfolio.cash, portfolio.equity))
        return Signal()


def run_buy_and_hold(candles: Sequence[Candle], *, symbol: str, interval: str,
                     config: BacktestConfig | None = None,
                     initial_capital: Decimal = Decimal("1000")) -> BacktestResult:
    """Buy at candle 2 open with the same fee/slippage; hold through final close."""
    return BacktestEngine().run(candles, _BuyAndHoldBenchmark(), symbol=symbol,
                                interval=interval, config=config, initial_capital=initial_capital)
