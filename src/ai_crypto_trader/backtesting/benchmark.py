"""Offline allocation-matched, full-allocation and cash benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from ..market_data.candle import Candle
from .engine import BacktestConfig, BacktestEngine, BacktestError, BacktestResult
from .strategy import PortfolioSnapshot, Signal


@dataclass(frozen=True)
class _BuyAndHoldBenchmark:
    allocation_pct: Decimal = Decimal("100")
    name: str = "Buy & Hold benchmark"
    parameters: tuple[tuple[str, str], ...] = (("entry", "first next-open"),)

    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        if len(history) == 1:
            return Signal("BUY", budget_usdt=min(portfolio.cash, portfolio.equity)
                          * self.allocation_pct / Decimal("100"))
        return Signal()


@dataclass(frozen=True)
class _CashBenchmark:
    name: str = "Cash benchmark"
    parameters: tuple[tuple[str, str], ...] = (("allocation_pct", "0"),)

    def on_candle(self, history: tuple[Candle, ...], portfolio: PortfolioSnapshot) -> Signal:
        return Signal()


def run_buy_and_hold(candles: Sequence[Candle], *, symbol: str, interval: str,
                     config: BacktestConfig | None = None,
                     initial_capital: Decimal = Decimal("1000"),
                     allocation_pct: Decimal = Decimal("100")) -> BacktestResult:
    """Invest the specified initial cash share at candle 2 open; never rebalance."""
    if (not isinstance(allocation_pct, Decimal) or not allocation_pct.is_finite()
            or not Decimal("0") < allocation_pct <= Decimal("100")):
        raise BacktestError("Buy & Hold allocation must be a Decimal in (0, 100]")
    benchmark = _BuyAndHoldBenchmark(
        allocation_pct, parameters=(("entry", "first next-open"),
                                    ("allocation_pct", str(allocation_pct))),
    )
    return BacktestEngine().run(candles, benchmark, symbol=symbol,
                                interval=interval, config=config, initial_capital=initial_capital)


def run_cash_benchmark(candles: Sequence[Candle], *, symbol: str, interval: str,
                       config: BacktestConfig | None = None,
                       initial_capital: Decimal = Decimal("1000")) -> BacktestResult:
    """Stay entirely in virtual USDT for the same evaluation candles."""
    return BacktestEngine().run(candles, _CashBenchmark(), symbol=symbol,
                                interval=interval, config=config, initial_capital=initial_capital)
