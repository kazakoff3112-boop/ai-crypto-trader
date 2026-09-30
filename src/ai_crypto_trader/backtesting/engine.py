"""Deterministic long-only spot backtest using only supplied closed Candle objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Sequence

from ..config import Settings
from ..market_data.candle import (
    Candle, HistoricalDataUnavailable, INTERVAL_MINUTES, epoch_ms, validate_candle,
)
from ..portfolio import Portfolio, PortfolioError
from ..safety.guard import TradingRequest, validate_request
from .periods import BacktestPeriod
from .strategy import PortfolioSnapshot, Signal, Strategy

ZERO = Decimal("0")
ONE = Decimal("1")
BPS = Decimal("10000")


class BacktestError(ValueError):
    """Untrustworthy inputs or an impossible simulated operation."""


@dataclass(frozen=True)
class BacktestConfig:
    fee_rate: Decimal = Decimal("0.001")
    slippage_bps: Decimal = Decimal("5")

    def __post_init__(self) -> None:
        if (not isinstance(self.fee_rate, Decimal) or not self.fee_rate.is_finite()
                or not ZERO <= self.fee_rate < ONE):
            raise BacktestError("Invalid backtest fee rate")
        if (not isinstance(self.slippage_bps, Decimal) or not self.slippage_bps.is_finite()
                or not ZERO <= self.slippage_bps < BPS):
            raise BacktestError("Invalid backtest slippage BPS")

    @classmethod
    def from_settings(cls, settings: Settings) -> "BacktestConfig":
        return cls(settings.spot_fee_rate, settings.paper_slippage_bps)


@dataclass(frozen=True)
class BacktestTrade:
    timestamp_utc: datetime
    symbol: str
    side: str
    signal_candle_timestamp_utc: datetime
    execution_candle_timestamp_utc: datetime
    quantity: Decimal
    execution_price: Decimal
    gross_value: Decimal
    commission: Decimal
    slippage_cost: Decimal
    realized_pnl: Decimal
    portfolio_equity_after_trade: Decimal


@dataclass(frozen=True)
class EquityPoint:
    timestamp_utc: datetime
    equity: Decimal


@dataclass(frozen=True)
class BacktestResult:
    initial_capital: Decimal
    final_equity: Decimal
    total_pnl: Decimal
    total_return_percent: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_commissions: Decimal
    total_slippage: Decimal
    buy_operations: int
    sell_operations: int
    max_equity: Decimal
    min_equity: Decimal
    max_drawdown_usdt: Decimal
    max_drawdown_percent: Decimal
    candles_processed: int
    ledger_entries: int
    equity_curve: tuple[EquityPoint, ...]
    ledger: tuple[BacktestTrade, ...]
    exposure_percent: Decimal
    strategy_name: str
    strategy_parameters: tuple[tuple[str, str], ...]
    symbol: str
    final_cash: Decimal
    final_quantity: Decimal
    final_close_price: Decimal
    evaluation_start: datetime
    evaluation_end: datetime
    warmup_start: datetime | None
    warmup_candles: int
    evaluation_candles: int


class BacktestEngine:
    def run(self, candles: Sequence[Candle], strategy: Strategy, *, symbol: str,
            interval: str, initial_capital: Decimal = Decimal("1000"),
            config: BacktestConfig | None = None, category: str = "spot",
            as_of_utc: datetime | None = None,
            period: BacktestPeriod | None = None) -> BacktestResult:
        safe = validate_request(TradingRequest(symbol=symbol, category=category))
        config = config if config is not None else BacktestConfig()
        if not isinstance(config, BacktestConfig):
            raise BacktestError("Invalid backtest configuration")
        if (not isinstance(initial_capital, Decimal) or not initial_capital.is_finite()
                or initial_capital <= ZERO):
            raise BacktestError("Initial capital must be a positive Decimal")
        if interval not in INTERVAL_MINUTES:
            raise BacktestError("Unsupported backtest interval")
        now = as_of_utc if as_of_utc is not None else datetime.now(timezone.utc)
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() != timedelta(0):
            raise BacktestError("Backtest clock must be UTC")
        history = tuple(candles)
        if len(history) < 2:
            raise BacktestError("At least two completed candles are required")
        minimum_candles = getattr(strategy, "minimum_candles", 2)
        if type(minimum_candles) is not int or minimum_candles < 2 or len(history) < minimum_candles:
            raise BacktestError("Insufficient candles for this strategy")
        step_ms = INTERVAL_MINUTES[interval] * 60_000
        previous_ms: int | None = None
        for candle in history:
            if not isinstance(candle, Candle) or candle.symbol != safe.symbol or candle.interval != interval:
                raise BacktestError("Candle symbol or interval mismatch")
            try:
                validate_candle(candle)
                ms = epoch_ms(candle.open_time_utc)
            except (HistoricalDataUnavailable, ValueError, AttributeError, TypeError):
                raise BacktestError("Invalid historical candle") from None
            if previous_ms is not None and ms != previous_ms + step_ms:
                raise BacktestError("Candles contain a gap, duplicate or ordering error")
            if candle.close_time_utc > now:
                raise BacktestError("Unclosed candle is forbidden")
            previous_ms = ms

        if period is None:
            period = BacktestPeriod.create(history[0].open_time_utc,
                                           history[-1].close_time_utc, interval)
        else:
            if not isinstance(period, BacktestPeriod) or period.interval != interval:
                raise BacktestError("Invalid evaluation period")
            try:
                expected_period = BacktestPeriod.create(
                    period.evaluation_start, period.evaluation_end, interval,
                    warmup_candles=period.warmup_candles,
                )
            except (ValueError, AttributeError, TypeError):
                raise BacktestError("Invalid evaluation period") from None
            if (period != expected_period
                    or period.evaluation_candles < 2
                    or period.warmup_candles != minimum_candles - 2
                    or len(history) != period.warmup_candles + period.evaluation_candles
                    or history[0].open_time_utc != (period.warmup_start or period.evaluation_start)
                    or history[period.warmup_candles].open_time_utc != period.evaluation_start
                    or history[-1].close_time_utc != period.evaluation_end):
                raise BacktestError("Candles do not exactly match evaluation and warm-up periods")

        # Never call Portfolio.open(): this in-memory instance has a fresh Ledger.
        portfolio = Portfolio(Settings(spot_fee_rate=config.fee_rate), initial_cash=initial_capital)
        trades: list[BacktestTrade] = []
        curve: list[EquityPoint] = []
        pending: tuple[Signal, Candle] | None = None
        peak = initial_capital
        maximum = initial_capital
        minimum = initial_capital
        max_drawdown = ZERO
        max_drawdown_pct = ZERO
        exposed_candles = 0

        for index in range(period.warmup_candles, len(history)):
            candle = history[index]
            if pending is not None:
                signal, signal_candle = pending
                price = candle.open * (ONE + config.slippage_bps / BPS) if signal.action == "BUY" else (
                    candle.open * (ONE - config.slippage_bps / BPS))
                quantity = signal.quantity
                if signal.action == "BUY" and signal.budget_usdt is not None:
                    budget = min(signal.budget_usdt, portfolio.cash)
                    quantity = budget / (price * (ONE + config.fee_rate))
                    while quantity * price * (ONE + config.fee_rate) > budget:
                        quantity = quantity.next_minus()
                    if quantity <= ZERO:
                        raise BacktestError("Backtest BUY budget is too small")
                slippage = quantity * abs(price - candle.open)
                try:
                    if signal.action == "BUY":
                        entry = portfolio.buy(safe.symbol, quantity, price)
                    else:
                        entry = portfolio.sell(safe.symbol, quantity, price)
                except PortfolioError:
                    raise BacktestError("Insufficient virtual balance or asset for backtest trade") from None
                # Price already includes slippage. Never debit the cost a second time.
                equity_after = portfolio.equity({safe.symbol: candle.open})
                trades.append(BacktestTrade(
                    candle.open_time_utc, safe.symbol, signal.action,
                    signal_candle.open_time_utc, candle.open_time_utc, quantity,
                    price, entry.gross_value, entry.commission, slippage,
                    entry.realized_pnl, equity_after,
                ))
                pending = None

            equity = portfolio.equity({safe.symbol: candle.close})
            if portfolio.quantity(safe.symbol) > ZERO:
                exposed_candles += 1
            curve.append(EquityPoint(candle.close_time_utc, equity))
            maximum = max(maximum, equity)
            minimum = min(minimum, equity)
            peak = max(peak, equity)
            drawdown = peak - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            drawdown_pct = drawdown / peak * Decimal("100")
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct

            view = PortfolioSnapshot(safe.symbol, portfolio.cash,
                                     portfolio.quantity(safe.symbol),
                                     portfolio.average_entry_price(safe.symbol),
                                     portfolio.realized_pnl, equity)
            signal = strategy.on_candle(history[:index + 1], view)
            self._validate_signal(signal)
            if signal.action != "HOLD" and index + 1 < len(history):
                pending = (signal, candle)

        final = curve[-1].equity
        pnl = final - initial_capital
        return BacktestResult(
            initial_capital, final, pnl, pnl / initial_capital * Decimal("100"),
            portfolio.realized_pnl, portfolio.unrealized_pnl(safe.symbol, history[-1].close),
            portfolio.total_commission, sum((trade.slippage_cost for trade in trades), ZERO),
            sum(trade.side == "BUY" for trade in trades),
            sum(trade.side == "SELL" for trade in trades),
            maximum, minimum, max_drawdown, max_drawdown_pct,
            period.evaluation_candles, len(trades),
            tuple(curve), tuple(trades),
            Decimal(exposed_candles) / Decimal(period.evaluation_candles) * Decimal("100"),
            getattr(strategy, "name", type(strategy).__name__),
            getattr(strategy, "parameters", ()),
            safe.symbol, portfolio.cash, portfolio.quantity(safe.symbol), history[-1].close,
            period.evaluation_start, period.evaluation_end,
            period.warmup_start, period.warmup_candles, period.evaluation_candles,
        )

    @staticmethod
    def _validate_signal(signal: Signal) -> None:
        if not isinstance(signal, Signal) or signal.action not in {"HOLD", "BUY", "SELL"}:
            raise BacktestError("Unknown backtest signal")
        if signal.action == "HOLD":
            if signal.quantity is not None or signal.budget_usdt is not None:
                raise BacktestError("HOLD cannot carry quantity or budget")
        elif signal.action == "SELL":
            if signal.budget_usdt is not None or not BacktestEngine._positive_decimal(signal.quantity):
                raise BacktestError("SELL quantity must be a positive Decimal")
        elif ((signal.quantity is None) == (signal.budget_usdt is None)
              or not BacktestEngine._positive_decimal(
                  signal.quantity if signal.quantity is not None else signal.budget_usdt
              )):
            raise BacktestError("BUY requires exactly one positive Decimal quantity or USDT budget")

    @staticmethod
    def _positive_decimal(value: object) -> bool:
        return isinstance(value, Decimal) and value.is_finite() and value > ZERO
