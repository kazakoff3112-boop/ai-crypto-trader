"""Read-only accounting diagnostics for long-only, non-pyramiding backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, getcontext

from ..safety.guard import TradingRequest, validate_request
from .engine import BacktestResult, BacktestTrade

ZERO = Decimal("0")
HUNDRED = Decimal("100")


class BacktestAnalyticsError(ValueError):
    """The simulated ledger and result cannot be reconciled safely."""


@dataclass(frozen=True)
class CompletedTrade:
    symbol: str
    entry_timestamp_utc: datetime
    exit_timestamp_utc: datetime
    holding_time: timedelta
    entry_quantity: Decimal
    entry_execution_price: Decimal
    exit_execution_price: Decimal
    gross_entry_notional: Decimal
    gross_exit_notional: Decimal
    entry_commission: Decimal
    exit_commission: Decimal
    entry_slippage: Decimal
    exit_slippage: Decimal
    total_trading_costs: Decimal
    gross_pnl_before_costs: Decimal
    net_pnl_after_costs: Decimal
    return_on_allocated_capital_percent: Decimal


@dataclass(frozen=True)
class OpenBacktestPosition:
    symbol: str
    entry_timestamp_utc: datetime
    quantity: Decimal
    entry_execution_price: Decimal
    final_close_price: Decimal
    unrealized_pnl: Decimal
    gross_unrealized_pnl_before_costs: Decimal


@dataclass(frozen=True)
class BacktestTradeAnalytics:
    completed_trades: tuple[CompletedTrade, ...]
    open_position: OpenBacktestPosition | None
    completed_trade_count: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate_percent: Decimal
    loss_rate_percent: Decimal
    average_net_pnl: Decimal | None
    median_net_pnl: Decimal | None
    average_winner: Decimal | None
    average_loser: Decimal | None
    largest_winner: Decimal | None
    largest_loser: Decimal | None
    gross_profit: Decimal
    gross_loss: Decimal
    profit_factor: Decimal | None
    average_holding_time: timedelta | None
    minimum_holding_time: timedelta | None
    maximum_holding_time: timedelta | None
    total_traded_notional: Decimal
    total_commissions: Decimal
    total_slippage: Decimal
    total_trading_costs: Decimal
    strategy_pnl_after_costs: Decimal
    strategy_pnl_before_costs: Decimal
    return_after_costs_percent: Decimal
    approximate_return_before_costs_percent: Decimal
    commission_drag: Decimal
    slippage_drag: Decimal
    combined_cost_drag: Decimal


def _tolerance(result: BacktestResult) -> Decimal:
    scale = max(abs(result.initial_capital), abs(result.final_equity), Decimal("1"))
    return Decimal("10").scaleb(scale.adjusted() - getcontext().prec + 3) * Decimal(
        max(1, len(result.ledger))
    )


def _near(actual: Decimal, expected: Decimal, tolerance: Decimal, name: str) -> None:
    if not isinstance(actual, Decimal) or not actual.is_finite() or abs(actual - expected) > tolerance:
        raise BacktestAnalyticsError(f"Backtest accounting mismatch: {name}")


def _validate_fill(fill: BacktestTrade, symbol: str, previous: datetime | None,
                   tolerance: Decimal) -> None:
    if not isinstance(fill, BacktestTrade) or fill.symbol != symbol or fill.side not in {"BUY", "SELL"}:
        raise BacktestAnalyticsError("Unknown backtest fill")
    if (not isinstance(fill.timestamp_utc, datetime)
            or not isinstance(fill.signal_candle_timestamp_utc, datetime)
            or not isinstance(fill.execution_candle_timestamp_utc, datetime)
            or fill.timestamp_utc.tzinfo is None or fill.timestamp_utc.utcoffset() != timedelta(0)
            or fill.signal_candle_timestamp_utc.tzinfo is None
            or fill.signal_candle_timestamp_utc.utcoffset() != timedelta(0)
            or fill.execution_candle_timestamp_utc != fill.timestamp_utc
            or fill.signal_candle_timestamp_utc >= fill.timestamp_utc
            or (previous is not None and fill.timestamp_utc <= previous)):
        raise BacktestAnalyticsError("Backtest fill timestamps are inconsistent")
    numbers = (fill.quantity, fill.execution_price, fill.gross_value, fill.commission,
               fill.slippage_cost, fill.realized_pnl, fill.portfolio_equity_after_trade)
    if any(not isinstance(value, Decimal) or not value.is_finite() for value in numbers):
        raise BacktestAnalyticsError("Invalid backtest fill number")
    if (fill.quantity <= ZERO or fill.execution_price <= ZERO or fill.gross_value <= ZERO
            or fill.commission < ZERO or fill.slippage_cost < ZERO):
        raise BacktestAnalyticsError("Invalid backtest fill value")
    _near(fill.gross_value, fill.quantity * fill.execution_price, tolerance, "fill gross value")
    if fill.side == "BUY" and abs(fill.realized_pnl) > tolerance:
        raise BacktestAnalyticsError("BUY cannot realize P&L")


def analyze_backtest(result: BacktestResult) -> BacktestTradeAnalytics:
    """Rebuild round trips and independently reconcile marked-to-market P&L."""
    if (not isinstance(result, BacktestResult)
            or not isinstance(result.initial_capital, Decimal)
            or not result.initial_capital.is_finite() or result.initial_capital <= ZERO
            or not isinstance(result.final_equity, Decimal)
            or not result.final_equity.is_finite()):
        raise BacktestAnalyticsError("Invalid backtest result")
    try:
        if validate_request(TradingRequest(symbol=result.symbol, category="spot")).symbol != result.symbol:
            raise BacktestAnalyticsError("Invalid backtest symbol")
    except (AttributeError, ValueError):
        raise BacktestAnalyticsError("Invalid backtest symbol") from None
    if not isinstance(result.ledger, tuple) or not all(isinstance(fill, BacktestTrade) for fill in result.ledger):
        raise BacktestAnalyticsError("Invalid backtest ledger")
    tolerance = _tolerance(result)
    if (result.ledger_entries != len(result.ledger)
            or result.buy_operations != sum(t.side == "BUY" for t in result.ledger)
            or result.sell_operations != sum(t.side == "SELL" for t in result.ledger)):
        raise BacktestAnalyticsError("Backtest fill counts are inconsistent")
    if not result.equity_curve:
        raise BacktestAnalyticsError("Backtest equity curve is empty")

    cash = result.initial_capital
    open_buy: BacktestTrade | None = None
    closed: list[CompletedTrade] = []
    commissions = ZERO
    slippage = ZERO
    turnover = ZERO
    last_timestamp: datetime | None = None
    for fill in result.ledger:
        _validate_fill(fill, result.symbol, last_timestamp, tolerance)
        last_timestamp = fill.timestamp_utc
        commissions += fill.commission
        slippage += fill.slippage_cost
        turnover += fill.gross_value
        if fill.side == "BUY":
            if open_buy is not None:
                raise BacktestAnalyticsError("Pyramiding is unsupported by round-trip analytics")
            cash -= fill.gross_value + fill.commission
            if cash < -tolerance:
                raise BacktestAnalyticsError("Backtest BUY creates negative cash")
            open_buy = fill
            continue
        if open_buy is None or fill.quantity != open_buy.quantity:
            raise BacktestAnalyticsError("SELL does not close exactly one long position")
        cash += fill.gross_value - fill.commission
        entry_cost = open_buy.gross_value + open_buy.commission
        net = fill.gross_value - fill.commission - entry_cost
        gross_before = fill.gross_value - open_buy.gross_value + (
            open_buy.slippage_cost + fill.slippage_cost
        )
        costs = open_buy.commission + fill.commission + open_buy.slippage_cost + fill.slippage_cost
        _near(gross_before - costs, net, tolerance, "completed trade costs")
        _near(fill.realized_pnl, net, tolerance, "realized trade P&L")
        closed.append(CompletedTrade(
            result.symbol, open_buy.timestamp_utc, fill.timestamp_utc,
            fill.timestamp_utc - open_buy.timestamp_utc, fill.quantity,
            open_buy.execution_price, fill.execution_price,
            open_buy.gross_value, fill.gross_value, open_buy.commission, fill.commission,
            open_buy.slippage_cost, fill.slippage_cost, costs, gross_before, net,
            net / entry_cost * HUNDRED,
        ))
        open_buy = None

    for value, expected, name in (
        (result.final_cash, cash, "final cash"),
        (result.total_commissions, commissions, "commissions"),
        (result.total_slippage, slippage, "slippage"),
    ):
        _near(value, expected, tolerance, name)
    if not isinstance(result.final_close_price, Decimal) or not result.final_close_price.is_finite() or result.final_close_price <= ZERO:
        raise BacktestAnalyticsError("Invalid final close price")
    expected_quantity = open_buy.quantity if open_buy is not None else ZERO
    _near(result.final_quantity, expected_quantity, tolerance, "final quantity")
    _near(result.final_equity, cash + expected_quantity * result.final_close_price,
          tolerance, "final equity")
    _near(result.equity_curve[-1].equity, result.final_equity, tolerance, "equity curve")
    _near(result.total_pnl, result.final_equity - result.initial_capital,
          tolerance, "total P&L")
    _near(result.total_return_percent,
          result.total_pnl / result.initial_capital * HUNDRED,
          tolerance, "total return")

    open_position = None
    open_gross = ZERO
    expected_unrealized = ZERO
    if open_buy is not None:
        mark = open_buy.quantity * result.final_close_price
        expected_unrealized = mark - (open_buy.gross_value + open_buy.commission)
        open_gross = mark - open_buy.gross_value + open_buy.slippage_cost
        open_position = OpenBacktestPosition(
            result.symbol, open_buy.timestamp_utc, open_buy.quantity,
            open_buy.execution_price, result.final_close_price,
            expected_unrealized, open_gross,
        )
    _near(result.unrealized_pnl, expected_unrealized, tolerance, "unrealized P&L")
    completed_net = sum((trade.net_pnl_after_costs for trade in closed), ZERO)
    _near(result.realized_pnl, completed_net, tolerance, "realized P&L")
    _near(result.total_pnl, completed_net + expected_unrealized, tolerance,
          "realized + unrealized P&L")

    gross_before = sum((trade.gross_pnl_before_costs for trade in closed), ZERO) + open_gross
    costs = commissions + slippage
    _near(gross_before - costs, result.total_pnl, tolerance,
          "gross P&L less all costs")

    wins = [trade.net_pnl_after_costs for trade in closed if trade.net_pnl_after_costs > ZERO]
    losses = [trade.net_pnl_after_costs for trade in closed if trade.net_pnl_after_costs < ZERO]
    values = sorted(trade.net_pnl_after_costs for trade in closed)
    count = len(values)
    median = None if not count else (values[count // 2] if count % 2 else
                                     (values[count // 2 - 1] + values[count // 2]) / Decimal("2"))
    durations = [trade.holding_time for trade in closed]
    gross_profit = sum(wins, ZERO)
    gross_loss = -sum(losses, ZERO)
    return BacktestTradeAnalytics(
        tuple(closed), open_position, count, len(wins), len(losses), count - len(wins) - len(losses),
        Decimal(len(wins)) / Decimal(count) * HUNDRED if count else ZERO,
        Decimal(len(losses)) / Decimal(count) * HUNDRED if count else ZERO,
        sum(values, ZERO) / Decimal(count) if count else None,
        median,
        gross_profit / Decimal(len(wins)) if wins else None,
        -gross_loss / Decimal(len(losses)) if losses else None,
        max(wins) if wins else None,
        min(losses) if losses else None,
        gross_profit, gross_loss, gross_profit / gross_loss if gross_loss > ZERO else None,
        sum(durations, timedelta()) / count if count else None,
        min(durations) if durations else None,
        max(durations) if durations else None,
        turnover, commissions, slippage, costs,
        result.total_pnl, gross_before,
        result.total_return_percent, gross_before / result.initial_capital * HUNDRED,
        commissions, slippage, costs,
    )
