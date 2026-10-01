"""Read-only attribution of existing backtest fills to known hourly regimes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, getcontext

from ..market_data.candle import INTERVAL_MINUTES
from ..market_data.market_regime import MarketRegime, MarketRegimeReport, REGIMES, RegimeUnavailable
from .analytics import CompletedTrade, OpenBacktestPosition, analyze_backtest
from .engine import BacktestResult, BacktestTrade

ZERO = Decimal("0")
HUNDRED = Decimal("100")


class RegimeAttributionUnavailable(ValueError):
    """A completed trade cannot be attributed without trustworthy regime data."""


@dataclass(frozen=True)
class RegimeAttributedTrade:
    trade: CompletedTrade
    entry_signal_time_utc: datetime
    entry_execution_time_utc: datetime
    exit_signal_time_utc: datetime
    exit_execution_time_utc: datetime
    entry_signal_regime: MarketRegime
    entry_execution_regime: MarketRegime
    exit_signal_regime: MarketRegime
    exit_execution_regime: MarketRegime


@dataclass(frozen=True)
class RegimeAttributedOpenPosition:
    position: OpenBacktestPosition
    entry_signal_time_utc: datetime
    entry_execution_time_utc: datetime
    entry_signal_regime: MarketRegime
    entry_execution_regime: MarketRegime


@dataclass(frozen=True)
class EntryRegimeStats:
    regime: MarketRegime
    completed_trades: int
    entry_share_percent: Decimal
    winning_trades: int
    losing_trades: int
    win_rate_percent: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    profit_factor: Decimal | None
    average_net_pnl: Decimal | None
    median_net_pnl: Decimal | None
    average_winner: Decimal | None
    average_loser: Decimal | None
    total_net_pnl: Decimal
    pnl_before_costs: Decimal
    commissions: Decimal
    slippage: Decimal
    total_costs: Decimal
    average_holding_time: timedelta | None
    total_turnover: Decimal


@dataclass(frozen=True)
class RegimeTradeTransition:
    entry_signal_regime: MarketRegime
    exit_signal_regime: MarketRegime
    trade_count: int
    total_net_pnl: Decimal


@dataclass(frozen=True)
class RegimeTradeAttribution:
    completed_trades: tuple[RegimeAttributedTrade, ...]
    groups: tuple[EntryRegimeStats, ...]
    transitions: tuple[RegimeTradeTransition, ...]
    open_position: RegimeAttributedOpenPosition | None
    total_completed_net_pnl: Decimal

    def group_for(self, regime: MarketRegime) -> EntryRegimeStats:
        return next(group for group in self.groups if group.regime == regime)

    def transition_for(self, entry: MarketRegime, exit_: MarketRegime) -> RegimeTradeTransition:
        return next(item for item in self.transitions
                    if item.entry_signal_regime == entry and item.exit_signal_regime == exit_)


def _tolerance(result: BacktestResult) -> Decimal:
    scale = max(abs(result.initial_capital), abs(result.final_equity), Decimal("1"))
    return Decimal("10").scaleb(scale.adjusted() - getcontext().prec + 3) * Decimal(
        max(1, len(result.ledger))
    )


def _require_near(actual: Decimal, expected: Decimal, tolerance: Decimal) -> None:
    if abs(actual - expected) > tolerance:
        raise RegimeAttributionUnavailable("Regime trade accounting does not reconcile")


def _fill_regimes(fill: BacktestTrade, report: MarketRegimeReport,
                  step: timedelta) -> tuple[datetime, MarketRegime, MarketRegime]:
    # The ledger stores the signal candle OPEN. Its CLOSE is the decision time.
    signal_time = fill.signal_candle_timestamp_utc + step
    if signal_time != fill.execution_candle_timestamp_utc or fill.timestamp_utc != signal_time:
        raise RegimeAttributionUnavailable("Backtest fill is not next-open execution")
    try:
        signal_regime = report.regime_at(signal_time)
        execution_regime = report.regime_at(fill.timestamp_utc)
    except RegimeUnavailable:
        raise RegimeAttributionUnavailable("Regime timestamp is invalid") from None
    if signal_regime is None or execution_regime is None:
        raise RegimeAttributionUnavailable("Hourly regime is unavailable at a trade timestamp")
    return signal_time, signal_regime, execution_regime


def _group(regime: MarketRegime, trades: tuple[RegimeAttributedTrade, ...],
           total_completed: int, tolerance: Decimal) -> EntryRegimeStats:
    selected = [item.trade for item in trades if item.entry_signal_regime == regime]
    values = sorted(item.net_pnl_after_costs for item in selected)
    wins = [value for value in values if value > ZERO]
    losses = [value for value in values if value < ZERO]
    count = len(values)
    profit = sum(wins, ZERO)
    loss = -sum(losses, ZERO)
    commissions = sum((item.entry_commission + item.exit_commission for item in selected), ZERO)
    slippage = sum((item.entry_slippage + item.exit_slippage for item in selected), ZERO)
    costs = commissions + slippage
    pnl_before = sum((item.gross_pnl_before_costs for item in selected), ZERO)
    net = sum(values, ZERO)
    _require_near(pnl_before - costs, net, tolerance)
    durations = [item.holding_time for item in selected]
    median = None
    if count:
        median = (values[count // 2] if count % 2 else
                  (values[count // 2 - 1] + values[count // 2]) / Decimal("2"))
    return EntryRegimeStats(
        regime, count, Decimal(count) / Decimal(total_completed) * HUNDRED if total_completed else ZERO,
        len(wins), len(losses), Decimal(len(wins)) / Decimal(count) * HUNDRED if count else ZERO,
        profit, loss, profit / loss if loss > ZERO else None,
        net / Decimal(count) if count else None,
        median,
        profit / Decimal(len(wins)) if wins else None,
        -loss / Decimal(len(losses)) if losses else None,
        net, pnl_before, commissions, slippage, costs,
        sum(durations, timedelta()) / count if count else None,
        sum((item.gross_entry_notional + item.gross_exit_notional for item in selected), ZERO),
    )


def attribute_backtest_by_regime(result: BacktestResult, report: MarketRegimeReport,
                                 *, interval: str) -> RegimeTradeAttribution:
    """Group existing fills only; never re-run a strategy or recompute a regime."""
    if result.symbol != "BTCUSDT" or not isinstance(report, MarketRegimeReport) or not report.points:
        raise RegimeAttributionUnavailable("Regime attribution requires BTCUSDT hourly data")
    if interval not in INTERVAL_MINUTES:
        raise RegimeAttributionUnavailable("Unsupported backtest interval")
    step = timedelta(minutes=INTERVAL_MINUTES[interval])
    quality = analyze_backtest(result)
    tolerance = _tolerance(result)
    fills = result.ledger
    if len(fills) != 2 * quality.completed_trade_count + (quality.open_position is not None):
        raise RegimeAttributionUnavailable("Backtest fill pairing is inconsistent")
    attributed: list[RegimeAttributedTrade] = []
    for index, completed in enumerate(quality.completed_trades):
        buy, sell = fills[2 * index:2 * index + 2]
        if (buy.side != "BUY" or sell.side != "SELL"
                or completed.entry_timestamp_utc != buy.timestamp_utc
                or completed.exit_timestamp_utc != sell.timestamp_utc):
            raise RegimeAttributionUnavailable("Backtest fill pairing is inconsistent")
        entry_time, entry_signal, entry_execution = _fill_regimes(buy, report, step)
        exit_time, exit_signal, exit_execution = _fill_regimes(sell, report, step)
        attributed.append(RegimeAttributedTrade(
            completed, entry_time, buy.timestamp_utc, exit_time, sell.timestamp_utc,
            entry_signal, entry_execution, exit_signal, exit_execution,
        ))
    open_position = None
    if quality.open_position is not None:
        buy = fills[-1]
        if buy.side != "BUY" or buy.timestamp_utc != quality.open_position.entry_timestamp_utc:
            raise RegimeAttributionUnavailable("Open backtest position is inconsistent")
        entry_time, entry_signal, entry_execution = _fill_regimes(buy, report, step)
        open_position = RegimeAttributedOpenPosition(
            quality.open_position, entry_time, buy.timestamp_utc,
            entry_signal, entry_execution,
        )
    completed_items = tuple(attributed)
    groups = tuple(_group(regime, completed_items, len(completed_items), tolerance)
                   for regime in REGIMES)
    transitions = tuple(RegimeTradeTransition(
        entry, exit_,
        sum(item.entry_signal_regime == entry and item.exit_signal_regime == exit_
            for item in completed_items),
        sum((item.trade.net_pnl_after_costs for item in completed_items
             if item.entry_signal_regime == entry and item.exit_signal_regime == exit_), ZERO),
    ) for entry in REGIMES for exit_ in REGIMES)
    expected_net = sum((trade.net_pnl_after_costs for trade in quality.completed_trades), ZERO)
    grouped_net = sum((group.total_net_pnl for group in groups), ZERO)
    matrix_net = sum((item.total_net_pnl for item in transitions), ZERO)
    expected_commissions = sum((trade.entry_commission + trade.exit_commission
                                for trade in quality.completed_trades), ZERO)
    expected_slippage = sum((trade.entry_slippage + trade.exit_slippage
                             for trade in quality.completed_trades), ZERO)
    _require_near(grouped_net, expected_net, tolerance)
    _require_near(matrix_net, expected_net, tolerance)
    _require_near(sum((group.commissions for group in groups), ZERO), expected_commissions, tolerance)
    _require_near(sum((group.slippage for group in groups), ZERO), expected_slippage, tolerance)
    if (sum(group.completed_trades for group in groups) != len(completed_items)
            or sum(item.trade_count for item in transitions) != len(completed_items)):
        raise RegimeAttributionUnavailable("Regime trade counts do not reconcile")
    return RegimeTradeAttribution(completed_items, groups, transitions, open_position, expected_net)
