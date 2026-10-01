"""One-shot public-market observer; never opens or mutates a Portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

from ..backtesting.strategy import PortfolioSnapshot
from ..database.shadow_store import (
    HOUR, ShadowDecision, ShadowStatus, ShadowStore, ShadowUnavailable, new_decision,
)
from ..market_data.historical import HistoricalMarketDataClient
from ..market_data.market_regime import (
    WARMUP_CANDLES, MarketRegimeAnalyzer, MarketRegimeReport, RegimeUnavailable,
)
from ..strategies.regime_bull_hold import RegimeBullHoldStrategy

REQUIRED_HOURS = RegimeBullHoldStrategy.minimum_candles - 1  # 224 prior + current.
MAX_SHADOW_CATCHUP_HOURS = 72


@dataclass(frozen=True)
class ShadowRunResult:
    processed: bool
    action: str
    status: ShadowStatus


@dataclass(frozen=True)
class ShadowCatchUpPlan:
    initial_status: ShadowStatus
    latest_closed_candle: datetime
    decisions: tuple[ShadowDecision, ...]

    @property
    def missing_candles(self) -> int:
        return len(self.decisions)

    @property
    def first_missing_candle(self) -> datetime:
        return self.decisions[0].candle_close_utc

    @property
    def last_missing_candle(self) -> datetime:
        return self.decisions[-1].candle_close_utc

    def action_count(self, action: str) -> int:
        return sum(decision.action == action for decision in self.decisions)


def latest_closed_hour(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise ShadowUnavailable("Shadow clock must be UTC")
    return now.replace(minute=0, second=0, microsecond=0)


def health(status: ShadowStatus, now: datetime) -> str:
    last = status.last_candle_utc
    if last is None:
        return "NOT_READY: no shadow decision yet"
    difference = latest_closed_hour(now) - last
    if difference < timedelta(0):
        return "CLOCK_BEHIND: last decision is ahead of this clock"
    if difference > HOUR:
        return "GAP: controlled catch-up required"
    if difference == HOUR:
        return "DUE: one closed hour is ready"
    return "HEALTHY: latest closed hour processed"


def _shadow_snapshot(state: str) -> PortfolioSnapshot:
    # Decimal fields only adapt the existing frozen strategy interface. They
    # are not balances and cannot reach Portfolio or the Paper Engine.
    return PortfolioSnapshot(
        "BTCUSDT", Decimal("1"), Decimal("1") if state == "LONG" else Decimal("0"),
        Decimal("0"), Decimal("0"), Decimal("1"),
    )


def _decision(*, report: MarketRegimeReport, candles: tuple, state: str,
              decision_utc: datetime) -> ShadowDecision:
    previous, current = report.points[-2:]
    expected_close = candles[-1].close_time_utc
    if (previous.effective_time_utc != expected_close - HOUR
            or current.effective_time_utc != expected_close):
        raise RegimeUnavailable("Latest completed regime points do not match candle closes")
    signal = RegimeBullHoldStrategy(report.regime_at).on_candle(
        candles, _shadow_snapshot(state),
    )
    if signal.action == "BUY":
        action, after, reason = "BUY_SIGNAL", "LONG", "non-BULL -> BULL"
    elif signal.action == "SELL":
        action, after, reason = "SELL_SIGNAL", "FLAT", "BULL -> non-BULL"
    elif signal.action == "HOLD":
        action, after, reason = "HOLD", state, "No qualifying regime transition"
    else:
        raise ShadowUnavailable("Unknown strategy signal")
    return new_decision(
        candle=candles[-1], decision_utc=decision_utc,
        previous_regime=previous.regime, current_regime=current.regime,
        action=action, reason=reason, sma50=current.sma50, sma200=current.sma200,
        previous_sma200=current.previous_sma200,
        state_before=state, state_after=after,
    )


class ShadowRunner:
    def __init__(self, store: ShadowStore, market: HistoricalMarketDataClient,
                 *, clock: Callable[[], datetime] | None = None) -> None:
        self._store = store
        self._market = market
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def once(self) -> ShadowRunResult:
        now = self._clock()
        end = latest_closed_hour(now)
        status = self._store.status()
        if status.last_candle_utc == end:
            return ShadowRunResult(False, "ALREADY_PROCESSED", status)
        if status.last_candle_utc is not None and end != status.last_candle_utc + HOUR:
            raise ShadowUnavailable("Gap or clock reversal; controlled catch-up required")
        start = end - REQUIRED_HOURS * HOUR
        candles = self._market.fetch_candles("BTCUSDT", "1h", start, end,
                                              category="spot", completed_only=True)
        if len(candles) != REQUIRED_HOURS or candles[-1].close_time_utc != end:
            raise ShadowUnavailable("Required closed hourly candles are unavailable")
        report = MarketRegimeAnalyzer().analyze(
            candles, evaluation_start=end - 2 * HOUR, evaluation_end=end,
            as_of_utc=now,
        )
        decision = _decision(report=report, candles=tuple(candles), state=status.state,
                             decision_utc=now)
        inserted, updated = self._store.append(decision)
        return ShadowRunResult(
            inserted, decision.action if inserted else "ALREADY_PROCESSED", updated,
        )

    def catch_up_plan(self) -> ShadowCatchUpPlan:
        """Validate and simulate every missing hour without writing SQLite."""
        now = self._clock()
        latest = latest_closed_hour(now)
        status = self._store.status()
        if status.last_candle_utc is None or status.last_regime is None:
            raise ShadowUnavailable("No shadow baseline exists")
        if latest <= status.last_candle_utc:
            if latest == status.last_candle_utc:
                raise ShadowUnavailable("No missing closed shadow candles")
            raise ShadowUnavailable("Shadow clock is behind stored history")
        missing = int((latest - status.last_candle_utc) / HOUR)
        if missing > MAX_SHADOW_CATCHUP_HOURS:
            raise ShadowUnavailable("Catch-up range exceeds safety limit")

        evaluation_start = status.last_candle_utc - HOUR
        start = evaluation_start - WARMUP_CANDLES * HOUR
        candles = tuple(self._market.fetch_candles(
            "BTCUSDT", "1h", start, latest, category="spot", completed_only=True,
        ))
        report = MarketRegimeAnalyzer().analyze(
            candles, evaluation_start=evaluation_start, evaluation_end=latest,
            as_of_utc=now,
        )
        if (report.points[0].effective_time_utc != status.last_candle_utc
                or report.points[0].regime is not status.last_regime):
            raise ShadowUnavailable("Downloaded history conflicts with shadow baseline")

        candle_by_close = {candle.close_time_utc: candle for candle in candles}
        decisions: list[ShadowDecision] = []
        state = status.state
        for offset in range(1, missing + 1):
            close_time = status.last_candle_utc + offset * HOUR
            previous_candle = candle_by_close.get(close_time - HOUR)
            current_candle = candle_by_close.get(close_time)
            if previous_candle is None or current_candle is None:
                raise ShadowUnavailable("Missing candle in controlled catch-up range")
            partial = MarketRegimeReport(
                evaluation_start=report.evaluation_start,
                evaluation_end=close_time,
                points=tuple(point for point in report.points
                             if point.effective_time_utc <= close_time),
                stats=report.stats,
                transitions=report.transitions,
                total_transitions=report.total_transitions,
            )
            decision = _decision(
                report=partial, candles=(previous_candle, current_candle), state=state,
                decision_utc=now,
            )
            decisions.append(decision)
            state = decision.state_after
        return ShadowCatchUpPlan(status, latest, tuple(decisions))

    def execute_catch_up(self, plan: ShadowCatchUpPlan) -> ShadowStatus:
        """Commit an approved plan as one all-or-nothing SQLite transaction."""
        return self._store.append_batch(
            plan.decisions, expected_status=plan.initial_status,
        )
