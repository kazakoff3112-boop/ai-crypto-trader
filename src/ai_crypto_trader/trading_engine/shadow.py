"""One-shot public-market observer; never opens or mutates a Portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

from ..backtesting.strategy import PortfolioSnapshot
from ..database.shadow_store import (
    HOUR, STRATEGY_VERSION, ShadowStatus, ShadowStore, ShadowUnavailable, new_decision,
)
from ..market_data.historical import HistoricalMarketDataClient
from ..market_data.market_regime import MarketRegimeAnalyzer, RegimeUnavailable
from ..strategies.regime_bull_hold import RegimeBullHoldStrategy

REQUIRED_HOURS = RegimeBullHoldStrategy.minimum_candles - 1  # 224 prior + current.


@dataclass(frozen=True)
class ShadowRunResult:
    processed: bool
    action: str
    status: ShadowStatus


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
        previous, current = report.points[-2:]
        if previous.effective_time_utc != end - HOUR or current.effective_time_utc != end:
            raise RegimeUnavailable("Latest completed regime points do not match candle closes")
        # Decimal fields in this immutable snapshot only adapt the existing
        # strategy's signal interface. They are not balances or an execution.
        is_long = status.state == "LONG"
        shadow_view = PortfolioSnapshot(
            "BTCUSDT", Decimal("1"), Decimal("1") if is_long else Decimal("0"),
            Decimal("0"), Decimal("0"), Decimal("1"),
        )
        signal = RegimeBullHoldStrategy(report.regime_at).on_candle(tuple(candles), shadow_view)
        if signal.action == "BUY":
            action, after, reason = "BUY_SIGNAL", "LONG", "non-BULL -> BULL"
        elif signal.action == "SELL":
            action, after, reason = "SELL_SIGNAL", "FLAT", "BULL -> non-BULL"
        elif signal.action == "HOLD":
            action, after, reason = "HOLD", status.state, "No qualifying regime transition"
        else:
            raise ShadowUnavailable("Unknown strategy signal")
        decision = new_decision(
            candle=candles[-1], decision_utc=now, previous_regime=previous.regime,
            current_regime=current.regime, action=action, reason=reason,
            sma50=current.sma50, sma200=current.sma200,
            previous_sma200=current.previous_sma200,
            state_before=status.state, state_after=after,
        )
        inserted, updated = self._store.append(decision)
        return ShadowRunResult(inserted, action if inserted else "ALREADY_PROCESSED", updated)
