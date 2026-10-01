"""Read-only BTCUSDT 1h market-regime diagnostics; never produces trade signals."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, DecimalException
from enum import Enum
from typing import Sequence

from .candle import Candle, HistoricalDataUnavailable, epoch_ms, validate_candle

ZERO = Decimal("0")
HUNDRED = Decimal("100")
HOUR = timedelta(hours=1)
FAST_REGIME_SMA = 50
SLOW_REGIME_SMA = 200
SLOW_SMA_SLOPE_LOOKBACK = 24
REQUIRED_CANDLES = SLOW_REGIME_SMA + SLOW_SMA_SLOPE_LOOKBACK
WARMUP_CANDLES = REQUIRED_CANDLES - 1


class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


REGIMES = (MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS)


class RegimeUnavailable(ValueError):
    """The hourly history or requested evaluation period is untrustworthy."""


@dataclass(frozen=True)
class RegimePoint:
    effective_time_utc: datetime
    symbol: str
    interval: str
    close: Decimal
    sma50: Decimal
    sma200: Decimal
    previous_sma200: Decimal
    regime: MarketRegime

    @property
    def slope_direction(self) -> str:
        if self.sma200 > self.previous_sma200:
            return "RISING"
        if self.sma200 < self.previous_sma200:
            return "FALLING"
        return "FLAT"


@dataclass(frozen=True)
class RegimeStats:
    regime: MarketRegime
    hours: int
    percentage: Decimal
    longest_run_hours: int
    average_next_hour_return: Decimal | None  # Fraction, not percent.
    next_hour_sample_count: int


@dataclass(frozen=True)
class RegimeTransition:
    source: MarketRegime
    destination: MarketRegime
    count: int


@dataclass(frozen=True)
class MarketRegimeReport:
    evaluation_start: datetime
    evaluation_end: datetime
    points: tuple[RegimePoint, ...]
    stats: tuple[RegimeStats, ...]
    transitions: tuple[RegimeTransition, ...]
    total_transitions: int

    @property
    def valid_classified_hours(self) -> int:
        return len(self.points)

    @property
    def first_valid_regime_timestamp(self) -> datetime:
        return self.points[0].effective_time_utc

    @property
    def last_regime_timestamp(self) -> datetime:
        return self.points[-1].effective_time_utc

    def stats_for(self, regime: MarketRegime) -> RegimeStats:
        return next(item for item in self.stats if item.regime == regime)

    def transition_count(self, source: MarketRegime, destination: MarketRegime) -> int:
        return next(item.count for item in self.transitions
                    if item.source == source and item.destination == destination)

    def point_at(self, timestamp: datetime) -> RegimePoint | None:
        """Latest completed 1h point known at a UTC timestamp, or unavailable."""
        _require_utc(timestamp)
        if timestamp < self.first_valid_regime_timestamp or timestamp >= self.evaluation_end + HOUR:
            return None
        index = bisect_right([point.effective_time_utc for point in self.points], timestamp) - 1
        return self.points[index] if index >= 0 else None

    def regime_at(self, timestamp: datetime) -> MarketRegime | None:
        """Never expose the regime of an hourly candle before its close."""
        point = self.point_at(timestamp)
        return point.regime if point is not None else None


def _require_utc(value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise RegimeUnavailable("A timezone-aware UTC timestamp is required")


def _validate_history(candles: Sequence[Candle], as_of_utc: datetime) -> tuple[Candle, ...]:
    _require_utc(as_of_utc)
    history = tuple(candles)
    previous_ms: int | None = None
    for candle in history:
        if not isinstance(candle, Candle) or candle.symbol != "BTCUSDT" or candle.interval != "1h":
            raise RegimeUnavailable("Only BTCUSDT 1h candles are supported")
        try:
            validate_candle(candle)
            current_ms = epoch_ms(candle.open_time_utc)
        except (HistoricalDataUnavailable, ValueError, AttributeError, TypeError, OverflowError, OSError):
            raise RegimeUnavailable("Invalid hourly candle") from None
        if previous_ms is not None and current_ms != previous_ms + 3_600_000:
            raise RegimeUnavailable("Hourly candles contain a gap, duplicate or ordering error")
        if candle.close_time_utc > as_of_utc:
            raise RegimeUnavailable("Unclosed hourly candle is forbidden")
        previous_ms = current_ms
    return history


def _sma(history: tuple[Candle, ...], end_index: int, period: int) -> Decimal:
    return sum((candle.close for candle in history[end_index - period + 1:end_index + 1]), ZERO) / Decimal(period)


def _classify(history: tuple[Candle, ...], index: int) -> RegimePoint:
    try:
        fast = _sma(history, index, FAST_REGIME_SMA)
        slow = _sma(history, index, SLOW_REGIME_SMA)
        previous_slow = _sma(history, index - SLOW_SMA_SLOPE_LOOKBACK, SLOW_REGIME_SMA)
    except DecimalException:
        raise RegimeUnavailable("Regime averages cannot be calculated") from None
    if any(not value.is_finite() or value <= ZERO for value in (fast, slow, previous_slow)):
        raise RegimeUnavailable("Regime averages are invalid")
    candle = history[index]
    if candle.close > slow and fast > slow and slow > previous_slow:
        regime = MarketRegime.BULL
    elif candle.close < slow and fast < slow and slow < previous_slow:
        regime = MarketRegime.BEAR
    else:
        regime = MarketRegime.SIDEWAYS
    return RegimePoint(candle.close_time_utc, candle.symbol, candle.interval,
                       candle.close, fast, slow, previous_slow, regime)


class MarketRegimeAnalyzer:
    """Pure calculation on supplied closed candles; no cache, API or Portfolio access."""

    def classify_latest(self, candles: Sequence[Candle], *,
                        as_of_utc: datetime | None = None) -> RegimePoint | None:
        now = datetime.now(timezone.utc) if as_of_utc is None else as_of_utc
        history = _validate_history(candles, now)
        if len(history) < REQUIRED_CANDLES:
            return None
        return _classify(history, len(history) - 1)

    def analyze(self, candles: Sequence[Candle], *, evaluation_start: datetime,
                evaluation_end: datetime,
                as_of_utc: datetime | None = None) -> MarketRegimeReport:
        _require_utc(evaluation_start)
        _require_utc(evaluation_end)
        if (evaluation_start >= evaluation_end or evaluation_start.microsecond
                or evaluation_end.microsecond or epoch_ms(evaluation_start) % 3_600_000
                or epoch_ms(evaluation_end) % 3_600_000):
            raise RegimeUnavailable("Evaluation period must be positive and hourly aligned")
        expected_hours = (epoch_ms(evaluation_end) - epoch_ms(evaluation_start)) // 3_600_000
        now = datetime.now(timezone.utc) if as_of_utc is None else as_of_utc
        history = _validate_history(candles, now)
        try:
            expected_first_open = evaluation_start - WARMUP_CANDLES * HOUR
        except OverflowError:
            raise RegimeUnavailable("Regime warm-up begins outside supported dates") from None
        if (len(history) != WARMUP_CANDLES + expected_hours
                or len(history) < REQUIRED_CANDLES
                or history[0].open_time_utc != expected_first_open
                or history[WARMUP_CANDLES].open_time_utc != evaluation_start
                or history[-1].close_time_utc != evaluation_end):
            raise RegimeUnavailable("Insufficient or mismatched regime warm-up/evaluation history")
        points = tuple(_classify(history, index)
                       for index in range(WARMUP_CANDLES, len(history)))
        counts = {regime: 0 for regime in REGIMES}
        longest = {regime: 0 for regime in REGIMES}
        return_sums = {regime: ZERO for regime in REGIMES}
        return_counts = {regime: 0 for regime in REGIMES}
        transition_counts = {(source, destination): 0 for source in REGIMES
                             for destination in REGIMES if source != destination}
        current_run_regime: MarketRegime | None = None
        current_run_length = 0
        for index, point in enumerate(points):
            counts[point.regime] += 1
            if point.regime == current_run_regime:
                current_run_length += 1
            else:
                if current_run_regime is not None:
                    transition_counts[(current_run_regime, point.regime)] += 1
                current_run_regime = point.regime
                current_run_length = 1
            longest[point.regime] = max(longest[point.regime], current_run_length)
            if index + 1 < len(points):
                try:
                    change = points[index + 1].close / point.close - Decimal("1")
                    return_sums[point.regime] += change
                except DecimalException:
                    raise RegimeUnavailable("Next-hour return cannot be calculated") from None
                if not return_sums[point.regime].is_finite():
                    raise RegimeUnavailable("Next-hour return is invalid")
                return_counts[point.regime] += 1
        try:
            stats = tuple(RegimeStats(
                regime, counts[regime], Decimal(counts[regime]) / Decimal(len(points)) * HUNDRED,
                longest[regime],
                return_sums[regime] / Decimal(return_counts[regime]) if return_counts[regime] else None,
                return_counts[regime],
            ) for regime in REGIMES)
        except DecimalException:
            raise RegimeUnavailable("Regime summary cannot be calculated") from None
        if any(not item.percentage.is_finite() or
               (item.average_next_hour_return is not None and
                not item.average_next_hour_return.is_finite()) for item in stats):
            raise RegimeUnavailable("Regime summary is invalid")
        transitions = tuple(RegimeTransition(source, destination, transition_counts[(source, destination)])
                            for source, destination in transition_counts)
        return MarketRegimeReport(evaluation_start, evaluation_end, points, stats,
                                  transitions, sum(transition_counts.values()))
