"""Offline candle model and validation; imports no transport or exchange client."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from ..safety.guard import TradingRequest, validate_request

INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}


class HistoricalDataUnavailable(RuntimeError):
    """Requested continuous, trustworthy candle history is unavailable."""


@dataclass(frozen=True)
class Candle:
    symbol: str
    interval: str
    open_time_utc: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    turnover: Decimal

    @property
    def close_time_utc(self) -> datetime:
        """Exclusive UTC end of this candle's interval."""
        return self.open_time_utc + timedelta(minutes=INTERVAL_MINUTES[self.interval])


def epoch_ms(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("A timezone-aware UTC datetime is required")
    return int(value.timestamp()) * 1000 + value.microsecond // 1000


def utc_from_ms(value: int) -> datetime:
    try:
        return datetime.fromtimestamp(value // 1000, tz=timezone.utc).replace(
            microsecond=(value % 1000) * 1000
        )
    except (ValueError, OverflowError, OSError):
        raise HistoricalDataUnavailable("Invalid candle timestamp") from None


def validate_range(symbol: str, category: str, interval: str, start: datetime,
                   end: datetime) -> tuple[str, int, int, int]:
    safe = validate_request(TradingRequest(symbol=symbol, category=category))
    if interval not in INTERVAL_MINUTES:
        raise ValueError("Unsupported historical interval")
    step = INTERVAL_MINUTES[interval] * 60_000
    if start.microsecond % 1000 or end.microsecond % 1000:
        raise ValueError("Historical range must have millisecond precision")
    start_ms, end_ms = epoch_ms(start), epoch_ms(end)
    if start_ms < 0 or end_ms <= start_ms or start_ms % step or end_ms % step:
        raise ValueError("Historical range must be positive and candle-aligned")
    if (end_ms - start_ms) // step > 100_000:
        raise ValueError("Historical range is too large")
    return safe.symbol, step, start_ms, end_ms


def validate_candle(candle: Candle) -> None:
    validate_request(TradingRequest(symbol=candle.symbol, category="spot"))
    if candle.interval not in INTERVAL_MINUTES:
        raise HistoricalDataUnavailable("Unsupported cached interval")
    step = INTERVAL_MINUTES[candle.interval] * 60_000
    if candle.open_time_utc.microsecond % 1000 or epoch_ms(candle.open_time_utc) % step:
        raise HistoricalDataUnavailable("Unaligned candle timestamp")
    for value in (candle.open, candle.high, candle.low, candle.close):
        if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
            raise HistoricalDataUnavailable("Invalid candle OHLC")
    for value in (candle.volume, candle.turnover):
        if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
            raise HistoricalDataUnavailable("Invalid candle volume or turnover")
    if candle.high < max(candle.open, candle.close, candle.low) or candle.low > min(
        candle.open, candle.close, candle.high
    ):
        raise HistoricalDataUnavailable("Inconsistent candle OHLC")
