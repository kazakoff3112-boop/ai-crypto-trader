"""Explicit UTC evaluation windows; warm-up is outside performance accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from ..market_data.candle import INTERVAL_MINUTES, epoch_ms


def parse_utc(value: str) -> datetime:
    """Accept ISO 8601 timestamps ending in Z or +00:00, never local time."""
    if not isinstance(value, str) or not (value.endswith("Z") or value.endswith("+00:00")):
        raise ValueError("Timestamp must end in Z or +00:00")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        raise ValueError("Invalid ISO 8601 UTC timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Timestamp must be UTC")
    return parsed


@dataclass(frozen=True)
class BacktestPeriod:
    evaluation_start: datetime
    evaluation_end: datetime
    warmup_start: datetime | None
    evaluation_candles: int
    warmup_candles: int
    interval: str

    @classmethod
    def create(cls, start: datetime, end: datetime, interval: str,
               *, warmup_candles: int = 0) -> "BacktestPeriod":
        if interval not in INTERVAL_MINUTES:
            raise ValueError("Unsupported evaluation interval")
        if type(warmup_candles) is not int or warmup_candles < 0:
            raise ValueError("Warm-up candle count must be non-negative")
        step_ms = INTERVAL_MINUTES[interval] * 60_000
        start_ms, end_ms = epoch_ms(start), epoch_ms(end)
        if (start.microsecond % 1000 or end.microsecond % 1000 or start_ms < 0
                or start_ms % step_ms or end_ms % step_ms or end_ms <= start_ms):
            raise ValueError("Evaluation range must be positive and candle-aligned UTC")
        count = (end_ms - start_ms) // step_ms
        try:
            warmup_start = start - timedelta(minutes=INTERVAL_MINUTES[interval] * warmup_candles)
        except OverflowError:
            raise ValueError("Warm-up starts before supported dates") from None
        if warmup_start.timestamp() < 0:
            raise ValueError("Warm-up starts before supported epoch")
        return cls(start, end, warmup_start if warmup_candles else None,
                   count, warmup_candles, interval)


@dataclass(frozen=True)
class PeriodComparison:
    overlap: bool
    strictly_non_overlapping: bool
    order: Literal["first-before-second", "second-before-first", "overlap"]
    gap: timedelta


def compare_periods(first: BacktestPeriod, second: BacktestPeriod) -> PeriodComparison:
    """Compare evaluation windows only; warm-up may overlap without leakage."""
    if first.evaluation_end <= second.evaluation_start:
        return PeriodComparison(False, True, "first-before-second",
                                second.evaluation_start - first.evaluation_end)
    if second.evaluation_end <= first.evaluation_start:
        return PeriodComparison(False, True, "second-before-first",
                                first.evaluation_start - second.evaluation_end)
    return PeriodComparison(True, False, "overlap", timedelta(0))
