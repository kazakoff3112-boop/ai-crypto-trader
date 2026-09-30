"""Strict, read-only Bybit V5 spot OHLCV downloader (no Portfolio coupling)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Callable

from ..bybit.client import BybitReadOnlyClient, BybitUnavailable
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
    start_ms, end_ms = epoch_ms(start), epoch_ms(end)
    if start_ms < 0 or end_ms <= start_ms or start_ms % step or end_ms % step:
        raise ValueError("Historical range must be positive and candle-aligned")
    if (end_ms - start_ms) // step > 100_000:
        raise ValueError("Historical range is too large")
    return safe.symbol, step, start_ms, end_ms


def _decimal(value: object, *, positive: bool) -> Decimal:
    if not isinstance(value, str):
        raise HistoricalDataUnavailable("Candle has a missing or invalid number")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise HistoricalDataUnavailable("Candle has an invalid number") from None
    if not number.is_finite() or (number <= 0 if positive else number < 0):
        raise HistoricalDataUnavailable("Candle has an invalid number")
    return number


def validate_candle(candle: Candle) -> None:
    validate_request(TradingRequest(symbol=candle.symbol, category="spot"))
    if candle.interval not in INTERVAL_MINUTES:
        raise HistoricalDataUnavailable("Unsupported cached interval")
    step = INTERVAL_MINUTES[candle.interval] * 60_000
    if epoch_ms(candle.open_time_utc) % step:
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


class HistoricalMarketDataClient:
    def __init__(self, bybit: BybitReadOnlyClient, *, clock: Callable[[], datetime] | None = None) -> None:
        self._bybit = bybit
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def fetch_candles(self, symbol: str, interval: str, start: datetime, end: datetime,
                      *, category: str = "spot", completed_only: bool = True) -> list[Candle]:
        """Fetch aligned [start,end), paginating forward; reject gaps and bad rows."""
        symbol, step, start_ms, end_ms = validate_range(symbol, category, interval, start, end)
        if completed_only:
            now_ms = epoch_ms(self._clock())
            end_ms = min(end_ms, now_ms - now_ms % step)
        if end_ms <= start_ms:
            return []
        candles: list[Candle] = []
        cursor = start_ms
        while cursor < end_ms:
            page_end = min(end_ms, cursor + step * 1000)
            expected_count = (page_end - cursor) // step
            try:
                payload = self._bybit.get_spot_klines(
                    symbol, str(INTERVAL_MINUTES[interval]), cursor, page_end - 1,
                    category="spot", limit=expected_count,
                )
            except BybitUnavailable as exc:
                raise HistoricalDataUnavailable(str(exc)) from None
            result = payload.get("result")
            if not isinstance(result, dict) or result.get("category") != "spot" or result.get("symbol") != symbol:
                raise HistoricalDataUnavailable("Unexpected kline category or symbol")
            rows = result.get("list")
            if not isinstance(rows, list) or len(rows) != expected_count:
                raise HistoricalDataUnavailable("Kline range has missing or extra candles")
            page: list[Candle] = []
            for index, row in enumerate(rows):
                if not isinstance(row, list) or len(row) != 7 or not isinstance(row[0], str) or not row[0].isdigit():
                    raise HistoricalDataUnavailable("Malformed kline row")
                ms = int(row[0])
                if ms != page_end - step * (index + 1):
                    raise HistoricalDataUnavailable("Kline timestamps are duplicated, missing or out of order")
                values = [_decimal(v, positive=i < 4) for i, v in enumerate(row[1:])]
                candle = Candle(symbol, interval, utc_from_ms(ms), *values)
                validate_candle(candle)
                page.append(candle)
            candles.extend(reversed(page))
            cursor = page_end
        return candles
