"""Strict, read-only Bybit V5 spot OHLCV downloader (no Portfolio coupling)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Callable

from ..bybit.client import BybitReadOnlyClient, BybitUnavailable
from .candle import (
    Candle, HistoricalDataUnavailable, INTERVAL_MINUTES, epoch_ms, utc_from_ms,
    validate_candle, validate_range,
)


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
