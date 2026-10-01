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


class HistoricalDownloadUnavailable(HistoricalDataUnavailable):
    """Safe, structured context for one failed public kline page."""

    def __init__(self, category: str, requested_start: datetime, requested_end: datetime,
                 *, page_number: int | None = None, page_start_ms: int | None = None,
                 page_end_ms: int | None = None, problem_ms: int | None = None,
                 detail: str = "") -> None:
        self.category = category
        self.requested_start = requested_start
        self.requested_end = requested_end
        self.page_number = page_number
        self.page_start_ms = page_start_ms
        self.page_end_ms = page_end_ms
        self.problem_ms = problem_ms
        self.detail = detail
        parts = [f"category={category}",
                 f"requested=[{requested_start.isoformat()}, {requested_end.isoformat()})"]
        if page_number is not None and page_start_ms is not None and page_end_ms is not None:
            parts.append(f"page={page_number}")
            parts.append(f"page_range=[{_safe_timestamp(page_start_ms)}, "
                         f"{_safe_timestamp(page_end_ms)})")
        if problem_ms is not None:
            parts.append(f"problem_timestamp={_safe_timestamp(problem_ms)}")
        if detail:
            parts.append(f"detail={detail}")
        super().__init__("; ".join(parts))


def _safe_timestamp(value_ms: int) -> str:
    try:
        return utc_from_ms(value_ms).isoformat()
    except HistoricalDataUnavailable:
        return f"invalid-ms:{value_ms}"


def _transport_category(message: str) -> str:
    if "timed out" in message:
        return "timeout"
    if "connection error" in message:
        return "connection_error"
    if "HTTP error" in message:
        return "http_error"
    if "API error code" in message:
        return "bybit_api_error"
    if "malformed" in message or "must be an object" in message or "status code" in message:
        return "malformed_response"
    return "transport_error"


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
        page_number = 0
        while cursor < end_ms:
            page_number += 1
            page_end = min(end_ms, cursor + step * 1000)
            expected_count = (page_end - cursor) // step
            try:
                payload = self._bybit.get_spot_klines(
                    symbol, str(INTERVAL_MINUTES[interval]), cursor, page_end - 1,
                    category="spot", limit=expected_count,
                )
            except BybitUnavailable as exc:
                message = str(exc)
                raise HistoricalDownloadUnavailable(
                    _transport_category(message), start, end, page_number=page_number,
                    page_start_ms=cursor, page_end_ms=page_end, detail=message,
                ) from None
            result = payload.get("result")
            if not isinstance(result, dict) or result.get("category") != "spot" or result.get("symbol") != symbol:
                raise HistoricalDownloadUnavailable(
                    "response_identity", start, end, page_number=page_number,
                    page_start_ms=cursor, page_end_ms=page_end,
                    detail="unexpected kline category or symbol",
                )
            rows = result.get("list")
            if not isinstance(rows, list):
                raise HistoricalDownloadUnavailable(
                    "malformed_response", start, end, page_number=page_number,
                    page_start_ms=cursor, page_end_ms=page_end,
                    detail="kline list is missing or invalid",
                )
            if len(rows) != expected_count:
                raise HistoricalDownloadUnavailable(
                    "page_count_mismatch", start, end, page_number=page_number,
                    page_start_ms=cursor, page_end_ms=page_end,
                    detail=f"expected={expected_count}, received={len(rows)}",
                )
            page: list[Candle] = []
            for index, row in enumerate(rows):
                if not isinstance(row, list) or len(row) != 7 or not isinstance(row[0], str) or not row[0].isdigit():
                    raise HistoricalDownloadUnavailable(
                        "malformed_candle", start, end, page_number=page_number,
                        page_start_ms=cursor, page_end_ms=page_end,
                        detail=f"invalid row at response index {index}",
                    )
                ms = int(row[0])
                expected_ms = page_end - step * (index + 1)
                if ms != expected_ms:
                    raise HistoricalDownloadUnavailable(
                        "candle_sequence", start, end, page_number=page_number,
                        page_start_ms=cursor, page_end_ms=page_end, problem_ms=ms,
                        detail=f"expected_timestamp={utc_from_ms(expected_ms).isoformat()}",
                    )
                try:
                    values = [_decimal(v, positive=i < 4) for i, v in enumerate(row[1:])]
                    candle = Candle(symbol, interval, utc_from_ms(ms), *values)
                    validate_candle(candle)
                except (HistoricalDataUnavailable, ValueError, TypeError):
                    raise HistoricalDownloadUnavailable(
                        "invalid_candle", start, end, page_number=page_number,
                        page_start_ms=cursor, page_end_ms=page_end, problem_ms=ms,
                        detail="invalid OHLCV values",
                    ) from None
                page.append(candle)
            candles.extend(reversed(page))
            cursor = page_end
        return candles
