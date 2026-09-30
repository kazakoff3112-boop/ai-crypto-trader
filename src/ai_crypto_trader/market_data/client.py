"""Validated public Bybit spot quotes, separate from Portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import time
from typing import Callable

from ..bybit.client import BybitReadOnlyClient, BybitUnavailable
from ..safety.guard import TradingRequest, validate_request


class MarketDataUnavailable(RuntimeError):
    """No trustworthy current quote is available."""


@dataclass(frozen=True)
class SpotQuote:
    symbol: str
    last_price: Decimal
    bid: Decimal
    ask: Decimal
    timestamp_utc: datetime


class MarketDataClient:
    def __init__(
        self,
        bybit: BybitReadOnlyClient,
        *,
        max_age_seconds: int = 60,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self._bybit = bybit
        self._max_age_ms = max_age_seconds * 1000
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def get_spot_quote(self, symbol: str, *, category: str = "spot") -> SpotQuote:
        safe = validate_request(TradingRequest(symbol=symbol, category=category))
        try:
            payload = self._bybit.get_spot_ticker(safe.symbol, category=safe.category)
        except BybitUnavailable as exc:
            raise MarketDataUnavailable(str(exc)) from None

        result = payload.get("result")
        if not isinstance(result, dict) or result.get("category") != "spot":
            raise MarketDataUnavailable("Spot market data is malformed")
        tickers = result.get("list")
        if not isinstance(tickers, list) or len(tickers) != 1:
            raise MarketDataUnavailable("Spot market data is missing the requested ticker")
        ticker = tickers[0]
        if not isinstance(ticker, dict) or ticker.get("symbol") != safe.symbol:
            raise MarketDataUnavailable("Spot market data has an unexpected symbol")
        last = self._price(ticker.get("lastPrice"), "last price")
        bid = self._price(ticker.get("bid1Price"), "bid")
        ask = self._price(ticker.get("ask1Price"), "ask")
        if bid > ask:
            raise MarketDataUnavailable("Spot market data has crossed bid and ask")

        timestamp_ms = payload.get("time")
        if type(timestamp_ms) is not int or timestamp_ms <= 0:
            raise MarketDataUnavailable("Spot market data has invalid timestamp")
        age = self._clock_ms() - timestamp_ms
        if age > self._max_age_ms or age < -5000:
            raise MarketDataUnavailable("Spot market data is stale or dated in the future")
        try:
            timestamp = datetime.fromtimestamp(timestamp_ms // 1000, tz=timezone.utc).replace(
                microsecond=(timestamp_ms % 1000) * 1000
            )
        except (ValueError, OverflowError, OSError):
            raise MarketDataUnavailable("Spot market data has invalid timestamp") from None
        return SpotQuote(safe.symbol, last, bid, ask, timestamp)

    @staticmethod
    def _price(value: object, name: str) -> Decimal:
        if not isinstance(value, str):
            raise MarketDataUnavailable(f"Spot market data is missing {name}")
        try:
            price = Decimal(value)
        except InvalidOperation:
            raise MarketDataUnavailable(f"Spot market data has invalid {name}") from None
        if not price.is_finite() or price <= 0:
            raise MarketDataUnavailable(f"Spot market data has invalid {name}")
        return price
