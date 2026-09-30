"""Narrow, read-only Bybit V5 transport for the Demo endpoint."""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..config import Settings
from ..safety.guard import TradingRequest, validate_request


class BybitUnavailable(RuntimeError):
    """A safe, credential-free description of an API failure."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


def _default_open(request: Request, timeout: float) -> Any:
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


class BybitReadOnlyClient:
    """Exposes only fixed GET endpoints; no generic public request method."""

    def __init__(
        self,
        settings: Settings,
        timeout_seconds: float = 10.0,
        *,
        opener: Callable[[Request, float], Any] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = settings.base_url
        self._api_key = settings.api_key
        self._api_secret = settings.api_secret
        self._timeout_seconds = timeout_seconds
        self._opener = opener or _default_open
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    def get_server_time(self) -> dict[str, Any]:
        return self._get("/v5/market/time")

    def get_spot_ticker(self, symbol: str, *, category: str = "spot") -> dict[str, Any]:
        safe = validate_request(TradingRequest(symbol=symbol, category=category))
        return self._get("/v5/market/tickers", {"category": "spot", "symbol": safe.symbol})

    def get_spot_klines(
        self, symbol: str, interval_code: str, start_ms: int, end_ms: int,
        *, category: str = "spot", limit: int = 1000,
    ) -> dict[str, Any]:
        """Fixed public V5 GET only; the caller validates the returned candles."""
        safe = validate_request(TradingRequest(symbol=symbol, category=category))
        if interval_code not in {"1", "5", "15", "60"}:
            raise ValueError("Unsupported spot kline interval")
        if (type(start_ms) is not int or type(end_ms) is not int or start_ms < 0
                or end_ms < start_ms or type(limit) is not int or not 1 <= limit <= 1000):
            raise ValueError("Invalid spot kline request range or limit")
        return self._get("/v5/market/kline", {
            "category": "spot", "symbol": safe.symbol, "interval": interval_code,
            "start": str(start_ms), "end": str(end_ms), "limit": str(limit),
        })

    def check_demo_credentials(self) -> bool:
        """GET account info solely to confirm Demo credentials; discard its payload."""
        if not self._api_key or not self._api_secret:
            raise BybitUnavailable("Demo API credentials are missing")
        timestamp = str(self._clock_ms())
        receive_window = "5000"
        to_sign = f"{timestamp}{self._api_key}{receive_window}"
        signature = hmac.new(
            self._api_secret.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": receive_window,
            "X-BAPI-SIGN": signature,
        }
        payload = self._get("/v5/account/info", headers=headers)
        if not isinstance(payload.get("result"), dict):
            raise BybitUnavailable("Private Demo response is malformed")
        return True

    def _get(
        self,
        path: str,
        params: dict[str, str] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        query = f"?{urlencode(params)}" if params else ""
        request = Request(f"{self._base_url}{path}{query}", headers=headers or {}, method="GET")
        try:
            with self._opener(request, self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise BybitUnavailable(f"Bybit HTTP error {exc.code}") from None
        except (TimeoutError, socket.timeout):
            raise BybitUnavailable("Bybit request timed out") from None
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise BybitUnavailable("Bybit request timed out") from None
            raise BybitUnavailable("Bybit connection error") from None
        except (ConnectionError, OSError):
            raise BybitUnavailable("Bybit connection error") from None
        except (UnicodeError, ValueError, TypeError):
            raise BybitUnavailable("Bybit response is malformed JSON") from None
        if not isinstance(payload, dict):
            raise BybitUnavailable("Bybit response must be an object")
        code = payload.get("retCode")
        if type(code) is not int:
            raise BybitUnavailable("Bybit response has no valid status code")
        if code != 0:
            raise BybitUnavailable(f"Bybit API error code {code}")
        return payload
