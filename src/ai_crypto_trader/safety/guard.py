"""Fail-closed trading policy shared by every future component."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

ALLOWED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT"})
ALLOWED_CATEGORIES = frozenset({"spot"})
ALLOWED_TRADING_MODES = frozenset({"demo"})
FORBIDDEN_CATEGORIES = frozenset({"linear", "inverse", "option", "future", "futures", "perpetual"})


class SafetyViolation(ValueError):
    """Raised whenever an action falls outside the explicit safety policy."""


@dataclass(frozen=True)
class TradingRequest:
    symbol: str
    category: str
    margin: bool = False
    leverage: Decimal | int | float | None = None


def validate_trading_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in ALLOWED_TRADING_MODES:
        raise SafetyViolation("Only Bybit Demo trading mode is allowed")
    return normalized


def validate_request(request: TradingRequest) -> TradingRequest:
    """Validate an intended action. Unknown or ambiguous values are rejected."""
    symbol = request.symbol.strip().upper()
    category = request.category.strip().lower()

    if category in FORBIDDEN_CATEGORIES or category not in ALLOWED_CATEGORIES:
        raise SafetyViolation(f"Category {category!r} is forbidden; only 'spot' is allowed")
    if symbol not in ALLOWED_SYMBOLS:
        raise SafetyViolation(f"Symbol {symbol!r} is not in the allowlist")
    if request.margin:
        raise SafetyViolation("Margin trading is forbidden")
    if request.leverage is not None:
        raise SafetyViolation("Leverage is forbidden, including leverage=1")

    return TradingRequest(symbol=symbol, category=category)
