"""Value open spot positions at fresh bids without modifying the Portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from typing import Callable, Protocol

from ..market_data import SpotQuote
from ..portfolio import Portfolio

ZERO = Decimal("0")


class AnalyticsUnavailable(RuntimeError):
    """The available inputs cannot support a trustworthy portfolio report."""


class QuoteProvider(Protocol):
    def get_spot_quote(self, symbol: str, *, category: str = "spot") -> SpotQuote: ...


@dataclass(frozen=True)
class AssetValuation:
    symbol: str
    quantity: Decimal
    average_entry_price: Decimal
    current_bid: Decimal | None
    market_value: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True)
class PortfolioReport:
    initial_capital: Decimal
    virtual_usdt: Decimal
    btc: AssetValuation
    eth: AssetValuation
    total_market_value: Decimal
    total_equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    total_return_percent: Decimal
    total_commissions: Decimal
    total_slippage: Decimal
    ledger_entries: int


def build_portfolio_report(
    portfolio: Portfolio,
    market_data: QuoteProvider,
    *,
    clock: Callable[[], datetime] | None = None,
    max_quote_age_seconds: int = 60,
) -> PortfolioReport:
    """Take one read-only snapshot; every open asset requires its own fresh bid."""
    if max_quote_age_seconds <= 0:
        raise ValueError("max_quote_age_seconds must be positive")
    if portfolio.initial_cash <= ZERO:
        raise AnalyticsUnavailable("Initial capital must be positive for return percentage")
    now = clock() if clock is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise AnalyticsUnavailable("Report clock must be timezone-aware")

    valuations: dict[str, AssetValuation] = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        quantity = portfolio.quantity(symbol)
        average = portfolio.average_entry_price(symbol)
        if quantity == ZERO:
            valuations[symbol] = AssetValuation(symbol, quantity, average, None, ZERO, ZERO)
            continue
        try:
            quote = market_data.get_spot_quote(symbol, category="spot")
        except Exception as exc:
            raise AnalyticsUnavailable(f"Fresh {symbol} quote is unavailable") from None
        if not isinstance(quote, SpotQuote) or quote.symbol != symbol:
            raise AnalyticsUnavailable(f"Invalid {symbol} quote")
        if not all(
            isinstance(value, Decimal) and value.is_finite() and value > ZERO
            for value in (quote.bid, quote.ask, quote.last_price)
        ) or quote.ask < quote.bid:
            raise AnalyticsUnavailable(f"Invalid {symbol} quote")
        timestamp = quote.timestamp_utc
        if (
            not isinstance(timestamp, datetime)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
        ):
            raise AnalyticsUnavailable(f"Invalid {symbol} quote timestamp")
        age = (now - timestamp).total_seconds()
        if age > max_quote_age_seconds or age < -5:
            raise AnalyticsUnavailable(f"Stale {symbol} quote")
        market_value = quantity * quote.bid
        unrealized = market_value - quantity * average
        valuations[symbol] = AssetValuation(
            symbol, quantity, average, quote.bid, market_value, unrealized
        )

    btc = valuations["BTCUSDT"]
    eth = valuations["ETHUSDT"]
    market_value = btc.market_value + eth.market_value
    equity = portfolio.cash + market_value
    unrealized = btc.unrealized_pnl + eth.unrealized_pnl
    total_pnl = equity - portfolio.initial_cash
    reconciliation_error = total_pnl - portfolio.realized_pnl - unrealized
    scale = max(abs(equity), abs(portfolio.initial_cash), Decimal("1"))
    rounding_bound = Decimal("10").scaleb(scale.adjusted() - getcontext().prec + 1)
    if abs(reconciliation_error) > rounding_bound:
        raise AnalyticsUnavailable("Portfolio P&L reconciliation failed")
    return PortfolioReport(
        initial_capital=portfolio.initial_cash,
        virtual_usdt=portfolio.cash,
        btc=btc,
        eth=eth,
        total_market_value=market_value,
        total_equity=equity,
        realized_pnl=portfolio.realized_pnl,
        unrealized_pnl=unrealized,
        total_pnl=total_pnl,
        total_return_percent=total_pnl / portfolio.initial_cash * Decimal("100"),
        total_commissions=portfolio.total_commission,
        total_slippage=portfolio.total_slippage,
        ledger_entries=len(portfolio.ledger),
    )
