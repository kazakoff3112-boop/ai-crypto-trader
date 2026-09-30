"""Explicit, persistent paper trades using read-only spot quotes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Protocol

from ..config import Settings
from ..market_data import SpotQuote
from ..portfolio import InsufficientAssets, InsufficientFunds, Portfolio, PortfolioError
from ..portfolio.portfolio import DecimalInput, _decimal
from ..safety.guard import TradingRequest, validate_request

ZERO = Decimal("0")
ONE = Decimal("1")
BPS = Decimal("10000")


class QuoteProvider(Protocol):
    def get_spot_quote(self, symbol: str, *, category: str = "spot") -> SpotQuote: ...


@dataclass(frozen=True)
class PaperExecutionConfig:
    fee_rate: Decimal = Decimal("0.001")
    slippage_bps: Decimal = Decimal("5")
    max_quote_age_seconds: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.fee_rate, Decimal) or not self.fee_rate.is_finite() or not ZERO <= self.fee_rate < ONE:
            raise ValueError("Invalid paper fee rate")
        if not isinstance(self.slippage_bps, Decimal) or not self.slippage_bps.is_finite() or not ZERO <= self.slippage_bps < BPS:
            raise ValueError("Invalid paper slippage BPS")
        if type(self.max_quote_age_seconds) is not int or self.max_quote_age_seconds <= 0:
            raise ValueError("Invalid maximum quote age")

    @classmethod
    def from_settings(cls, settings: Settings) -> "PaperExecutionConfig":
        return cls(fee_rate=settings.spot_fee_rate, slippage_bps=settings.paper_slippage_bps)


@dataclass(frozen=True)
class PaperPreview:
    symbol: str
    side: str
    quote_timestamp_utc: datetime
    quantity: Decimal
    bid: Decimal
    ask: Decimal
    last_price: Decimal
    execution_price: Decimal
    gross_value: Decimal
    commission: Decimal
    slippage_cost: Decimal
    realized_pnl: Decimal
    usdt_balance_after: Decimal
    asset_quantity_after: Decimal
    equity: Decimal | None


@dataclass(frozen=True)
class PaperTradeResult:
    trade_id: str
    timestamp_utc: datetime
    symbol: str
    side: str
    quantity: Decimal
    bid: Decimal
    ask: Decimal
    execution_price: Decimal
    gross_value: Decimal
    commission: Decimal
    slippage_cost: Decimal
    realized_pnl: Decimal
    usdt_balance_after: Decimal
    asset_quantity_after: Decimal
    equity: Decimal | None


class PaperTradingEngine:
    """Consumes quotes and writes only to the internal Portfolio/SQLite Ledger."""

    def __init__(
        self,
        market_data: QuoteProvider,
        portfolio: Portfolio,
        config: PaperExecutionConfig,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if portfolio.fee_rate != config.fee_rate:
            raise ValueError("Paper fee rate differs from Portfolio fee rate")
        self._market_data = market_data
        self._portfolio = portfolio
        self._config = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def preview_buy(self, symbol: str, usdt_amount: DecimalInput, *, category: str = "spot") -> PaperPreview:
        safe = validate_request(TradingRequest(symbol=symbol, category=category))
        budget = _decimal(usdt_amount, "usdt_amount")
        if budget <= ZERO:
            raise PortfolioError("USDT amount must be positive")
        if budget > self._portfolio.cash:
            raise InsufficientFunds("Paper BUY exceeds virtual USDT")
        quote = self._quote(safe.symbol)
        execution = quote.ask * (ONE + self._config.slippage_bps / BPS)
        quantity = budget / (execution * (ONE + self._config.fee_rate))
        if quantity <= ZERO:
            raise PortfolioError("Paper BUY quantity is too small")
        gross = quantity * execution
        commission = gross * self._config.fee_rate
        while gross + commission > budget:
            quantity = quantity.next_minus()
            gross = quantity * execution
            commission = gross * self._config.fee_rate
        slippage = quantity * (execution - quote.ask)
        cash_after = self._portfolio.cash - gross - commission
        position_after = self._portfolio.quantity(safe.symbol) + quantity
        return PaperPreview(
            symbol=safe.symbol, side="BUY", quote_timestamp_utc=quote.timestamp_utc,
            quantity=quantity, bid=quote.bid, ask=quote.ask, last_price=quote.last_price,
            execution_price=execution, gross_value=gross, commission=commission,
            slippage_cost=slippage, realized_pnl=ZERO, usdt_balance_after=cash_after,
            asset_quantity_after=position_after,
            equity=self._projected_equity(safe.symbol, cash_after, position_after, quote.bid),
        )

    def preview_sell(
        self,
        symbol: str,
        quantity: DecimalInput | None = None,
        *,
        sell_all: bool = False,
        category: str = "spot",
    ) -> PaperPreview:
        safe = validate_request(TradingRequest(symbol=symbol, category=category))
        if sell_all:
            if quantity is not None:
                raise PortfolioError("Choose quantity or sell_all, not both")
            qty = self._portfolio.quantity(safe.symbol)
        else:
            if quantity is None:
                raise PortfolioError("SELL quantity is required")
            qty = _decimal(quantity, "quantity")
        if qty <= ZERO:
            raise PortfolioError("SELL quantity must be positive")
        if qty > self._portfolio.quantity(safe.symbol):
            raise InsufficientAssets("Paper SELL exceeds virtual asset position")
        quote = self._quote(safe.symbol)
        execution = quote.bid * (ONE - self._config.slippage_bps / BPS)
        gross = qty * execution
        commission = gross * self._config.fee_rate
        slippage = qty * (quote.bid - execution)
        proceeds = gross - commission
        if proceeds < ZERO:
            raise PortfolioError("Paper SELL proceeds are negative")
        cash_after = self._portfolio.cash + proceeds
        position_after = self._portfolio.quantity(safe.symbol) - qty
        realized = proceeds - qty * self._portfolio.average_entry_price(safe.symbol)
        return PaperPreview(
            symbol=safe.symbol, side="SELL", quote_timestamp_utc=quote.timestamp_utc,
            quantity=qty, bid=quote.bid, ask=quote.ask, last_price=quote.last_price,
            execution_price=execution, gross_value=gross, commission=commission,
            slippage_cost=slippage, realized_pnl=realized, usdt_balance_after=cash_after,
            asset_quantity_after=position_after,
            equity=self._projected_equity(safe.symbol, cash_after, position_after, quote.bid),
        )

    def paper_buy(self, symbol: str, usdt_amount: DecimalInput, *, category: str = "spot") -> PaperTradeResult:
        plan = self.preview_buy(symbol, usdt_amount, category=category)
        return self.execute_preview(plan)

    def paper_sell(
        self,
        symbol: str,
        quantity: DecimalInput | None = None,
        *,
        sell_all: bool = False,
        category: str = "spot",
    ) -> PaperTradeResult:
        plan = self.preview_sell(symbol, quantity, sell_all=sell_all, category=category)
        return self.execute_preview(plan)

    def execute_preview(self, plan: PaperPreview) -> PaperTradeResult:
        """Execute exactly a previously shown plan after revalidating it."""
        self._require_persistence()
        safe = validate_request(TradingRequest(symbol=plan.symbol, category="spot"))
        self._validate_quote(
            SpotQuote(safe.symbol, plan.last_price, plan.bid, plan.ask, plan.quote_timestamp_utc)
        )
        if plan.quantity <= ZERO:
            raise PortfolioError("Paper quantity must be positive")
        expected_execution = (
            plan.ask * (ONE + self._config.slippage_bps / BPS)
            if plan.side == "BUY"
            else plan.bid * (ONE - self._config.slippage_bps / BPS)
        )
        expected_gross = plan.quantity * expected_execution
        expected_commission = expected_gross * self._config.fee_rate
        expected_slippage = plan.quantity * (
            expected_execution - plan.ask if plan.side == "BUY" else plan.bid - expected_execution
        )
        if (
            plan.side not in {"BUY", "SELL"}
            or plan.execution_price != expected_execution
            or plan.gross_value != expected_gross
            or plan.commission != expected_commission
            or plan.slippage_cost != expected_slippage
        ):
            raise PortfolioError("Paper preview is inconsistent")

        current_quantity = self._portfolio.quantity(plan.symbol)
        if plan.side == "BUY":
            expected_cash = self._portfolio.cash - expected_gross - expected_commission
            expected_position = current_quantity + plan.quantity
            expected_realized = ZERO
            if expected_cash < ZERO:
                raise InsufficientFunds("Paper BUY exceeds virtual USDT")
        else:
            if plan.quantity > current_quantity:
                raise InsufficientAssets("Paper SELL exceeds virtual asset position")
            expected_cash = self._portfolio.cash + expected_gross - expected_commission
            expected_position = current_quantity - plan.quantity
            expected_realized = (
                expected_gross
                - expected_commission
                - plan.quantity * self._portfolio.average_entry_price(plan.symbol)
            )
        if (
            plan.usdt_balance_after != expected_cash
            or plan.asset_quantity_after != expected_position
            or plan.realized_pnl != expected_realized
        ):
            raise PortfolioError("Portfolio changed after preview or preview is inconsistent")

        if plan.side == "BUY":
            entry = self._portfolio.buy(
                plan.symbol, plan.quantity, plan.execution_price,
                slippage_cost=plan.slippage_cost, market_bid=plan.bid, market_ask=plan.ask,
            )
        else:
            entry = self._portfolio.sell(
                plan.symbol, plan.quantity, plan.execution_price,
                slippage_cost=plan.slippage_cost, market_bid=plan.bid, market_ask=plan.ask,
            )
        return self._result(plan, entry.id, entry.timestamp_utc, entry.usdt_balance_after,
                            entry.position_after, entry.realized_pnl)

    def _quote(self, symbol: str) -> SpotQuote:
        quote = self._market_data.get_spot_quote(symbol, category="spot")
        return self._validate_quote(quote, expected_symbol=symbol)

    def _validate_quote(self, quote: SpotQuote, expected_symbol: str | None = None) -> SpotQuote:
        symbol = expected_symbol or quote.symbol
        if quote.symbol != symbol or not all(
            isinstance(value, Decimal) and value.is_finite() and value > ZERO
            for value in (quote.bid, quote.ask, quote.last_price)
        ) or quote.ask < quote.bid:
            raise PortfolioError("Invalid spot quote")
        timestamp = quote.timestamp_utc
        now = self._clock()
        if (timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
                or now.tzinfo is None):
            raise PortfolioError("Quote timestamp must be UTC")
        age = (now - timestamp).total_seconds()
        if age > self._config.max_quote_age_seconds or age < -5:
            raise PortfolioError("Spot quote is stale or dated in the future")
        return quote

    def _projected_equity(
        self, symbol: str, cash_after: Decimal, position_after: Decimal, bid: Decimal
    ) -> Decimal | None:
        other = "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
        if self._portfolio.quantity(other) != ZERO:
            return None
        return cash_after + position_after * bid

    def _require_persistence(self) -> None:
        if not self._portfolio.ledger.is_persistent:
            raise PortfolioError("Paper execution requires a persistent SQLite Portfolio")

    @staticmethod
    def _result(
        plan: PaperPreview, trade_id: str, timestamp: datetime,
        cash_after: Decimal, position_after: Decimal, realized: Decimal,
    ) -> PaperTradeResult:
        return PaperTradeResult(
            trade_id=trade_id, timestamp_utc=timestamp, symbol=plan.symbol, side=plan.side,
            quantity=plan.quantity, bid=plan.bid, ask=plan.ask,
            execution_price=plan.execution_price, gross_value=plan.gross_value,
            commission=plan.commission, slippage_cost=plan.slippage_cost,
            realized_pnl=realized, usdt_balance_after=cash_after,
            asset_quantity_after=position_after, equity=plan.equity,
        )
