"""Virtual spot portfolio, fully independent from any exchange balance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Mapping, Union
from uuid import uuid4

from ..config import Settings
from ..safety.guard import TradingRequest, validate_request
from .ledger import Ledger, LedgerEntry

DecimalInput = Union[Decimal, str, int]
ZERO = Decimal("0")


class PortfolioError(ValueError):
    """Base exception for rejected virtual portfolio operations."""


class InsufficientFunds(PortfolioError):
    """Raised when a BUY would exceed available virtual USDT."""


class InsufficientAssets(PortfolioError):
    """Raised when a SELL would create a short position."""


class PortfolioIntegrityError(PortfolioError):
    """Raised when replayed ledger data violates accounting invariants."""


@dataclass(frozen=True)
class Position:
    quantity: Decimal = ZERO
    average_entry_price: Decimal = ZERO


def _decimal(value: DecimalInput, name: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise PortfolioError(f"{name} must use Decimal, string, or integer; float is forbidden")
    try:
        result = Decimal(value)
    except Exception as exc:
        raise PortfolioError(f"Invalid {name}") from exc
    if not result.is_finite():
        raise PortfolioError(f"{name} must be finite")
    return result


class Portfolio:
    """Simulates BUY/SELL accounting; it never reads a Bybit account balance."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        initial_cash: DecimalInput | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        settings = settings or Settings()
        starting_cash = settings.initial_capital_usdt if initial_cash is None else _decimal(initial_cash, "initial_cash")
        if starting_cash < ZERO:
            raise PortfolioError("initial_cash cannot be negative")
        self._cash = starting_cash
        self._fee_rate = settings.spot_fee_rate
        self._positions = {"BTCUSDT": Position(), "ETHUSDT": Position()}
        self._realized_pnl = ZERO
        self._total_commission = ZERO
        self._total_slippage = ZERO
        self._ledger = ledger if ledger is not None else Ledger()
        if len(self._ledger):
            self._restore_from_ledger()

    @classmethod
    def open(
        cls,
        settings: Settings | None = None,
        *,
        database_path: str | None = None,
        initial_cash: DecimalInput | None = None,
    ) -> "Portfolio":
        """Open or create a persistent portfolio and deterministically replay it."""
        from ..database.sqlite_store import SQLiteLedgerStore

        settings = settings or Settings()
        starting_cash = settings.initial_capital_usdt if initial_cash is None else _decimal(initial_cash, "initial_cash")
        store = SQLiteLedgerStore(database_path or settings.database_path, starting_cash)
        try:
            ledger = Ledger(
                store.load_entries(),
                persist_entry=store.append,
                close_storage=store.close,
            )
            return cls(settings, initial_cash=store.initial_cash, ledger=ledger)
        except Exception:
            store.close()
            raise

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def free_usdt(self) -> Decimal:
        return self._cash

    @property
    def btc(self) -> Decimal:
        return self._positions["BTCUSDT"].quantity

    @property
    def eth(self) -> Decimal:
        return self._positions["ETHUSDT"].quantity

    @property
    def realized_pnl(self) -> Decimal:
        return self._realized_pnl

    @property
    def total_commission(self) -> Decimal:
        return self._total_commission

    @property
    def total_slippage(self) -> Decimal:
        return self._total_slippage

    @property
    def ledger(self) -> Ledger:
        return self._ledger

    @property
    def fee_rate(self) -> Decimal:
        return self._fee_rate

    def close(self) -> None:
        self._ledger.close()

    def quantity(self, symbol: str) -> Decimal:
        safe_symbol = self._safe_symbol(symbol)
        return self._positions[safe_symbol].quantity

    def average_entry_price(self, symbol: str) -> Decimal:
        safe_symbol = self._safe_symbol(symbol)
        return self._positions[safe_symbol].average_entry_price

    def buy(
        self,
        symbol: str,
        quantity: DecimalInput,
        execution_price: DecimalInput,
        *,
        slippage_cost: DecimalInput = ZERO,
        market_bid: DecimalInput | None = None,
        market_ask: DecimalInput | None = None,
    ) -> LedgerEntry:
        safe_symbol = self._safe_symbol(symbol)
        qty, price, slippage = self._validated_trade_values(quantity, execution_price, slippage_cost)
        bid, ask = self._validated_market_reference("BUY", qty, price, slippage, market_bid, market_ask)
        gross = qty * price
        commission = gross * self._fee_rate
        total_cost = gross + commission + (ZERO if ask is not None else slippage)
        if total_cost > self._cash:
            raise InsufficientFunds("Insufficient virtual USDT for BUY including fees and slippage")

        old_position = self._positions[safe_symbol]
        new_quantity = old_position.quantity + qty
        old_cost_basis = old_position.quantity * old_position.average_entry_price
        new_average = (old_cost_basis + total_cost) / new_quantity
        new_cash = self._cash - total_cost
        self._ensure_non_negative(new_cash, new_quantity)

        entry = self._entry(
            symbol=safe_symbol,
            side="BUY",
            quantity=qty,
            price=price,
            gross=gross,
            commission=commission,
            slippage=slippage,
            usdt_change=-total_cost,
            base_change=qty,
            realized_pnl=ZERO,
            cash_after=new_cash,
            position_after=new_quantity,
            average_after=new_average,
            market_bid=bid,
            market_ask=ask,
        )
        self._ledger._append(entry)
        self._cash = new_cash
        self._positions[safe_symbol] = Position(new_quantity, new_average)
        self._total_commission += commission
        self._total_slippage += slippage
        return entry

    def sell(
        self,
        symbol: str,
        quantity: DecimalInput,
        execution_price: DecimalInput,
        *,
        slippage_cost: DecimalInput = ZERO,
        market_bid: DecimalInput | None = None,
        market_ask: DecimalInput | None = None,
    ) -> LedgerEntry:
        safe_symbol = self._safe_symbol(symbol)
        qty, price, slippage = self._validated_trade_values(quantity, execution_price, slippage_cost)
        bid, ask = self._validated_market_reference("SELL", qty, price, slippage, market_bid, market_ask)
        old_position = self._positions[safe_symbol]
        if qty > old_position.quantity:
            raise InsufficientAssets("Insufficient asset quantity; short positions are forbidden")

        gross = qty * price
        commission = gross * self._fee_rate
        net_proceeds = gross - commission - (ZERO if bid is not None else slippage)
        if net_proceeds < ZERO:
            raise PortfolioError("Commission and slippage cannot exceed SELL gross value")
        realized = net_proceeds - (qty * old_position.average_entry_price)
        new_quantity = old_position.quantity - qty
        new_average = ZERO if new_quantity == ZERO else old_position.average_entry_price
        new_cash = self._cash + net_proceeds
        self._ensure_non_negative(new_cash, new_quantity)

        entry = self._entry(
            symbol=safe_symbol,
            side="SELL",
            quantity=qty,
            price=price,
            gross=gross,
            commission=commission,
            slippage=slippage,
            usdt_change=net_proceeds,
            base_change=-qty,
            realized_pnl=realized,
            cash_after=new_cash,
            position_after=new_quantity,
            average_after=new_average,
            market_bid=bid,
            market_ask=ask,
        )
        self._ledger._append(entry)
        self._cash = new_cash
        self._positions[safe_symbol] = Position(new_quantity, new_average)
        self._realized_pnl += realized
        self._total_commission += commission
        self._total_slippage += slippage
        return entry

    def unrealized_pnl(self, symbol: str, current_price: DecimalInput) -> Decimal:
        safe_symbol = self._safe_symbol(symbol)
        price = _decimal(current_price, "current_price")
        if price <= ZERO:
            raise PortfolioError("current_price must be positive")
        position = self._positions[safe_symbol]
        return position.quantity * (price - position.average_entry_price)

    def equity(self, current_prices: Mapping[str, DecimalInput]) -> Decimal:
        normalized_prices = {self._safe_symbol(symbol): _decimal(price, "current_price") for symbol, price in current_prices.items()}
        equity = self._cash
        for symbol, position in self._positions.items():
            if position.quantity == ZERO:
                continue
            if symbol not in normalized_prices or normalized_prices[symbol] <= ZERO:
                raise PortfolioError(f"Positive current price required for open position {symbol}")
            equity += position.quantity * normalized_prices[symbol]
        return equity

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        try:
            return validate_request(TradingRequest(symbol=symbol, category="spot")).symbol
        except (AttributeError, ValueError) as exc:
            raise PortfolioError(str(exc)) from exc

    @staticmethod
    def _validated_trade_values(
        quantity: DecimalInput, execution_price: DecimalInput, slippage_cost: DecimalInput
    ) -> tuple[Decimal, Decimal, Decimal]:
        qty = _decimal(quantity, "quantity")
        price = _decimal(execution_price, "execution_price")
        slippage = _decimal(slippage_cost, "slippage_cost")
        if qty <= ZERO:
            raise PortfolioError("quantity must be positive")
        if price <= ZERO:
            raise PortfolioError("execution_price must be positive")
        if slippage < ZERO:
            raise PortfolioError("slippage_cost cannot be negative")
        return qty, price, slippage

    @staticmethod
    def _ensure_non_negative(cash: Decimal, quantity: Decimal) -> None:
        if cash < ZERO or quantity < ZERO:
            raise PortfolioError("Operation would create a negative balance")

    @staticmethod
    def _validated_market_reference(
        side: str,
        quantity: Decimal,
        execution_price: Decimal,
        slippage_cost: Decimal,
        market_bid: DecimalInput | None,
        market_ask: DecimalInput | None,
    ) -> tuple[Decimal | None, Decimal | None]:
        if market_bid is None and market_ask is None:
            return None, None
        if market_bid is None or market_ask is None:
            raise PortfolioError("Both market bid and ask are required")
        bid = _decimal(market_bid, "market_bid")
        ask = _decimal(market_ask, "market_ask")
        if bid <= ZERO or ask < bid:
            raise PortfolioError("Invalid market bid/ask")
        if side == "BUY":
            if execution_price < ask or slippage_cost != quantity * (execution_price - ask):
                raise PortfolioError("BUY execution/slippage does not match market ask")
        elif execution_price > bid or slippage_cost != quantity * (bid - execution_price):
            raise PortfolioError("SELL execution/slippage does not match market bid")
        return bid, ask

    def _restore_from_ledger(self) -> None:
        """Replay immutable events and reject any inconsistent persisted state."""
        cash = self._cash
        positions = {"BTCUSDT": Position(), "ETHUSDT": Position()}
        realized_total = ZERO
        commission_total = ZERO
        slippage_total = ZERO
        seen_ids: set[str] = set()

        for entry in self._ledger:
            if not entry.id or entry.id in seen_ids:
                raise PortfolioIntegrityError("Ledger IDs must be non-empty and unique")
            seen_ids.add(entry.id)
            if entry.operation_type != "SIMULATED_SPOT_TRADE":
                raise PortfolioIntegrityError("Unknown ledger operation type")
            safe_symbol = self._safe_symbol(entry.symbol)
            if safe_symbol != entry.symbol or entry.side not in {"BUY", "SELL"}:
                raise PortfolioIntegrityError("Invalid ledger symbol or side")
            values = (
                entry.quantity,
                entry.execution_price,
                entry.gross_value,
                entry.commission,
                entry.slippage_cost,
                entry.usdt_change,
                entry.base_asset_change,
                entry.realized_pnl,
                entry.usdt_balance_after,
                entry.position_after,
                entry.average_entry_price_after,
            )
            if any(not value.is_finite() for value in values):
                raise PortfolioIntegrityError("Ledger contains non-finite numbers")
            if entry.quantity <= ZERO or entry.execution_price <= ZERO:
                raise PortfolioIntegrityError("Ledger quantity and price must be positive")
            if entry.commission < ZERO or entry.slippage_cost < ZERO:
                raise PortfolioIntegrityError("Ledger costs cannot be negative")
            if entry.gross_value != entry.quantity * entry.execution_price:
                raise PortfolioIntegrityError("Ledger gross value is inconsistent")
            try:
                bid, ask = self._validated_market_reference(
                    entry.side,
                    entry.quantity,
                    entry.execution_price,
                    entry.slippage_cost,
                    entry.market_bid,
                    entry.market_ask,
                )
            except PortfolioError as exc:
                raise PortfolioIntegrityError("Ledger market reference is inconsistent") from exc

            old = positions[safe_symbol]
            if entry.side == "BUY":
                total_cost = entry.gross_value + entry.commission + (ZERO if ask is not None else entry.slippage_cost)
                expected_cash = cash - total_cost
                expected_quantity = old.quantity + entry.quantity
                expected_average = (
                    old.quantity * old.average_entry_price + total_cost
                ) / expected_quantity
                expected_delta = -total_cost
                expected_asset_delta = entry.quantity
                expected_realized = ZERO
            else:
                if entry.quantity > old.quantity:
                    raise PortfolioIntegrityError("Ledger would create a short position")
                net_proceeds = entry.gross_value - entry.commission - (ZERO if bid is not None else entry.slippage_cost)
                if net_proceeds < ZERO:
                    raise PortfolioIntegrityError("Ledger SELL costs exceed gross value")
                expected_cash = cash + net_proceeds
                expected_quantity = old.quantity - entry.quantity
                expected_average = ZERO if expected_quantity == ZERO else old.average_entry_price
                expected_delta = net_proceeds
                expected_asset_delta = -entry.quantity
                expected_realized = net_proceeds - entry.quantity * old.average_entry_price

            if expected_cash < ZERO or expected_quantity < ZERO:
                raise PortfolioIntegrityError("Ledger produces a negative balance")
            expected = (
                expected_delta,
                expected_asset_delta,
                expected_realized,
                expected_cash,
                expected_quantity,
                expected_average,
            )
            actual = (
                entry.usdt_change,
                entry.base_asset_change,
                entry.realized_pnl,
                entry.usdt_balance_after,
                entry.position_after,
                entry.average_entry_price_after,
            )
            if actual != expected:
                raise PortfolioIntegrityError("Ledger balances or P&L are inconsistent")

            cash = expected_cash
            positions[safe_symbol] = Position(expected_quantity, expected_average)
            realized_total += expected_realized
            commission_total += entry.commission
            slippage_total += entry.slippage_cost

        self._cash = cash
        self._positions = positions
        self._realized_pnl = realized_total
        self._total_commission = commission_total
        self._total_slippage = slippage_total

    @staticmethod
    def _entry(
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        gross: Decimal,
        commission: Decimal,
        slippage: Decimal,
        usdt_change: Decimal,
        base_change: Decimal,
        realized_pnl: Decimal,
        cash_after: Decimal,
        position_after: Decimal,
        average_after: Decimal,
        market_bid: Decimal | None = None,
        market_ask: Decimal | None = None,
    ) -> LedgerEntry:
        return LedgerEntry(
            id=str(uuid4()),
            timestamp_utc=datetime.now(timezone.utc),
            operation_type="SIMULATED_SPOT_TRADE",
            symbol=symbol,
            side=side,
            quantity=quantity,
            execution_price=price,
            gross_value=gross,
            commission=commission,
            slippage_cost=slippage,
            usdt_change=usdt_change,
            base_asset_change=base_change,
            realized_pnl=realized_pnl,
            usdt_balance_after=cash_after,
            position_after=position_after,
            average_entry_price_after=average_after,
            market_bid=market_bid,
            market_ask=market_ask,
        )
