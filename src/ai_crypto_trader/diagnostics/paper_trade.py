"""Manual preview, status, and explicitly confirmed internal paper execution."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from decimal import Decimal, DecimalException
import sqlite3

from ..bybit import BybitReadOnlyClient
from ..config import load_settings
from ..database import DataIntegrityError
from ..market_data import MarketDataClient
from ..market_data import MarketDataUnavailable
from ..portfolio import InsufficientAssets, InsufficientFunds, Portfolio, PortfolioError
from ..safety import SafetyViolation
from ..trading_engine import PaperExecutionConfig, PaperPreview, PaperTradingEngine

CONFIRMATION = "YES"
PAPER_WARNING = "PAPER TRADE — NO BYBIT ORDER WILL BE SENT"


class ManualTradeLimitExceeded(ValueError):
    pass


class PriceChangedBeyondTolerance(ValueError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe internal paper trading diagnostic")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preview", action="store_true")
    modes.add_argument("--execute-paper", action="store_true")
    modes.add_argument("--status", action="store_true")
    parser.add_argument("--side", choices=("buy", "sell"))
    parser.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"))
    parser.add_argument("--usdt", help="Virtual USDT budget for BUY")
    parser.add_argument("--quantity", help="Asset quantity for SELL")
    parser.add_argument("--all", action="store_true", help="SELL the full virtual position")
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    trade_arguments = (args.side, args.symbol, args.usdt, args.quantity)
    if args.status:
        if any(value is not None for value in trade_arguments) or args.all:
            parser.error("--status cannot be combined with trade arguments")
        return
    if args.side is None or args.symbol is None:
        parser.error("--side and --symbol are required")
    if args.side == "buy" and (args.usdt is None or args.quantity is not None or args.all):
        parser.error("BUY requires --usdt only")
    if args.side == "sell" and ((args.quantity is None) == (not args.all) or args.usdt is not None):
        parser.error("SELL requires exactly one of --quantity or --all")


def _show_status(portfolio: Portfolio, output: Callable[[str], None]) -> None:
    output(f"Virtual USDT: {portfolio.cash}")
    output(f"BTC quantity: {portfolio.btc}")
    output(f"ETH quantity: {portfolio.eth}")
    output(f"BTC average entry: {portfolio.average_entry_price('BTCUSDT')}")
    output(f"ETH average entry: {portfolio.average_entry_price('ETHUSDT')}")
    output(f"Realized P&L: {portfolio.realized_pnl}")
    output(f"Total commissions: {portfolio.total_commission}")
    output(f"Total slippage: {portfolio.total_slippage}")
    output(f"Ledger entries: {len(portfolio.ledger)}")


def _show_plan(plan: PaperPreview, portfolio: Portfolio, output: Callable[[str], None]) -> None:
    output(f"Symbol: {plan.symbol}")
    output(f"Side: {plan.side}")
    output(f"Market bid: {plan.bid}")
    output(f"Market ask: {plan.ask}")
    output(f"Execution price: {plan.execution_price}")
    output(f"Quantity: {plan.quantity}")
    output(f"Gross: {plan.gross_value}")
    output(f"Commission: {plan.commission}")
    output(f"Slippage cost: {plan.slippage_cost}")
    output(f"Virtual USDT now: {portfolio.cash}")
    output(f"Virtual USDT after: {plan.usdt_balance_after}")
    output(f"Position now: {portfolio.quantity(plan.symbol)}")
    output(f"Position after: {plan.asset_quantity_after}")
    if plan.equity is not None:
        output(f"Projected equity: {plan.equity}")


def _enforce_limit(plan: PaperPreview, maximum: Decimal) -> None:
    notional = plan.gross_value + plan.commission if plan.side == "BUY" else plan.quantity * plan.bid
    if notional > maximum:
        raise ManualTradeLimitExceeded


def _refresh_plan(
    engine: PaperTradingEngine,
    original: PaperPreview,
    *,
    buy_usdt: str | None,
    maximum: Decimal,
    price_tolerance_bps: Decimal,
) -> PaperPreview:
    """Fetch a new quote after confirmation and reject an unexpected price move."""
    if original.side == "BUY":
        refreshed = engine.preview_buy(original.symbol, buy_usdt)
    else:
        refreshed = engine.preview_sell(original.symbol, original.quantity)
    _enforce_limit(refreshed, maximum)
    price_change_bps = (
        abs(refreshed.execution_price - original.execution_price)
        / original.execution_price
        * Decimal("10000")
    )
    if price_change_bps > price_tolerance_bps:
        raise PriceChangedBeyondTolerance
    return refreshed


def _safe_failure_message(error: Exception) -> str:
    if isinstance(error, PriceChangedBeyondTolerance):
        return "Paper trade rejected: price changed beyond allowed tolerance. Run preview again."
    if isinstance(error, ManualTradeLimitExceeded):
        return "Paper trade rejected: manual trade limit exceeded."
    if isinstance(error, InsufficientFunds):
        return "Paper trade rejected: insufficient virtual USDT balance."
    if isinstance(error, InsufficientAssets):
        return "Paper trade rejected: insufficient virtual asset balance."
    if isinstance(error, sqlite3.Error):
        return "Paper trade rejected: local database unavailable."
    if isinstance(error, DataIntegrityError):
        return "Paper trade rejected: stored Portfolio state is invalid."
    if isinstance(error, MarketDataUnavailable):
        return "Paper trade rejected: market data unavailable."
    if isinstance(error, SafetyViolation):
        return "Paper trade rejected: symbol or category is not allowed."
    if isinstance(error, PortfolioError):
        message = str(error).lower()
        if "stale" in message or "future" in message:
            return "Paper trade rejected: quote expired or has an invalid timestamp."
        if "changed after preview" in message:
            return "Paper trade rejected: Portfolio state changed after preview."
        return "Paper trade rejected: Portfolio validation failed."
    if isinstance(error, (ValueError, DecimalException, OSError)):
        return "Paper trade rejected: local configuration is invalid."
    return "Paper trade rejected: unexpected safe failure."


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    portfolio: Portfolio | None = None
    try:
        settings = load_settings()
        portfolio = Portfolio.open(settings)
        if args.status:
            _show_status(portfolio, output)
            return 0
        provider = MarketDataClient(BybitReadOnlyClient(settings))
        engine = PaperTradingEngine(provider, portfolio, PaperExecutionConfig.from_settings(settings))
        if args.side == "buy":
            plan = engine.preview_buy(args.symbol, args.usdt)
        else:
            plan = engine.preview_sell(args.symbol, args.quantity, sell_all=args.all)
        _enforce_limit(plan, settings.max_manual_paper_trade_usdt)
        _show_plan(plan, portfolio, output)
        if args.preview:
            output("Preview only: Portfolio and Ledger unchanged; no Bybit order sent.")
            return 0
        output(PAPER_WARNING)
        try:
            confirmation = input_fn(f"Type {CONFIRMATION} to execute this internal paper trade: ")
        except (EOFError, KeyboardInterrupt):
            confirmation = ""
        if confirmation != CONFIRMATION:
            output("Paper trade cancelled. Portfolio and Ledger unchanged.")
            return 2
        refreshed = _refresh_plan(
            engine,
            plan,
            buy_usdt=args.usdt,
            maximum=settings.max_manual_paper_trade_usdt,
            price_tolerance_bps=settings.max_manual_price_change_bps,
        )
        if refreshed.execution_price != plan.execution_price:
            output(f"Fresh execution price after confirmation: {refreshed.execution_price}")
        result = engine.execute_preview(refreshed)
        output(f"Paper trade saved: {result.trade_id}")
        output(f"Virtual USDT after: {result.usdt_balance_after}")
        output(f"Position after: {result.asset_quantity_after}")
        return 0
    except Exception as exc:
        output(_safe_failure_message(exc))
        return 1
    finally:
        if portfolio is not None:
            portfolio.close()


if __name__ == "__main__":
    raise SystemExit(main())
