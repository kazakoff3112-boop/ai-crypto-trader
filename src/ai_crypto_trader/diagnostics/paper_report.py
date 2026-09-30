"""Read-only valuation of the virtual paper Portfolio at current spot bids."""

from __future__ import annotations

from ..analytics import AssetValuation, build_portfolio_report
from ..bybit import BybitReadOnlyClient
from ..config import load_settings
from ..market_data import MarketDataClient
from ..portfolio import Portfolio


def _show_asset(label: str, asset: AssetValuation) -> None:
    print(f"{label}:")
    print(f"  Quantity: {asset.quantity}")
    print(f"  Average entry: {asset.average_entry_price}")
    print(f"  Current bid: {asset.current_bid if asset.current_bid is not None else 'N/A (no position)'}")
    print(f"  Market value: {asset.market_value}")
    print(f"  Unrealized P&L: {asset.unrealized_pnl}")


def main() -> int:
    portfolio: Portfolio | None = None
    try:
        settings = load_settings()
        portfolio = Portfolio.open(settings, read_only=True)
        market = MarketDataClient(BybitReadOnlyClient(settings))
        report = build_portfolio_report(portfolio, market)
        print("PAPER PORTFOLIO REPORT")
        print(f"Initial capital: {report.initial_capital}")
        print(f"Virtual USDT: {report.virtual_usdt}")
        _show_asset("BTC", report.btc)
        _show_asset("ETH", report.eth)
        print(f"Total market value: {report.total_market_value}")
        print(f"Total equity: {report.total_equity}")
        print(f"Realized P&L: {report.realized_pnl}")
        print(f"Unrealized P&L: {report.unrealized_pnl}")
        print(f"Total P&L: {report.total_pnl}")
        print(f"Return: {report.total_return_percent} %")
        print(f"Total commissions: {report.total_commissions}")
        print(f"Total slippage: {report.total_slippage}")
        print(f"Ledger entries: {report.ledger_entries}")
        return 0
    except Exception:
        print("Paper report unavailable: verify local database integrity and fresh spot quotes.")
        return 1
    finally:
        if portfolio is not None:
            portfolio.close()


if __name__ == "__main__":
    raise SystemExit(main())
