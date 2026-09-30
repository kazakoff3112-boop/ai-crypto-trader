"""Read-only chronological history of internal paper operations."""

from __future__ import annotations

from ..analytics import build_trade_history
from ..config import load_settings
from ..portfolio import Portfolio


def main() -> int:
    portfolio: Portfolio | None = None
    try:
        settings = load_settings()
        portfolio = Portfolio.open(settings, read_only=True)
        history = build_trade_history(portfolio.ledger)
        print("PAPER TRADE HISTORY")
        for entry in history.entries:
            print(
                f"{entry.timestamp_utc.isoformat()} | {entry.id} | {entry.symbol} | {entry.side} | "
                f"quantity={entry.quantity} | bid={entry.market_bid if entry.market_bid is not None else 'N/A'} | "
                f"ask={entry.market_ask if entry.market_ask is not None else 'N/A'} | "
                f"execution={entry.execution_price} | gross={entry.gross_value} | "
                f"commission={entry.commission} | slippage={entry.slippage_cost} | "
                f"realized P&L={entry.realized_pnl}"
            )
        print(f"BUY operations: {history.summary.buy_operations}")
        print(f"SELL operations: {history.summary.sell_operations}")
        print(f"Total commissions: {history.summary.total_commissions}")
        print(f"Total slippage: {history.summary.total_slippage}")
        print(f"Cumulative realized P&L: {history.summary.cumulative_realized_pnl}")
        return 0
    except Exception:
        print("Paper history unavailable: verify local database integrity.")
        return 1
    finally:
        if portfolio is not None:
            portfolio.close()


if __name__ == "__main__":
    raise SystemExit(main())
