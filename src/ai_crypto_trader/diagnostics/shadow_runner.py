"""Manual one-shot shadow observation and local status, with no trade path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sqlite3

from ..bybit.client import BybitReadOnlyClient
from ..config import Settings, load_settings
from ..database.shadow_store import STRATEGY_VERSION, ShadowStore, ShadowUnavailable
from ..market_data.candle import HistoricalDataUnavailable
from ..market_data.historical import HistoricalMarketDataClient
from ..market_data.market_regime import RegimeUnavailable
from ..trading_engine.shadow import ShadowRunner, health


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BTCUSDT 1h shadow decisions; no trading")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--once", action="store_true", help="observe only the latest closed hour")
    modes.add_argument("--status", action="store_true", help="read local shadow state; no network")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        store = ShadowStore(settings.shadow_database_path)
        if args.status:
            status = store.status()
            print(f"Strategy version: {STRATEGY_VERSION}")
            print(f"Shadow state: {status.state}")
            print(f"Last processed candle: {status.last_candle_utc.isoformat() if status.last_candle_utc else 'none'}")
            print(f"Last regime: {status.last_regime.value if status.last_regime else 'none'}")
            print(f"Last action: {status.last_action or 'none'}")
            print(f"Total processed candles: {status.processed}")
            print(f"BUY_SIGNAL count: {status.buy_signals}")
            print(f"SELL_SIGNAL count: {status.sell_signals}")
            print(f"Health: {health(status, datetime.now(timezone.utc))}")
            return 0
        # Public GET only; no user credentials are passed to this client.
        market = HistoricalMarketDataClient(BybitReadOnlyClient(Settings()))
        result = ShadowRunner(store, market).once()
        print(f"Strategy version: {STRATEGY_VERSION}")
        print(f"Result: {result.action}")
        print(f"Last processed candle: {result.status.last_candle_utc.isoformat()}")
        print(f"Shadow state: {result.status.state}")
        return 0
    except (ShadowUnavailable, HistoricalDataUnavailable, RegimeUnavailable) as exc:
        print(f"Shadow unavailable: {exc}")
        return 1
    except (sqlite3.Error, OSError, ValueError):
        print("Shadow unavailable: local configuration or SQLite error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
