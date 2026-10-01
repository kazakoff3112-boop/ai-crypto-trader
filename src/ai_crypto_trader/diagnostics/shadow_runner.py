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
from ..trading_engine.shadow import ShadowCatchUpPlan, ShadowRunner, health


def _print_catch_up_plan(plan: ShadowCatchUpPlan, now: datetime) -> None:
    status = plan.initial_status
    print("SHADOW CATCH-UP — NO PAPER OR BYBIT ORDERS WILL BE SENT")
    print(f"Strategy version: {STRATEGY_VERSION}")
    print(f"Current shadow state: {status.state}")
    print(f"Last processed candle: {status.last_candle_utc.isoformat()}")
    print(f"Latest fully closed candle: {plan.latest_closed_candle.isoformat()}")
    print(f"Missing candles: {plan.missing_candles}")
    print(f"First missing candle: {plan.first_missing_candle.isoformat()}")
    print(f"Last missing candle: {plan.last_missing_candle.isoformat()}")
    print(f"Current health: {health(status, now)}")
    print(f"Decisions to simulate: {plan.missing_candles}")


def _print_catch_up_summary(plan: ShadowCatchUpPlan, final_state: str) -> None:
    print("Controlled catch-up complete")
    print(f"Processed candles: {plan.missing_candles}")
    print(f"From: {plan.first_missing_candle.isoformat()}")
    print(f"To: {plan.last_missing_candle.isoformat()}")
    print(f"BUY_SIGNAL: {plan.action_count('BUY_SIGNAL')}")
    print(f"SELL_SIGNAL: {plan.action_count('SELL_SIGNAL')}")
    print(f"HOLD: {plan.action_count('HOLD')}")
    print(f"Initial state: {plan.initial_status.state}")
    print(f"Final state: {final_state}")
    print("Paper trades sent: 0")
    print("Bybit orders sent: 0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BTCUSDT 1h shadow decisions; no trading")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--once", action="store_true", help="observe only the latest closed hour")
    modes.add_argument("--status", action="store_true", help="read local shadow state; no network")
    modes.add_argument("--catch-up", action="store_true", help="controlled processing of missed hours")
    catch_up_modes = parser.add_mutually_exclusive_group()
    catch_up_modes.add_argument("--preview", action="store_true", help="validate and show catch-up only")
    catch_up_modes.add_argument("--execute", action="store_true", help="require YES, then atomically commit")
    args = parser.parse_args(argv)
    if args.catch_up and not (args.preview or args.execute):
        parser.error("--catch-up requires --preview or --execute")
    if not args.catch_up and (args.preview or args.execute):
        parser.error("--preview/--execute require --catch-up")
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
        runner = ShadowRunner(store, market)
        if args.catch_up:
            plan = runner.catch_up_plan()
            now = datetime.now(timezone.utc)
            _print_catch_up_plan(plan, now)
            if args.preview:
                print("Preview only: no shadow decisions were written")
                return 0
            if input("Type YES to execute this shadow catch-up: ") != "YES":
                print("Catch-up cancelled; no changes were made")
                return 0
            updated = runner.execute_catch_up(plan)
            _print_catch_up_summary(plan, updated.state)
            return 0
        result = runner.once()
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
