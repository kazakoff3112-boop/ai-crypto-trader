"""Manual one-shot shadow observation and local status, with no trade path."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from enum import IntEnum
import sqlite3

from ..bybit.client import BybitReadOnlyClient
from ..config import Settings, load_settings
from ..database.shadow_store import (
    STRATEGY_VERSION, ShadowGapUnavailable, ShadowStore, ShadowUnavailable,
)
from ..market_data.candle import HistoricalDataUnavailable
from ..market_data.historical import HistoricalDownloadUnavailable, HistoricalMarketDataClient
from ..market_data.market_regime import RegimeUnavailable
from ..trading_engine.shadow import ShadowCatchUpPlan, ShadowRunner, health
from .shadow_lock import ShadowLockBusy, shadow_process_lock


class ShadowExitCode(IntEnum):
    SUCCESS_NEW_DECISION = 0
    SUCCESS_ALREADY_PROCESSED = 10
    NETWORK_TEMPORARY_FAILURE = 20
    GAP_REQUIRES_CATCHUP = 21
    DATA_VALIDATION_FAILURE = 22
    LOCAL_STORAGE_FAILURE = 23
    CONFIGURATION_INVALID = 24
    SKIPPED_LOCK_BUSY = 75


_NETWORK_CATEGORIES = frozenset({
    "timeout", "connection_error", "http_error", "bybit_api_error", "transport_error",
})


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
    parser.add_argument("--require-baseline", action="store_true",
                        help="with --status, fail unless an existing shadow history is present")
    args = parser.parse_args(argv)
    if args.catch_up and not (args.preview or args.execute):
        parser.error("--catch-up requires --preview or --execute")
    if not args.catch_up and (args.preview or args.execute):
        parser.error("--preview/--execute require --catch-up")
    if args.require_baseline and not args.status:
        parser.error("--require-baseline requires --status")
    try:
        settings = load_settings()
    except (ValueError, OSError):
        print("Outcome: CONFIGURATION_INVALID; check local Demo configuration")
        return ShadowExitCode.CONFIGURATION_INVALID
    try:
        if args.require_baseline and (
            not settings.shadow_database_path.is_file()
            or settings.shadow_database_path.stat().st_size == 0
        ):
            print("Outcome: LOCAL_STORAGE_FAILURE; transferred shadow baseline is missing")
            return ShadowExitCode.LOCAL_STORAGE_FAILURE
        guard = nullcontext() if args.status else shadow_process_lock(settings.shadow_database_path)
        with guard:
            store = ShadowStore(settings.shadow_database_path)
            if args.status:
                status = store.status()
                if args.require_baseline and status.processed == 0:
                    print("Outcome: LOCAL_STORAGE_FAILURE; shadow baseline has no decisions")
                    return ShadowExitCode.LOCAL_STORAGE_FAILURE
                print(f"Strategy version: {STRATEGY_VERSION}")
                print(f"Shadow state: {status.state}")
                print(f"Last processed candle: {status.last_candle_utc.isoformat() if status.last_candle_utc else 'none'}")
                print(f"Last regime: {status.last_regime.value if status.last_regime else 'none'}")
                print(f"Last action: {status.last_action or 'none'}")
                print(f"Total processed candles: {status.processed}")
                print(f"BUY_SIGNAL count: {status.buy_signals}")
                print(f"SELL_SIGNAL count: {status.sell_signals}")
                print(f"Health: {health(status, datetime.now(timezone.utc))}")
                return ShadowExitCode.SUCCESS_NEW_DECISION
            # Public GET only; no user credentials are passed to this client.
            market = HistoricalMarketDataClient(BybitReadOnlyClient(Settings()))
            runner = ShadowRunner(store, market)
            if args.catch_up:
                plan = runner.catch_up_plan()
                now = datetime.now(timezone.utc)
                _print_catch_up_plan(plan, now)
                if args.preview:
                    print("Preview only: no shadow decisions were written")
                    return ShadowExitCode.SUCCESS_NEW_DECISION
                if input("Type YES to execute this shadow catch-up: ") != "YES":
                    print("Catch-up cancelled; no changes were made")
                    return ShadowExitCode.SUCCESS_NEW_DECISION
                updated = runner.execute_catch_up(plan)
                _print_catch_up_summary(plan, updated.state)
                return ShadowExitCode.SUCCESS_NEW_DECISION
            result = runner.once()
            print(f"Strategy version: {STRATEGY_VERSION}")
            print(f"Result: {result.action}")
            print(f"Last processed candle: {result.status.last_candle_utc.isoformat()}")
            print(f"Shadow state: {result.status.state}")
            outcome = (ShadowExitCode.SUCCESS_NEW_DECISION if result.processed
                       else ShadowExitCode.SUCCESS_ALREADY_PROCESSED)
            print(f"Outcome: {outcome.name}")
            return outcome
    except ShadowLockBusy:
        print("Outcome: SKIPPED_LOCK_BUSY; another shadow run is active")
        return ShadowExitCode.SKIPPED_LOCK_BUSY
    except ShadowGapUnavailable:
        print("Outcome: GAP_REQUIRES_CATCHUP; use the manual controlled catch-up procedure")
        return ShadowExitCode.GAP_REQUIRES_CATCHUP
    except HistoricalDownloadUnavailable as exc:
        outcome = (ShadowExitCode.NETWORK_TEMPORARY_FAILURE
                   if exc.category in _NETWORK_CATEGORIES
                   else ShadowExitCode.DATA_VALIDATION_FAILURE)
        print(f"Outcome: {outcome.name}; {exc}")
        return outcome
    except (ShadowUnavailable, HistoricalDataUnavailable, RegimeUnavailable) as exc:
        print(f"Outcome: DATA_VALIDATION_FAILURE; {exc}")
        return ShadowExitCode.DATA_VALIDATION_FAILURE
    except (sqlite3.Error, OSError):
        print("Outcome: LOCAL_STORAGE_FAILURE; check shadow SQLite and filesystem")
        return ShadowExitCode.LOCAL_STORAGE_FAILURE
    except ValueError:
        print("Outcome: CONFIGURATION_INVALID; check local Demo configuration")
        return ShadowExitCode.CONFIGURATION_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
