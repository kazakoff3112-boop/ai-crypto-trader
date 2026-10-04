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
from ..notifications.store import NotificationStore, notification_summary
from ..notifications.telegram import TelegramNotifier
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


def _notification_context(settings: Settings, store: ShadowStore):
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return None
    try:
        status = store.status()
        return (NotificationStore(settings.shadow_notifications_path, status.last_candle_utc),
                TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id))
    except (sqlite3.Error, OSError, ValueError):
        print("Telegram notifications unavailable: local notification storage")
        return None


def _deliver(context, store: ShadowStore, *, failure: str | None = None,
             details: str = "", candle: str | None = None,
             healthy: bool = False) -> None:
    if context is None:
        return
    outbox, notifier = context
    try:
        if failure:
            outbox.record_failure(failure, details, candle)
        try:
            outbox.observe_decisions(store.decisions_since(outbox.cursor()))
        except (sqlite3.Error, OSError, ValueError, RuntimeError):
            if not failure:
                raise
        if healthy:
            outbox.record_healthy()
        outbox.flush(notifier)
    except (sqlite3.Error, OSError, ValueError, RuntimeError):
        # Shadow decisions and their exit codes are authoritative. Never print
        # an exception from notification storage or a network client.
        print("Telegram notifications unavailable; shadow result preserved")


def _deliver_without_shadow(settings: Settings, category: str, details: str) -> None:
    """Report a broken shadow database through an already established outbox."""
    if (not settings.telegram_bot_token or not settings.telegram_chat_id
            or not settings.shadow_notifications_path.is_file()):
        return
    try:
        outbox = NotificationStore(settings.shadow_notifications_path, None)
        outbox.record_failure(category, details)
        outbox.flush(TelegramNotifier(settings.telegram_bot_token,
                                      settings.telegram_chat_id))
    except (sqlite3.Error, OSError, ValueError, RuntimeError):
        print("Telegram notifications unavailable; shadow result preserved")


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
    store = None
    context = None
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
            store = ShadowStore(settings.shadow_database_path,
                                read_only=args.status and settings.shadow_database_path.is_file())
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
                enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)
                print(f"Telegram notifications: {'enabled' if enabled else 'disabled'}")
                try:
                    pending, kind, sent = (notification_summary(settings.shadow_notifications_path)
                                           if enabled else (0, None, None))
                    print(f"Pending notifications: {pending}")
                    print(f"Last notification: {kind or 'none'} / {sent or 'not sent'}")
                except (sqlite3.Error, OSError):
                    print("Pending notifications: unavailable")
                    print("Last notification: unavailable")
                return ShadowExitCode.SUCCESS_NEW_DECISION
            if args.once or args.execute:
                context = _notification_context(settings, store)
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
                if context:
                    try:
                        context[0].prepare_catchup(plan.decisions)
                    except (sqlite3.Error, OSError):
                        print("Telegram notifications unavailable: catch-up provenance")
                updated = runner.execute_catch_up(plan)
                _print_catch_up_summary(plan, updated.state)
                _deliver(context, store, healthy=health(updated, datetime.now(timezone.utc)).startswith("HEALTHY"))
                return ShadowExitCode.SUCCESS_NEW_DECISION
            result = runner.once()
            print(f"Strategy version: {STRATEGY_VERSION}")
            print(f"Result: {result.action}")
            print(f"Last processed candle: {result.status.last_candle_utc.isoformat()}")
            print(f"Shadow state: {result.status.state}")
            outcome = (ShadowExitCode.SUCCESS_NEW_DECISION if result.processed
                       else ShadowExitCode.SUCCESS_ALREADY_PROCESSED)
            print(f"Outcome: {outcome.name}")
            _deliver(context, store,
                     healthy=health(result.status, datetime.now(timezone.utc)).startswith("HEALTHY"))
            return outcome
    except ShadowLockBusy:
        print("Outcome: SKIPPED_LOCK_BUSY; another shadow run is active")
        return ShadowExitCode.SKIPPED_LOCK_BUSY
    except ShadowGapUnavailable:
        if store is not None:
            status = store.status()
            _deliver(context, store, failure="GAP",
                     details=(f"Last processed candle: {status.last_candle_utc.isoformat() if status.last_candle_utc else 'none'}\n"
                              f"Latest closed candle: {datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()}\n"
                              "Automatic processing stopped. Controlled catch-up required."),
                     candle=status.last_candle_utc.isoformat() if status.last_candle_utc else None)
        print("Outcome: GAP_REQUIRES_CATCHUP; use the manual controlled catch-up procedure")
        return ShadowExitCode.GAP_REQUIRES_CATCHUP
    except HistoricalDownloadUnavailable as exc:
        outcome = (ShadowExitCode.NETWORK_TEMPORARY_FAILURE
                   if exc.category in _NETWORK_CATEGORIES
                   else ShadowExitCode.DATA_VALIDATION_FAILURE)
        if store is not None:
            _deliver(context, store, failure=("NETWORK_ERROR" if outcome == ShadowExitCode.NETWORK_TEMPORARY_FAILURE
                                              else "DATA_ERROR"),
                     details="Public market data unavailable; shadow processing paused.")
        print(f"Outcome: {outcome.name}; {exc}")
        return outcome
    except (ShadowUnavailable, HistoricalDataUnavailable, RegimeUnavailable) as exc:
        if store is not None:
            _deliver(context, store, failure="DATA_ERROR",
                     details="Market data validation failed; shadow processing paused.")
        else:
            _deliver_without_shadow(settings, "DATA_ERROR",
                                    "Shadow data validation failed; operator check required.")
        print(f"Outcome: DATA_VALIDATION_FAILURE; {exc}")
        return ShadowExitCode.DATA_VALIDATION_FAILURE
    except (sqlite3.Error, OSError):
        if store is not None:
            _deliver(context, store, failure="STORAGE_ERROR",
                     details="Local shadow storage unavailable; operator check required.")
        else:
            _deliver_without_shadow(settings, "STORAGE_ERROR",
                                    "Local shadow storage unavailable; operator check required.")
        print("Outcome: LOCAL_STORAGE_FAILURE; check shadow SQLite and filesystem")
        return ShadowExitCode.LOCAL_STORAGE_FAILURE
    except ValueError:
        print("Outcome: CONFIGURATION_INVALID; check local Demo configuration")
        return ShadowExitCode.CONFIGURATION_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
