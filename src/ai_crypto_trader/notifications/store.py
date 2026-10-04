"""Independent persistent outbox and health transitions for Shadow Mode."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from uuid import uuid4

from ..database.shadow_store import ShadowDecision, STRATEGY_VERSION
from .telegram import TelegramDeliveryError, TelegramNotifier


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _signal_key(decision: ShadowDecision) -> str:
    return f"signal:{decision.strategy_version}:{decision.action}:{decision.candle_close_utc.isoformat()}"


def _signal_message(decision: ShadowDecision, recovered: bool) -> str:
    heading = "🟢 BUY SIGNAL" if decision.action == "BUY_SIGNAL" else "🔴 SELL SIGNAL"
    prefix = "SHADOW SIGNAL RECOVERED DURING CATCH-UP\n" if recovered else ""
    return (f"{prefix}AI Crypto Trader — {heading}\n\n"
            f"{decision.symbol}\nStrategy: {decision.strategy_version}\n"
            f"Timeframe: {decision.timeframe}\n"
            f"Regime: {decision.previous_regime.value} → {decision.current_regime.value}\n"
            f"Shadow state: {decision.state_before} → {decision.state_after}\n"
            f"Closed candle: {decision.candle_close_utc.isoformat()}\n"
            f"Price: {decision.close}\n\nNO TRADE EXECUTED\nShadow Mode only")


class NotificationStore:
    def __init__(self, path: Path, initial_candle: datetime | None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS notification_events (
                notification_id TEXT PRIMARY KEY, event_key TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL, candle_utc TEXT, message TEXT NOT NULL,
                created_utc TEXT NOT NULL, sent_utc TEXT,
                delivery_status TEXT NOT NULL CHECK(delivery_status IN ('pending','sent')),
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_safe_error_category TEXT)""")
            db.execute("""CREATE TABLE IF NOT EXISTS notification_state (
                id INTEGER PRIMARY KEY CHECK(id = 1), last_observed_candle TEXT,
                active_health TEXT, incident INTEGER NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE IF NOT EXISTS catchup_intents (
                event_key TEXT PRIMARY KEY)""")
            db.execute("""INSERT OR IGNORE INTO notification_state
                (id, last_observed_candle, active_health, incident) VALUES (1, ?, NULL, 0)""",
                       (initial_candle.isoformat() if initial_candle else None,))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def cursor(self) -> datetime | None:
        with closing(self._connect()) as db:
            value = db.execute("SELECT last_observed_candle FROM notification_state WHERE id=1").fetchone()[0]
        return datetime.fromisoformat(value) if value else None

    @staticmethod
    def _enqueue(db: sqlite3.Connection, key: str, kind: str,
                 candle: str | None, message: str) -> None:
        db.execute("""INSERT OR IGNORE INTO notification_events
            (notification_id,event_key,event_type,candle_utc,message,created_utc,delivery_status)
            VALUES (?,?,?,?,?,?,'pending')""",
                   (str(uuid4()), key, kind, candle, message, _utc()))

    def prepare_catchup(self, decisions: tuple[ShadowDecision, ...]) -> None:
        """Record signal provenance before the independent shadow DB commits."""
        with closing(self._connect()) as db, db:
            db.executemany("INSERT OR IGNORE INTO catchup_intents(event_key) VALUES (?)",
                           ((_signal_key(d),) for d in decisions
                            if d.action in ("BUY_SIGNAL", "SELL_SIGNAL")))

    def observe_decisions(self, decisions: tuple[ShadowDecision, ...]) -> None:
        """Reconcile committed decisions after a crash or a Telegram outage."""
        with closing(self._connect()) as db, db:
            cursor = db.execute("SELECT last_observed_candle FROM notification_state WHERE id=1").fetchone()[0]
            for decision in decisions:
                candle = decision.candle_close_utc.isoformat()
                if cursor is not None and candle <= cursor:
                    continue
                if decision.action in ("BUY_SIGNAL", "SELL_SIGNAL"):
                    key = _signal_key(decision)
                    recovered = db.execute("SELECT 1 FROM catchup_intents WHERE event_key=?",
                                           (key,)).fetchone() is not None
                    self._enqueue(db, key, decision.action, candle,
                                  _signal_message(decision, recovered))
                    db.execute("DELETE FROM catchup_intents WHERE event_key=?", (key,))
                cursor = candle
            db.execute("UPDATE notification_state SET last_observed_candle=? WHERE id=1", (cursor,))

    def record_failure(self, category: str, message: str, candle: str | None = None) -> None:
        if category not in {"GAP", "NETWORK_ERROR", "DATA_ERROR", "STORAGE_ERROR"}:
            raise ValueError("Unsupported notification health category")
        with closing(self._connect()) as db, db:
            active, incident = db.execute(
                "SELECT active_health,incident FROM notification_state WHERE id=1").fetchone()
            if active == category:
                return
            incident += 1
            db.execute("UPDATE notification_state SET active_health=?, incident=? WHERE id=1",
                       (category, incident))
            self._enqueue(db, f"health:{incident}:{category}", category, candle,
                          f"⚠️ AI Crypto Trader — {category}\n{message}\n\nNO TRADE EXECUTED")

    def record_healthy(self) -> None:
        with closing(self._connect()) as db, db:
            active, incident = db.execute(
                "SELECT active_health,incident FROM notification_state WHERE id=1").fetchone()
            if active is None:
                return
            db.execute("UPDATE notification_state SET active_health=NULL WHERE id=1")
            self._enqueue(db, f"health:{incident}:RECOVERED", "RECOVERED", None,
                          "✅ AI Crypto Trader — RECOVERED\nHealth restored. "
                          "Shadow runner is processing normally.\nNO TRADE EXECUTED")

    def flush(self, notifier: TelegramNotifier, *, limit: int = 5) -> None:
        with closing(self._connect()) as db:
            pending = db.execute("""SELECT notification_id,message FROM notification_events
                WHERE delivery_status='pending' ORDER BY created_utc, rowid LIMIT ?""",
                                 (limit,)).fetchall()
        for notification_id, message in pending:
            category = None
            try:
                notifier.send(message)
            except TelegramDeliveryError as exc:
                category = (str(exc) if str(exc) in {
                    "http_error", "network_error", "invalid_response", "api_error"
                } else "delivery_error")
            with closing(self._connect()) as db, db:
                if category is None:
                    db.execute("""UPDATE notification_events SET delivery_status='sent',
                        sent_utc=?, attempt_count=attempt_count+1,
                        last_safe_error_category=NULL WHERE notification_id=?""",
                               (_utc(), notification_id))
                else:
                    db.execute("""UPDATE notification_events SET attempt_count=attempt_count+1,
                        last_safe_error_category=? WHERE notification_id=?""",
                               (category, notification_id))


def notification_summary(path: Path) -> tuple[int, str | None, str | None]:
    """Status never creates or mutates the notification database."""
    if not path.is_file():
        return 0, None, None
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        pending = db.execute("SELECT COUNT(*) FROM notification_events WHERE delivery_status='pending'").fetchone()[0]
        last = db.execute("""SELECT event_type,sent_utc FROM notification_events
            ORDER BY created_utc DESC,rowid DESC LIMIT 1""").fetchone()
        return pending, last[0] if last else None, last[1] if last else None
