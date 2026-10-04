"""Telegram shadow alerts are offline, persistent and independent of trading."""

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
import hashlib
import json
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.shadow_store import ShadowStore, new_decision
from ai_crypto_trader.diagnostics import shadow_runner as shadow_cli
from ai_crypto_trader.diagnostics.shadow_runner import main
from ai_crypto_trader.diagnostics.telegram_notifications import main as test_message_main
from ai_crypto_trader.market_data.candle import Candle
from ai_crypto_trader.market_data.historical import HistoricalDownloadUnavailable, HistoricalMarketDataClient
from ai_crypto_trader.market_data.market_regime import MarketRegime
from ai_crypto_trader.notifications.store import NotificationStore, notification_summary
from ai_crypto_trader.notifications.telegram import TelegramDeliveryError, TelegramNotifier
from ai_crypto_trader.trading_engine.shadow import ShadowCatchUpPlan, ShadowRunResult, ShadowRunner

D = Decimal
END = datetime(2024, 2, 1, 12, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def decision(at, action, before, after, previous, current):
    candle = Candle("BTCUSDT", "1h", at - HOUR, D("100"), D("200"),
                    D("100"), D("200"), D("1"), D("1"))
    return new_decision(candle=candle, decision_utc=at + timedelta(minutes=3),
                        previous_regime=previous, current_regime=current,
                        action=action, reason="test", sma50=D("100"),
                        sma200=D("100"), previous_sma200=D("100"),
                        state_before=before, state_after=after)


BUY = lambda: decision(END, "BUY_SIGNAL", "FLAT", "LONG",
                       MarketRegime.SIDEWAYS, MarketRegime.BULL)
HOLD = lambda: decision(END + HOUR, "HOLD", "LONG", "LONG",
                        MarketRegime.BULL, MarketRegime.BULL)
SELL = lambda: decision(END + 2 * HOUR, "SELL_SIGNAL", "LONG", "FLAT",
                        MarketRegime.BULL, MarketRegime.BEAR)


class FakeSender:
    def __init__(self, fail=False):
        self.messages = []
        self.fail = fail

    def send(self, message):
        self.messages.append(message)
        if self.fail:
            raise TelegramDeliveryError("network_error")


class NotificationStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "alerts.sqlite3"
        self.paper = Path(self.tmp.name) / "paper.sqlite3"
        self.paper.write_bytes(b"paper-sentinel")
        self.outbox = NotificationStore(self.path, None)

    def rows(self):
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT event_key,event_type,delivery_status,attempt_count,"
                              "last_safe_error_category,message FROM notification_events "
                              "ORDER BY rowid").fetchall()

    def test_buy_sell_hold_dedup_and_separate_paper(self):
        self.outbox.observe_decisions((BUY(), HOLD(), SELL()))
        self.outbox.observe_decisions((BUY(), HOLD(), SELL()))
        rows = self.rows()
        self.assertEqual([r[1] for r in rows], ["BUY_SIGNAL", "SELL_SIGNAL"])
        self.assertIn("SIDEWAYS → BULL", rows[0][5])
        self.assertIn("LONG → FLAT", rows[1][5])
        self.assertTrue(all("NO TRADE EXECUTED" in r[5] for r in rows))
        self.assertEqual(self.paper.read_bytes(), b"paper-sentinel")
        self.assertEqual(self.outbox.cursor(), END + 2 * HOUR)

    def test_pending_retry_sent_and_restart(self):
        self.outbox.observe_decisions((BUY(),))
        failed = FakeSender(fail=True)
        self.outbox.flush(failed)
        self.assertEqual((self.rows()[0][2], self.rows()[0][3], self.rows()[0][4]),
                         ("pending", 1, "network_error"))
        restarted = NotificationStore(self.path, END)
        good = FakeSender()
        restarted.flush(good)
        self.assertEqual((self.rows()[0][2], self.rows()[0][3]), ("sent", 2))
        restarted.flush(good)
        self.assertEqual(len(good.messages), 1)
        self.assertEqual(notification_summary(self.path)[0], 0)

    def test_health_incidents_are_persistent_and_recover_once(self):
        self.outbox.record_failure("GAP", "controlled catch-up required", END.isoformat())
        NotificationStore(self.path, None).record_failure("GAP", "same gap")
        self.outbox.record_healthy()
        self.outbox.record_healthy()
        self.outbox.record_failure("NETWORK_ERROR", "public GET unavailable")
        self.outbox.record_failure("NETWORK_ERROR", "same network issue")
        self.outbox.record_healthy()
        self.outbox.record_failure("DATA_ERROR", "invalid candle")
        self.outbox.record_failure("STORAGE_ERROR", "SQLite unavailable")
        kinds = [r[1] for r in self.rows()]
        self.assertEqual(kinds, ["GAP", "RECOVERED", "NETWORK_ERROR", "RECOVERED",
                                 "DATA_ERROR", "STORAGE_ERROR"])
        self.assertEqual(kinds.count("RECOVERED"), 2)

    def test_catchup_provenance_and_repeated_reconciliation(self):
        self.outbox.prepare_catchup((BUY(), HOLD(), SELL()))
        self.outbox.observe_decisions((BUY(), HOLD(), SELL()))
        self.outbox.observe_decisions((BUY(), HOLD(), SELL()))
        self.assertEqual(len(self.rows()), 2)
        self.assertTrue(all("SHADOW SIGNAL RECOVERED DURING CATCH-UP" in r[5]
                            for r in self.rows()))

    def test_uncommitted_catchup_intent_is_not_delivered(self):
        self.outbox.prepare_catchup((BUY(),))
        self.outbox.observe_decisions(())
        sender = FakeSender()
        self.outbox.flush(sender)
        self.assertEqual(sender.messages, [])
        self.assertEqual(self.rows(), [])

    def test_status_is_read_only_and_no_credentials_in_sqlite(self):
        self.assertEqual(notification_summary(Path(self.tmp.name) / "missing.sqlite3"),
                         (0, None, None))
        self.outbox.observe_decisions((BUY(),))
        before = self.path.read_bytes()
        self.assertEqual(notification_summary(self.path)[0], 1)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertNotIn(b"test-secret-token", before)
        self.assertNotIn(b"test-chat-id", before)


class TelegramTransportTests(unittest.TestCase):
    def test_only_sendmessage_post_and_safe_response(self):
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, count): return b'{"ok":true,"result":{}}'
        observed = []
        def opener(req, timeout):
            observed.append((req.full_url, req.get_method(), req.data, timeout))
            return Response()
        TelegramNotifier("test-secret-token", "test-chat-id", opener=opener).send("hello")
        self.assertEqual(len(observed), 1)
        self.assertTrue(observed[0][0].endswith("/sendMessage"))
        self.assertEqual(observed[0][1], "POST")
        self.assertIn(b"chat_id=test-chat-id", observed[0][2])

    def test_secret_and_chat_never_leak_from_transport_errors(self):
        def bad(_req, timeout):
            raise URLError("test-secret-token test-chat-id")
        sender = TelegramNotifier("test-secret-token", "test-chat-id", opener=bad)
        self.assertNotIn("test-secret-token", repr(sender))
        with self.assertRaises(TelegramDeliveryError) as caught:
            sender.send("test")
        self.assertEqual(str(caught.exception), "network_error")
        self.assertNotIn("test-secret-token", repr(caught.exception))
        self.assertNotIn("test-chat-id", repr(caught.exception))

    def test_http_malformed_api_and_timeout_are_safe(self):
        class Response:
            status = 200
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, count): return self.body
        for body, category in ((b"no-json", "invalid_response"),
                               (b'{"ok":false}', "api_error")):
            with self.subTest(body=body):
                with self.assertRaisesRegex(TelegramDeliveryError, category):
                    TelegramNotifier("token", "chat", opener=lambda *a, **kw: Response(body)).send("x")
        with self.assertRaisesRegex(TelegramDeliveryError, "http_error"):
            TelegramNotifier("token", "chat", opener=lambda *a, **kw: (_ for _ in ()).throw(
                HTTPError("secret-url", 500, "bad", None, None))).send("x")
        with self.assertRaisesRegex(TelegramDeliveryError, "network_error"):
            TelegramNotifier("token", "chat", opener=lambda *a, **kw: (_ for _ in ()).throw(
                TimeoutError("secret-url"))).send("x")


class ShadowNotificationIntegrationTests(unittest.TestCase):
    def test_gap_once_repeated_and_recovery_once(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "shadow.sqlite3"
            alerts = Path(directory) / "alerts.sqlite3"
            store = ShadowStore(shadow)
            store.append(BUY())
            env = {"SHADOW_DATABASE_PATH": str(shadow), "SHADOW_NOTIFICATIONS_PATH": str(alerts),
                   "AI_CRYPTO_TELEGRAM_BOT_TOKEN": "test-secret-token",
                   "AI_CRYPTO_TELEGRAM_CHAT_ID": "test-chat-id"}
            with patch.dict("os.environ", env), \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network")), \
                 patch.object(TelegramNotifier, "send") as send:
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--once"]), 21)
                    self.assertEqual(main(["--once"]), 21)
                with patch.object(ShadowRunner, "once", return_value=ShadowRunResult(
                    False, "ALREADY_PROCESSED", store.status())), \
                     patch.object(shadow_cli, "health", return_value="HEALTHY"):
                    with redirect_stdout(StringIO()):
                        self.assertEqual(main(["--once"]), 10)
                        self.assertEqual(main(["--once"]), 10)
            with sqlite3.connect(alerts) as db:
                kinds = [r[0] for r in db.execute("SELECT event_type FROM notification_events ORDER BY rowid")]
            self.assertEqual(kinds, ["GAP", "RECOVERED"])
            self.assertEqual(send.call_count, 2)
            self.assertEqual(store.status().processed, 1)

    def test_network_failure_dedup_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "shadow.sqlite3"
            alerts = Path(directory) / "alerts.sqlite3"
            store = ShadowStore(shadow)
            store.append(BUY())
            env = {"SHADOW_DATABASE_PATH": str(shadow), "SHADOW_NOTIFICATIONS_PATH": str(alerts),
                   "AI_CRYPTO_TELEGRAM_BOT_TOKEN": "test-secret-token",
                   "AI_CRYPTO_TELEGRAM_CHAT_ID": "test-chat-id"}
            failure = HistoricalDownloadUnavailable("timeout", END, END + HOUR)
            with patch.dict("os.environ", env), \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network")), \
                 patch.object(TelegramNotifier, "send"), \
                 patch.object(ShadowRunner, "once", side_effect=failure):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--once"]), 20)
                    self.assertEqual(main(["--once"]), 20)
            with sqlite3.connect(alerts) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM notification_events").fetchone()[0], 1)
            self.assertEqual(store.status().processed, 1)

    def test_catchup_preview_is_silent_and_execute_marks_historical_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "shadow.sqlite3"
            alerts = Path(directory) / "alerts.sqlite3"
            store = ShadowStore(shadow)
            store.append(BUY())
            plan = ShadowCatchUpPlan(store.status(), END + 2 * HOUR, (HOLD(), SELL()))
            env = {"SHADOW_DATABASE_PATH": str(shadow), "SHADOW_NOTIFICATIONS_PATH": str(alerts),
                   "AI_CRYPTO_TELEGRAM_BOT_TOKEN": "test-secret-token",
                   "AI_CRYPTO_TELEGRAM_CHAT_ID": "test-chat-id"}
            with patch.dict("os.environ", env), \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network")), \
                 patch.object(ShadowRunner, "catch_up_plan", return_value=plan), \
                 patch.object(ShadowRunner, "execute_catch_up",
                              side_effect=lambda p: store.append_batch(p.decisions,
                                                                       expected_status=p.initial_status)), \
                 patch.object(TelegramNotifier, "send") as send, \
                 patch.object(shadow_cli, "health", return_value="HEALTHY"):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--catch-up", "--preview"]), 0)
                self.assertFalse(alerts.exists())
                self.assertEqual(send.call_count, 0)
                NotificationStore(alerts, END).record_failure("GAP", "controlled catch-up")
                with patch("builtins.input", return_value="YES"), redirect_stdout(StringIO()):
                    self.assertEqual(main(["--catch-up", "--execute"]), 0)
            with sqlite3.connect(alerts) as db:
                rows = db.execute("SELECT event_type,message FROM notification_events ORDER BY rowid").fetchall()
            self.assertEqual([r[0] for r in rows], ["GAP", "SELL_SIGNAL", "RECOVERED"])
            self.assertIn("SHADOW SIGNAL RECOVERED DURING CATCH-UP", rows[1][1])
            self.assertEqual(store.status().processed, 3)

    def test_storage_failure_uses_existing_independent_outbox(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "shadow.sqlite3"
            alerts = Path(directory) / "alerts.sqlite3"
            NotificationStore(alerts, END)
            env = {"SHADOW_DATABASE_PATH": str(shadow), "SHADOW_NOTIFICATIONS_PATH": str(alerts),
                   "AI_CRYPTO_TELEGRAM_BOT_TOKEN": "test-secret-token",
                   "AI_CRYPTO_TELEGRAM_CHAT_ID": "test-chat-id"}
            with patch.dict("os.environ", env), \
                 patch.object(shadow_cli, "ShadowStore", side_effect=sqlite3.DatabaseError("broken")), \
                 patch.object(TelegramNotifier, "send") as send, \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--once"]), 23)
            self.assertEqual(send.call_count, 1)
            with sqlite3.connect(alerts) as db:
                self.assertEqual(db.execute("SELECT event_type FROM notification_events").fetchone()[0],
                                 "STORAGE_ERROR")
            self.assertFalse(shadow.exists())

    def test_committed_signal_is_reconciled_after_crash_before_enqueue(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "shadow.sqlite3"
            alerts = Path(directory) / "alerts.sqlite3"
            store = ShadowStore(shadow)
            NotificationStore(alerts, None)  # Created before the shadow commit.
            store.append(BUY())                 # Simulate crash immediately after commit.
            env = {"SHADOW_DATABASE_PATH": str(shadow), "SHADOW_NOTIFICATIONS_PATH": str(alerts),
                   "AI_CRYPTO_TELEGRAM_BOT_TOKEN": "test-secret-token",
                   "AI_CRYPTO_TELEGRAM_CHAT_ID": "test-chat-id"}
            with patch.dict("os.environ", env), \
                 patch.object(ShadowRunner, "once", return_value=ShadowRunResult(
                     False, "ALREADY_PROCESSED", store.status())), \
                 patch.object(shadow_cli, "health", return_value="HEALTHY"), \
                 patch.object(TelegramNotifier, "send") as send, \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--once"]), 10)
            self.assertEqual(send.call_count, 1)
            self.assertEqual(notification_summary(alerts)[0], 0)

    def test_status_keeps_existing_shadow_db_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "shadow.sqlite3"
            ShadowStore(shadow).append(BUY())
            before = hashlib.sha256(shadow.read_bytes()).hexdigest()
            with patch.dict("os.environ", {"SHADOW_DATABASE_PATH": str(shadow),
                                        "AI_CRYPTO_TELEGRAM_BOT_TOKEN": "",
                                        "AI_CRYPTO_TELEGRAM_CHAT_ID": ""}), \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--status"]), 0)
            self.assertEqual(hashlib.sha256(shadow.read_bytes()).hexdigest(), before)

    def test_status_reports_pending_without_disclosing_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "shadow.sqlite3"
            alerts = Path(directory) / "alerts.sqlite3"
            ShadowStore(shadow).append(BUY())
            NotificationStore(alerts, None).observe_decisions((BUY(),))
            env = {"SHADOW_DATABASE_PATH": str(shadow), "SHADOW_NOTIFICATIONS_PATH": str(alerts),
                   "AI_CRYPTO_TELEGRAM_BOT_TOKEN": "test-secret-token",
                   "AI_CRYPTO_TELEGRAM_CHAT_ID": "test-chat-id"}
            output = StringIO()
            with patch.dict("os.environ", env), \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network")), \
                 redirect_stdout(output):
                self.assertEqual(main(["--status"]), 0)
            self.assertIn("Telegram notifications: enabled", output.getvalue())
            self.assertIn("Pending notifications: 1", output.getvalue())
            self.assertIn("Last notification: BUY_SIGNAL / not sent", output.getvalue())
            self.assertNotIn("test-secret-token", output.getvalue())
            self.assertNotIn("test-chat-id", output.getvalue())

    def test_failed_telegram_does_not_rollback_signal_and_already_processed_dedups(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "shadow.sqlite3"
            alerts = Path(directory) / "alerts.sqlite3"
            paper = Path(directory) / "paper.sqlite3"
            paper.write_bytes(b"paper-sentinel")
            def candles(symbol, interval, start, end, **kwargs):
                return [Candle("BTCUSDT", "1h", start + i * HOUR,
                               D("100"), D("200") if i == 224 else D("100"),
                               D("100"), D("200") if i == 224 else D("100"),
                               D("1"), D("1")) for i in range(225)]
            env = {"SHADOW_DATABASE_PATH": str(shadow), "SHADOW_NOTIFICATIONS_PATH": str(alerts),
                   "DATABASE_PATH": str(paper), "AI_CRYPTO_TELEGRAM_BOT_TOKEN": "test-secret-token",
                   "AI_CRYPTO_TELEGRAM_CHAT_ID": "test-chat-id"}
            with patch.dict("os.environ", env), \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network")), \
                 patch.object(HistoricalMarketDataClient, "fetch_candles", side_effect=candles), \
                 patch.object(TelegramNotifier, "send", side_effect=TelegramDeliveryError("network_error")):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--once"]), 0)
                    self.assertEqual(main(["--once"]), 10)
            self.assertEqual(ShadowStore(shadow).status().processed, 1)
            with sqlite3.connect(alerts) as db:
                self.assertEqual(db.execute("SELECT COUNT(*),MAX(attempt_count) FROM notification_events")
                                 .fetchone(), (1, 2))
            self.assertEqual(paper.read_bytes(), b"paper-sentinel")
            for name in ("create_order", "amend_order", "cancel_order", "place_order"):
                self.assertFalse(hasattr(BybitReadOnlyClient, name))

    def test_missing_config_and_manual_test_do_not_touch_shadow(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "shadow.sqlite3"
            with patch.dict("os.environ", {"SHADOW_DATABASE_PATH": str(shadow),
                                        "SHADOW_NOTIFICATIONS_PATH": str(Path(directory)/"alerts.sqlite3"),
                                        "AI_CRYPTO_TELEGRAM_BOT_TOKEN": "",
                                        "AI_CRYPTO_TELEGRAM_CHAT_ID": ""}), \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["--status"]), 0)
                    self.assertEqual(test_message_main(["--test"]), 2)
                self.assertIn("Telegram notifications: disabled", output.getvalue())
                self.assertFalse((Path(directory)/"alerts.sqlite3").exists())
                self.assertEqual(ShadowStore(shadow).status().processed, 0)

    def test_manual_test_uses_exact_message_without_shadow_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "shadow.sqlite3"
            env = {"SHADOW_DATABASE_PATH": str(shadow),
                   "AI_CRYPTO_TELEGRAM_BOT_TOKEN": "test-secret-token",
                   "AI_CRYPTO_TELEGRAM_CHAT_ID": "test-chat-id"}
            with patch.dict("os.environ", env), \
                 patch.object(TelegramNotifier, "send") as send, \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
                with redirect_stdout(StringIO()):
                    self.assertEqual(test_message_main(["--test"]), 0)
            self.assertEqual(send.call_count, 1)
            self.assertIn("NO TRADE EXECUTED", send.call_args.args[0])
            self.assertFalse(shadow.exists())
