"""Offline checks for scheduled shadow outcomes and process exclusivity."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.shadow_store import ShadowGapUnavailable, ShadowStore
from ai_crypto_trader.diagnostics.shadow_lock import ShadowLockBusy, shadow_process_lock
from ai_crypto_trader.diagnostics.shadow_runner import ShadowExitCode, main
from ai_crypto_trader.market_data.historical import HistoricalDownloadUnavailable
from ai_crypto_trader.market_data.market_regime import RegimeUnavailable
from ai_crypto_trader.trading_engine.shadow import ShadowRunner


class ScheduledShadowTests(unittest.TestCase):
    def _once_with_failure(self, failure):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.sqlite3"
            output = StringIO()
            settings = Settings(shadow_database_path=path)
            with patch("ai_crypto_trader.diagnostics.shadow_runner.load_settings",
                       return_value=settings):
                with patch.object(socket.socket, "connect",
                                  side_effect=AssertionError("network called")):
                    with patch.object(ShadowRunner, "once", side_effect=failure):
                        with redirect_stdout(output):
                            code = main(["--once"])
            self.assertEqual(ShadowStore(path).status().processed, 0)
            return code, output.getvalue()

    def test_scheduled_failure_categories_are_distinct_and_safe(self):
        start = datetime(2026, 10, 1, 8, tzinfo=timezone.utc)
        end = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
        cases = (
            (HistoricalDownloadUnavailable("timeout", start, end,
                                           detail="Bybit request timed out"),
             ShadowExitCode.NETWORK_TEMPORARY_FAILURE),
            (HistoricalDownloadUnavailable("page_count_mismatch", start, end),
             ShadowExitCode.DATA_VALIDATION_FAILURE),
            (ShadowGapUnavailable("controlled catch-up required"),
             ShadowExitCode.GAP_REQUIRES_CATCHUP),
            (RegimeUnavailable("Invalid hourly candle"),
             ShadowExitCode.DATA_VALIDATION_FAILURE),
            (sqlite3.OperationalError("synthetic disk failure"),
             ShadowExitCode.LOCAL_STORAGE_FAILURE),
        )
        for failure, expected in cases:
            with self.subTest(outcome=expected.name):
                code, output = self._once_with_failure(failure)
                self.assertEqual(code, expected)
                self.assertIn(f"Outcome: {expected.name}", output)
                self.assertNotIn("synthetic disk failure", output)

    def test_busy_lock_prevents_second_cli_run_and_does_not_touch_paper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.sqlite3"
            paper = Path(directory) / "paper.sqlite3"
            paper.write_bytes(b"existing paper ledger")
            settings = Settings(shadow_database_path=path, database_path=paper)
            output = StringIO()
            with shadow_process_lock(path):
                with self.assertRaises(ShadowLockBusy):
                    with shadow_process_lock(path):
                        pass
                with patch("ai_crypto_trader.diagnostics.shadow_runner.load_settings",
                           return_value=settings):
                    with patch.object(ShadowRunner, "once",
                                      side_effect=AssertionError("concurrent run")):
                        with redirect_stdout(output):
                            self.assertEqual(main(["--once"]), ShadowExitCode.SKIPPED_LOCK_BUSY)
            self.assertIn("SKIPPED_LOCK_BUSY", output.getvalue())
            self.assertFalse(path.exists())
            self.assertEqual(paper.read_bytes(), b"existing paper ledger")

    def test_systemd_templates_schedule_only_shadow_once(self):
        directory = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
        service = (directory / "ai-crypto-shadow.service").read_text(encoding="utf-8")
        timer = (directory / "ai-crypto-shadow.timer").read_text(encoding="utf-8")
        self.assertIn("Type=oneshot", service)
        self.assertIn("ExecStart=/opt/ai-crypto-trader/.venv/bin/python -m "
                      "ai_crypto_trader.diagnostics.shadow_runner --once", service)
        self.assertIn("SuccessExitStatus=10 75", service)
        self.assertIn("--status --require-baseline", service)
        self.assertIn("ProtectSystem=strict", service)
        self.assertIn("ReadWritePaths=/opt/ai-crypto-trader/data", service)
        self.assertIn("OnCalendar=*-*-* *:00/10:00", timer)
        self.assertIn("Unit=ai-crypto-shadow.service", timer)
        for forbidden in ("--catch-up", "--execute", "paper_trade", "place_order",
                          "wallet", "withdraw"):
            self.assertNotIn(forbidden, service + timer)
        for name in ("create_order", "place_order", "amend_order", "cancel_order"):
            self.assertFalse(hasattr(BybitReadOnlyClient, name))

    def test_service_preflight_refuses_missing_or_empty_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.sqlite3"
            settings = Settings(shadow_database_path=path)
            with patch("ai_crypto_trader.diagnostics.shadow_runner.load_settings",
                       return_value=settings):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--status", "--require-baseline"]),
                                     ShadowExitCode.LOCAL_STORAGE_FAILURE)
                self.assertFalse(path.exists())
                ShadowStore(path)
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(["--status", "--require-baseline"]),
                                     ShadowExitCode.LOCAL_STORAGE_FAILURE)


if __name__ == "__main__":
    unittest.main()
