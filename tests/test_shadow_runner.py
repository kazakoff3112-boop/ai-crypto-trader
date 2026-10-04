"""Shadow processing uses synthetic hours and never contacts Bybit or PAPER."""

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.shadow_store import ShadowStore, ShadowUnavailable
from ai_crypto_trader.diagnostics.shadow_runner import ShadowExitCode, main
from ai_crypto_trader.market_data.candle import Candle, HistoricalDataUnavailable
from ai_crypto_trader.market_data.historical import HistoricalMarketDataClient
from ai_crypto_trader.market_data.market_regime import RegimeUnavailable
from ai_crypto_trader.trading_engine.shadow import (
    MAX_SHADOW_CATCHUP_HOURS, ShadowRunner, health,
)

D = Decimal
HOUR = timedelta(hours=1)
FIRST_END = datetime(2024, 2, 1, 12, tzinfo=timezone.utc)


def history(end, *, spike_close=True):
    first = end - 225 * HOUR
    rows = []
    for index in range(225):
        price = D("200") if spike_close and index == 224 else D("100")
        rows.append(Candle("BTCUSDT", "1h", first + index * HOUR,
                           D("100"), max(D("100"), price), D("100"), price,
                           D("1"), D("1")))
    return rows


def continuation(end, closes):
    rows = history(end, spike_close=False)
    for offset, close in enumerate(closes):
        open_time = end + offset * HOUR
        rows.append(Candle("BTCUSDT", "1h", open_time, close, close, close, close,
                           D("1"), D("1")))
    return rows


class FakeMarket:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def fetch_candles(self, symbol, interval, start, end, **kwargs):
        self.calls += 1
        assert symbol == "BTCUSDT" and interval == "1h"
        assert kwargs == {"category": "spot", "completed_only": True}
        return [row for row in self.rows if start <= row.open_time_utc < end]


class ShadowTests(unittest.TestCase):
    def setUp(self):
        disabled = patch.dict("os.environ", {"AI_CRYPTO_TELEGRAM_BOT_TOKEN": "",
                                           "AI_CRYPTO_TELEGRAM_CHAT_ID": ""})
        disabled.start()
        self.addCleanup(disabled.stop)

    def test_buy_restart_sell_and_exactly_once_without_paper_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow_path = Path(directory) / "shadow.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            paper_path.write_bytes(b"paper-ledger-sentinel")
            store = ShadowStore(shadow_path)
            first_market = FakeMarket(history(FIRST_END))
            first_runner = ShadowRunner(store, first_market,
                                        clock=lambda: FIRST_END + timedelta(minutes=5))
            buy = first_runner.once()
            self.assertEqual((buy.action, buy.status.state, buy.status.buy_signals),
                             ("BUY_SIGNAL", "LONG", 1))
            self.assertEqual(store.status().processed, 1)
            self.assertEqual(first_runner.once().action, "ALREADY_PROCESSED")
            self.assertEqual(first_market.calls, 1)
            self.assertEqual(store.status().processed, 1)

            restarted = ShadowStore(shadow_path)
            self.assertEqual(restarted.status().state, "LONG")
            next_end = FIRST_END + HOUR
            bull_candle = Candle("BTCUSDT", "1h", FIRST_END, D("200"), D("200"),
                                 D("200"), D("200"), D("1"), D("1"))
            second_market = FakeMarket([*history(FIRST_END), bull_candle])
            hold = ShadowRunner(restarted, second_market,
                                clock=lambda: next_end + timedelta(minutes=5)).once()
            self.assertEqual((hold.action, hold.status.state), ("HOLD", "LONG"))
            third_end = next_end + HOUR
            bear_candle = Candle("BTCUSDT", "1h", next_end, D("100"), D("100"),
                                 D("100"), D("100"), D("1"), D("1"))
            third_market = FakeMarket([*history(FIRST_END), bull_candle, bear_candle])
            sell = ShadowRunner(ShadowStore(shadow_path), third_market,
                                clock=lambda: third_end + timedelta(minutes=5)).once()
            self.assertEqual((sell.action, sell.status.state, sell.status.sell_signals),
                             ("SELL_SIGNAL", "FLAT", 1))
            self.assertEqual(sell.status.processed, 3)
            self.assertEqual(paper_path.read_bytes(), b"paper-ledger-sentinel")
            with sqlite3.connect(shadow_path) as db:
                rows = db.execute("SELECT action, state_before, state_after, "
                                  "typeof(open), typeof(sma50), position_would_be_open "
                                  "FROM shadow_decisions ORDER BY candle_close_utc").fetchall()
                self.assertEqual(rows[0], ("BUY_SIGNAL", "FLAT", "LONG", "text", "text", 1))
                self.assertEqual(rows[1], ("HOLD", "LONG", "LONG", "text", "text", 1))
                self.assertEqual(rows[2], ("SELL_SIGNAL", "LONG", "FLAT", "text", "text", 0))
                with self.assertRaises(sqlite3.DatabaseError):
                    db.execute("UPDATE shadow_decisions SET action='HOLD'")
                with self.assertRaises(sqlite3.DatabaseError):
                    db.execute("DELETE FROM shadow_decisions")

    def test_hold_transitions_and_future_candle_cannot_change_prior_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ShadowStore(Path(directory) / "shadow.sqlite3")
            flat = FakeMarket(history(FIRST_END, spike_close=False))
            result = ShadowRunner(store, flat, clock=lambda: FIRST_END).once()
            self.assertEqual((result.action, result.status.state), ("HOLD", "FLAT"))
            self.assertEqual(result.status.buy_signals, 0)
            # The next hour is outside the requested [start, end); it cannot alter this decision.
            future = Candle("BTCUSDT", "1h", FIRST_END, D("100"), D("900"), D("100"),
                            D("900"), D("1"), D("1"))
            other = ShadowStore(Path(directory) / "other.sqlite3")
            market = FakeMarket([*history(FIRST_END, spike_close=False), future])
            repeat = ShadowRunner(other, market, clock=lambda: FIRST_END).once()
            self.assertEqual(repeat.action, "HOLD")
            self.assertEqual(other.status().last_regime, result.status.last_regime)

    def test_gap_and_incomplete_hour_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ShadowStore(Path(directory) / "shadow.sqlite3")
            ShadowRunner(store, FakeMarket(history(FIRST_END)), clock=lambda: FIRST_END).once()
            market = FakeMarket(history(FIRST_END + 3 * HOUR))
            with self.assertRaises(ShadowUnavailable):
                ShadowRunner(store, market, clock=lambda: FIRST_END + 3 * HOUR).once()
            self.assertEqual(market.calls, 0)
            self.assertEqual(store.status().processed, 1)
            self.assertIn("GAP", health(store.status(), FIRST_END + 3 * HOUR))

            new_store = ShadowStore(Path(directory) / "incomplete.sqlite3")
            incomplete = FakeMarket(history(FIRST_END)[:-1])
            with self.assertRaises(ShadowUnavailable):
                ShadowRunner(new_store, incomplete, clock=lambda: FIRST_END).once()
            self.assertEqual(new_store.status().processed, 0)

    def test_sqlite_failure_rolls_back_decision_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.sqlite3"
            store = ShadowStore(path)
            with sqlite3.connect(path) as db:
                db.execute("""CREATE TRIGGER fail_shadow_insert BEFORE INSERT ON shadow_decisions
                    BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
            with self.assertRaises(sqlite3.DatabaseError):
                ShadowRunner(store, FakeMarket(history(FIRST_END)), clock=lambda: FIRST_END).once()
            self.assertEqual(ShadowStore(path).status().processed, 0)
            self.assertEqual(store.status().state, "FLAT")

    def test_status_is_offline_and_paths_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow_path = Path(directory) / "shadow.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            history_path = Path(directory) / "history.sqlite3"
            paper_path.write_bytes(b"paper-ledger-sentinel")
            history_path.write_bytes(b"history-sentinel")
            with self.assertRaises(ValueError):
                Settings(database_path=paper_path, shadow_database_path=paper_path)
            with self.assertRaises(ValueError):
                Settings(historical_cache_path=history_path, shadow_database_path=history_path)
            output = StringIO()
            with patch.dict("os.environ", {"SHADOW_DATABASE_PATH": str(shadow_path),
                                        "DATABASE_PATH": str(paper_path),
                                        "HISTORICAL_CACHE_PATH": str(history_path),
                                        "BYBIT_API_KEY": "private-test-key",
                                        "BYBIT_API_SECRET": "private-test-secret"}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                    with redirect_stdout(output):
                        self.assertEqual(main(["--status"]), 0)
            self.assertIn("Strategy version: regime-bull-hold-v1", output.getvalue())
            self.assertIn("Shadow state: FLAT", output.getvalue())
            self.assertIn("Total processed candles: 0", output.getvalue())
            self.assertNotIn("private-test-key", output.getvalue())
            self.assertNotIn("private-test-secret", output.getvalue())
            self.assertNotIn(b"private-test-key", shadow_path.read_bytes())
            self.assertNotIn(b"private-test-secret", shadow_path.read_bytes())
            self.assertEqual(paper_path.read_bytes(), b"paper-ledger-sentinel")
            self.assertEqual(history_path.read_bytes(), b"history-sentinel")
        for name in ("create_order", "place_order", "amend_order", "cancel_order",
                     "submit_order", "withdraw"):
            self.assertFalse(hasattr(BybitReadOnlyClient, name))

    def test_once_cli_uses_only_mocked_public_candles_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            shadow_path = Path(directory) / "shadow.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            paper_path.write_bytes(b"paper-ledger-sentinel")

            def mocked_candles(symbol, interval, start, end, **kwargs):
                self.assertEqual((symbol, interval), ("BTCUSDT", "1h"))
                self.assertEqual(kwargs, {"category": "spot", "completed_only": True})
                self.assertEqual(start, end - 225 * HOUR)
                return history(end)

            with patch.dict("os.environ", {"SHADOW_DATABASE_PATH": str(shadow_path),
                                        "DATABASE_PATH": str(paper_path)}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                    with patch.object(HistoricalMarketDataClient, "fetch_candles",
                                      side_effect=mocked_candles) as fetch:
                        first_output = StringIO()
                        with redirect_stdout(first_output):
                            self.assertEqual(main(["--once"]), ShadowExitCode.SUCCESS_NEW_DECISION)
                        self.assertIn("Result: BUY_SIGNAL", first_output.getvalue())
                        second_output = StringIO()
                        with redirect_stdout(second_output):
                            self.assertEqual(main(["--once"]), ShadowExitCode.SUCCESS_ALREADY_PROCESSED)
                        self.assertIn("Result: ALREADY_PROCESSED", second_output.getvalue())
                        self.assertEqual(fetch.call_count, 1)
            self.assertEqual(ShadowStore(shadow_path).status().processed, 1)
            self.assertEqual(paper_path.read_bytes(), b"paper-ledger-sentinel")


class ShadowCatchUpTests(unittest.TestCase):
    def setUp(self):
        disabled = patch.dict("os.environ", {"AI_CRYPTO_TELEGRAM_BOT_TOKEN": "",
                                           "AI_CRYPTO_TELEGRAM_CHAT_ID": ""})
        disabled.start()
        self.addCleanup(disabled.stop)

    def _baseline(self, path):
        store = ShadowStore(path)
        ShadowRunner(store, FakeMarket(history(FIRST_END, spike_close=False)),
                     clock=lambda: FIRST_END).once()
        return store

    def test_preview_one_missing_does_not_write_and_batch_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.sqlite3"
            store = self._baseline(path)
            runner = ShadowRunner(
                store, FakeMarket(continuation(FIRST_END, [D("200")])),
                clock=lambda: FIRST_END + HOUR + timedelta(minutes=59),
            )
            before = store.status()
            plan = runner.catch_up_plan()
            self.assertEqual(plan.missing_candles, 1)
            self.assertEqual(plan.decisions[0].action, "BUY_SIGNAL")
            self.assertEqual(store.status(), before)
            updated = runner.execute_catch_up(plan)
            self.assertEqual((updated.state, updated.processed), ("LONG", 2))
            restarted = ShadowStore(path).status()
            self.assertEqual((restarted.state, restarted.processed), ("LONG", 2))
            with self.assertRaisesRegex(ShadowUnavailable, "No missing"):
                runner.catch_up_plan()
            self.assertEqual(store.status().processed, 2)

    def test_multi_hour_buy_hold_sell_is_chronological_and_carries_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.sqlite3"
            paper = Path(directory) / "paper.sqlite3"
            paper.write_bytes(b"paper-ledger-sentinel")
            store = self._baseline(path)
            rows = continuation(FIRST_END, [D("200"), D("200"), D("100")])
            plan = ShadowRunner(
                store, FakeMarket(rows), clock=lambda: FIRST_END + 3 * HOUR,
            ).catch_up_plan()
            self.assertEqual([item.candle_close_utc for item in plan.decisions],
                             [FIRST_END + HOUR, FIRST_END + 2 * HOUR,
                              FIRST_END + 3 * HOUR])
            self.assertEqual([item.action for item in plan.decisions],
                             ["BUY_SIGNAL", "HOLD", "SELL_SIGNAL"])
            self.assertEqual([item.state_before for item in plan.decisions],
                             ["FLAT", "LONG", "LONG"])
            self.assertEqual([item.state_after for item in plan.decisions],
                             ["LONG", "LONG", "FLAT"])
            final = store.append_batch(plan.decisions, expected_status=plan.initial_status)
            self.assertEqual((final.state, final.buy_signals, final.sell_signals),
                             ("FLAT", 1, 1))
            self.assertEqual(paper.read_bytes(), b"paper-ledger-sentinel")

    def test_no_baseline_and_more_than_72_hours_are_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            empty_market = FakeMarket([])
            empty = ShadowStore(Path(directory) / "empty.sqlite3")
            with self.assertRaisesRegex(ShadowUnavailable, "No shadow baseline"):
                ShadowRunner(empty, empty_market, clock=lambda: FIRST_END).catch_up_plan()
            self.assertEqual(empty_market.calls, 0)

            store = self._baseline(Path(directory) / "limited.sqlite3")
            market = FakeMarket([])
            with self.assertRaisesRegex(ShadowUnavailable, "exceeds safety limit"):
                ShadowRunner(
                    store, market,
                    clock=lambda: FIRST_END + (MAX_SHADOW_CATCHUP_HOURS + 1) * HOUR,
                ).catch_up_plan()
            self.assertEqual(market.calls, 0)

    def test_incomplete_current_hour_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self._baseline(Path(directory) / "shadow.sqlite3")
            rows = continuation(FIRST_END, [D("200"), D("200"), D("900")])
            plan = ShadowRunner(
                store, FakeMarket(rows),
                clock=lambda: FIRST_END + 2 * HOUR + timedelta(minutes=59),
            ).catch_up_plan()
            self.assertEqual(plan.last_missing_candle, FIRST_END + 2 * HOUR)
            self.assertEqual(plan.missing_candles, 2)

    def test_bad_source_sequences_and_values_fail_closed(self):
        variants = {}
        valid = continuation(FIRST_END, [D("200"), D("200")])
        variants["missing"] = valid[:-1]
        variants["duplicate"] = [*valid, valid[-1]]
        variants["out-of-order"] = [*valid[:-2], valid[-1], valid[-2]]
        bad = list(valid)
        last = bad[-1]
        bad[-1] = Candle(last.symbol, last.interval, last.open_time_utc,
                         D("200"), D("100"), D("200"), D("200"), D("1"), D("1"))
        variants["invalid OHLC"] = bad
        variants["insufficient"] = valid[-10:]
        for name, rows in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                store = self._baseline(Path(directory) / "shadow.sqlite3")
                with self.assertRaises((ShadowUnavailable, HistoricalDataUnavailable,
                                        RegimeUnavailable)):
                    ShadowRunner(store, FakeMarket(rows),
                                 clock=lambda: FIRST_END + 2 * HOUR).catch_up_plan()
                self.assertEqual(store.status().processed, 1)

    def test_middle_sqlite_failure_rolls_back_entire_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.sqlite3"
            store = self._baseline(path)
            plan = ShadowRunner(
                store, FakeMarket(continuation(FIRST_END, [D("200"), D("200"), D("100")])),
                clock=lambda: FIRST_END + 3 * HOUR,
            ).catch_up_plan()
            failing_close = plan.decisions[1].candle_close_utc.isoformat()
            with sqlite3.connect(path) as db:
                db.execute(f"""CREATE TRIGGER fail_catchup_middle BEFORE INSERT ON shadow_decisions
                    WHEN NEW.candle_close_utc = '{failing_close}'
                    BEGIN SELECT RAISE(ABORT, 'test middle failure'); END""")
            with self.assertRaises(sqlite3.DatabaseError):
                store.append_batch(plan.decisions, expected_status=plan.initial_status)
            self.assertEqual(ShadowStore(path).status().processed, 1)
            self.assertEqual(ShadowStore(path).status().state, "FLAT")

    def test_non_yes_confirmation_cancels_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.sqlite3"
            store = self._baseline(path)
            plan = ShadowRunner(
                store, FakeMarket(continuation(FIRST_END, [D("200")])),
                clock=lambda: FIRST_END + HOUR,
            ).catch_up_plan()
            output = StringIO()
            with patch.dict("os.environ", {"SHADOW_DATABASE_PATH": str(path)}):
                with patch.object(ShadowRunner, "catch_up_plan", return_value=plan):
                    with redirect_stdout(output):
                        self.assertEqual(main(["--catch-up", "--preview"]), 0)
                    self.assertIn("Preview only", output.getvalue())
                    self.assertEqual(store.status().processed, 1)
                    output = StringIO()
                    with patch("builtins.input", return_value="NO"):
                        with redirect_stdout(output):
                            self.assertEqual(main(["--catch-up", "--execute"]), 0)
            self.assertIn("Catch-up cancelled", output.getvalue())
            self.assertEqual(store.status().processed, 1)


if __name__ == "__main__":
    unittest.main()
