from datetime import datetime, timedelta, timezone
from decimal import Decimal
from contextlib import redirect_stdout
from io import BytesIO
from io import StringIO
import json
from pathlib import Path
from unittest.mock import patch
import sqlite3
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.config import Settings
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.historical_data import main
from ai_crypto_trader.market_data.historical import (
    HistoricalDataUnavailable, HistoricalDownloadUnavailable, HistoricalMarketDataClient, epoch_ms,
)
from ai_crypto_trader.safety.guard import SafetyViolation


START = datetime(2024, 1, 1, tzinfo=timezone.utc)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def row(ms, price="1.1234567890123456789012345"):
    return [str(ms), price, "2.9", "1.0", "2.0", "0.000000000000000000000001", "0.1"]


class HistoricalTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.alter = lambda payload: payload

        def opener(request, timeout):
            self.requests.append(request)
            query = parse_qs(urlparse(request.full_url).query)
            step = int(query["interval"][0]) * 60_000
            start = int(query["start"][0])
            end = int(query["end"][0])
            payload = {"retCode": 0, "result": {"category": "spot", "symbol": query["symbol"][0],
                       "list": [row(ms) for ms in range(start, end + 1, step)][::-1]}}
            return Response(self.alter(payload))

        self.bybit = BybitReadOnlyClient(Settings(api_key="must-not-go", api_secret="secret"), opener=opener)
        self.market = HistoricalMarketDataClient(self.bybit, clock=lambda: START + timedelta(days=500))

    def fetch(self, symbol="BTCUSDT", interval="1m", count=2, category="spot"):
        return self.market.fetch_candles(symbol, interval, START, START + timedelta(minutes=count), category=category)

    def test_btc_and_eth_decimal_utc_ascending_and_public_get(self):
        for symbol in ("BTCUSDT", "ETHUSDT"):
            candles = self.fetch(symbol)
            self.assertEqual([c.open_time_utc for c in candles], [START, START + timedelta(minutes=1)])
            self.assertEqual(candles[0].close_time_utc, START + timedelta(minutes=1))
            self.assertEqual(candles[0].open, Decimal("1.1234567890123456789012345"))
            self.assertEqual(candles[0].volume, Decimal("0.000000000000000000000001"))
            request = self.requests[-1]
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(urlparse(request.full_url).path, "/v5/market/kline")
            self.assertNotIn("X-bapi-api-key", request.headers)
            self.assertIn("category=spot", request.full_url)

    def test_intervals_and_pagination_over_1000(self):
        for interval, minutes in (("5m", 5), ("15m", 15), ("1h", 60)):
            candles = self.market.fetch_candles("ETHUSDT", interval, START,
                                                START + timedelta(minutes=minutes * 2))
            self.assertEqual(len(candles), 2)
        candles = self.fetch(count=1002)
        self.assertEqual(len(candles), 1002)
        self.assertEqual(len(self.requests), 5)
        self.assertEqual(candles[-1].open_time_utc, START + timedelta(minutes=1001))
        self.assertEqual(parse_qs(urlparse(self.requests[-2].full_url).query)["limit"], ["1000"])

    def test_second_page_failure_reports_exact_safe_context(self):
        def alter_second_page(payload):
            if len(self.requests) == 2:
                payload["result"]["list"].pop()
            return payload

        self.alter = alter_second_page
        with self.assertRaises(HistoricalDownloadUnavailable) as caught:
            self.fetch(count=1002)
        error = caught.exception
        self.assertEqual(error.category, "page_count_mismatch")
        self.assertEqual(error.page_number, 2)
        self.assertEqual(error.page_start_ms, epoch_ms(START + timedelta(minutes=1000)))
        self.assertEqual(error.page_end_ms, epoch_ms(START + timedelta(minutes=1002)))
        rendered = str(error)
        self.assertIn("expected=2, received=1", rendered)
        self.assertIn("page=2", rendered)
        self.assertIn("page_range=[2024-01-01T16:40:00+00:00, 2024-01-01T16:42:00+00:00)", rendered)

    def test_completed_only_excludes_current_interval(self):
        market = HistoricalMarketDataClient(self.bybit, clock=lambda: START + timedelta(minutes=2, seconds=30))
        candles = market.fetch_candles("BTCUSDT", "1m", START, START + timedelta(minutes=3))
        self.assertEqual(len(candles), 2)

    def test_symbol_category_interval_and_range_fail_before_network(self):
        for symbol, category, interval in (("SOLUSDT", "spot", "1m"),
                                           ("BTCUSDT", "linear", "1m"),
                                           ("BTCUSDT", "spot", "1d")):
            with self.assertRaises((SafetyViolation, ValueError)):
                self.market.fetch_candles(symbol, interval, START, START + timedelta(minutes=1), category=category)
        with self.assertRaises(ValueError):
            self.market.fetch_candles("BTCUSDT", "1m", START + timedelta(seconds=1),
                                      START + timedelta(minutes=2))
        self.assertEqual(self.requests, [])

    def test_gaps_duplicates_order_and_wrong_identity_fail(self):
        mutations = (
            lambda p: p["result"]["list"].pop(),
            lambda p: p["result"]["list"].reverse(),
            lambda p: p["result"]["list"].__setitem__(1, p["result"]["list"][0]),
            lambda p: p["result"].__setitem__("category", "linear"),
            lambda p: p["result"].__setitem__("symbol", "ETHUSDT"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                def altered(p):
                    mutate(p)
                    return p
                self.alter = altered
                with self.assertRaises(HistoricalDataUnavailable):
                    self.fetch()

    def test_invalid_ohlcv_and_malformed_rows_fail(self):
        bad_rows = (["bad", "1", "2", "1", "2", "1", "1"],
                    row(epoch_ms(START), "NaN"),
                    [str(epoch_ms(START)), "1", "0.5", "1", "2", "1", "1"],
                    [str(epoch_ms(START)), "1", "2", "1", "2", "-1", "1"],
                    [str(epoch_ms(START)), "1", "2", "1", "2", "1"])
        for bad in bad_rows:
            with self.subTest(bad=bad):
                self.alter = lambda p, bad=bad: p["result"]["list"].__setitem__(-1, bad) or p
                with self.assertRaises(HistoricalDataUnavailable):
                    self.fetch()

    def test_http_and_bybit_errors_fail(self):
        bad = BybitReadOnlyClient(Settings(), opener=lambda req, timeout: (_ for _ in ()).throw(
            HTTPError(req.full_url, 503, "bad", {}, BytesIO())))
        with self.assertRaises(HistoricalDataUnavailable):
            HistoricalMarketDataClient(bad).fetch_candles("BTCUSDT", "1m", START, START + timedelta(minutes=1))
        self.alter = lambda p: {"retCode": 10001, "retMsg": "private text"}
        with self.assertRaises(HistoricalDataUnavailable) as caught:
            self.fetch()
        self.assertNotIn("private text", str(caught.exception))
        disconnected = BybitReadOnlyClient(Settings(), opener=lambda req, timeout: (_ for _ in ()).throw(URLError("offline")))
        with self.assertRaises(HistoricalDataUnavailable):
            HistoricalMarketDataClient(disconnected).fetch_candles(
                "BTCUSDT", "1m", START, START + timedelta(minutes=1))
        malformed = BybitReadOnlyClient(Settings(), opener=lambda req, timeout: Response({"retCode": 0}))
        with self.assertRaises(HistoricalDataUnavailable):
            HistoricalMarketDataClient(malformed).fetch_candles(
                "BTCUSDT", "1m", START, START + timedelta(minutes=1))

    def test_cli_preserves_structured_failure_without_sensitive_values(self):
        failure = HistoricalDownloadUnavailable(
            "bybit_api_error", START, START + timedelta(days=920), page_number=7,
            page_start_ms=epoch_ms(START + timedelta(hours=6000)),
            page_end_ms=epoch_ms(START + timedelta(hours=7000)),
            detail="Bybit API error code 10006",
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.sqlite3"
            output = StringIO()
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(path),
                                           "BYBIT_API_KEY": "never-print-key",
                                           "BYBIT_API_SECRET": "never-print-secret"}):
                with patch.object(HistoricalMarketDataClient, "fetch_candles", side_effect=failure):
                    with redirect_stdout(output):
                        self.assertEqual(main(["--symbol", "BTCUSDT", "--interval", "1h",
                                               "--days", "920"]), 1)
        rendered = output.getvalue()
        for expected in ("category=bybit_api_error", "page=7", "page_range=",
                         "Bybit API error code 10006"):
            self.assertIn(expected, rendered)
        self.assertNotIn("never-print-key", rendered)
        self.assertNotIn("never-print-secret", rendered)

    def test_cache_restart_precision_conflict_and_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.sqlite3"
            cache = HistoricalCache(path)
            cache.save(self.fetch())
            cache.save(self.fetch())
            reloaded = HistoricalCache(path)
            self.assertEqual(reloaded.load("BTCUSDT", "1m", START, START + timedelta(minutes=2))[0].open,
                             Decimal("1.1234567890123456789012345"))
            self.assertEqual(reloaded.summary()[0][2], 2)
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(path)}):
                self.assertEqual(main(["--offline-summary", "--symbol", "ETHUSDT"]), 0)
            self.assertEqual(len(reloaded.batches()), 1)
            self.assertEqual(reloaded.batches()[0][4], 2)
            with self.assertRaises(HistoricalDataUnavailable):
                reloaded.load("BTCUSDT", "1m", START, START + timedelta(minutes=3))
            self.alter = lambda p: p["result"]["list"][0].__setitem__(1, "1.2") or p
            with self.assertRaises(HistoricalDataUnavailable) as caught:
                cache.save(self.fetch())
            self.assertIn("2024-01-01T00:01:00+00:00", str(caught.exception))
            self.assertEqual(reloaded.summary()[0][2], 2)

    def test_cache_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.sqlite3"
            cache = HistoricalCache(path)
            cache.save(self.fetch())
            with sqlite3.connect(path) as db:
                db.execute("UPDATE candles SET high='-1' WHERE open_time_ms=?", (epoch_ms(START),))
            with self.assertRaises(HistoricalDataUnavailable):
                cache.load("BTCUSDT", "1m", START, START + timedelta(minutes=2))

    def test_cache_is_separate_from_trade_db_and_no_order_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            trade = Path(temp) / "trade.sqlite3"
            trade.write_bytes(b"trade-ledger-sentinel")
            history = Path(temp) / "history.sqlite3"
            with self.assertRaises(ValueError):
                Settings(database_path=trade, historical_cache_path=trade)
            HistoricalCache(history).save(self.fetch())
            self.assertEqual(trade.read_bytes(), b"trade-ledger-sentinel")
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(self.bybit, name))

    def test_download_days_accepts_existing_and_new_boundaries_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.sqlite3"
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(path)}):
                with patch.object(HistoricalMarketDataClient, "fetch_candles", return_value=[]) as fetch:
                    for days in (1, 400, 500, 600, 740, 800, 920, 950):
                        with self.subTest(days=days):
                            self.assertEqual(main(["--symbol", "BTCUSDT", "--interval", "1h",
                                                   "--days", str(days)]), 0)
                    self.assertEqual(fetch.call_count, 8)

    def test_download_days_rejects_zero_and_951_before_network(self):
        with patch.object(HistoricalMarketDataClient, "fetch_candles") as fetch:
            for days in (0, 951):
                with self.subTest(days=days), self.assertRaises(SystemExit) as caught:
                    main(["--symbol", "BTCUSDT", "--interval", "1h", "--days", str(days)])
                self.assertEqual(caught.exception.code, 2)
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
