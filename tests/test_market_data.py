from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import hmac
from io import BytesIO
import json
import socket
import unittest
from urllib.error import HTTPError, URLError

from ai_crypto_trader.bybit import BybitReadOnlyClient, BybitUnavailable
from ai_crypto_trader.config import Settings
from ai_crypto_trader.market_data import MarketDataClient, MarketDataUnavailable
from ai_crypto_trader.safety import SafetyViolation


NOW_MS = 1_800_000_000_000


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.body


def ticker(symbol: str, **changes) -> dict:
    row = {
        "symbol": symbol,
        "lastPrice": "12345.123456789123456789",
        "bid1Price": "12344.000000000000000001",
        "ask1Price": "12346.999999999999999999",
    }
    row.update(changes)
    return {"retCode": 0, "result": {"category": "spot", "list": [row]}, "time": NOW_MS}


class MarketDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests = []
        self.payload = ticker("BTCUSDT")

        def opener(request, timeout):
            self.requests.append(request)
            return FakeResponse(json.dumps(self.payload).encode())

        self.bybit = BybitReadOnlyClient(Settings(), opener=opener, clock_ms=lambda: NOW_MS)
        self.market = MarketDataClient(self.bybit, clock_ms=lambda: NOW_MS)

    def test_btc_quote_and_decimal_precision(self) -> None:
        quote = self.market.get_spot_quote("BTCUSDT")
        self.assertEqual(quote.last_price, Decimal("12345.123456789123456789"))
        self.assertEqual(quote.bid, Decimal("12344.000000000000000001"))
        self.assertEqual(quote.ask, Decimal("12346.999999999999999999"))
        self.assertEqual(quote.timestamp_utc, datetime.fromtimestamp(NOW_MS // 1000, tz=timezone.utc))
        self.assertEqual(self.requests[0].get_method(), "GET")
        self.assertIn("category=spot&symbol=BTCUSDT", self.requests[0].full_url)

    def test_eth_quote(self) -> None:
        self.payload = ticker("ETHUSDT")
        self.assertEqual(self.market.get_spot_quote("ETHUSDT").symbol, "ETHUSDT")

    def test_unknown_symbol_and_non_spot_fail_before_network(self) -> None:
        with self.assertRaises(SafetyViolation):
            self.market.get_spot_quote("SOLUSDT")
        with self.assertRaises(SafetyViolation):
            self.market.get_spot_quote("BTCUSDT", category="linear")
        self.assertEqual(self.requests, [])

    def test_timeout_connection_and_http_errors(self) -> None:
        for error in (TimeoutError(), socket.timeout(), URLError(socket.timeout()), URLError("offline"),
                      HTTPError("https://api-demo.bybit.com", 503, "bad", {}, BytesIO())):
            with self.subTest(error=type(error).__name__):
                bybit = BybitReadOnlyClient(Settings(), opener=lambda request, timeout: (_ for _ in ()).throw(error))
                with self.assertRaises(MarketDataUnavailable):
                    MarketDataClient(bybit).get_spot_quote("BTCUSDT")

    def test_api_error_code_is_rejected(self) -> None:
        self.payload = {"retCode": 10003, "retMsg": "secret-text"}
        with self.assertRaisesRegex(MarketDataUnavailable, "10003") as caught:
            self.market.get_spot_quote("BTCUSDT")
        self.assertNotIn("secret-text", str(caught.exception))

    def test_malformed_json_is_rejected(self) -> None:
        bybit = BybitReadOnlyClient(Settings(), opener=lambda request, timeout: FakeResponse(b"{bad"))
        with self.assertRaises(MarketDataUnavailable):
            MarketDataClient(bybit).get_spot_quote("BTCUSDT")

    def test_missing_fields_and_wrong_shape_are_rejected(self) -> None:
        for payload in (
            {"retCode": 0, "result": {}, "time": NOW_MS},
            {"retCode": 0, "result": {"category": "linear", "list": []}, "time": NOW_MS},
            ticker("ETHUSDT"),
            {"retCode": 0, "result": {"category": "spot", "list": [{"symbol": "BTCUSDT"}]}, "time": NOW_MS},
        ):
            with self.subTest(payload=payload):
                self.payload = payload
                with self.assertRaises(MarketDataUnavailable):
                    self.market.get_spot_quote("BTCUSDT")

    def test_invalid_prices_are_rejected(self) -> None:
        for price in ("0", "-1", "NaN", "Infinity", "bad", 123.4):
            with self.subTest(price=price):
                self.payload = ticker("BTCUSDT", lastPrice=price)
                with self.assertRaises(MarketDataUnavailable):
                    self.market.get_spot_quote("BTCUSDT")

    def test_stale_future_and_invalid_timestamp_are_rejected(self) -> None:
        for timestamp in (NOW_MS - 60_001, NOW_MS + 5_001, 0, "bad", None):
            with self.subTest(timestamp=timestamp):
                self.payload = ticker("BTCUSDT")
                self.payload["time"] = timestamp
                with self.assertRaises(MarketDataUnavailable):
                    self.market.get_spot_quote("BTCUSDT")

    def test_missing_credentials_rejected_before_network(self) -> None:
        with self.assertRaises(BybitUnavailable):
            self.bybit.check_demo_credentials()
        self.assertEqual(self.requests, [])

    def test_private_check_is_get_signed_and_secret_redacted(self) -> None:
        secret = "very-private-secret"
        key = "demo-key"
        requests = []

        def opener(request, timeout):
            requests.append(request)
            return FakeResponse(b'{"retCode":0,"result":{"marginMode":"REGULAR_MARGIN"}}')

        bybit = BybitReadOnlyClient(
            Settings(api_key=key, api_secret=secret), opener=opener, clock_ms=lambda: NOW_MS
        )
        self.assertTrue(bybit.check_demo_credentials())
        request = requests[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, "https://api-demo.bybit.com/v5/account/info")
        self.assertEqual(request.get_header("X-bapi-api-key"), key)
        expected_signature = hmac.new(
            secret.encode(), f"{NOW_MS}{key}5000".encode(), hashlib.sha256
        ).hexdigest()
        self.assertEqual(request.get_header("X-bapi-sign"), expected_signature)
        self.assertNotIn(secret, repr(bybit))
        self.assertNotIn(secret, repr(request))

        failing = BybitReadOnlyClient(
            Settings(api_key=key, api_secret=secret),
            opener=lambda request, timeout: (_ for _ in ()).throw(URLError(secret)),
            clock_ms=lambda: NOW_MS,
        )
        with self.assertRaises(BybitUnavailable) as caught:
            failing.check_demo_credentials()
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, repr(caught.exception))

    def test_private_api_error_does_not_leak_response_message(self) -> None:
        bybit = BybitReadOnlyClient(
            Settings(api_key="key", api_secret="secret"),
            opener=lambda request, timeout: FakeResponse(
                b'{"retCode":10003,"retMsg":"secret"}'
            ),
        )
        with self.assertRaises(BybitUnavailable) as caught:
            bybit.check_demo_credentials()
        self.assertNotIn("secret", str(caught.exception))

    def test_no_order_mutation_methods(self) -> None:
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(self.bybit, name))


if __name__ == "__main__":
    unittest.main()
