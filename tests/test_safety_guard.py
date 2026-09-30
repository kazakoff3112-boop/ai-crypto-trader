from decimal import Decimal
import unittest

from ai_crypto_trader.config import Settings
from ai_crypto_trader.safety.guard import SafetyViolation, TradingRequest, validate_request


class SafetyGuardTests(unittest.TestCase):
    def test_allowed_spot_symbols(self) -> None:
        for symbol in ("BTCUSDT", "ETHUSDT"):
            with self.subTest(symbol=symbol):
                result = validate_request(TradingRequest(symbol=symbol, category="spot"))
                self.assertEqual(result.symbol, symbol)

    def test_linear_is_forbidden(self) -> None:
        with self.assertRaises(SafetyViolation):
            validate_request(TradingRequest(symbol="BTCUSDT", category="linear"))

    def test_non_allowlisted_symbols_are_forbidden(self) -> None:
        for symbol in ("SOLUSDT", "BTCUSD", "DOGEUSDT", ""):
            with self.subTest(symbol=symbol), self.assertRaises(SafetyViolation):
                validate_request(TradingRequest(symbol=symbol, category="spot"))

    def test_margin_is_forbidden(self) -> None:
        with self.assertRaises(SafetyViolation):
            validate_request(TradingRequest(symbol="BTCUSDT", category="spot", margin=True))

    def test_any_explicit_leverage_is_forbidden(self) -> None:
        for leverage in (Decimal("1"), Decimal("2"), 1, 0):
            with self.subTest(leverage=leverage), self.assertRaises(SafetyViolation):
                validate_request(TradingRequest(symbol="BTCUSDT", category="spot", leverage=leverage))

    def test_non_demo_mode_is_forbidden(self) -> None:
        for mode in ("live", "production", "prod", "testnet", ""):
            with self.subTest(mode=mode), self.assertRaises(SafetyViolation):
                Settings(trading_mode=mode)

    def test_secret_is_redacted_from_settings_repr(self) -> None:
        rendered = repr(Settings(api_key="public-key", api_secret="super-secret"))
        self.assertNotIn("public-key", rendered)
        self.assertNotIn("super-secret", rendered)


if __name__ == "__main__":
    unittest.main()
