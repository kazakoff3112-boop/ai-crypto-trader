from pathlib import Path
from decimal import Decimal
import tempfile
import unittest

from ai_crypto_trader.config import DEMO_BASE_URL, load_settings


class ConfigTests(unittest.TestCase):
    def test_safe_defaults_without_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(Path(directory) / ".env", environ={})
        self.assertEqual(settings.trading_mode, "demo")
        self.assertEqual(settings.base_url, DEMO_BASE_URL)
        self.assertEqual(settings.max_manual_paper_trade_usdt, Decimal("250"))
        self.assertEqual(settings.max_manual_price_change_bps, Decimal("10"))

    def test_environment_has_priority_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text("BYBIT_API_KEY=file-key\n", encoding="utf-8")
            settings = load_settings(dotenv, environ={"BYBIT_API_KEY": "environment-key"})
        self.assertEqual(settings.api_key, "environment-key")

    def test_non_demo_endpoint_is_forbidden(self) -> None:
        with self.assertRaises(ValueError):
            load_settings(environ={"BYBIT_BASE_URL": "https://api.bybit.com"})


if __name__ == "__main__":
    unittest.main()
