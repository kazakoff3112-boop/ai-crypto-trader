"""Environment-only configuration with safe Demo defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from .safety.guard import ALLOWED_CATEGORIES, ALLOWED_SYMBOLS, validate_trading_mode

DEMO_BASE_URL = "https://api-demo.bybit.com"


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without executing file contents."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry on line {number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid .env key on line {number}")
        values[key] = value.strip().strip("'\"")
    return values


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    api_secret: str = field(default="", repr=False)
    base_url: str = DEMO_BASE_URL
    trading_mode: str = "demo"
    allowed_symbols: frozenset[str] = ALLOWED_SYMBOLS
    allowed_categories: frozenset[str] = ALLOWED_CATEGORIES
    initial_capital_usdt: Decimal = Decimal("1000")
    spot_fee_rate: Decimal = Decimal("0.001")
    paper_slippage_bps: Decimal = Decimal("5")
    max_manual_paper_trade_usdt: Decimal = Decimal("250")
    max_manual_price_change_bps: Decimal = Decimal("10")
    database_path: Path = Path("data/ai_crypto_trader.sqlite3")
    historical_cache_path: Path = Path("data/historical_market_data.sqlite3")

    def __post_init__(self) -> None:
        validate_trading_mode(self.trading_mode)
        if self.base_url.rstrip("/") != DEMO_BASE_URL:
            raise ValueError("Only the official Bybit Demo endpoint is allowed")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or parsed.hostname != "api-demo.bybit.com":
            raise ValueError("Invalid Bybit Demo URL")
        if self.allowed_symbols != ALLOWED_SYMBOLS or self.allowed_categories != ALLOWED_CATEGORIES:
            raise ValueError("Safety allowlists cannot be overridden")
        if self.database_path.resolve() == self.historical_cache_path.resolve():
            raise ValueError("Historical cache must be separate from the trade ledger database")
        if self.initial_capital_usdt != Decimal("1000"):
            raise ValueError("Internal starting capital must be 1000 USDT")
        if not self.spot_fee_rate.is_finite() or self.spot_fee_rate < 0 or self.spot_fee_rate >= 1:
            raise ValueError("SPOT fee rate must be between 0 (inclusive) and 1 (exclusive)")
        if not self.paper_slippage_bps.is_finite() or not 0 <= self.paper_slippage_bps < 10_000:
            raise ValueError("Paper slippage BPS must be in [0, 10000)")
        if not self.max_manual_paper_trade_usdt.is_finite() or self.max_manual_paper_trade_usdt <= 0:
            raise ValueError("Manual paper trade limit must be positive")
        if (
            not self.max_manual_price_change_bps.is_finite()
            or not 0 <= self.max_manual_price_change_bps < 10_000
        ):
            raise ValueError("Manual price change tolerance BPS must be in [0, 10000)")


def load_settings(dotenv_path: str | Path = ".env", environ: Mapping[str, str] | None = None) -> Settings:
    """Load process environment first, then use .env only for missing values."""
    file_values = _read_dotenv(Path(dotenv_path))
    source = dict(os.environ if environ is None else environ)
    for key, value in file_values.items():
        source.setdefault(key, value)
    return Settings(
        api_key=source.get("BYBIT_API_KEY", ""),
        api_secret=source.get("BYBIT_API_SECRET", ""),
        base_url=source.get("BYBIT_BASE_URL", DEMO_BASE_URL).rstrip("/"),
        trading_mode=source.get("TRADING_MODE", "demo"),
        spot_fee_rate=Decimal(source.get("SPOT_FEE_RATE", "0.001")),
        paper_slippage_bps=Decimal(source.get("PAPER_SLIPPAGE_BPS", "5")),
        max_manual_paper_trade_usdt=Decimal(
            source.get("MAX_MANUAL_PAPER_TRADE_USDT", "250")
        ),
        max_manual_price_change_bps=Decimal(
            source.get("MAX_MANUAL_PRICE_CHANGE_BPS", "10")
        ),
        database_path=Path(source.get("DATABASE_PATH", "data/ai_crypto_trader.sqlite3")),
        historical_cache_path=Path(source.get("HISTORICAL_CACHE_PATH", "data/historical_market_data.sqlite3")),
    )
