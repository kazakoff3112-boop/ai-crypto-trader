"""Manual read-only Bybit Demo diagnostic: python -m ...bybit_connection."""

from __future__ import annotations

from ..bybit import BybitReadOnlyClient, BybitUnavailable
from ..config import load_settings
from ..market_data import MarketDataClient, MarketDataUnavailable


def main() -> int:
    try:
        settings = load_settings()
    except (ValueError, OSError):
        print("Ошибка локальной конфигурации. Проверьте Demo URL и .env.")
        return 1

    bybit = BybitReadOnlyClient(settings)
    market = MarketDataClient(bybit)
    failed = False
    for symbol in ("BTCUSDT", "ETHUSDT"):
        try:
            quote = market.get_spot_quote(symbol)
            print(
                f"{symbol} spot: last={quote.last_price}, bid={quote.bid}, "
                f"ask={quote.ask}, UTC={quote.timestamp_utc.isoformat()}"
            )
        except MarketDataUnavailable as exc:
            print(f"{symbol} spot: данные недоступны ({exc})")
            failed = True

    if settings.api_key and settings.api_secret:
        try:
            bybit.check_demo_credentials()
            print("Приватное Demo API-соединение: успешно (только чтение).")
        except BybitUnavailable as exc:
            print(f"Приватное Demo API-соединение: недоступно ({exc}).")
            failed = True
    else:
        print("Приватная проверка пропущена: Demo credentials не заданы.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
