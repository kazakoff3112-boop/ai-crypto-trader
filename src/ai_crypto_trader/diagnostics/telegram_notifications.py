"""Explicit, outgoing-only manual Telegram connectivity test."""

from __future__ import annotations

import argparse

from ..config import load_settings
from ..notifications.telegram import TelegramDeliveryError, TelegramNotifier


TEST_MESSAGE = "🧪 AI Crypto Trader\nTelegram notifications test.\nNO TRADE EXECUTED."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send one manual Telegram notification")
    parser.add_argument("--test", required=True, action="store_true")
    parser.parse_args(argv)
    try:
        settings = load_settings()
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            print("Telegram notifications disabled: set both local environment values")
            return 2
        TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id).send(TEST_MESSAGE)
    except (TelegramDeliveryError, OSError, ValueError):
        print("Telegram test unavailable; check local configuration or network")
        return 1
    print("Telegram test message sent; no trade executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
