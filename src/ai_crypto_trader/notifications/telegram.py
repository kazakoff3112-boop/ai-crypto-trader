"""Outgoing Telegram Bot API sendMessage only; never receives commands."""

from __future__ import annotations

import json
from urllib import error, parse, request


class TelegramDeliveryError(RuntimeError):
    """A safe category with no URL, token, chat ID, or server response."""


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, *, opener=None) -> None:
        if not token or not chat_id:
            raise ValueError("Telegram notifications are disabled")
        self._token = token
        self._chat_id = chat_id
        self._opener = opener or request.urlopen

    def send(self, message: str) -> None:
        # The token is required in Telegram's Bot API URL. Never log the URL or
        # propagate urllib exceptions, whose repr may contain credentials.
        url = "https://api.telegram.org/bot" + self._token + "/sendMessage"
        body = parse.urlencode({"chat_id": self._chat_id, "text": message}).encode("utf-8")
        outgoing = request.Request(url, data=body, method="POST")
        try:
            with self._opener(outgoing, timeout=8) as response:
                if response.status != 200:
                    raise TelegramDeliveryError("http_error")
                payload = json.loads(response.read(16384).decode("utf-8"))
        except TelegramDeliveryError:
            raise
        except error.HTTPError:
            raise TelegramDeliveryError("http_error") from None
        except error.URLError:
            raise TelegramDeliveryError("network_error") from None
        except (TimeoutError, OSError):
            raise TelegramDeliveryError("network_error") from None
        except (UnicodeError, ValueError):
            raise TelegramDeliveryError("invalid_response") from None
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise TelegramDeliveryError("api_error")
