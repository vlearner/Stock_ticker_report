"""Telegram-specific :class:`MessagingAdapter` implementation.

Implemented in Implementation Order step 11 (Telegram bot).
"""

from __future__ import annotations

from src.bot.base import MessagingAdapter


class TelegramAdapter(MessagingAdapter):
    """python-telegram-bot backed adapter (stub)."""

    name = "telegram"

    async def start(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    async def stop(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    async def send_message(self, chat_id: str, text: str) -> None:  # pragma: no cover - stub
        raise NotImplementedError
