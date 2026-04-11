"""Messaging adapter abstraction.

Defines the contract that every messaging platform (Telegram, WhatsApp,
iMessage, Slack, ...) must implement so the LangGraph pipeline stays
transport-agnostic. New platforms plug in by subclassing ``MessagingAdapter``
— no changes to the graph or agents.

Implemented in Implementation Order step 11 (Telegram bot).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class IncomingMessage:
    """Platform-neutral representation of an inbound user message."""

    user_id: str
    chat_id: str
    text: str
    platform: str


class MessagingAdapter(ABC):
    """Abstract base class for all messaging-platform adapters.

    Concrete adapters translate platform-specific events into
    :class:`IncomingMessage` instances, invoke the LangGraph pipeline,
    and send the formatted response back via :meth:`send_message`.
    """

    name: str = "base"

    @abstractmethod
    async def start(self) -> None:
        """Start the adapter (connect, begin long-polling / webhook, ...)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the adapter and release resources gracefully."""

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> None:
        """Send ``text`` to ``chat_id`` using the platform's native API."""
