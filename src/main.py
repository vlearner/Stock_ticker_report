"""Top-level entry point — wires config, pipeline, and Telegram adapter.

Run with::

    python -m src.main

The bot uses long-polling (no webhook setup required for local/demo use).
Graceful shutdown on SIGINT / SIGTERM via asyncio signal handling.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from src.bot.telegram_handler import TelegramAdapter

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def main() -> None:
    adapter = TelegramAdapter()
    loop = asyncio.get_running_loop()

    stop_event = asyncio.Event()

    def _request_shutdown(*_):
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)

    logger.info("Starting Stock Ticker Bot")
    await adapter.start()
    logger.info("Bot is running — press Ctrl+C to stop")

    await stop_event.wait()

    logger.info("Shutting down")
    await adapter.stop()
    logger.info("Bye")


if __name__ == "__main__":
    asyncio.run(main())
