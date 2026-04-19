"""Top-level entry point — runs Telegram bot and FastAPI server together.

Both services share the same asyncio event loop via ``asyncio.gather``:

- **Telegram bot** — long-polling, handles stock queries from Telegram users
- **FastAPI / Uvicorn** — web demo UI + REST API on port 8000

Run with::

    python -m src.main

Graceful shutdown on SIGINT / SIGTERM stops both services cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import signal

import uvicorn

from src.api.app import app
from src.bot.telegram_handler import TelegramAdapter
from src.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


async def main() -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_shutdown(*_):
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)

    # --- Telegram bot ---
    adapter = TelegramAdapter()

    # --- Uvicorn (FastAPI) ---
    uv_config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",   # keep uvicorn quiet; our logger handles info
        reload=False,
    )
    uv_server = uvicorn.Server(uv_config)
    # Prevent uvicorn from installing its own signal handlers (we own them)
    uv_server.install_signal_handlers = lambda: None

    logger.info("Starting Telegram bot + FastAPI server (port 8000)")
    await adapter.start()

    async def _run_uvicorn():
        await uv_server.serve()

    async def _wait_for_stop():
        await stop_event.wait()
        uv_server.should_exit = True

    await asyncio.gather(_run_uvicorn(), _wait_for_stop())

    logger.info("Shutting down Telegram bot")
    await adapter.stop()
    logger.info("Bye")


if __name__ == "__main__":
    asyncio.run(main())
