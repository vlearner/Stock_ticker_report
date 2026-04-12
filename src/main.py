"""Top-level entry point.

Reads :mod:`src.config`, selects a :class:`~src.bot.base.MessagingAdapter`
based on ``settings.messaging_adapter``, wires it to the compiled LangGraph
pipeline, and runs the event loop. Swap adapters (Telegram, WhatsApp,
iMessage, ...) here without touching agents or the graph.

Implemented in Implementation Order step 11 (Telegram bot).
"""

from __future__ import annotations


async def main() -> None:  # pragma: no cover - stub
    """Application entry point (stub).

    Will:
        1. Load ``settings`` from :mod:`src.config`.
        2. Build the LangGraph pipeline with a SQLite checkpointer.
        3. Instantiate the configured messaging adapter.
        4. Run ``adapter.start()`` and await shutdown.
    """

    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(main())
