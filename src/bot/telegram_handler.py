"""Telegram adapter — receives messages, runs the pipeline, sends replies.

Flow per message
----------------
1. Check SQLite rate limit → over limit: send ⏳ message and stop.
2. Send typing action (immediate UX feedback).
3. Invoke the LangGraph pipeline via ``ainvoke``.
4. Split the reply at section dividers and send each part separately
   (Telegram's 4096-char limit per message).
5. On any pipeline failure, send an appropriate ⚠️ / 😔 message.

Error messages
--------------
- Groq rate limited  → "😔 Our AI service is temporarily busy. Please try again later."
- Any other failure  → "⚠️ Something went wrong. Please try again."
- User rate limited  → "⏳ You've reached the limit of {N} queries per hour. Try again later."
"""

from __future__ import annotations

import logging
import os
import pathlib

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src.bot.base import MessagingAdapter
from src.bot.rate_limiter import RateLimiter
from src.config import settings

logger = logging.getLogger(__name__)

# Telegram hard limit per message
_MAX_MSG_LEN = 4096

# Section divider produced by the formatter — split multi-ticker replies here
_SECTION_DIVIDER = "─" * 28


def _split_message(text: str) -> list[str]:
    """Split a formatted message into Telegram-safe chunks.

    Splits first at section dividers (for comparison replies), then hard-
    truncates any remaining chunk that still exceeds 4096 chars.
    """
    parts = [p.strip() for p in text.split(_SECTION_DIVIDER) if p.strip()]
    if not parts:
        parts = [text.strip()]

    result: list[str] = []
    for part in parts:
        while len(part) > _MAX_MSG_LEN:
            # Find last newline before the limit to avoid mid-word cuts
            cut = part.rfind("\n", 0, _MAX_MSG_LEN)
            if cut == -1:
                cut = _MAX_MSG_LEN
            result.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            result.append(part)

    return result or ["No response generated."]


class TelegramAdapter(MessagingAdapter):
    """python-telegram-bot v21 adapter wired to the LangGraph pipeline."""

    name = "telegram"

    def __init__(self) -> None:
        self._app: Application | None = None
        self._rate_limiter = RateLimiter(
            db_path=settings.sqlite_db_path,
            limit=settings.rate_limit_per_hour,
            window_s=3600,
        )
        # Import here to avoid circular imports at module level
        from src.graph.pipeline import compile_graph
        self._pipeline = compile_graph()

    async def start(self) -> None:
        """Build the Application, register handlers, and start polling."""
        # Ensure DB directory exists
        pathlib.Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
        await self._rate_limiter.setup()

        self._app = (
            Application.builder()
            .token(settings.telegram_bot_token)
            .build()
        )

        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        logger.info("TelegramAdapter: starting polling")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        """Stop polling and release resources."""
        if self._app:
            logger.info("TelegramAdapter: stopping")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a single Telegram message with Markdown formatting."""
        if not self._app:
            raise RuntimeError("TelegramAdapter not started")
        await self._app.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
        )

    # ---------------------------------------------------------------------------
    # Command handlers
    # ---------------------------------------------------------------------------

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            "👋 *Stock Analysis Bot*\n\n"
            "Ask me about any stock ticker:\n\n"
            "• `AAPL` — quick summary\n"
            "• `AAPL analysis` — full report\n"
            "• `AAPL vs MSFT` — comparison\n"
            "• `TSLA latest news` — headlines\n\n"
            f"_Limit: {settings.rate_limit_per_hour} queries per hour._",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await update.message.reply_text(
            "📖 *How to use*\n\n"
            "*Quick summary:* just send a ticker — `AAPL`\n"
            "*Full report:* add a detail word — `NVDA analysis`\n"
            "*Compare:* `AAPL vs MSFT`\n"
            "*News only:* `TSLA news`\n\n"
            "Data is fetched live from yfinance and Brave Search.\n"
            "Analysis is generated by Groq (llama-3.3-70b).",
            parse_mode=ParseMode.MARKDOWN,
        )

    # ---------------------------------------------------------------------------
    # Message handler
    # ---------------------------------------------------------------------------

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.message
        if not message or not message.text:
            return

        user_id = str(message.from_user.id)
        chat_id = str(message.chat_id)
        text = message.text.strip()

        # 1. Rate limit check
        allowed, remaining = await self._rate_limiter.check_and_record(user_id)
        if not allowed:
            await message.reply_text(
                f"⏳ You've reached the limit of {settings.rate_limit_per_hour} "
                "queries per hour. Please try again later.",
            )
            return

        # 2. Typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        # 3. Run pipeline
        try:
            state = {
                "user_message": text,
                "user_id": user_id,
                "chat_id": chat_id,
                "rate_limit_remaining": remaining,
            }
            result = await self._pipeline.ainvoke(state)
            reply = result.get("final_message") or "⚠️ No response generated."

        except Exception as exc:
            logger.error("Pipeline error for user %s: %s", user_id, exc)
            exc_str = str(exc).lower()
            if "rate" in exc_str and ("limit" in exc_str or "429" in exc_str):
                reply = (
                    "😔 Our AI service is temporarily busy. "
                    "Please try again later."
                )
            else:
                reply = "⚠️ Something went wrong. Please try again."

        # 4. Send reply (split if needed)
        chunks = _split_message(reply)
        for chunk in chunks:
            try:
                await message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                # Markdown parse failure — retry as plain text
                await message.reply_text(chunk)
