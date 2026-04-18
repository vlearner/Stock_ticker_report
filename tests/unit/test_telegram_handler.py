"""Unit tests for TelegramAdapter helpers and RateLimiter.

Pipeline and Telegram API calls are mocked — no real tokens or network
requests needed.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("BRAVE_API_KEY", "test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")

from src.bot.rate_limiter import RateLimiter
from src.bot.telegram_handler import _split_message


# ---------------------------------------------------------------------------
# _split_message
# ---------------------------------------------------------------------------

class TestSplitMessage:
    def test_short_message_returns_single_chunk(self):
        result = _split_message("*AAPL*\n\nApple looks good.")
        assert len(result) == 1
        assert "AAPL" in result[0]

    def test_splits_at_divider(self):
        msg = "*AAPL*\n\nSummary A.\n\n" + "─" * 28 + "\n\n*MSFT*\n\nSummary B."
        result = _split_message(msg)
        assert len(result) == 2
        assert "AAPL" in result[0]
        assert "MSFT" in result[1]

    def test_long_chunk_is_split(self):
        long_text = "A" * 5000
        result = _split_message(long_text)
        assert all(len(chunk) <= 4096 for chunk in result)

    def test_empty_string_returns_fallback(self):
        result = _split_message("")
        assert len(result) == 1

    def test_only_dividers_returns_fallback(self):
        result = _split_message("─" * 28)
        assert len(result) == 1

    def test_three_tickers_splits_into_three(self):
        divider = "\n\n" + "─" * 28 + "\n\n"
        msg = "*A*\n\nSummary." + divider + "*B*\n\nSummary." + divider + "*C*\n\nSummary."
        result = _split_message(msg)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def _make_limiter(self, db_path: str, limit: int = 3) -> RateLimiter:
        return RateLimiter(db_path=db_path, limit=limit, window_s=3600)

    def test_first_request_allowed(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            limiter = self._make_limiter(f.name)
            asyncio.get_event_loop().run_until_complete(limiter.setup())
            allowed, remaining = asyncio.get_event_loop().run_until_complete(
                limiter.check_and_record("user1")
            )
        assert allowed is True
        assert remaining == 2  # limit=3, used=1

    def test_hits_limit(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            limiter = self._make_limiter(f.name, limit=2)

            async def _run():
                await limiter.setup()
                r1 = await limiter.check_and_record("u1")
                r2 = await limiter.check_and_record("u1")
                r3 = await limiter.check_and_record("u1")  # should be denied
                return r1, r2, r3

            r1, r2, r3 = asyncio.get_event_loop().run_until_complete(_run())

        assert r1[0] is True
        assert r2[0] is True
        assert r3[0] is False
        assert r3[1] == 0

    def test_different_users_independent(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            limiter = self._make_limiter(f.name, limit=1)

            async def _run():
                await limiter.setup()
                a = await limiter.check_and_record("userA")
                b = await limiter.check_and_record("userB")
                return a, b

            a, b = asyncio.get_event_loop().run_until_complete(_run())

        assert a[0] is True
        assert b[0] is True

    def test_expired_timestamps_pruned(self):
        """Timestamps older than window_s should not count."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            # window of 1 second — entries inserted manually in the past
            limiter = RateLimiter(db_path=f.name, limit=2, window_s=1)

            import time
            import aiosqlite

            async def _run():
                await limiter.setup()
                # Insert an old entry directly
                async with aiosqlite.connect(f.name) as db:
                    await db.execute(
                        "INSERT INTO rate_limit (user_id, ts) VALUES (?, ?)",
                        ("u1", time.time() - 10),  # 10s ago, window=1s → expired
                    )
                    await db.commit()
                # Should still be allowed (old entry is outside window)
                return await limiter.check_and_record("u1")

            allowed, remaining = asyncio.get_event_loop().run_until_complete(_run())

        assert allowed is True
