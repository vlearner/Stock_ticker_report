"""SQLite-backed rolling-window rate limiter.

Tracks query timestamps per user in a lightweight SQLite table.  The
rolling window (default 1 hour) is checked on every incoming message
before the pipeline runs.

Usage::

    limiter = RateLimiter(db_path="./data/state.db", limit=10, window_s=3600)
    await limiter.setup()

    allowed, remaining = await limiter.check_and_record(user_id="123")
    if not allowed:
        # tell the user they're over limit
        ...
"""

from __future__ import annotations

import time

import aiosqlite


class RateLimiter:
    """Rolling-window rate limiter backed by SQLite.

    Args:
        db_path: Path to the SQLite database file.
        limit: Maximum queries allowed per user within ``window_s`` seconds.
        window_s: Rolling window size in seconds (default 3600 = 1 hour).
    """

    def __init__(self, db_path: str, limit: int, window_s: int = 3600) -> None:
        self._db_path = db_path
        self._limit = limit
        self._window_s = window_s

    async def setup(self) -> None:
        """Create the rate_limit table if it does not exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limit (
                    user_id TEXT NOT NULL,
                    ts      REAL  NOT NULL
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_rate_limit_user ON rate_limit(user_id)"
            )
            await db.commit()

    async def check_and_record(self, user_id: str) -> tuple[bool, int]:
        """Check the rate limit and record this query if allowed.

        Prunes expired timestamps, counts remaining queries in the window,
        and inserts the current timestamp only when the request is allowed.

        Args:
            user_id: Stable identifier for the user.

        Returns:
            ``(allowed, remaining)`` — ``allowed`` is True when the query
            may proceed; ``remaining`` is queries left after this one.
        """
        now = time.time()
        cutoff = now - self._window_s

        async with aiosqlite.connect(self._db_path) as db:
            # Remove timestamps outside the rolling window
            await db.execute(
                "DELETE FROM rate_limit WHERE user_id = ? AND ts < ?",
                (user_id, cutoff),
            )

            # Count queries in the current window
            cursor = await db.execute(
                "SELECT COUNT(*) FROM rate_limit WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            count = row[0] if row else 0

            if count >= self._limit:
                await db.commit()
                return False, 0

            # Record this query
            await db.execute(
                "INSERT INTO rate_limit (user_id, ts) VALUES (?, ?)",
                (user_id, now),
            )
            await db.commit()
            remaining = self._limit - count - 1
            return True, remaining
