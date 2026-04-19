"""SQLite-backed cache for :class:`AnalystOutput` results.

Why cache?
----------
The Analyst agent makes a ChatGroq call per ticker on every run.  For a
popular stock queried by many users, this means N identical LLM requests
per day — each paid for, each adding latency.  Caching the analyst output
by ``ticker + UTC date`` reduces these to one LLM call per ticker per day.

Cache key
---------
``"{TICKER}:{YYYY-MM-DD}"`` — UTC date, uppercase ticker.  The daily
rollover is a natural TTL: fresh market data each trading day produces a
new cache entry.  A configurable absolute TTL (``ttl_seconds``) is a
backstop that keeps the cache from growing unbounded.

Cache value
-----------
The ``AnalystOutput`` serialised as JSON (via ``model_dump_json``).  Invalid
/ unreadable payloads are treated as cache misses — no exception leaks out
of :meth:`get`.

Thread safety
-------------
The singleton ``analyst_cache`` is safe to use from any coroutine — each
call opens its own aiosqlite connection (SQLite handles its own file
locking).  Schema setup is guarded by an :class:`asyncio.Lock` so the first
concurrent callers don't race on ``CREATE TABLE``.

Scope
-----
We intentionally do **not** include input data (fundamentals / news) in
the cache key.  Market data changes continuously; if it did influence the
key, we would effectively never hit.  Daily granularity is the right
trade-off for a financial summary: it's fresh each market open without
re-paying per user.  Disable the cache globally via
``settings.analyst_cache_enabled=False`` for a full bypass.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

import aiosqlite

from src.schemas.ticker_data import AnalystOutput

logger = logging.getLogger(__name__)


class AnalystCache:
    """SQLite KV cache keyed by ``ticker:date``."""

    def __init__(self, db_path: str, ttl_seconds: int = 86_400) -> None:
        """
        Args:
            db_path:     Path to the SQLite file.  Reuses the same DB as the
                         rate limiter / LangGraph checkpointer.
            ttl_seconds: Absolute max age of a cache entry in seconds.
                         Default 86_400 (24h).
        """
        self._db_path = db_path
        self._ttl = ttl_seconds
        self._setup_done = False
        self._setup_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Schema setup (lazy)
    # ------------------------------------------------------------------

    async def _ensure_setup(self) -> None:
        """Create the ``analyst_cache`` table the first time it's needed."""
        if self._setup_done:
            return
        async with self._setup_lock:
            if self._setup_done:
                return
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analyst_cache (
                        cache_key  TEXT PRIMARY KEY,
                        payload    TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                await db.commit()
            self._setup_done = True

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(ticker: str, when: datetime | None = None) -> str:
        """Return the cache key for ``ticker`` on ``when`` (UTC date).

        Args:
            ticker: Stock ticker symbol (case-insensitive).
            when:   Optional timestamp. Defaults to ``datetime.now(UTC)``.

        Returns:
            String of the form ``"AAPL:2026-04-18"``.
        """
        dt = when or datetime.now(timezone.utc)
        return f"{ticker.upper()}:{dt.strftime('%Y-%m-%d')}"

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    async def get(self, cache_key: str) -> AnalystOutput | None:
        """Return the cached ``AnalystOutput`` or ``None`` on miss/expired."""
        await self._ensure_setup()
        cutoff = time.time() - self._ttl
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT payload FROM analyst_cache "
                "WHERE cache_key = ? AND created_at > ?",
                (cache_key, cutoff),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        try:
            return AnalystOutput.model_validate_json(row[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AnalystCache: failed to deserialise payload for %s — %s",
                cache_key,
                exc,
            )
            return None

    async def set(self, cache_key: str, output: AnalystOutput) -> None:
        """Write an :class:`AnalystOutput` into the cache (overwrites)."""
        await self._ensure_setup()
        payload = output.model_dump_json()
        now = time.time()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO analyst_cache "
                "(cache_key, payload, created_at) VALUES (?, ?, ?)",
                (cache_key, payload, now),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def clear_expired(self) -> int:
        """Delete rows older than ``ttl_seconds``.

        Returns:
            Number of rows removed.
        """
        await self._ensure_setup()
        cutoff = time.time() - self._ttl
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM analyst_cache WHERE created_at <= ?",
                (cutoff,),
            )
            await db.commit()
            return cursor.rowcount

    async def clear_all(self) -> int:
        """Delete every row.  Primarily useful in tests."""
        await self._ensure_setup()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("DELETE FROM analyst_cache")
            await db.commit()
            return cursor.rowcount


# ---------------------------------------------------------------------------
# Module-level singleton — used by the analyst node
# ---------------------------------------------------------------------------

def _build_singleton() -> AnalystCache:
    from src.config import settings

    ttl_h = getattr(settings, "analyst_cache_ttl_hours", 24)
    return AnalystCache(
        db_path=settings.sqlite_db_path,
        ttl_seconds=int(ttl_h) * 3600,
    )


analyst_cache = _build_singleton()
"""Process-wide singleton reused by the analyst node."""
