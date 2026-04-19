"""Unit tests for ``src.cache.analyst_cache.AnalystCache``.

Each test gets its own temp SQLite DB via the ``cache_db`` fixture so tests
are fully isolated.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone

import pytest

# Set env before any import that triggers Settings construction.
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("BRAVE_API_KEY", "test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")

from src.cache.analyst_cache import AnalystCache  # noqa: E402
from src.schemas.ticker_data import AnalystOutput  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_db(tmp_path) -> str:
    return str(tmp_path / "cache.db")


@pytest.fixture
def cache(cache_db) -> AnalystCache:
    return AnalystCache(db_path=cache_db, ttl_seconds=3600)


@pytest.fixture
def sample_output() -> AnalystOutput:
    return AnalystOutput(
        summary="Apple had a solid quarter with strong iPhone sales.",
        key_points=["Revenue up 5%", "Services segment growing", "Stock near highs"],
        iteration=1,
    )


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


class TestMakeKey:
    def test_uppercases_ticker(self):
        key = AnalystCache.make_key("aapl", datetime(2026, 4, 18, tzinfo=timezone.utc))
        assert key == "AAPL:2026-04-18"

    def test_uses_utc_date_format(self):
        key = AnalystCache.make_key("msft", datetime(2026, 1, 5, 23, 59, tzinfo=timezone.utc))
        assert key == "MSFT:2026-01-05"

    def test_defaults_to_now_utc(self):
        key = AnalystCache.make_key("TSLA")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert key == f"TSLA:{today}"


# ---------------------------------------------------------------------------
# get / set round-trip
# ---------------------------------------------------------------------------


class TestGetSet:
    async def test_miss_on_empty_cache(self, cache: AnalystCache):
        result = await cache.get("AAPL:2026-04-18")
        assert result is None

    async def test_set_then_get_returns_same_output(
        self, cache: AnalystCache, sample_output: AnalystOutput
    ):
        key = "AAPL:2026-04-18"
        await cache.set(key, sample_output)
        got = await cache.get(key)

        assert got is not None
        assert got.summary == sample_output.summary
        assert got.key_points == sample_output.key_points
        assert got.iteration == sample_output.iteration

    async def test_set_overwrites_existing(
        self, cache: AnalystCache, sample_output: AnalystOutput
    ):
        key = "AAPL:2026-04-18"
        await cache.set(key, sample_output)

        updated = sample_output.model_copy(update={"summary": "Updated."})
        await cache.set(key, updated)

        got = await cache.get(key)
        assert got is not None
        assert got.summary == "Updated."

    async def test_different_keys_do_not_collide(
        self, cache: AnalystCache, sample_output: AnalystOutput
    ):
        await cache.set("AAPL:2026-04-18", sample_output)
        msft_output = sample_output.model_copy(update={"summary": "Microsoft summary"})
        await cache.set("MSFT:2026-04-18", msft_output)

        a = await cache.get("AAPL:2026-04-18")
        m = await cache.get("MSFT:2026-04-18")

        assert a is not None and a.summary.startswith("Apple")
        assert m is not None and m.summary == "Microsoft summary"


# ---------------------------------------------------------------------------
# TTL / expiration
# ---------------------------------------------------------------------------


class TestTTL:
    async def test_expired_entry_returns_none(
        self, cache_db: str, sample_output: AnalystOutput
    ):
        # TTL of 0 seconds — any entry is immediately expired
        cache = AnalystCache(db_path=cache_db, ttl_seconds=0)
        await cache.set("AAPL:2026-04-18", sample_output)
        # Sleep past the TTL
        await asyncio.sleep(0.01)
        result = await cache.get("AAPL:2026-04-18")
        assert result is None

    async def test_fresh_entry_is_returned(
        self, cache: AnalystCache, sample_output: AnalystOutput
    ):
        await cache.set("AAPL:2026-04-18", sample_output)
        result = await cache.get("AAPL:2026-04-18")
        assert result is not None


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


class TestMaintenance:
    async def test_clear_expired_removes_only_old_rows(
        self, cache_db: str, sample_output: AnalystOutput
    ):
        cache = AnalystCache(db_path=cache_db, ttl_seconds=60)
        # Manually insert an expired row
        await cache.set("AAPL:2026-04-18", sample_output)

        # Directly poke the row's created_at into the past (simulating staleness)
        import aiosqlite

        async with aiosqlite.connect(cache_db) as db:
            await db.execute(
                "UPDATE analyst_cache SET created_at = ? WHERE cache_key = ?",
                (time.time() - 120, "AAPL:2026-04-18"),
            )
            await db.commit()

        # Insert a fresh row
        await cache.set("MSFT:2026-04-18", sample_output)

        removed = await cache.clear_expired()
        assert removed == 1

        # Fresh row still there
        assert await cache.get("MSFT:2026-04-18") is not None
        # Expired row gone
        assert await cache.get("AAPL:2026-04-18") is None

    async def test_clear_all_removes_everything(
        self, cache: AnalystCache, sample_output: AnalystOutput
    ):
        await cache.set("AAPL:2026-04-18", sample_output)
        await cache.set("MSFT:2026-04-18", sample_output)
        removed = await cache.clear_all()
        assert removed == 2
        assert await cache.get("AAPL:2026-04-18") is None


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    async def test_corrupt_payload_returns_none(
        self, cache: AnalystCache, cache_db: str
    ):
        import aiosqlite

        # Trigger setup then manually insert garbage
        await cache.get("dummy")
        async with aiosqlite.connect(cache_db) as db:
            await db.execute(
                "INSERT INTO analyst_cache (cache_key, payload, created_at) VALUES (?, ?, ?)",
                ("AAPL:2026-04-18", "this-is-not-valid-json", time.time()),
            )
            await db.commit()

        assert await cache.get("AAPL:2026-04-18") is None

    async def test_setup_is_idempotent(self, cache: AnalystCache):
        # Calling get / set many times should not fail on repeated setup
        await cache.get("x")
        await cache.get("y")
        await cache.get("z")
