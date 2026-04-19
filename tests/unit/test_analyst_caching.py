"""Integration tests for the analyst node's cache behaviour.

We don't exercise the real LLM — ``_structured_llm.ainvoke`` is patched —
but we do exercise the real ``AnalystCache`` against a temp SQLite DB so
both sides of the integration are covered.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("BRAVE_API_KEY", "test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")

from src.agents import analyst as analyst_agent  # noqa: E402
from src.cache.analyst_cache import AnalystCache  # noqa: E402
from src.schemas.ticker_data import AnalystOutput, Fundamentals, TickerData  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Swap the module singleton for one backed by a temp DB per test."""
    cache = AnalystCache(db_path=str(tmp_path / "cache.db"), ttl_seconds=3600)
    monkeypatch.setattr(analyst_agent, "analyst_cache", cache)
    return cache


@pytest.fixture
def sample_data() -> TickerData:
    return TickerData(
        ticker="AAPL",
        fundamentals=Fundamentals(ticker="AAPL", company_name="Apple Inc."),
    )


@pytest.fixture
def mock_llm_output() -> AnalystOutput:
    return AnalystOutput(
        summary="Apple: strong fundamentals.",
        key_points=["P/E in line with peers"],
        iteration=1,
    )


def _patch_llm(return_value: AnalystOutput) -> AsyncMock:
    """Build a patchable mock for ``_structured_llm.ainvoke``."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=return_value)
    return mock


# ---------------------------------------------------------------------------
# Caching enabled — first call misses, second call hits
# ---------------------------------------------------------------------------


class TestAnalystCacheIntegration:
    async def test_first_call_is_cache_miss_and_writes_cache(
        self, isolated_cache, sample_data, mock_llm_output
    ):
        mock_llm = _patch_llm(mock_llm_output)
        state = {"data_by_ticker": {"AAPL": sample_data}, "iteration_count": {}}

        with patch("src.agents.analyst._structured_llm", mock_llm):
            result = await analyst_agent.run(state)

        # LLM was called exactly once
        assert mock_llm.ainvoke.await_count == 1

        # Cache now has the entry
        key = AnalystCache.make_key("AAPL")
        cached = await isolated_cache.get(key)
        assert cached is not None
        assert cached.summary == mock_llm_output.summary

        # Output is present in the result
        assert "AAPL" in result["analyst_outputs"]

    async def test_second_call_is_cache_hit_and_skips_llm(
        self, isolated_cache, sample_data, mock_llm_output
    ):
        mock_llm = _patch_llm(mock_llm_output)
        state = {"data_by_ticker": {"AAPL": sample_data}, "iteration_count": {}}

        with patch("src.agents.analyst._structured_llm", mock_llm):
            # First call — populates cache
            await analyst_agent.run(state)
            # Second call — should hit cache
            result2 = await analyst_agent.run(state)

        # LLM was called only once despite two runs
        assert mock_llm.ainvoke.await_count == 1
        assert result2["analyst_outputs"]["AAPL"].summary == mock_llm_output.summary

    async def test_iteration_2_bypasses_cache(
        self, isolated_cache, sample_data, mock_llm_output
    ):
        """Post-critic revisions (iteration >= 2) must always call the LLM,
        even if a cached iteration-1 answer exists."""
        mock_llm = _patch_llm(mock_llm_output)
        # Pre-populate the cache as if iteration 1 already ran
        key = AnalystCache.make_key("AAPL")
        await isolated_cache.set(key, mock_llm_output)

        state = {
            "data_by_ticker": {"AAPL": sample_data},
            "iteration_count": {"AAPL": 1},  # will become 2 in this run
        }

        with patch("src.agents.analyst._structured_llm", mock_llm):
            await analyst_agent.run(state)

        # LLM WAS called because iteration=2 skips cache
        assert mock_llm.ainvoke.await_count == 1

    async def test_caching_disabled_always_calls_llm(
        self, isolated_cache, sample_data, mock_llm_output, monkeypatch
    ):
        from src.config import settings as app_settings

        monkeypatch.setattr(app_settings, "analyst_cache_enabled", False)
        mock_llm = _patch_llm(mock_llm_output)
        state = {"data_by_ticker": {"AAPL": sample_data}, "iteration_count": {}}

        with patch("src.agents.analyst._structured_llm", mock_llm):
            await analyst_agent.run(state)
            await analyst_agent.run(state)

        # Both runs hit the LLM — cache is off
        assert mock_llm.ainvoke.await_count == 2

        # Nothing written to cache either
        assert await isolated_cache.get(AnalystCache.make_key("AAPL")) is None

    async def test_llm_failure_does_not_write_cache(
        self, isolated_cache, sample_data
    ):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("groq down"))
        state = {"data_by_ticker": {"AAPL": sample_data}, "iteration_count": {}}

        with patch("src.agents.analyst._structured_llm", mock_llm):
            result = await analyst_agent.run(state)

        # No cache entry was written
        assert await isolated_cache.get(AnalystCache.make_key("AAPL")) is None
        # Fallback AnalystOutput was produced
        fallback = result["analyst_outputs"]["AAPL"]
        assert "temporarily unavailable" in fallback.summary.lower()

    async def test_cached_output_has_current_iteration(
        self, isolated_cache, sample_data, mock_llm_output
    ):
        """Cached summaries should reflect the current iteration counter,
        not the one that was stored."""
        # Store a cached entry with iteration=1
        key = AnalystCache.make_key("AAPL")
        await isolated_cache.set(key, mock_llm_output)  # iteration=1 inside

        mock_llm = _patch_llm(mock_llm_output)
        state = {
            "data_by_ticker": {"AAPL": sample_data},
            "iteration_count": {},  # will become 1 this pass
        }

        with patch("src.agents.analyst._structured_llm", mock_llm):
            result = await analyst_agent.run(state)

        assert mock_llm.ainvoke.await_count == 0  # served from cache
        assert result["analyst_outputs"]["AAPL"].iteration == 1
