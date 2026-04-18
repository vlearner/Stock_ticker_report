"""Integration tests for the orchestrator — hits the real Groq API.

Prerequisites
-------------
Set ``GROQ_API_KEY`` in your environment or ``.env`` file.

Run command
-----------
    pytest tests/integration/test_orchestrator_integration.py -v -m integration

These tests are excluded from the default ``pytest`` run to avoid burning API
credits in CI.  Opt in with ``-m integration``.

Budget note
-----------
Groq free tier allows only 5 000 tokens/day.  Each structured-output call uses
~500 tokens, so 5 tests ≈ 2 500 tokens per run — leaves room for two runs/day.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

import pytest
from langchain_groq import ChatGroq
from pydantic import SecretStr

from src.agents import orchestrator
from src.agents.orchestrator import OrchestratorOutput

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# API key — read from environment / .env
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    pytest.skip(
        "GROQ_API_KEY not set — export it or add it to .env to run integration tests",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
async def rate_limit_guard():
    """Brief pause between tests to stay within Groq free-tier rate limits."""
    yield
    await asyncio.sleep(5)


@pytest.fixture(autouse=True)
def real_groq_client():
    """Swap module-level _structured_llm for one using the env API key.

    Also patches _heuristic_fallback to raise so tests fail loudly if the
    LLM call fails instead of silently passing via the fallback.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=SecretStr(GROQ_API_KEY),
        temperature=0,
        max_retries=1,
    )
    real_client = llm.with_structured_output(OrchestratorOutput)

    def _no_fallback(message: str) -> OrchestratorOutput:
        raise AssertionError(
            f"Heuristic fallback triggered for {message!r} — "
            "LLM call failed. Check API key or Groq quota."
        )

    with (
        patch.object(orchestrator, "_structured_llm", real_client),
        patch.object(orchestrator, "_heuristic_fallback", _no_fallback),
    ):
        yield


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _classify(message: str) -> dict:
    return await orchestrator.run({"user_message": message})


# ---------------------------------------------------------------------------
# Tests — one per route, covering all format_style values
# ---------------------------------------------------------------------------

async def test_single_minimal():
    """Plain ticker → single / minimal; start_time is populated."""
    result = await _classify("AAPL")
    assert result["route"] == "single"
    assert result["tickers"] == ["AAPL"]
    assert result["format_style"] == "minimal"
    assert isinstance(result["start_time"], float) and result["start_time"] > 0


async def test_single_rich():
    """Ticker + detail keyword → single / rich."""
    result = await _classify("AAPL analysis")
    assert result["route"] == "single"
    assert result["tickers"] == ["AAPL"]
    assert result["format_style"] == "rich"


async def test_comparison():
    """Two tickers with 'vs' → comparison / rich."""
    result = await _classify("AAPL vs MSFT")
    assert result["route"] == "comparison"
    assert set(result["tickers"]) == {"AAPL", "MSFT"}
    assert result["format_style"] == "rich"


async def test_news_only():
    """Ticker + news keyword → news_only / news."""
    result = await _classify("TSLA latest news")
    assert result["route"] == "news_only"
    assert result["tickers"] == ["TSLA"]
    assert result["format_style"] == "news"


async def test_invalid():
    """No ticker → invalid / minimal with empty tickers list."""
    result = await _classify("weather tomorrow")
    assert result["route"] == "invalid"
    assert result["tickers"] == []
