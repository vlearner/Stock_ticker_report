"""Unit tests for the LLM-based orchestrator agent.

LLM calls are mocked via AsyncMock so tests run without a Groq API key.
Separate tests verify the heuristic fallback behaves correctly when the
LLM raises an exception.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.orchestrator import OrchestratorOutput, _heuristic_fallback, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output(route, tickers, format_style) -> OrchestratorOutput:
    return OrchestratorOutput(route=route, tickers=tickers, format_style=format_style)


def _make_mock_llm(return_value: OrchestratorOutput | None = None, side_effect=None) -> MagicMock:
    """Build a mock that quacks like _structured_llm (has an async ainvoke)."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=return_value, side_effect=side_effect)
    return mock


def _run(message: str, llm_output: OrchestratorOutput | None = None) -> dict:
    """Run the orchestrator with a mocked LLM response."""
    mock_response = llm_output or _make_output("invalid", [], "minimal")
    with patch("src.agents.orchestrator._structured_llm", _make_mock_llm(mock_response)):
        return asyncio.get_event_loop().run_until_complete(run({"user_message": message}))


# ---------------------------------------------------------------------------
# LLM path — happy cases
# ---------------------------------------------------------------------------

class TestLLMPath:
    def test_single_minimal(self):
        result = _run("AAPL", _make_output("single", ["AAPL"], "minimal"))
        assert result["route"] == "single"
        assert result["tickers"] == ["AAPL"]
        assert result["format_style"] == "minimal"

    def test_single_rich(self):
        result = _run("AAPL analysis", _make_output("single", ["AAPL"], "rich"))
        assert result["route"] == "single"
        assert result["format_style"] == "rich"

    def test_comparison(self):
        result = _run("AAPL vs MSFT", _make_output("comparison", ["AAPL", "MSFT"], "rich"))
        assert result["route"] == "comparison"
        assert result["tickers"] == ["AAPL", "MSFT"]
        assert result["format_style"] == "rich"

    def test_news_only(self):
        result = _run("TSLA latest news", _make_output("news_only", ["TSLA"], "news"))
        assert result["route"] == "news_only"
        assert result["tickers"] == ["TSLA"]
        assert result["format_style"] == "news"

    def test_invalid(self):
        result = _run("hello world", _make_output("invalid", [], "minimal"))
        assert result["route"] == "invalid"
        assert result["tickers"] == []

    def test_tickers_uppercased(self):
        """Tickers returned by LLM should always be uppercased."""
        result = _run("aapl", _make_output("single", ["aapl"], "minimal"))
        assert result["tickers"] == ["AAPL"]

    def test_start_time_set(self):
        result = _run("AAPL", _make_output("single", ["AAPL"], "minimal"))
        assert isinstance(result["start_time"], float)
        assert result["start_time"] > 0


# ---------------------------------------------------------------------------
# Fallback path — LLM raises, heuristic kicks in
# ---------------------------------------------------------------------------

class TestFallback:
    def _run_with_llm_error(self, message: str) -> dict:
        with patch(
            "src.agents.orchestrator._structured_llm",
            _make_mock_llm(side_effect=RuntimeError("rate limited")),
        ):
            return asyncio.get_event_loop().run_until_complete(run({"user_message": message}))

    def test_fallback_single(self):
        result = self._run_with_llm_error("AAPL")
        assert result["route"] == "single"
        assert "AAPL" in result["tickers"]

    def test_fallback_comparison(self):
        result = self._run_with_llm_error("AAPL vs MSFT")
        assert result["route"] == "comparison"
        assert set(result["tickers"]) == {"AAPL", "MSFT"}

    def test_fallback_news_only(self):
        result = self._run_with_llm_error("TSLA latest news")
        assert result["route"] == "news_only"

    def test_fallback_invalid(self):
        result = self._run_with_llm_error("random gibberish")
        assert result["route"] == "invalid"
        assert result["tickers"] == []


# ---------------------------------------------------------------------------
# Heuristic fallback directly
# ---------------------------------------------------------------------------

class TestHeuristicFallback:
    def test_plain_ticker(self):
        out = _heuristic_fallback("NVDA")
        assert out.route == "single"
        assert out.tickers == ["NVDA"]
        assert out.format_style == "minimal"

    def test_detail_keyword(self):
        out = _heuristic_fallback("NVDA report")
        assert out.route == "single"
        assert out.format_style == "rich"

    def test_compare(self):
        out = _heuristic_fallback("AAPL vs GOOGL")
        assert out.route == "comparison"
        assert out.format_style == "rich"

    def test_news(self):
        out = _heuristic_fallback("AMZN news")
        assert out.route == "news_only"
        assert out.format_style == "news"

    def test_no_ticker(self):
        out = _heuristic_fallback("what is the weather")
        assert out.route == "invalid"
        assert out.tickers == []

    def test_stopwords_not_extracted_as_tickers(self):
        out = _heuristic_fallback("give me TSLA report")
        assert "GIVE" not in out.tickers
        assert "TSLA" in out.tickers
