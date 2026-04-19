"""Unit tests for src.tools.ticker_search.

All yfinance.Search calls are mocked — no network requests are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.ticker_search import search_ticker, search_ticker_async, ticker_search_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_search(quotes: list[dict]) -> MagicMock:
    """Return a mock yf.Search result with the given quotes list."""
    m = MagicMock()
    m.quotes = quotes
    return m


# ---------------------------------------------------------------------------
# search_ticker — synchronous
# ---------------------------------------------------------------------------


class TestSearchTicker:
    def test_returns_none_for_empty_string(self):
        assert search_ticker("") is None

    def test_returns_none_for_whitespace(self):
        assert search_ticker("   ") is None

    def test_returns_none_when_no_quotes(self):
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search([])):
            assert search_ticker("unkown corp xyz") is None

    def test_returns_major_exchange_symbol(self):
        quotes = [
            {"symbol": "AAPL.DE", "exchange": "GER"},
            {"symbol": "AAPL", "exchange": "NMS"},
        ]
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search(quotes)):
            assert search_ticker("Apple") == "AAPL"

    def test_prefers_first_major_exchange_match(self):
        quotes = [
            {"symbol": "XYZ", "exchange": "OTC"},
            {"symbol": "MSFT", "exchange": "NMS"},
            {"symbol": "MSFX", "exchange": "NYQ"},
        ]
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search(quotes)):
            assert search_ticker("Microsoft") == "MSFT"

    def test_falls_back_to_first_result_when_no_major_exchange(self):
        quotes = [
            {"symbol": "XYZ", "exchange": "OTC"},
            {"symbol": "ABC", "exchange": "PINK"},
        ]
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search(quotes)):
            assert search_ticker("XYZ Corp") == "XYZ"

    def test_returns_none_when_symbol_missing(self):
        quotes = [{"exchange": "NMS"}]  # no "symbol" key
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search(quotes)):
            assert search_ticker("Broken Corp") is None

    def test_returns_uppercase_symbol(self):
        quotes = [{"symbol": "nvda", "exchange": "NMS"}]
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search(quotes)):
            assert search_ticker("Nvidia") == "NVDA"

    def test_handles_yfinance_exception_gracefully(self):
        with patch("src.tools.ticker_search.yf.Search", side_effect=RuntimeError("network error")):
            assert search_ticker("Tesla") is None

    def test_nyq_exchange_is_major(self):
        quotes = [{"symbol": "F", "exchange": "NYQ"}]
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search(quotes)):
            assert search_ticker("Ford") == "F"


# ---------------------------------------------------------------------------
# search_ticker_async
# ---------------------------------------------------------------------------


class TestSearchTickerAsync:
    async def test_async_returns_same_as_sync(self):
        quotes = [{"symbol": "TSLA", "exchange": "NMS"}]
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search(quotes)):
            result = await search_ticker_async("Tesla")
        assert result == "TSLA"

    async def test_async_returns_none_on_no_match(self):
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search([])):
            result = await search_ticker_async("nosuchthing")
        assert result is None


# ---------------------------------------------------------------------------
# ticker_search_tool (LangChain @tool wrapper)
# ---------------------------------------------------------------------------


class TestTickerSearchTool:
    def test_returns_symbol_for_known_company(self):
        quotes = [{"symbol": "GOOGL", "exchange": "NMS"}]
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search(quotes)):
            assert ticker_search_tool.invoke({"company_name": "Alphabet"}) == "GOOGL"

    def test_returns_not_found_on_empty_results(self):
        with patch("src.tools.ticker_search.yf.Search", return_value=_mock_search([])):
            assert ticker_search_tool.invoke({"company_name": "ghostcorp"}) == "NOT_FOUND"
