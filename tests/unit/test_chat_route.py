"""Unit tests for src.api.routes.chat.

All tool calls are patched — no real network traffic. Tests cover the
four intent paths (help / fundamentals / comparison / news), graceful
degradation when a single tool raises, and FastAPI's 422 on bad input.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.routes.chat import (
    _classify_intent,
    _extract_tickers,
)
from src.schemas.ticker_data import (
    Fundamentals,
    MovingAverages,
    NewsBundle,
    NewsItem,
    VolumeData,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the full-app Settings singleton can initialize in tests."""
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")


@pytest.fixture
def aapl_fundamentals() -> Fundamentals:
    return Fundamentals(
        ticker="AAPL",
        company_name="Apple Inc.",
        pe_ratio=29.5,
        eps=6.12,
        market_cap=3_000_000_000_000.0,
        week_52_high=230.0,
        week_52_low=165.0,
        dividend_yield=0.005,
        currency="USD",
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def msft_fundamentals() -> Fundamentals:
    return Fundamentals(
        ticker="MSFT",
        company_name="Microsoft Corp.",
        pe_ratio=34.1,
        eps=11.05,
        market_cap=3_100_000_000_000.0,
        week_52_high=470.0,
        week_52_low=350.0,
        dividend_yield=0.008,
        currency="USD",
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def aapl_moving_averages() -> MovingAverages:
    return MovingAverages(
        ticker="AAPL",
        sma_50=185.0,
        sma_100=180.0,
        sma_200=170.0,
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def aapl_volume() -> VolumeData:
    return VolumeData(
        ticker="AAPL",
        current_volume=55_000_000,
        avg_volume=80_000_000,
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def tsla_news() -> NewsBundle:
    return NewsBundle(
        ticker="TSLA",
        items=[
            NewsItem(
                title="Tesla beats Q4 delivery estimates",
                url="https://example.com/1",
                source="Reuters",
                snippet="Tesla delivered more cars than analysts expected…",
            ),
            NewsItem(
                title="Cybertruck ramp update",
                url="https://example.com/2",
                source="Bloomberg",
            ),
        ],
        sentiment="unknown",
    )


async def _client() -> AsyncClient:
    from src.api.app import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestExtractTickers:
    def test_single_ticker(self) -> None:
        assert _extract_tickers("AAPL") == ["AAPL"]

    def test_lowercased_input_uppercased(self) -> None:
        assert _extract_tickers("aapl vs msft") == ["AAPL", "MSFT"]

    def test_stopwords_filtered(self) -> None:
        assert _extract_tickers("hi") == []
        assert _extract_tickers("what is AAPL") == ["AAPL"]

    def test_deduplicated(self) -> None:
        assert _extract_tickers("AAPL AAPL AAPL") == ["AAPL"]

    def test_capped_at_four(self) -> None:
        tickers = _extract_tickers("AAPL MSFT GOOG AMZN META NVDA")
        assert len(tickers) == 4


class TestClassifyIntent:
    def test_no_tickers_is_help(self) -> None:
        assert _classify_intent("hello there", []) == "help"

    def test_news_keyword(self) -> None:
        assert _classify_intent("news TSLA", ["TSLA"]) == "news"
        assert _classify_intent("any headlines on AAPL?", ["AAPL"]) == "news"

    def test_comparison_needs_two_tickers_and_keyword(self) -> None:
        assert (
            _classify_intent("AAPL vs MSFT", ["AAPL", "MSFT"]) == "comparison"
        )
        # only one ticker ⇒ fall back to fundamentals
        assert _classify_intent("AAPL vs bond", ["AAPL"]) == "fundamentals"

    def test_default_is_fundamentals(self) -> None:
        assert _classify_intent("AAPL", ["AAPL"]) == "fundamentals"


# ---------------------------------------------------------------------------
# HTTP integration
# ---------------------------------------------------------------------------


class TestChatRoute:
    pytestmark = pytest.mark.asyncio

    async def test_single_ticker_calls_three_tools(
        self,
        aapl_fundamentals: Fundamentals,
        aapl_moving_averages: MovingAverages,
        aapl_volume: VolumeData,
    ) -> None:
        f_mock = AsyncMock(return_value=aapl_fundamentals)
        ma_mock = AsyncMock(return_value=aapl_moving_averages)
        v_mock = AsyncMock(return_value=aapl_volume)
        news_mock = AsyncMock()

        with patch("src.api.routes.chat.fetch_fundamentals", f_mock), patch(
            "src.api.routes.chat.fetch_moving_averages", ma_mock
        ), patch("src.api.routes.chat.fetch_volume_data", v_mock), patch(
            "src.api.routes.chat.fetch_ticker_news", news_mock
        ):
            async with await _client() as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "AAPL", "session_id": "sid-1"},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["used_tools"] == ["fundamentals", "moving_averages", "volume"]
        assert body["session_id"] == "sid-1"
        assert "AAPL" in body["reply"]
        assert "Apple Inc." in body["reply"]
        assert "185" in body["reply"]  # sma_50
        f_mock.assert_awaited_once_with("AAPL")
        ma_mock.assert_awaited_once_with("AAPL")
        v_mock.assert_awaited_once_with("AAPL")
        news_mock.assert_not_awaited()

    async def test_news_query_only_calls_news_tool(
        self, tsla_news: NewsBundle
    ) -> None:
        news_mock = AsyncMock(return_value=tsla_news)
        f_mock = AsyncMock()
        ma_mock = AsyncMock()
        v_mock = AsyncMock()

        with patch("src.api.routes.chat.fetch_ticker_news", news_mock), patch(
            "src.api.routes.chat.fetch_fundamentals", f_mock
        ), patch("src.api.routes.chat.fetch_moving_averages", ma_mock), patch(
            "src.api.routes.chat.fetch_volume_data", v_mock
        ):
            async with await _client() as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "news TSLA", "session_id": "sid-2"},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["used_tools"] == ["news"]
        assert "Tesla beats Q4" in body["reply"]
        news_mock.assert_awaited_once_with("TSLA")
        f_mock.assert_not_awaited()
        ma_mock.assert_not_awaited()
        v_mock.assert_not_awaited()

    async def test_comparison_calls_fundamentals_for_each(
        self,
        aapl_fundamentals: Fundamentals,
        msft_fundamentals: Fundamentals,
    ) -> None:
        by_ticker = {"AAPL": aapl_fundamentals, "MSFT": msft_fundamentals}
        f_mock = AsyncMock(side_effect=lambda t: by_ticker[t])

        with patch("src.api.routes.chat.fetch_fundamentals", f_mock):
            async with await _client() as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={
                        "message": "AAPL vs MSFT",
                        "session_id": "sid-3",
                    },
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["used_tools"] == ["fundamentals"]
        assert "AAPL" in body["reply"]
        assert "MSFT" in body["reply"]
        assert "Apple" in body["reply"]
        assert "Microsoft" in body["reply"]
        assert f_mock.await_count == 2

    async def test_no_ticker_returns_help_text(self) -> None:
        f_mock = AsyncMock()
        news_mock = AsyncMock()

        with patch("src.api.routes.chat.fetch_fundamentals", f_mock), patch(
            "src.api.routes.chat.fetch_ticker_news", news_mock
        ):
            async with await _client() as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "hi", "session_id": "sid-4"},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["used_tools"] == []
        assert "AAPL" in body["reply"]  # help text mentions the example
        f_mock.assert_not_awaited()
        news_mock.assert_not_awaited()

    async def test_one_tool_failure_degrades_gracefully(
        self,
        aapl_moving_averages: MovingAverages,
        aapl_volume: VolumeData,
    ) -> None:
        """When fetch_fundamentals raises, the other two sections still render."""
        from src.tools.yfinance_tools import YFinanceFetchError

        f_mock = AsyncMock(
            side_effect=YFinanceFetchError("AAPL", "info", IOError("boom"))
        )
        ma_mock = AsyncMock(return_value=aapl_moving_averages)
        v_mock = AsyncMock(return_value=aapl_volume)

        with patch("src.api.routes.chat.fetch_fundamentals", f_mock), patch(
            "src.api.routes.chat.fetch_moving_averages", ma_mock
        ), patch("src.api.routes.chat.fetch_volume_data", v_mock):
            async with await _client() as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "AAPL", "session_id": "sid-5"},
                )

        assert resp.status_code == 200
        body = resp.json()
        reply = body["reply"]
        assert "Fundamentals unavailable" in reply
        assert "185" in reply  # sma_50 still rendered
        assert "55,000,000" in reply  # volume still rendered

    async def test_missing_message_returns_422(self) -> None:
        async with await _client() as client:
            resp = await client.post(
                "/api/v1/chat", json={"session_id": "sid-6"}
            )
        assert resp.status_code == 422

    async def test_response_includes_session_query_limit(
        self,
        aapl_fundamentals: Fundamentals,
        aapl_moving_averages: MovingAverages,
        aapl_volume: VolumeData,
    ) -> None:
        """UI depends on server echoing the configured limit."""
        with patch(
            "src.api.routes.chat.fetch_fundamentals",
            AsyncMock(return_value=aapl_fundamentals),
        ), patch(
            "src.api.routes.chat.fetch_moving_averages",
            AsyncMock(return_value=aapl_moving_averages),
        ), patch(
            "src.api.routes.chat.fetch_volume_data",
            AsyncMock(return_value=aapl_volume),
        ):
            async with await _client() as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "AAPL", "session_id": "sid-7"},
                )
        assert resp.status_code == 200
        assert resp.json()["session_query_limit"] >= 1
