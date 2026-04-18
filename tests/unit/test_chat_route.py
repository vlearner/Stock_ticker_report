"""Unit tests for src.api.routes.chat.

The chat route is now a thin wrapper around the LangGraph pipeline.
All tests patch ``_pipeline.ainvoke`` so no real network traffic occurs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import src.api.routes.chat as chat_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")


async def _client() -> AsyncClient:
    from src.api.app import app
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _pipeline_result(final_message: str, route: str = "single") -> dict:
    return {"final_message": final_message, "route": route}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChatRoute:
    async def test_successful_single_ticker(self) -> None:
        mock = AsyncMock(return_value=_pipeline_result("*AAPL*\n\nSummary here.", "single"))
        with patch.object(chat_module._pipeline, "ainvoke", mock):
            async with await _client() as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "AAPL", "session_id": "sid-1"},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["route"] == "single"
        assert "AAPL" in body["reply"]
        assert body["session_id"] == "sid-1"
        assert body["session_query_limit"] >= 1
        mock.assert_awaited_once_with({"user_message": "AAPL"})

    async def test_comparison_route(self) -> None:
        mock = AsyncMock(return_value=_pipeline_result("*AAPL* vs *MSFT*", "comparison"))
        with patch.object(chat_module._pipeline, "ainvoke", mock):
            async with await _client() as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "AAPL vs MSFT", "session_id": "sid-2"},
                )

        assert resp.status_code == 200
        assert resp.json()["route"] == "comparison"

    async def test_news_only_route(self) -> None:
        mock = AsyncMock(return_value=_pipeline_result("*TSLA — Latest News*\n1. Headline", "news_only"))
        with patch.object(chat_module._pipeline, "ainvoke", mock):
            async with await _client() as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "TSLA news", "session_id": "sid-3"},
                )

        assert resp.status_code == 200
        assert resp.json()["route"] == "news_only"

    async def test_invalid_route_returns_message(self) -> None:
        mock = AsyncMock(return_value=_pipeline_result(
            "I couldn't identify a valid ticker symbol.", "invalid"
        ))
        with patch.object(chat_module._pipeline, "ainvoke", mock):
            async with await _client() as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "hello", "session_id": "sid-4"},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["route"] == "invalid"
        assert len(body["reply"]) > 0

    async def test_missing_message_returns_422(self) -> None:
        async with await _client() as client:
            resp = await client.post(
                "/api/v1/chat", json={"session_id": "sid-6"}
            )
        assert resp.status_code == 422

    async def test_empty_message_returns_422(self) -> None:
        async with await _client() as client:
            resp = await client.post(
                "/api/v1/chat",
                json={"message": "", "session_id": "sid-7"},
            )
        assert resp.status_code == 422
