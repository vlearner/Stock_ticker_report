"""Unit tests for src.tools.brave_tools.

All HTTP calls are monkeypatched — no real Brave API requests are made.
Environment variables for the Settings singleton are injected via
monkeypatch so that ``src.config.settings`` can be constructed without a
real ``.env`` file.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from src.schemas.ticker_data import NewsBundle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Brave API result fixture — mimics the shape of a real response.
def _make_brave_result(
    title: str = "AAPL hits new high",
    url: str = "https://example.com/article",
    hostname: str = "example.com",
    description: str = "Apple stock surged today.",
    page_age: str = "2025-12-01T10:00:00Z",
) -> dict[str, Any]:
    return {
        "title": title,
        "url": url,
        "meta_url": {"hostname": hostname},
        "description": description,
        "page_age": page_age,
    }


def _make_brave_response(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"web": {"results": results}}


def _mock_httpx_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
) -> httpx.Response:
    """Build a fake httpx.Response with the given status and JSON body."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data or {},
        request=httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search"),
    )
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid Brave JSON with 3 results -> NewsBundle with 3 items."""
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

    results = [
        _make_brave_result(title=f"Article {i}", url=f"https://example.com/{i}")
        for i in range(3)
    ]
    response_data = _make_brave_response(results)

    mock_get = AsyncMock(
        return_value=_mock_httpx_response(200, response_data),
    )
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    from src.tools.brave_tools import search_ticker_news

    bundle: NewsBundle = await search_ticker_news.ainvoke(
        {"ticker": "aapl", "num_results": 3}
    )

    assert isinstance(bundle, NewsBundle)
    assert len(bundle.items) == 3
    assert bundle.ticker == "AAPL"
    assert bundle.sentiment == "unknown"
    for i, item in enumerate(bundle.items):
        assert item.title == f"Article {i}"


@pytest.mark.asyncio
async def test_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty results array -> empty NewsBundle, no crash."""
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

    mock_get = AsyncMock(
        return_value=_mock_httpx_response(200, _make_brave_response([])),
    )
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    from src.tools.brave_tools import search_ticker_news

    bundle: NewsBundle = await search_ticker_news.ainvoke(
        {"ticker": "MSFT", "num_results": 5}
    )

    assert isinstance(bundle, NewsBundle)
    assert len(bundle.items) == 0
    assert bundle.ticker == "MSFT"


@pytest.mark.asyncio
async def test_5xx_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """First call returns 500, second returns valid JSON -> retry works."""
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

    results = [_make_brave_result()]
    good_resp = _mock_httpx_response(200, _make_brave_response(results))
    bad_resp = _mock_httpx_response(500, {"error": "internal"})

    call_count = 0

    async def side_effect(*args: Any, **kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return bad_resp
        return good_resp

    mock_get = AsyncMock(side_effect=side_effect)
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    from src.tools.brave_tools import search_ticker_news

    bundle: NewsBundle = await search_ticker_news.ainvoke(
        {"ticker": "GOOG", "num_results": 1}
    )

    assert call_count == 2
    assert len(bundle.items) == 1
    assert bundle.ticker == "GOOG"


@pytest.mark.asyncio
async def test_4xx_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """429 rate limit -> empty bundle returned immediately, only 1 call."""
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

    call_count = 0

    async def side_effect(*args: Any, **kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _mock_httpx_response(429, {"error": "rate limited"})

    mock_get = AsyncMock(side_effect=side_effect)
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    from src.tools.brave_tools import search_ticker_news

    bundle: NewsBundle = await search_ticker_news.ainvoke(
        {"ticker": "TSLA", "num_results": 5}
    )

    assert call_count == 1
    assert isinstance(bundle, NewsBundle)
    assert len(bundle.items) == 0
    assert bundle.ticker == "TSLA"


@pytest.mark.asyncio
async def test_network_error_empty_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx.ConnectError -> empty NewsBundle returned."""
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

    mock_get = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused"),
    )
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    from src.tools.brave_tools import search_ticker_news

    bundle: NewsBundle = await search_ticker_news.ainvoke(
        {"ticker": "AMZN", "num_results": 5}
    )

    assert isinstance(bundle, NewsBundle)
    assert len(bundle.items) == 0
    assert bundle.ticker == "AMZN"


@pytest.mark.asyncio
async def test_malformed_result_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """One result missing 'title' is skipped, others parsed fine."""
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

    results = [
        _make_brave_result(title="Good article 1"),
        {"url": "https://example.com/bad", "description": "No title here"},  # missing title
        _make_brave_result(title="Good article 2"),
    ]
    response_data = _make_brave_response(results)

    mock_get = AsyncMock(
        return_value=_mock_httpx_response(200, response_data),
    )
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    from src.tools.brave_tools import search_ticker_news

    bundle: NewsBundle = await search_ticker_news.ainvoke(
        {"ticker": "META", "num_results": 5}
    )

    assert len(bundle.items) == 2
    assert bundle.items[0].title == "Good article 1"
    assert bundle.items[1].title == "Good article 2"
