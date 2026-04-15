"""Unit tests for src.api.routes.ticker.

All yfinance calls are monkeypatched — no real network requests are made.
Each endpoint is tested via the FastAPI test client (httpx.AsyncClient +
ASGITransport) so we exercise the full HTTP stack without a live server.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.schemas.ticker_data import (
    Fundamentals,
    MovingAverages,
    TickerValidationResult,
    VolumeData,
)
from src.tools.yfinance_tools import YFinanceFetchError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fundamentals_payload() -> Fundamentals:
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
def moving_averages_payload() -> MovingAverages:
    return MovingAverages(
        ticker="AAPL",
        sma_50=185.0,
        sma_100=180.0,
        sma_200=170.0,
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def volume_payload() -> VolumeData:
    return VolumeData(
        ticker="AAPL",
        current_volume=55_000_000,
        avg_volume=80_000_000,
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def validation_valid() -> TickerValidationResult:
    return TickerValidationResult(
        ticker="AAPL",
        valid=True,
        company_name="Apple Inc.",
        reason=None,
    )


@pytest.fixture
def validation_invalid() -> TickerValidationResult:
    return TickerValidationResult(
        ticker="FAKE",
        valid=False,
        company_name=None,
        reason="empty info response",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _client() -> AsyncClient:
    """Build an async test client bound to the FastAPI app."""
    from src.api.app import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# GET /api/v1/ticker/fundamentals
# ---------------------------------------------------------------------------


class TestFundamentalsRoute:
    pytestmark = pytest.mark.asyncio

    async def test_happy_path(
        self, monkeypatch: pytest.MonkeyPatch, fundamentals_payload: Fundamentals
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        with patch(
            "src.api.routes.ticker.fetch_fundamentals",
            AsyncMock(return_value=fundamentals_payload),
        ):
            async with await _client() as client:
                resp = await client.get("/api/v1/ticker/fundamentals", params={"ticker": "AAPL"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["ticker"] == "AAPL"
        assert body["company_name"] == "Apple Inc."
        assert body["pe_ratio"] == pytest.approx(29.5)
        assert body["currency"] == "USD"

    async def test_ticker_missing_returns_422(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        async with await _client() as client:
            resp = await client.get("/api/v1/ticker/fundamentals")

        assert resp.status_code == 422

    async def test_yfinance_error_returns_502(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        with patch(
            "src.api.routes.ticker.fetch_fundamentals",
            AsyncMock(
                side_effect=YFinanceFetchError("AAPL", "info", IOError("network"))
            ),
        ):
            async with await _client() as client:
                resp = await client.get(
                    "/api/v1/ticker/fundamentals", params={"ticker": "AAPL"}
                )

        assert resp.status_code == 502
        assert "AAPL" in resp.json()["detail"]

    async def test_ticker_lowercased_accepted(
        self, monkeypatch: pytest.MonkeyPatch, fundamentals_payload: Fundamentals
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        with patch(
            "src.api.routes.ticker.fetch_fundamentals",
            AsyncMock(return_value=fundamentals_payload),
        ):
            async with await _client() as client:
                resp = await client.get(
                    "/api/v1/ticker/fundamentals", params={"ticker": "aapl"}
                )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/v1/ticker/moving-averages
# ---------------------------------------------------------------------------


class TestMovingAveragesRoute:
    pytestmark = pytest.mark.asyncio

    async def test_happy_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        moving_averages_payload: MovingAverages,
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        with patch(
            "src.api.routes.ticker.fetch_moving_averages",
            AsyncMock(return_value=moving_averages_payload),
        ):
            async with await _client() as client:
                resp = await client.get(
                    "/api/v1/ticker/moving-averages", params={"ticker": "AAPL"}
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ticker"] == "AAPL"
        assert body["sma_50"] == pytest.approx(185.0)
        assert body["sma_100"] == pytest.approx(180.0)
        assert body["sma_200"] == pytest.approx(170.0)

    async def test_partial_smas_null(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        partial = MovingAverages(
            ticker="NEW",
            sma_50=50.0,
            sma_100=None,
            sma_200=None,
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        with patch(
            "src.api.routes.ticker.fetch_moving_averages",
            AsyncMock(return_value=partial),
        ):
            async with await _client() as client:
                resp = await client.get(
                    "/api/v1/ticker/moving-averages", params={"ticker": "NEW"}
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["sma_50"] == pytest.approx(50.0)
        assert body["sma_100"] is None
        assert body["sma_200"] is None

    async def test_yfinance_error_returns_502(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        with patch(
            "src.api.routes.ticker.fetch_moving_averages",
            AsyncMock(
                side_effect=YFinanceFetchError("AAPL", "history", IOError("timeout"))
            ),
        ):
            async with await _client() as client:
                resp = await client.get(
                    "/api/v1/ticker/moving-averages", params={"ticker": "AAPL"}
                )

        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# GET /api/v1/ticker/volume
# ---------------------------------------------------------------------------


class TestVolumeRoute:
    pytestmark = pytest.mark.asyncio

    async def test_happy_path(
        self, monkeypatch: pytest.MonkeyPatch, volume_payload: VolumeData
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        with patch(
            "src.api.routes.ticker.fetch_volume_data",
            AsyncMock(return_value=volume_payload),
        ):
            async with await _client() as client:
                resp = await client.get(
                    "/api/v1/ticker/volume", params={"ticker": "AAPL"}
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ticker"] == "AAPL"
        assert body["current_volume"] == 55_000_000
        assert body["avg_volume"] == 80_000_000

    async def test_null_volumes_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        payload = VolumeData(
            ticker="XYZ",
            current_volume=None,
            avg_volume=None,
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        with patch(
            "src.api.routes.ticker.fetch_volume_data",
            AsyncMock(return_value=payload),
        ):
            async with await _client() as client:
                resp = await client.get(
                    "/api/v1/ticker/volume", params={"ticker": "XYZ"}
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["current_volume"] is None
        assert body["avg_volume"] is None

    async def test_yfinance_error_returns_502(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        with patch(
            "src.api.routes.ticker.fetch_volume_data",
            AsyncMock(
                side_effect=YFinanceFetchError("AAPL", "volume", IOError("both failed"))
            ),
        ):
            async with await _client() as client:
                resp = await client.get(
                    "/api/v1/ticker/volume", params={"ticker": "AAPL"}
                )

        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# GET /api/v1/ticker/validate
# ---------------------------------------------------------------------------


class TestValidateRoute:
    pytestmark = pytest.mark.asyncio

    async def test_valid_ticker(
        self,
        monkeypatch: pytest.MonkeyPatch,
        validation_valid: TickerValidationResult,
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        with patch(
            "src.api.routes.ticker.fetch_ticker_validation",
            AsyncMock(return_value=validation_valid),
        ):
            async with await _client() as client:
                resp = await client.get(
                    "/api/v1/ticker/validate", params={"ticker": "AAPL"}
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is True
        assert body["company_name"] == "Apple Inc."
        assert body["reason"] is None

    async def test_invalid_ticker_returns_200_with_valid_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        validation_invalid: TickerValidationResult,
    ) -> None:
        """validate endpoint never raises — invalid ticker returns 200 with valid=False."""
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        with patch(
            "src.api.routes.ticker.fetch_ticker_validation",
            AsyncMock(return_value=validation_invalid),
        ):
            async with await _client() as client:
                resp = await client.get(
                    "/api/v1/ticker/validate", params={"ticker": "FAKE"}
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert body["reason"] == "empty info response"

    async def test_ticker_missing_returns_422(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        async with await _client() as client:
            resp = await client.get("/api/v1/ticker/validate")

        assert resp.status_code == 422

    async def test_ticker_too_long_returns_422(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "test")
        monkeypatch.setenv("BRAVE_API_KEY", "test")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")

        async with await _client() as client:
            resp = await client.get(
                "/api/v1/ticker/validate", params={"ticker": "A" * 16}
            )

        assert resp.status_code == 422
