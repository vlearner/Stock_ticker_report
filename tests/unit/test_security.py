"""Unit tests for Sprint 1 security features.

Covers:
- sanitize_message: strips injection chars, truncates at 200, collapses whitespace
- require_api_key: passes when key matches, returns 401 when wrong, passes when api_key=None
- ErrorResponse: serializes correctly with and without retry_after_s
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.middleware import sanitize_message
from src.schemas.errors import ErrorCode, ErrorResponse


# ---------------------------------------------------------------------------
# sanitize_message
# ---------------------------------------------------------------------------


class TestSanitizeMessage:
    def test_strips_angle_brackets(self) -> None:
        assert "<script>" not in sanitize_message("<script>alert(1)</script>")
        assert sanitize_message("<b>hello</b>") == "bhellob"

    def test_strips_curly_braces(self) -> None:
        result = sanitize_message("{injection} test {}")
        assert "{" not in result
        assert "}" not in result
        assert "injection" in result

    def test_strips_all_injection_chars(self) -> None:
        result = sanitize_message("<>{}")
        assert result == ""

    def test_truncates_at_200_chars(self) -> None:
        long_text = "a" * 300
        result = sanitize_message(long_text)
        assert len(result) == 200

    def test_truncates_exactly_at_200(self) -> None:
        text = "x" * 200
        assert sanitize_message(text) == text

    def test_truncates_injection_chars_after_strip(self) -> None:
        # Characters are removed before truncation
        text = "<" * 100 + "a" * 250
        result = sanitize_message(text)
        # After removing <, we have 250 a's, truncated to 200
        assert result == "a" * 200

    def test_collapses_multiple_spaces(self) -> None:
        result = sanitize_message("hello   world")
        assert result == "hello world"

    def test_collapses_newlines(self) -> None:
        result = sanitize_message("hello\n\n\nworld")
        assert result == "hello world"

    def test_collapses_tabs(self) -> None:
        result = sanitize_message("hello\t\tworld")
        assert result == "hello world"

    def test_strips_leading_trailing_whitespace(self) -> None:
        result = sanitize_message("  hello  ")
        assert result == "hello"

    def test_mixed_whitespace_collapse(self) -> None:
        result = sanitize_message("  hello  \n  world  ")
        assert result == "hello world"

    def test_empty_string(self) -> None:
        assert sanitize_message("") == ""

    def test_normal_message_unchanged(self) -> None:
        msg = "What is the AAPL stock price today?"
        assert sanitize_message(msg) == msg

    def test_ticker_query_passthrough(self) -> None:
        result = sanitize_message("AAPL vs MSFT analysis")
        assert result == "AAPL vs MSFT analysis"


# ---------------------------------------------------------------------------
# require_api_key (via FastAPI test client)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test")


class TestRequireApiKey:
    async def test_passes_when_api_key_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When settings.api_key is None, all requests pass through."""
        import src.api.middleware as mw_module
        import src.config as config_module

        monkeypatch.setattr(mw_module.settings, "api_key", None)
        monkeypatch.setattr(config_module.settings, "api_key", None)

        from unittest.mock import AsyncMock, patch
        import src.api.routes.chat as chat_module

        mock = AsyncMock(return_value={"final_message": "ok", "route": "single"})
        with patch.object(chat_module._pipeline, "ainvoke", mock):
            from src.api.app import app
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "AAPL", "session_id": "s1"},
                    # no X-API-Key header
                )
        assert resp.status_code == 200

    async def test_passes_when_correct_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Correct X-API-Key header is accepted."""
        import src.api.middleware as mw_module
        import src.config as config_module

        monkeypatch.setattr(mw_module.settings, "api_key", "secret-key")
        monkeypatch.setattr(config_module.settings, "api_key", "secret-key")

        from unittest.mock import AsyncMock, patch
        import src.api.routes.chat as chat_module

        mock = AsyncMock(return_value={"final_message": "ok", "route": "single"})
        with patch.object(chat_module._pipeline, "ainvoke", mock):
            from src.api.app import app
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/chat",
                    json={"message": "AAPL", "session_id": "s1"},
                    headers={"X-API-Key": "secret-key"},
                )
        assert resp.status_code == 200

    async def test_returns_401_when_wrong_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrong X-API-Key header returns HTTP 401."""
        import src.api.middleware as mw_module
        import src.config as config_module

        monkeypatch.setattr(mw_module.settings, "api_key", "secret-key")
        monkeypatch.setattr(config_module.settings, "api_key", "secret-key")

        from src.api.app import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/chat",
                json={"message": "AAPL", "session_id": "s1"},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401

    async def test_returns_401_when_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing X-API-Key header when auth is enabled returns HTTP 401."""
        import src.api.middleware as mw_module
        import src.config as config_module

        monkeypatch.setattr(mw_module.settings, "api_key", "secret-key")
        monkeypatch.setattr(config_module.settings, "api_key", "secret-key")

        from src.api.app import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/chat",
                json={"message": "AAPL", "session_id": "s1"},
                # no X-API-Key header
            )
        assert resp.status_code == 401

    async def test_401_body_has_error_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """401 response body contains ErrorCode.UNAUTHORIZED."""
        import src.api.middleware as mw_module
        import src.config as config_module

        monkeypatch.setattr(mw_module.settings, "api_key", "secret-key")
        monkeypatch.setattr(config_module.settings, "api_key", "secret-key")

        from src.api.app import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/chat",
                json={"message": "AAPL", "session_id": "s1"},
                headers={"X-API-Key": "bad"},
            )
        assert resp.status_code == 401
        body = resp.json()
        # FastAPI wraps HTTPException detail under "detail"
        detail = body.get("detail", body)
        assert detail.get("code") == ErrorCode.UNAUTHORIZED.value


# ---------------------------------------------------------------------------
# ErrorResponse serialization
# ---------------------------------------------------------------------------


class TestErrorResponse:
    def test_basic_serialization(self) -> None:
        err = ErrorResponse(code=ErrorCode.INVALID_TICKER, message="Unknown ticker: XYZ")
        data = err.model_dump()
        assert data["code"] == ErrorCode.INVALID_TICKER
        assert data["message"] == "Unknown ticker: XYZ"
        assert data["retry_after_s"] is None

    def test_rate_limited_with_retry_after(self) -> None:
        err = ErrorResponse(
            code=ErrorCode.RATE_LIMITED,
            message="Too many requests.",
            retry_after_s=3600,
        )
        data = err.model_dump()
        assert data["code"] == ErrorCode.RATE_LIMITED
        assert data["retry_after_s"] == 3600

    def test_json_serialization(self) -> None:
        err = ErrorResponse(code=ErrorCode.LLM_UNAVAILABLE, message="Groq is down")
        json_str = err.model_dump_json()
        assert "LLM_UNAVAILABLE" in json_str
        assert "Groq is down" in json_str

    def test_all_error_codes_valid(self) -> None:
        for code in ErrorCode:
            err = ErrorResponse(code=code, message="test")
            assert err.code == code

    def test_internal_error_no_retry_after(self) -> None:
        err = ErrorResponse(code=ErrorCode.INTERNAL_ERROR, message="Unexpected error")
        assert err.retry_after_s is None

    def test_validation_error_code(self) -> None:
        err = ErrorResponse(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid input provided.",
        )
        data = err.model_dump()
        assert data["code"] == ErrorCode.VALIDATION_ERROR
