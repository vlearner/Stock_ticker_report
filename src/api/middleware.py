"""API security middleware — auth dependency and input sanitization.

Functions
---------
- :func:`require_api_key` — FastAPI dependency that validates ``X-API-Key``.
  Passes through (no-op) when ``settings.api_key`` is ``None``.
- :func:`sanitize_message` — strips dangerous chars, collapses whitespace,
  and truncates to 200 characters before the message reaches the pipeline.
"""

from __future__ import annotations

import re

from fastapi import Header, HTTPException

from src.config import settings
from src.schemas.errors import ErrorCode, ErrorResponse

# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency — enforce X-API-Key header when auth is enabled.

    When ``settings.api_key`` is ``None`` the dependency is a no-op, which
    keeps local dev friction-free. When set, the header must match exactly.

    Raises:
        HTTPException: 401 with :class:`~src.schemas.errors.ErrorResponse`
            body if the key is missing or wrong.
    """
    if settings.api_key is None:
        # Auth disabled — pass through
        return

    if x_api_key != settings.api_key:
        error = ErrorResponse(
            code=ErrorCode.UNAUTHORIZED,
            message="Invalid or missing API key.",
        )
        raise HTTPException(
            status_code=401,
            detail=error.model_dump(),
        )


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------

_INJECTION_CHARS = re.compile(r"[<>/{}]")
_MULTI_WHITESPACE = re.compile(r"[ \t\r\n]+")
_MAX_MESSAGE_LEN = 200


def sanitize_message(text: str) -> str:
    """Clean user-supplied text before passing it to the pipeline.

    Steps (in order):
    1. Strip leading/trailing whitespace.
    2. Remove injection-prone characters: ``< > { }``.
    3. Collapse runs of whitespace/newlines into a single space.
    4. Truncate to 200 characters.

    Args:
        text: Raw user message string.

    Returns:
        Sanitized string safe for pipeline consumption.
    """
    text = text.strip()
    text = _INJECTION_CHARS.sub("", text)
    text = _MULTI_WHITESPACE.sub(" ", text)
    text = text[:_MAX_MESSAGE_LEN]
    return text
