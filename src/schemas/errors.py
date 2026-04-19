"""Standard error codes and response model for the API.

All API error responses should use :class:`ErrorResponse` with an
:class:`ErrorCode` so clients can react to specific failure modes.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ErrorCode(str, Enum):
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_TICKER = "INVALID_TICKER"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    """Structured error payload returned by all API error responses."""

    code: ErrorCode
    message: str
    retry_after_s: int | None = None  # populated for RATE_LIMITED
