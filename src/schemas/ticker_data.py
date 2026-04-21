"""Pydantic v2 data contracts for the ticker-report pipeline.

These models are the canonical inter-agent wire format. Tools populate
their slice (:class:`Fundamentals`, :class:`MovingAverages`,
:class:`VolumeData`, :class:`NewsBundle`); the DataFetcher / NewsFetcher
agents merge slices into a single :class:`TickerData` instance that the
Analyst consumes. Every numeric field is ``Optional`` so partial failures
degrade gracefully (Pattern 6 — Exception Handling and Recovery).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Timezone-aware UTC ``now`` — avoids naive datetimes on the wire."""

    return datetime.now(timezone.utc)


class _StrictBase(BaseModel):
    """Base model with strict configuration shared by all schemas."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
    )


# ---------------------------------------------------------------------------
# Per-slice models (populated by individual tools)
# ---------------------------------------------------------------------------


class Fundamentals(_StrictBase):
    """Point-in-time fundamentals fetched via yfinance ``.info``."""

    ticker: str
    company_name: str | None = None
    pe_ratio: float | None = None
    eps: float | None = None
    market_cap: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None
    dividend_yield: float | None = None
    currency: str | None = None
    as_of: datetime = Field(default_factory=_utcnow)

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.upper()


class MovingAverages(_StrictBase):
    """Simple moving averages over daily closes."""

    ticker: str
    sma_50: float | None = None
    sma_100: float | None = None
    sma_200: float | None = None
    as_of: datetime = Field(default_factory=_utcnow)

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.upper()


class MAChartData(_StrictBase):
    """Time-series data for rendering a moving average chart."""

    ticker: str
    dates: list[str] = Field(default_factory=list)
    prices: list[float] = Field(default_factory=list)
    sma50: list[float | None] = Field(default_factory=list)
    sma200: list[float | None] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=_utcnow)

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.upper()


class VolumeData(_StrictBase):
    """Current and average trading volume."""

    ticker: str
    current_volume: int | None = None
    avg_volume: int | None = None
    as_of: datetime = Field(default_factory=_utcnow)

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.upper()


class NewsItem(_StrictBase):
    """A single news article returned by the Brave Search adapter."""

    title: str
    url: str
    source: str | None = None
    published: datetime | None = None
    snippet: str | None = None


Sentiment = Literal["positive", "neutral", "negative", "unknown"]


class NewsBundle(_StrictBase):
    """Aggregated news for a single ticker."""

    ticker: str
    items: list[NewsItem] = Field(default_factory=list)
    sentiment: Sentiment = "unknown"
    as_of: datetime = Field(default_factory=_utcnow)

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.upper()


# ---------------------------------------------------------------------------
# Container passed between agents
# ---------------------------------------------------------------------------


class TickerData(_StrictBase):
    """Unified per-ticker container.

    Sub-slices are independently ``Optional`` so that partial failures
    (e.g. yfinance ok, Brave rate-limited) still produce a usable payload
    downstream. Warnings collected during fetching are surfaced here so the
    Analyst can mention them in the final summary.
    """

    ticker: str
    fundamentals: Fundamentals | None = None
    moving_averages: MovingAverages | None = None
    volume: VolumeData | None = None
    news: NewsBundle | None = None
    warnings: list[str] = Field(default_factory=list)
    data_completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.upper()


# ---------------------------------------------------------------------------
# Tool / agent outputs
# ---------------------------------------------------------------------------


class TickerValidationResult(_StrictBase):
    """Return type of :func:`src.tools.yfinance_tools.validate_ticker`."""

    ticker: str
    valid: bool
    company_name: str | None = None
    reason: str | None = None

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.upper()


class AnalystOutput(_StrictBase):
    """Structured output from the Analyst agent (pre-formatting)."""

    summary: str = Field(..., max_length=500)
    key_points: list[str] = Field(default_factory=list)
    iteration: int = Field(default=1, ge=1)


class CritiqueResult(_StrictBase):
    """Verdict from the Critic step (Pattern 5 — Reflection)."""

    passed: bool
    grounded: bool
    no_advice: bool
    concise: bool
    issues: list[str] = Field(default_factory=list)


class GoalMetrics(_StrictBase):
    """Observability metrics evaluated at the end of every run (Pattern 9)."""

    data_fresh: bool
    completeness_score: float = Field(ge=0.0, le=1.0)
    total_latency_ms: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
