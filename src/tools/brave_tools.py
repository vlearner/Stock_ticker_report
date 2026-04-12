"""Brave Web Search tools for fetching recent ticker news.

Uses the Brave Search API (``/res/v1/web/search``) to retrieve recent
news articles for a given stock ticker. Results are parsed into a
:class:`~src.schemas.ticker_data.NewsBundle` that downstream agents
(Analyst) consume for sentiment classification and summary generation.

Network calls use :mod:`httpx` natively async — no thread offloading
needed (unlike yfinance). Transient 5xx / transport errors are retried
once; 4xx errors return an empty bundle immediately. The tool itself
never raises — callers check ``len(bundle.items) == 0`` to detect
failure (Pattern 6 — Exception Handling and Recovery).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import settings
from src.schemas.ticker_data import NewsBundle, NewsItem

logger = logging.getLogger(__name__)

_BRAVE_BASE_URL = "https://api.search.brave.com/res/v1/web/search"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BraveFetchError(RuntimeError):
    """Raised when a Brave Search API call fails after retry attempts.

    Upstream code (the ``search_ticker_news`` tool) catches this and returns
    an empty :class:`NewsBundle` rather than propagating.
    """

    def __init__(
        self,
        ticker: str,
        status_code: int | None,
        original: Exception | None = None,
    ) -> None:
        detail = f"status={status_code}" if status_code else str(original)
        super().__init__(
            f"Brave search failed for {ticker!r}: {detail}"
        )
        self.ticker = ticker
        self.status_code = status_code
        self.original = original


# ---------------------------------------------------------------------------
# Tool argument schema
# ---------------------------------------------------------------------------


class NewsSearchArg(BaseModel):
    """Argument schema for :func:`search_ticker_news`."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=15,
        description="Stock ticker symbol (e.g. 'AAPL'). Uppercased internally.",
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of search results to request (1–20).",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_published(result: dict[str, Any]) -> datetime | None:
    """Best-effort extraction of a publication timestamp from a Brave result.

    Brave may provide ``age`` (e.g. "2 hours ago") or ``page_age`` (ISO-ish
    date string). We attempt ISO parse first; on failure we return ``None``.
    """
    for key in ("page_age", "age"):
        raw = result.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    return None


def _parse_source(result: dict[str, Any]) -> str | None:
    """Extract the source name from a Brave result."""
    meta_url = result.get("meta_url")
    if isinstance(meta_url, dict):
        hostname = meta_url.get("hostname")
        if hostname:
            return hostname

    profile = result.get("profile")
    if isinstance(profile, dict):
        name = profile.get("name")
        if name:
            return name

    return None


def _parse_results(ticker: str, raw_results: list[dict[str, Any]]) -> NewsBundle:
    """Parse Brave web results into a NewsBundle.

    Best-effort: malformed items (e.g. missing ``title``) are silently
    skipped so a single bad result never breaks the whole bundle.
    """
    items: list[NewsItem] = []
    for result in raw_results:
        try:
            title = result.get("title")
            if not title:
                logger.debug("Skipping Brave result with missing title")
                continue

            url = result.get("url", "")
            items.append(
                NewsItem(
                    title=title,
                    url=url,
                    source=_parse_source(result),
                    published=_parse_published(result),
                    snippet=result.get("description"),
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("Skipping malformed Brave result", exc_info=True)
            continue

    return NewsBundle(ticker=ticker.upper(), items=items, sentiment="unknown")


async def _brave_search(query: str, count: int) -> dict[str, Any]:
    """Raw Brave API call with retry on 5xx / transport errors.

    Retries exactly once on 5xx status codes or ``httpx.TransportError``.
    4xx errors raise :class:`BraveFetchError` immediately (no retry).

    Returns:
        The full parsed JSON response dict.

    Raises:
        BraveFetchError: On any non-recoverable HTTP or transport failure.
    """
    headers = {
        "X-Subscription-Token": settings.brave_api_key,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "count": count,
        "freshness": "pw",
    }
    timeout = httpx.Timeout(timeout=settings.brave_search_timeout_s)

    last_error: Exception | None = None
    last_status: int | None = None

    for attempt in range(2):  # initial + 1 retry
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    _BRAVE_BASE_URL, headers=headers, params=params
                )

            if response.status_code >= 500:
                last_status = response.status_code
                last_error = RuntimeError(
                    f"Brave API returned {response.status_code}"
                )
                logger.warning(
                    "Brave search 5xx (attempt %d/2): %d",
                    attempt + 1,
                    response.status_code,
                )
                continue  # retry

            if response.status_code >= 400:
                # 4xx — do NOT retry
                raise BraveFetchError(
                    ticker=query,
                    status_code=response.status_code,
                    original=None,
                )

            return response.json()

        except BraveFetchError:
            raise
        except httpx.TransportError as exc:
            last_error = exc
            last_status = None
            logger.warning(
                "Brave search transport error (attempt %d/2): %s",
                attempt + 1,
                exc,
            )
            continue  # retry
        except Exception as exc:  # noqa: BLE001
            raise BraveFetchError(
                ticker=query, status_code=None, original=exc
            ) from exc

    raise BraveFetchError(
        ticker=query, status_code=last_status, original=last_error
    )


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


@tool("search_ticker_news", args_schema=NewsSearchArg)
async def search_ticker_news(ticker: str, num_results: int = 5) -> NewsBundle:
    """Search for recent news about a stock ticker via Brave Web Search.

    Returns a :class:`~src.schemas.ticker_data.NewsBundle` with up to
    ``num_results`` items. Sentiment is always ``"unknown"`` — the Analyst
    agent classifies it later. On any API error, returns an empty bundle
    (never raises).

    Args:
        ticker: Stock symbol (case-insensitive, uppercased internally).
        num_results: How many results to fetch (1–20, default 5).

    Returns:
        A :class:`~src.schemas.ticker_data.NewsBundle` instance.
    """
    symbol = ticker.upper()
    query = f"{symbol} stock news"

    try:
        data = await _brave_search(query, num_results)
    except BraveFetchError as exc:
        logger.warning("Brave search failed for %s: %s", symbol, exc)
        return NewsBundle(ticker=symbol, items=[], sentiment="unknown")

    raw_results = data.get("web", {}).get("results", [])
    return _parse_results(symbol, raw_results)


__all__ = ["search_ticker_news", "BraveFetchError", "NewsSearchArg"]
