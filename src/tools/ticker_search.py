"""Ticker symbol lookup by company name using Yahoo Finance Search.

Public API
----------
- :func:`search_ticker`       — synchronous lookup, returns symbol or ``None``
- :func:`search_ticker_async` — async wrapper (runs in thread pool)
- ``ticker_search_tool``      — LangChain ``@tool`` for chain use

Exchange priority
-----------------
Results are scored by whether they trade on a major exchange (NASDAQ / NYSE
variants).  The highest-scoring major-exchange match is returned first; if no
major-exchange result exists the first result is used as a fallback.

Graceful degradation
--------------------
All network errors are caught and logged; callers receive ``None`` rather than
an exception so the orchestrator pipeline is never blocked by a lookup failure.
"""

from __future__ import annotations

import asyncio
import logging

import yfinance as yf
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Exchanges considered "major" — prefer these over regional / OTC listings.
_MAJOR_EXCHANGES: frozenset[str] = frozenset(
    {"NMS", "NYQ", "NGM", "PCX", "XNYS", "XNAS", "BTS"}
)


def search_ticker(query: str, max_results: int = 5) -> str | None:
    """Return the best-match ticker symbol for a company name query.

    Args:
        query:       Company name or partial name (e.g. ``"Apple"``).
        max_results: Maximum search results to evaluate (default 5).

    Returns:
        Uppercase ticker symbol (e.g. ``"AAPL"``), or ``None`` if no match
        is found or the search call fails.

    Example::

        >>> search_ticker("Apple")
        'AAPL'
        >>> search_ticker("Nvidia")
        'NVDA'
    """
    if not query or not query.strip():
        return None

    try:
        results = yf.Search(query.strip(), max_results=max_results)
        quotes: list[dict] = results.quotes or []
    except Exception as exc:
        logger.warning("ticker_search: yf.Search failed for %r — %s", query, exc)
        return None

    if not quotes:
        return None

    # Prefer major-exchange listings for reliability
    for q in quotes:
        if q.get("exchange", "") in _MAJOR_EXCHANGES:
            symbol = q.get("symbol", "")
            if symbol:
                return symbol.upper()

    # Fallback to the first result regardless of exchange
    symbol = quotes[0].get("symbol", "")
    return symbol.upper() if symbol else None


async def search_ticker_async(query: str, max_results: int = 5) -> str | None:
    """Async wrapper around :func:`search_ticker` (runs in a thread pool).

    Args:
        query:       Company name or partial name.
        max_results: Passed through to :func:`search_ticker`.

    Returns:
        Ticker symbol or ``None``.
    """
    return await asyncio.to_thread(search_ticker, query, max_results)


@tool
def ticker_search_tool(company_name: str) -> str:
    """Look up a stock ticker symbol for a company name or partial name.

    Args:
        company_name: Company name to search for (e.g. ``"Apple"``).

    Returns:
        Uppercase ticker symbol (e.g. ``"AAPL"``), or ``"NOT_FOUND"`` when
        no match can be located.
    """
    result = search_ticker(company_name)
    return result if result is not None else "NOT_FOUND"
