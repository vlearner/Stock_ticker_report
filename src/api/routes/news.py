"""News route — Brave Search news for a stock ticker.

Only requires ``BRAVE_API_KEY`` in ``.env``. No LLM or Telegram keys needed.

Endpoint
--------
``GET /api/v1/news?ticker=AAPL&count=5``

Response
--------
A :class:`~src.schemas.ticker_data.NewsBundle` JSON object containing
up to ``count`` news items. ``sentiment`` is always ``"unknown"`` at this
stage — the Analyst agent classifies it in the full pipeline.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from src.schemas.ticker_data import NewsBundle
from src.tools.brave_tools import fetch_ticker_news

router = APIRouter()


@router.get(
    "/news",
    response_model=NewsBundle,
    summary="Fetch recent news for a stock ticker",
    description=(
        "Searches Brave Web Search for recent news about the given ticker symbol. "
        "Returns up to `count` results. Requires only `BRAVE_API_KEY` in `.env`."
    ),
)
async def get_ticker_news(
    ticker: str = Query(
        ...,
        description="Stock ticker symbol (case-insensitive).",
        example="AAPL",
        min_length=1,
        max_length=15,
    ),
    count: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of news results to return (1–20).",
    ),
) -> NewsBundle:
    return await fetch_ticker_news(ticker, count)
