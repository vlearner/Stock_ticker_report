"""NewsFetcher agent — queries Brave Search and merges news into TickerData.

Brave is only called when the data is actually needed — avoiding unnecessary
API usage:

  - ``news_only`` route  → always fetch (that's the whole point)
  - ``single`` / ``comparison`` → fetch only when the user's message
    contains a news keyword (e.g. "news", "latest", "headline")
  - All other cases → skip; ``TickerData.news`` remains ``None``

Fetches all tickers concurrently.  The tool itself never raises — failures
return an empty :class:`~src.schemas.ticker_data.NewsBundle` (Pattern 6).

Sentiment classification is intentionally left as ``"unknown"`` here; the
Analyst agent incorporates news content into the summary narrative instead
of assigning a discrete label.
"""

from __future__ import annotations

import asyncio
import logging

from src.schemas.agent_state import AgentState
from src.schemas.ticker_data import TickerData
from src.tools.brave_tools import fetch_ticker_news

logger = logging.getLogger(__name__)

# Keywords that indicate the user wants news included in a non-news_only query
_NEWS_KEYWORDS: frozenset[str] = frozenset(
    {"news", "latest", "headline", "headlines", "recent", "happening", "update"}
)

# Number of Brave results to request per ticker
_NUM_RESULTS: int = 5


def _should_fetch_news(route: str, message: str) -> bool:
    """Return True if the NewsFetcher should call the Brave API."""
    if route == "news_only":
        return True
    lowered = message.lower()
    return any(kw in lowered for kw in _NEWS_KEYWORDS)


async def run(state: AgentState) -> dict:
    """LangGraph node — fetch news for tickers when relevant.

    Reads ``route`` and ``user_message`` to decide whether to call Brave,
    then merges the resulting :class:`~src.schemas.ticker_data.NewsBundle`
    into each ticker's existing :class:`~src.schemas.ticker_data.TickerData`.

    Args:
        state: Current pipeline state.

    Returns:
        ``{"data_by_ticker": ...}`` with news merged in, or ``{}`` if Brave
        was skipped entirely.
    """
    route: str = state.get("route", "single")
    message: str = state.get("user_message", "")
    tickers: list[str] = state.get("tickers", [])
    data_by_ticker: dict[str, TickerData] = dict(state.get("data_by_ticker", {}))

    if not _should_fetch_news(route, message):
        logger.info("NewsFetcher: skipping Brave — route=%s, no news keywords", route)
        return {}

    if not tickers:
        logger.warning("NewsFetcher: no tickers in state — skipping")
        return {}

    logger.info("NewsFetcher: fetching news for %s", tickers)
    bundles = await asyncio.gather(
        *[fetch_ticker_news(t, _NUM_RESULTS) for t in tickers]
    )

    for ticker, bundle in zip(tickers, bundles):
        existing = data_by_ticker.get(ticker) or TickerData(ticker=ticker)
        data_by_ticker[ticker] = existing.model_copy(update={"news": bundle})
        logger.info(
            "NewsFetcher: %s — %d articles fetched", ticker, len(bundle.items)
        )

    return {"data_by_ticker": data_by_ticker}
