"""Chat dispatcher endpoint — intent-routed demo for the web UI.

The LangGraph pipeline (orchestrator → fetchers → analyst → critic →
formatter) is the production path wired through the Telegram bot. For
the Vercel Hobby demo we bypass it: the 10s function timeout plus cold
start is too tight to safely run the full pipeline.

Instead, this endpoint parses a free-form message, extracts tickers,
classifies intent by keyword, and fans out to the existing tool
functions — same data, no Groq call on the critical path.

Public surface
--------------
- ``POST /api/v1/chat`` — one endpoint. Body is :class:`ChatRequest`,
  response is :class:`ChatResponse`.

Reuse (no duplication)
----------------------
- :func:`src.tools.yfinance_tools.fetch_fundamentals`
- :func:`src.tools.yfinance_tools.fetch_moving_averages`
- :func:`src.tools.yfinance_tools.fetch_volume_data`
- :func:`src.tools.brave_tools.fetch_ticker_news`

All fan-out uses ``asyncio.gather(..., return_exceptions=True)`` so a
single failing tool never blocks the reply (Pattern 6 — Exception
Handling and Recovery).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings
from src.schemas.ticker_data import (
    Fundamentals,
    MovingAverages,
    NewsBundle,
    VolumeData,
)
from src.tools.brave_tools import fetch_ticker_news
from src.tools.yfinance_tools import (
    fetch_fundamentals,
    fetch_moving_averages,
    fetch_volume_data,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Inbound chat payload from the web UI."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field(..., min_length=1, max_length=64)


class ChatResponse(BaseModel):
    """Outbound chat reply."""

    model_config = ConfigDict(extra="forbid")

    reply: str
    used_tools: list[str]
    session_id: str
    session_query_limit: int


# ---------------------------------------------------------------------------
# Intent parsing
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
_NEWS_KEYWORDS = {"news", "article", "articles", "headline", "headlines"}
_COMPARE_KEYWORDS = {"vs", "versus", "compare", "compared"}

# Plain English words that match the ticker regex but are almost never
# real tickers. Filtering these prevents the dispatcher from treating
# "HI" or "VS" as a stock query.
_STOPWORDS = {
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "DO", "FOR", "FROM",
    "HI", "I", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "NOT", "NOW", "OF",
    "ON", "OR", "SO", "THE", "TO", "UP", "US", "VS", "WE", "WHAT", "WHEN",
    "WHERE", "WHY", "YOU", "YOUR", "NEWS", "HEY", "HELLO", "HELP",
    "COMPARE", "VERSUS",
}


def _extract_tickers(message: str) -> list[str]:
    """Return uppercase tickers from the message, minus stopwords.

    Order-preserving and deduplicated; caps at 4 tickers so a runaway
    ALL-CAPS message can't explode the fan-out.
    """
    found: list[str] = []
    for match in _TICKER_RE.findall(message.upper()):
        if match in _STOPWORDS:
            continue
        if match in found:
            continue
        found.append(match)
        if len(found) >= 4:
            break
    return found


def _classify_intent(message: str, tickers: list[str]) -> str:
    """Return one of ``"news"``, ``"comparison"``, ``"fundamentals"``,
    or ``"help"``.
    """
    if not tickers:
        return "help"

    words = set(message.lower().split())
    if words & _NEWS_KEYWORDS:
        return "news"
    if len(tickers) >= 2 and words & _COMPARE_KEYWORDS:
        return "comparison"
    return "fundamentals"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_num(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int) or decimals == 0:
        return f"{value:,}"
    return f"{value:,.{decimals}f}"


def _fmt_large(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,.0f}"


def _format_fundamentals_block(
    ticker: str,
    fundamentals: Fundamentals | Exception | None,
    moving_averages: MovingAverages | Exception | None,
    volume: VolumeData | Exception | None,
) -> str:
    """Render a single ticker's fundamentals + MA + volume snapshot.

    Each sub-block degrades independently: a tool that raised is shown
    as ``_unavailable_`` rather than dropping the whole ticker.
    """
    lines: list[str] = [f"**{ticker}**"]

    if isinstance(fundamentals, Fundamentals):
        name = fundamentals.company_name or ticker
        lines.append(f"{name}")
        lines.append(
            f"- P/E: {_fmt_num(fundamentals.pe_ratio)}  "
            f"EPS: {_fmt_num(fundamentals.eps)}"
        )
        lines.append(
            f"- Market cap: {_fmt_large(fundamentals.market_cap)} "
            f"({fundamentals.currency or '?'})"
        )
        lines.append(
            f"- 52w: {_fmt_num(fundamentals.week_52_low)} – "
            f"{_fmt_num(fundamentals.week_52_high)}"
        )
    else:
        lines.append("- Fundamentals unavailable")

    if isinstance(moving_averages, MovingAverages):
        lines.append(
            f"- SMA 50d: {_fmt_num(moving_averages.sma_50)}  "
            f"100d: {_fmt_num(moving_averages.sma_100)}  "
            f"200d: {_fmt_num(moving_averages.sma_200)}"
        )
    else:
        lines.append("- Moving averages unavailable")

    if isinstance(volume, VolumeData):
        lines.append(
            f"- Volume: {_fmt_num(volume.current_volume, 0)} "
            f"(avg {_fmt_num(volume.avg_volume, 0)})"
        )
    else:
        lines.append("- Volume unavailable")

    return "\n".join(lines)


def _format_comparison_block(
    ticker: str, fundamentals: Fundamentals | Exception | None
) -> str:
    """Compact fundamentals-only block used for side-by-side comparisons."""
    if not isinstance(fundamentals, Fundamentals):
        return f"**{ticker}** — fundamentals unavailable"

    name = fundamentals.company_name or ticker
    return (
        f"**{ticker}** ({name})\n"
        f"- P/E: {_fmt_num(fundamentals.pe_ratio)}  "
        f"EPS: {_fmt_num(fundamentals.eps)}\n"
        f"- Market cap: {_fmt_large(fundamentals.market_cap)} "
        f"({fundamentals.currency or '?'})\n"
        f"- 52w: {_fmt_num(fundamentals.week_52_low)} – "
        f"{_fmt_num(fundamentals.week_52_high)}"
    )


def _format_news_block(ticker: str, news: NewsBundle | Exception | None) -> str:
    """Render a ticker's latest headlines."""
    if not isinstance(news, NewsBundle) or not news.items:
        return f"**{ticker}** — no recent news found."

    lines = [f"**{ticker} — Latest News**"]
    for i, item in enumerate(news.items[:5], 1):
        source = f" _{item.source}_" if item.source else ""
        lines.append(f"{i}. {item.title}{source}")
    return "\n".join(lines)


_HELP_TEXT = (
    "I can look up stock fundamentals and recent news. Try one of:\n"
    "- `AAPL` — snapshot for a ticker\n"
    "- `AAPL vs MSFT` — side-by-side comparison\n"
    "- `news TSLA` — latest headlines"
)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


async def _run_fundamentals(tickers: list[str]) -> tuple[str, list[str]]:
    """Parallel fan-out: fundamentals + MA + volume for every ticker.

    Returns the joined reply text and the list of ``used_tools`` tags.
    """
    tasks: list[asyncio.Future[Any]] = []
    for ticker in tickers:
        tasks.append(asyncio.ensure_future(fetch_fundamentals(ticker)))
        tasks.append(asyncio.ensure_future(fetch_moving_averages(ticker)))
        tasks.append(asyncio.ensure_future(fetch_volume_data(ticker)))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    blocks: list[str] = []
    for i, ticker in enumerate(tickers):
        f, ma, vol = results[i * 3 : i * 3 + 3]
        blocks.append(_format_fundamentals_block(ticker, f, ma, vol))

    return "\n\n".join(blocks), ["fundamentals", "moving_averages", "volume"]


async def _run_comparison(tickers: list[str]) -> tuple[str, list[str]]:
    """Parallel fundamentals-only fetch for a side-by-side comparison."""
    results = await asyncio.gather(
        *(fetch_fundamentals(t) for t in tickers),
        return_exceptions=True,
    )
    blocks = [_format_comparison_block(t, r) for t, r in zip(tickers, results)]
    return "\n\n".join(blocks), ["fundamentals"]


async def _run_news(tickers: list[str]) -> tuple[str, list[str]]:
    """Parallel news fetch for every ticker mentioned."""
    results = await asyncio.gather(
        *(fetch_ticker_news(t) for t in tickers),
        return_exceptions=True,
    )
    blocks = [_format_news_block(t, r) for t, r in zip(tickers, results)]
    return "\n\n".join(blocks), ["news"]


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Dispatch a free-form chat message to the right tool(s)",
    description=(
        "Parses intent, extracts tickers, and fans out to yfinance "
        "and/or Brave Search. Used by the web demo UI — the full "
        "LangGraph pipeline remains available via the Telegram path."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    tickers = _extract_tickers(request.message)
    intent = _classify_intent(request.message, tickers)

    logger.info(
        "Chat: sid=%s  intent=%s  tickers=%s",
        request.session_id,
        intent,
        tickers,
    )

    if intent == "help":
        reply = _HELP_TEXT
        used_tools: list[str] = []
    elif intent == "news":
        reply, used_tools = await _run_news(tickers)
    elif intent == "comparison":
        reply, used_tools = await _run_comparison(tickers)
    else:
        reply, used_tools = await _run_fundamentals(tickers)

    return ChatResponse(
        reply=reply,
        used_tools=used_tools,
        session_id=request.session_id,
        session_query_limit=settings.demo_session_query_limit,
    )
