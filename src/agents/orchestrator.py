"""Orchestrator agent — classifies intent via LLM and sets route + format_style.

Uses ChatGroq (llama-3.1-8b-instant) with structured output to classify the
user's message in a single LLM call.  A synchronous heuristic fallback
activates automatically if the LLM call fails, so the pipeline never crashes
on the orchestrator node.

Classification output
---------------------
- ``route``        — ``single`` | ``comparison`` | ``news_only`` | ``invalid``
- ``tickers``      — list of uppercase symbols extracted from the message (max 2)
- ``format_style`` — ``minimal`` | ``rich`` | ``news``
- ``start_time``   — ``time.monotonic()`` captured at node entry

Routing rules (taught to the LLM via system prompt)
----------------------------------------------------
- ``comparison`` + ``rich``   — message compares two tickers ("vs", "compare")
- ``news_only``  + ``news``   — message asks for news/headlines only
- ``single``     + ``rich``   — message has detail keywords (analysis, report…)
- ``single``     + ``minimal``— plain ticker query
- ``invalid``    + ``minimal``— no valid ticker found
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import settings
from src.schemas.agent_state import AgentState, FormatStyle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class OrchestratorOutput(BaseModel):
    """Structured classification returned by the LLM."""

    route: Literal["single", "comparison", "news_only", "invalid"] = Field(
        description=(
            "single      — one ticker, general query\n"
            "comparison  — two tickers being compared\n"
            "news_only   — user wants news/headlines only\n"
            "invalid     — no recognisable ticker found"
        )
    )
    tickers: list[str] = Field(
        description=(
            "Uppercase stock ticker symbols extracted from the message. "
            "Maximum 2. Empty list when route is 'invalid'."
        ),
        max_length=2,
    )
    format_style: Literal["minimal", "rich", "news"] = Field(
        description=(
            "minimal — plain ticker query, just summary\n"
            "rich    — detail/analysis/report keywords present, or comparison\n"
            "news    — news_only route"
        )
    )


# ---------------------------------------------------------------------------
# LLM client — built once, reused across calls
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an intent classifier for a stock analysis bot.

Given a user message, extract:
1. route — one of: single, comparison, news_only, invalid
2. tickers — uppercase stock symbols only (e.g. AAPL, MSFT, TSLA). Max 2. Empty if invalid.
3. format_style — one of: minimal, rich, news

Rules:
- comparison + rich   → message compares two tickers ("vs", "compare", "versus")
- news_only  + news   → message asks for news, headlines, or recent updates ONLY
- single     + rich   → message has words like: analysis, report, detail, full, deep, overview
- single     + minimal→ plain ticker query with no detail keywords
- invalid    + minimal→ no valid ticker symbol found

Ticker extraction rules:
- Only extract real stock ticker symbols (1–5 uppercase letters)
- Ignore common words even if they look like tickers (e.g. NEWS, GIVE, SHOW, CHECK, TELL)
- If unsure whether a word is a ticker, leave it out

Examples:
  "AAPL"                      → single,   ["AAPL"],         minimal
  "AAPL analysis"             → single,   ["AAPL"],         rich
  "AAPL vs MSFT"              → comparison, ["AAPL","MSFT"], rich
  "TSLA latest news"          → news_only, ["TSLA"],         news
  "give me NVDA report"       → single,   ["NVDA"],         rich
  "what is the price of AMZN" → single,   ["AMZN"],         minimal
  "hello world"               → invalid,  [],               minimal
"""

_llm = ChatGroq(
    model=settings.groq_fast_model,
    api_key=settings.groq_api_key,
    temperature=0,        # classification must be deterministic
    max_retries=1,
)

_structured_llm = _llm.with_structured_output(OrchestratorOutput)


# ---------------------------------------------------------------------------
# Heuristic fallback (used when LLM call fails)
# ---------------------------------------------------------------------------

_NEWS_KEYWORDS: frozenset[str] = frozenset(
    {"news", "latest", "headline", "headlines", "recent", "happening", "update"}
)
_DETAIL_KEYWORDS: frozenset[str] = frozenset(
    {"analysis", "analyse", "analyze", "report", "detail", "full", "deep", "overview"}
)
_STOPWORDS: frozenset[str] = frozenset(
    {
        "vs", "and", "or", "the", "for", "in", "on", "at", "a", "an",
        "me", "my", "is", "it", "its", "be", "to", "do", "of", "up",
        "give", "get", "show", "tell", "what", "how", "why", "who",
        "can", "you", "your", "we", "our", "i", "am", "are", "was",
        "has", "had", "not", "no", "so", "if", "by", "as", "from",
        "with", "this", "that", "they", "have", "about", "stock",
        "price", "share", "shares", "today", "now", "check",
    }
    | _NEWS_KEYWORDS
    | _DETAIL_KEYWORDS
)


def _heuristic_fallback(message: str) -> OrchestratorOutput:
    """Classify message using keyword heuristics — no LLM required.

    Called automatically when the LLM call fails.  Kept intentionally simple
    so it never raises.
    """
    lowered = message.lower()
    words = message.strip().split()

    tickers = [
        w.upper()
        for w in words
        if w.isalpha() and 1 <= len(w) <= 5 and w.lower() not in _STOPWORDS
    ]

    if not tickers:
        return OrchestratorOutput(route="invalid", tickers=[], format_style="minimal")

    is_comparison = "vs" in lowered or "compare" in lowered
    is_news_only = any(kw in lowered for kw in _NEWS_KEYWORDS) and not is_comparison
    has_detail = any(kw in lowered for kw in _DETAIL_KEYWORDS)

    if is_comparison:
        return OrchestratorOutput(route="comparison", tickers=tickers[:2], format_style="rich")
    if is_news_only:
        return OrchestratorOutput(route="news_only", tickers=tickers[:1], format_style="news")

    return OrchestratorOutput(
        route="single",
        tickers=tickers[:1],
        format_style="rich" if has_detail else "minimal",
    )


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------

async def run(state: AgentState) -> dict:
    """LangGraph node — classify intent via LLM with heuristic fallback.

    Args:
        state: Current pipeline state (reads ``user_message``).

    Returns:
        Partial state update with ``route``, ``tickers``, ``format_style``,
        and ``start_time``.
    """
    message: str = state.get("user_message", "")
    start_time = time.monotonic()

    try:
        result: OrchestratorOutput = await _structured_llm.ainvoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ]
        )
        logger.info(
            "Orchestrator LLM: route=%s tickers=%s style=%s",
            result.route, result.tickers, result.format_style,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Orchestrator LLM failed, using heuristic fallback: %s", exc)
        result = _heuristic_fallback(message)

    return {
        "route": result.route,
        "tickers": [t.upper() for t in result.tickers],
        "format_style": result.format_style,
        "start_time": start_time,
    }
