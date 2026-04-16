"""Orchestrator agent — classifies intent and sets route + format_style.

This is a heuristic stub (step 3 / step 4 logic).  Real LLM-based
classification is added in Implementation Order step 6.

Routing rules
-------------
- No valid tickers found        → ``invalid``
- "vs" / "compare" in message   → ``comparison`` + ``rich``
- "news" / news keywords        → ``news_only``  + ``news``
- Detail keywords present       → ``single``     + ``rich``
- Otherwise                     → ``single``     + ``minimal``

Format style is set here so the Formatter stays a pure rendering node
with no routing logic of its own.
"""

from __future__ import annotations

import time

from src.schemas.agent_state import AgentState, FormatStyle

# Keywords that trigger the "news_only" route
_NEWS_KEYWORDS: frozenset[str] = frozenset(
    {"news", "latest", "headline", "headlines", "recent", "happening", "update"}
)

# Keywords that trigger "rich" format on a single-ticker query
_DETAIL_KEYWORDS: frozenset[str] = frozenset(
    {"analysis", "analyse", "analyze", "report", "detail", "full", "deep", "overview"}
)


def run(state: AgentState) -> dict:
    """Classify intent and return route + tickers + format_style.

    Args:
        state: Current pipeline state (reads ``user_message``).

    Returns:
        Partial state update with ``route``, ``tickers``,
        ``format_style``, and ``start_time``.
    """
    message: str = state.get("user_message", "")
    lowered = message.lower()
    words = message.strip().split()

    # Exclude common English words that are never tickers
    _STOPWORDS = (
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
    tickers = [
        w.upper()
        for w in words
        if w.isalpha() and 1 <= len(w) <= 5 and w.lower() not in _STOPWORDS
    ]

    start_time = time.monotonic()

    if not tickers:
        return {
            "route": "invalid",
            "tickers": [],
            "format_style": "minimal",
            "start_time": start_time,
        }

    # Determine route
    is_comparison = "vs" in lowered or "compare" in lowered
    is_news_only = any(kw in lowered for kw in _NEWS_KEYWORDS) and not is_comparison
    has_detail = any(kw in lowered for kw in _DETAIL_KEYWORDS)

    if is_comparison:
        route = "comparison"
        format_style: FormatStyle = "rich"
    elif is_news_only:
        route = "news_only"
        format_style = "news"
    else:
        route = "single"
        format_style = "rich" if has_detail else "minimal"

    return {
        "route": route,
        "tickers": tickers[:2],
        "format_style": format_style,
        "start_time": start_time,
    }
