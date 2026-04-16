"""Formatter agent — converts pipeline results into a Telegram-ready message.

Three styles, set by ``state["format_style"]`` (decided by the Orchestrator):

  ``"minimal"``  — ticker header + analyst summary only.
                   Used for plain ticker queries (e.g. ``AAPL``).

  ``"rich"``     — header, key fundamentals snapshot, analyst summary,
                   and bullet key-points.
                   Used when detail keywords are present or for comparisons.

  ``"news"``     — ticker header + news headlines with sources/snippets.
                   Used for ``news_only`` route.

Telegram supports a limited markdown subset: ``*bold*``, ``_italic_``,
`` `code` ``, plain URLs.  No ``**`` or ``###`` headings — those are
stripped to ``*`` and plain text respectively.

All formatting helpers are pure functions with no I/O — easy to unit-test.
"""

from __future__ import annotations

import logging
from datetime import timezone

from src.schemas.agent_state import AgentState
from src.schemas.ticker_data import AnalystOutput, Fundamentals, NewsBundle, TickerData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 28


def _fmt_float(value: float | None, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def _fmt_large(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,.0f}"


def _completeness_bar(score: float) -> str:
    """ASCII progress bar for data completeness, e.g. '████░░ 67%'."""
    filled = round(score * 6)
    return "█" * filled + "░" * (6 - filled) + f"  {score:.0%}"


# ---------------------------------------------------------------------------
# Style: minimal
# ---------------------------------------------------------------------------

def _format_minimal(ticker: str, output: AnalystOutput) -> str:
    """Ticker header + one-paragraph summary."""
    return f"*{ticker}*\n\n{output.summary}"


# ---------------------------------------------------------------------------
# Style: rich
# ---------------------------------------------------------------------------

def _fundamentals_snapshot(f: Fundamentals) -> str:
    lines = [
        f"🏢 {f.company_name or ticker}" ,
        f"💹 P/E: `{_fmt_float(f.pe_ratio)}`  EPS: `{_fmt_float(f.eps)}`",
        f"📊 Market cap: `{_fmt_large(f.market_cap)}`  ({f.currency or '?'})",
        f"📅 52w `{_fmt_float(f.week_52_low)}` – `{_fmt_float(f.week_52_high)}`",
    ]
    if f.dividend_yield:
        lines.append(f"💰 Div. yield: `{_fmt_float(f.dividend_yield * 100, 2)}%`")
    return "\n".join(lines)


def _format_rich(ticker: str, data: TickerData, output: AnalystOutput) -> str:
    """Header + fundamentals snapshot + summary + key points."""
    parts: list[str] = [f"*{ticker}*  {_completeness_bar(data.data_completeness_score)}"]

    if data.fundamentals:
        parts.append(_fundamentals_snapshot(data.fundamentals))

    if data.moving_averages:
        ma = data.moving_averages
        parts.append(
            f"📈 SMA  50d `{_fmt_float(ma.sma_50)}`  "
            f"100d `{_fmt_float(ma.sma_100)}`  "
            f"200d `{_fmt_float(ma.sma_200)}`"
        )

    parts.append(_SEPARATOR)
    parts.append(output.summary)

    if output.key_points:
        points = "\n".join(f"• {kp}" for kp in output.key_points)
        parts.append(points)

    if data.warnings:
        parts.append(f"⚠️ _{'; '.join(data.warnings)}_")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Style: news
# ---------------------------------------------------------------------------

def _format_news(ticker: str, news: NewsBundle) -> str:
    """Ticker header + news headlines."""
    if not news.items:
        return f"*{ticker}* — no recent news found."

    lines: list[str] = [f"*{ticker} — Latest News*", ""]
    for i, item in enumerate(news.items[:5], 1):
        source = f" _{item.source}_" if item.source else ""
        lines.append(f"{i}. {item.title}{source}")
        if item.snippet:
            lines.append(f"   _{item.snippet[:100].rstrip()}_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------

def run(state: AgentState) -> dict:
    """LangGraph node — format pipeline results for Telegram delivery.

    Reads ``format_style``, ``analyst_outputs``, and ``data_by_ticker``
    from state. For comparisons, renders each ticker block and joins them
    with a divider.

    Args:
        state: Current pipeline state.

    Returns:
        ``{"final_message": "<telegram-ready string>"}``
    """
    format_style: str = state.get("format_style", "minimal")
    analyst_outputs: dict[str, AnalystOutput] = state.get("analyst_outputs", {})
    data_by_ticker: dict[str, TickerData] = state.get("data_by_ticker", {})
    tickers: list[str] = state.get("tickers", [])

    if not tickers:
        return {"final_message": "No tickers found in request."}

    blocks: list[str] = []

    for ticker in tickers:
        output = analyst_outputs.get(ticker)
        data = data_by_ticker.get(ticker) or TickerData(ticker=ticker)

        if format_style == "news":
            news = data.news
            if news:
                blocks.append(_format_news(ticker, news))
            else:
                blocks.append(f"*{ticker}* — no news data available.")

        elif format_style == "rich":
            if output:
                blocks.append(_format_rich(ticker, data, output))
            else:
                blocks.append(f"*{ticker}* — analysis unavailable.")

        else:  # "minimal"
            if output:
                blocks.append(_format_minimal(ticker, output))
            else:
                blocks.append(f"*{ticker}* — analysis unavailable.")

    divider = f"\n\n{_SEPARATOR}\n\n"
    final_message = divider.join(blocks) if len(blocks) > 1 else blocks[0]

    logger.info(
        "Formatter: style=%s  tickers=%s  message_len=%d",
        format_style,
        tickers,
        len(final_message),
    )

    return {"final_message": final_message}
