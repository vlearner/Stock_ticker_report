"""Analyst agent — generates structured summaries via ChatGroq (llama-3.3-70b).

For each ticker the agent builds a prompt from the available
:class:`~src.schemas.ticker_data.TickerData` and calls Groq with
``.with_structured_output(AnalystOutput)`` so the LLM response is
validated directly into the Pydantic schema — no string parsing.

For comparison queries (2 tickers) both LLM calls run concurrently
(``asyncio.gather``) to keep total latency close to a single-ticker run.

If an LLM call fails, a graceful fallback ``AnalystOutput`` is returned
and the error is appended to ``state["errors"]`` for LangSmith tracing
(Pattern 6).
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq

from src.cache import analyst_cache
from src.config import settings
from src.schemas.agent_state import AgentState
from src.schemas.ticker_data import AnalystOutput, Fundamentals, MovingAverages, NewsBundle, TickerData, VolumeData

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_ANALYST_INSTRUCTIONS: str = (_PROMPTS_DIR / "analyst_instructions.txt").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# LLM client — constructed once per process (thread-safe, connection-pooled)
# ---------------------------------------------------------------------------

_llm = ChatGroq(
    model=settings.groq_analysis_model,
    api_key=settings.groq_api_key,
    temperature=0.2,          # low temperature for factual financial reports
    max_retries=1,
)

_structured_llm = _llm.with_structured_output(AnalystOutput)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _fmt_float(value: float | None, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,}"


def _build_fundamentals_block(f: Fundamentals) -> str:
    return textwrap.dedent(f"""
        Company:         {f.company_name or 'N/A'}
        Currency:        {f.currency or 'N/A'}
        P/E ratio:       {_fmt_float(f.pe_ratio)}
        EPS:             {_fmt_float(f.eps)}
        Market cap:      {_fmt_int(int(f.market_cap)) if f.market_cap else 'N/A'}
        52-week high:    {_fmt_float(f.week_52_high)}
        52-week low:     {_fmt_float(f.week_52_low)}
        Dividend yield:  {_fmt_float(f.dividend_yield, 4) if f.dividend_yield else 'N/A'}
    """).strip()


def _build_moving_averages_block(ma: MovingAverages) -> str:
    return textwrap.dedent(f"""
        SMA 50-day:   {_fmt_float(ma.sma_50)}
        SMA 100-day:  {_fmt_float(ma.sma_100)}
        SMA 200-day:  {_fmt_float(ma.sma_200)}
    """).strip()


def _build_volume_block(v: VolumeData) -> str:
    return textwrap.dedent(f"""
        Current volume:  {_fmt_int(v.current_volume)}
        Average volume:  {_fmt_int(v.avg_volume)}
    """).strip()


def _build_news_block(news: NewsBundle) -> str:
    if not news.items:
        return "No recent news available."
    lines = []
    for item in news.items[:5]:
        source = f" ({item.source})" if item.source else ""
        lines.append(f"- {item.title}{source}")
        if item.snippet:
            lines.append(f"  {item.snippet[:120]}")
    return "\n".join(lines)


def _build_prompt(ticker: str, data: TickerData, iteration: int) -> str:
    """Build the analyst prompt from available TickerData slices."""
    sections: list[str] = [f"You are a financial data analyst. Write a factual report for {ticker}."]

    if data.fundamentals:
        sections.append(f"## Fundamentals\n{_build_fundamentals_block(data.fundamentals)}")
    else:
        sections.append("## Fundamentals\nNot available.")

    if data.moving_averages:
        sections.append(f"## Moving Averages\n{_build_moving_averages_block(data.moving_averages)}")

    if data.volume:
        sections.append(f"## Volume\n{_build_volume_block(data.volume)}")

    if data.news:
        sections.append(f"## Recent News\n{_build_news_block(data.news)}")

    if data.warnings:
        sections.append(f"## Data Warnings\n" + "\n".join(f"- {w}" for w in data.warnings))

    sections.append(_ANALYST_INSTRUCTIONS.format(iteration=iteration).strip())
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Per-ticker async analysis
# ---------------------------------------------------------------------------

async def _analyse_one(
    ticker: str,
    data: TickerData,
    iteration: int,
) -> tuple[str, AnalystOutput | Exception, bool]:
    """Run one structured LLM call for ``ticker`` (with answer caching).

    Cache behaviour
    ---------------
    * Only iteration 1 is served from cache — post-critic revisions (iter ≥ 2)
      always re-run so they can act on critic feedback.
    * Cache key is ``ticker:YYYY-MM-DD`` (UTC) — daily rollover.
    * Cache writes only happen after a successful LLM call.

    Returns:
        ``(ticker, result, cache_hit)`` where ``result`` is either a valid
        ``AnalystOutput`` or the exception that was raised, and ``cache_hit``
        reports whether the response came from cache.
    """
    cache_enabled = settings.analyst_cache_enabled and iteration == 1
    cache_key = analyst_cache.make_key(ticker) if cache_enabled else ""

    # --- Cache read ---
    if cache_enabled:
        try:
            cached = await analyst_cache.get(cache_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AnalystCache.get failed for %s: %s", ticker, exc)
            cached = None

        if cached is not None:
            # Preserve cached summary but enforce current iteration counter
            hit = cached.model_copy(update={"iteration": iteration})
            logger.info("Analyst cache HIT  ticker=%s key=%s", ticker, cache_key)
            return ticker, hit, True

    # --- LLM call (cache miss or caching disabled) ---
    prompt = _build_prompt(ticker, data, iteration)
    try:
        output: Any = await _structured_llm.ainvoke(prompt)
        output = output.model_copy(update={"iteration": iteration})
    except Exception as exc:  # noqa: BLE001
        logger.error("Analyst LLM call failed for %s: %s", ticker, exc)
        return ticker, exc, False

    # --- Cache write ---
    if cache_enabled:
        try:
            await analyst_cache.set(cache_key, output)
            logger.info("Analyst cache SET  ticker=%s key=%s", ticker, cache_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AnalystCache.set failed for %s: %s", ticker, exc)

    return ticker, output, False


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------

async def run(state: AgentState) -> dict:
    """LangGraph node — generate AnalystOutput for all tickers concurrently.

    Reads ``data_by_ticker`` and ``iteration_count`` from state. Runs one
    ChatGroq call per ticker in parallel. Failed calls produce a graceful
    fallback output and append to ``state["errors"]``.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with ``analyst_outputs``, ``iteration_count``,
        and (if any failures) appended ``errors``.
    """
    data_by_ticker: dict[str, TickerData] = state.get("data_by_ticker", {})
    iteration_count: dict[str, int] = state.get("iteration_count", {})
    existing_errors: list[str] = list(state.get("errors", []))
    run_id: str = state.get("run_id", "")

    logger.info("Analyst: run_id=%s tickers=%s", run_id, list(data_by_ticker.keys()))

    if not data_by_ticker:
        logger.warning("Analyst: no data_by_ticker in state — skipping LLM calls")
        return {}

    # Build per-ticker iteration counters (increment for this pass)
    new_iterations = {t: iteration_count.get(t, 0) + 1 for t in data_by_ticker}

    # Run all LLM calls concurrently
    results = await asyncio.gather(
        *[
            _analyse_one(ticker, data, new_iterations[ticker])
            for ticker, data in data_by_ticker.items()
        ]
    )

    analyst_outputs: dict[str, AnalystOutput] = {}
    new_errors: list[str] = []
    cache_hits: list[str] = []

    for ticker, result, cache_hit in results:
        if isinstance(result, Exception):
            new_errors.append(f"analyst:{ticker}:{result}")
            analyst_outputs[ticker] = AnalystOutput(
                summary=f"Analysis temporarily unavailable for {ticker}.",
                key_points=["Data fetch succeeded but LLM call failed."],
                iteration=new_iterations[ticker],
            )
        else:
            analyst_outputs[ticker] = result
            if cache_hit:
                cache_hits.append(ticker)
            logger.info(
                "Analyst: %s  iteration=%d  summary_len=%d  cached=%s",
                ticker,
                result.iteration,
                len(result.summary),
                cache_hit,
            )

    if cache_hits:
        logger.info("Analyst: %d/%d cache hits (%s)", len(cache_hits), len(results), cache_hits)

    return {
        "analyst_outputs": analyst_outputs,
        "iteration_count": new_iterations,
        "errors": existing_errors + new_errors,
    }
