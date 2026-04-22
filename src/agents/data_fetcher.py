"""DataFetcher agent — fetches yfinance data and populates TickerData.

Calls three yfinance tools concurrently per ticker:
  - fetch_fundamentals     → TickerData.fundamentals    (weight 0.50)
  - fetch_moving_averages  → TickerData.moving_averages (weight 0.30)
  - fetch_volume_data      → TickerData.volume          (weight 0.20)

For comparison queries (2 tickers) both tickers are also fetched
concurrently, giving up to 6 simultaneous yfinance calls bounded by the
configured timeout and retry settings.

Any individual call failure is caught, recorded as a warning on
``TickerData.warnings``, and excluded from the completeness score — the
pipeline continues with whatever data arrived (Pattern 6).
"""

from __future__ import annotations

import asyncio
import logging

from src.schemas.agent_state import AgentState
from src.schemas.ticker_data import Fundamentals, MAChartData, MovingAverages, TickerData, VolumeData
from src.tools.yfinance_tools import (
    fetch_fundamentals,
    fetch_ma_chart_data,
    fetch_moving_averages,
    fetch_volume_data,
)

logger = logging.getLogger(__name__)

# Completeness weights — must sum to 1.0
_W_FUNDAMENTALS: float = 0.50
_W_MOVING_AVERAGES: float = 0.30
_W_VOLUME: float = 0.20


async def _fetch_one(ticker: str) -> TickerData:
    """Fetch all three data slices for a single ticker concurrently.

    Returns a fully-populated :class:`TickerData` where any slice that
    failed is ``None`` and its weight is absent from
    ``data_completeness_score``.
    """
    fundamentals_result, ma_result, volume_result = await asyncio.gather(
        fetch_fundamentals(ticker),
        fetch_moving_averages(ticker),
        fetch_volume_data(ticker),
        return_exceptions=True,
    )

    warnings: list[str] = []
    score: float = 0.0

    fundamentals: Fundamentals | None = None
    if isinstance(fundamentals_result, Exception):
        warnings.append(f"fundamentals unavailable: {fundamentals_result}")
        logger.warning(
            "DataFetcher: fundamentals failed for %s — %s", ticker, fundamentals_result
        )
    else:
        fundamentals = fundamentals_result
        score += _W_FUNDAMENTALS

    moving_averages: MovingAverages | None = None
    if isinstance(ma_result, Exception):
        warnings.append(f"moving averages unavailable: {ma_result}")
        logger.warning(
            "DataFetcher: moving_averages failed for %s — %s", ticker, ma_result
        )
    else:
        moving_averages = ma_result
        score += _W_MOVING_AVERAGES

    volume: VolumeData | None = None
    if isinstance(volume_result, Exception):
        warnings.append(f"volume data unavailable: {volume_result}")
        logger.warning(
            "DataFetcher: volume failed for %s — %s", ticker, volume_result
        )
    else:
        volume = volume_result
        score += _W_VOLUME

    return TickerData(
        ticker=ticker,
        fundamentals=fundamentals,
        moving_averages=moving_averages,
        volume=volume,
        warnings=warnings,
        data_completeness_score=round(score, 2),
    )


async def run(state: AgentState) -> dict:
    """LangGraph node — fetch yfinance data for all tickers in state.

    Fetches all tickers concurrently. Returns a partial ``AgentState``
    update with ``data_by_ticker`` populated.

    Args:
        state: Current pipeline state (reads ``tickers``).

    Returns:
        ``{"data_by_ticker": {ticker: TickerData, ...}}``
    """
    tickers: list[str] = state.get("tickers", [])
    format_style: str = state.get("format_style", "minimal")
    run_id: str = state.get("run_id", "")
    logger.info("DataFetcher: run_id=%s tickers=%s", run_id, tickers)
    if not tickers:
        logger.warning("DataFetcher: no tickers in state — returning empty")
        return {"data_by_ticker": {}}

    # For rich single-ticker queries, fetch chart data concurrently
    fetch_chart = format_style == "rich" and len(tickers) == 1

    tasks: list = [_fetch_one(t) for t in tickers]
    if fetch_chart:
        tasks.append(fetch_ma_chart_data(tickers[0]))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    ticker_data_list = results[:len(tickers)]
    data_by_ticker: dict[str, TickerData] = {
        t: td for t, td in zip(tickers, ticker_data_list)
        if not isinstance(td, Exception)
    }

    for ticker, td in data_by_ticker.items():
        logger.info(
            "DataFetcher: %s  completeness=%.0f%%  warnings=%d",
            ticker,
            td.data_completeness_score * 100,
            len(td.warnings),
        )

    result: dict = {"data_by_ticker": data_by_ticker}

    if fetch_chart:
        chart_result = results[len(tickers)]
        if isinstance(chart_result, MAChartData):
            result["chart_data"] = chart_result
        else:
            logger.warning("DataFetcher: chart data fetch failed — %s", chart_result)

    return result
