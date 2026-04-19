"""Ticker routes — yfinance market data for a stock ticker.

No API key required. yfinance fetches public market data directly.

Endpoints
---------
``GET /api/v1/ticker/fundamentals?ticker=AAPL``
``GET /api/v1/ticker/moving-averages?ticker=AAPL``
``GET /api/v1/ticker/volume?ticker=AAPL``
``GET /api/v1/ticker/validate?ticker=AAPL``
``GET /api/v1/ticker/search?q=Apple``

Responses
---------
Each endpoint returns the corresponding Pydantic model as JSON.
``YFinanceFetchError`` is translated to a 502 Bad Gateway so callers get a
structured error rather than a raw 500.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.schemas.ticker_data import (
    Fundamentals,
    MovingAverages,
    TickerValidationResult,
    VolumeData,
)
from src.tools.ticker_search import search_ticker_async
from src.tools.yfinance_tools import (
    YFinanceFetchError,
    fetch_fundamentals,
    fetch_moving_averages,
    fetch_ticker_validation,
    fetch_volume_data,
)

router = APIRouter()

_TICKER_QUERY = Query(
    ...,
    description="Stock ticker symbol (case-insensitive).",
    examples=["AAPL"],
    min_length=1,
    max_length=15,
)


@router.get(
    "/ticker/fundamentals",
    response_model=Fundamentals,
    summary="Fetch fundamentals for a stock ticker",
    description=(
        "Returns PE ratio, EPS, market cap, 52-week high/low, dividend yield, "
        "company name, and currency. Fields that yfinance cannot populate are null."
    ),
)
async def get_fundamentals(
    ticker: str = _TICKER_QUERY,
) -> Fundamentals:
    try:
        return await fetch_fundamentals(ticker)
    except YFinanceFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get(
    "/ticker/moving-averages",
    response_model=MovingAverages,
    summary="Compute 50/100/200-day simple moving averages",
    description=(
        "Pulls ~1 year of daily closes and computes rolling SMAs. "
        "Windows shorter than available history are returned as null."
    ),
)
async def get_moving_averages_route(
    ticker: str = _TICKER_QUERY,
) -> MovingAverages:
    try:
        return await fetch_moving_averages(ticker)
    except YFinanceFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get(
    "/ticker/volume",
    response_model=VolumeData,
    summary="Fetch current and average trading volume",
    description=(
        "Returns the most recent daily volume and the 10/30-day average volume. "
        "Degrades gracefully when only one data source is available."
    ),
)
async def get_volume_route(
    ticker: str = _TICKER_QUERY,
) -> VolumeData:
    try:
        return await fetch_volume_data(ticker)
    except YFinanceFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get(
    "/ticker/validate",
    response_model=TickerValidationResult,
    summary="Validate a stock ticker symbol",
    description=(
        "Checks whether the ticker resolves to a real, tradeable instrument. "
        "Always returns a result — never raises. Check the ``valid`` field."
    ),
)
async def validate_ticker_route(
    ticker: str = _TICKER_QUERY,
) -> TickerValidationResult:
    return await fetch_ticker_validation(ticker)


@router.get(
    "/ticker/search",
    summary="Look up a ticker symbol by company name",
    description=(
        "Searches Yahoo Finance for a matching ticker symbol given a company "
        "name or partial name (e.g. ``Apple`` → ``AAPL``). Returns the best "
        "match on a major exchange (NASDAQ / NYSE) when available."
    ),
)
async def search_ticker_route(
    q: str = Query(
        ...,
        description="Company name or partial name to search for.",
        examples=["Apple", "Tesla", "Nvidia"],
        min_length=1,
        max_length=100,
    ),
) -> dict:
    symbol = await search_ticker_async(q)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"No ticker found for '{q}'")
    return {"query": q, "symbol": symbol}
