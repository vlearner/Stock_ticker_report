"""Async yfinance tools exposed as LangChain ``BaseTool`` instances.

yfinance itself is a blocking library. To satisfy the project's strict
"no synchronous blocking calls" rule, every network-touching call is
dispatched to a worker thread via :func:`asyncio.to_thread` and bounded
by :func:`asyncio.wait_for` using
``yfinance_settings.yfinance_timeout_s``. Transient failures are retried
``yfinance_settings.yfinance_retries`` times; terminal failures raise a
typed :class:`YFinanceFetchError` so upstream agents can translate them
into structured warnings rather than crashing the pipeline
(Pattern 6 — Exception Handling and Recovery).

Public surface
--------------
- :func:`fetch_fundamentals`     — plain async function; called by API routes
  and any code that does not need the LangChain tool wrapper.
- :func:`fetch_moving_averages`  — plain async function.
- :func:`fetch_volume_data`      — plain async function.
- :func:`fetch_ticker_validation` — plain async function.
- :func:`get_ticker_fundamentals` — LangChain ``@tool`` wrapper.
- :func:`get_moving_averages`    — LangChain ``@tool`` wrapper.
- :func:`get_volume_data`        — LangChain ``@tool`` wrapper.
- :func:`validate_ticker`        — LangChain ``@tool`` wrapper.

Tools return Pydantic models (never raw dicts) — this keeps the inter-agent
contract enforced at the tool boundary (Pattern 4 — Tool Use).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TypeVar

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config.yfinance_settings import yfinance_settings
from src.schemas.ticker_data import (
    Fundamentals,
    MovingAverages,
    TickerValidationResult,
    VolumeData,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class YFinanceFetchError(RuntimeError):
    """Raised when a yfinance call fails after all retries.

    Upstream agents (DataFetcher) catch this and record it as a warning on
    :class:`~src.schemas.ticker_data.TickerData` rather than propagating.
    """

    def __init__(self, ticker: str, operation: str, original: Exception) -> None:
        super().__init__(
            f"yfinance {operation} failed for {ticker!r}: "
            f"{type(original).__name__}: {original}"
        )
        self.ticker = ticker
        self.operation = operation
        self.original = original


# ---------------------------------------------------------------------------
# Tool argument schema
# ---------------------------------------------------------------------------


class TickerArg(BaseModel):
    """Shared argument schema for all single-ticker yfinance tools."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=15,
        description=(
            "Stock ticker symbol (e.g. 'AAPL', 'MSFT'). Case-insensitive — "
            "will be uppercased internally. May include an exchange suffix "
            "such as 'RIO.L' for London or 'SAP.DE' for Xetra."
        ),
    )


# ---------------------------------------------------------------------------
# Low-level async helpers (thread-offloaded + retry + timeout)
# ---------------------------------------------------------------------------


async def _run_with_retry(
    fn: Callable[..., T],
    *args: Any,
    operation: str,
    ticker: str,
    **kwargs: Any,
) -> T:
    """Run a blocking callable in a worker thread with timeout + retry.

    Args:
        fn: Blocking function to invoke (e.g. a ``lambda`` wrapping a
            yfinance call).
        *args: Positional arguments forwarded to ``fn``.
        operation: Short label used in error messages and logs (e.g.
            ``"info"``, ``"history"``). Not forwarded to ``fn``.
        ticker: Ticker symbol — used purely for error context.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        YFinanceFetchError: If every attempt (initial + retries) fails.
    """

    attempts = yfinance_settings.yfinance_retries + 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(fn, *args, **kwargs),
                timeout=yfinance_settings.yfinance_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            last_error = exc
            logger.warning(
                "yfinance %s timeout for %s (attempt %d/%d)",
                operation,
                ticker,
                attempt,
                attempts,
            )
        except Exception as exc:  # noqa: BLE001 — yfinance raises a grab-bag
            last_error = exc
            logger.warning(
                "yfinance %s error for %s (attempt %d/%d): %s",
                operation,
                ticker,
                attempt,
                attempts,
                exc,
            )

    assert last_error is not None  # for type-checkers
    raise YFinanceFetchError(ticker=ticker, operation=operation, original=last_error)


def _fetch_info_sync(ticker: str) -> dict[str, Any]:
    """Blocking fetch of ``yf.Ticker(ticker).info``.

    Kept as a named function (rather than a lambda) so it is traceable in
    stack dumps and easier to monkeypatch in tests.
    """

    return yf.Ticker(ticker).info or {}


def _fetch_history_sync(ticker: str, period: str) -> pd.DataFrame:
    """Blocking fetch of daily-close history for ``ticker``."""

    return yf.Ticker(ticker).history(period=period, auto_adjust=False)


async def _fetch_info(ticker: str) -> dict[str, Any]:
    """Async wrapper around :func:`_fetch_info_sync` with retry + timeout."""

    return await _run_with_retry(
        _fetch_info_sync, ticker, operation="info", ticker=ticker
    )


async def _fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Async wrapper around :func:`_fetch_history_sync` with retry + timeout."""

    return await _run_with_retry(
        _fetch_history_sync,
        ticker,
        period,
        operation="history",
        ticker=ticker,
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion that returns ``None`` for junk values."""

    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check (NaN != NaN)
        return None
    return f


def _coerce_int(value: Any) -> int | None:
    """Best-effort int coercion that returns ``None`` for junk values."""

    f = _coerce_float(value)
    if f is None:
        return None
    return int(f)


def _first_present(info: dict[str, Any], *keys: str) -> Any:
    """Return the first non-None value across a sequence of dict keys."""

    for k in keys:
        v = info.get(k)
        if v is not None:
            return v
    return None


def _sma_or_none(closes: pd.Series, window: int) -> float | None:
    """Compute the latest SMA of ``closes`` over ``window`` days, or ``None``.

    Returns ``None`` when there are fewer than ``window`` observations.
    """

    if closes is None or len(closes) < window:
        return None
    value = closes.rolling(window=window).mean().iloc[-1]
    return _coerce_float(value)


# ---------------------------------------------------------------------------
# Core async functions — used by API routes and the LangChain tools
# ---------------------------------------------------------------------------


async def fetch_fundamentals(ticker: str) -> Fundamentals:
    """Fetch fundamentals for a single ticker.

    This is the primary callable for non-agent code (API routes, CLI tests,
    notebooks). The LangChain :func:`get_ticker_fundamentals` tool delegates
    here.

    Returns a :class:`~src.schemas.ticker_data.Fundamentals` instance. Any
    field yfinance cannot populate is returned as ``None``.

    Args:
        ticker: Stock symbol (case-insensitive, uppercased internally).

    Raises:
        YFinanceFetchError: If the underlying yfinance call fails after all
            retries.
    """

    symbol = ticker.upper()
    info = await _fetch_info(symbol)

    return Fundamentals(
        ticker=symbol,
        company_name=_first_present(info, "longName", "shortName"),
        pe_ratio=_coerce_float(_first_present(info, "trailingPE", "forwardPE")),
        eps=_coerce_float(_first_present(info, "trailingEps", "forwardEps")),
        market_cap=_coerce_float(info.get("marketCap")),
        week_52_high=_coerce_float(info.get("fiftyTwoWeekHigh")),
        week_52_low=_coerce_float(info.get("fiftyTwoWeekLow")),
        dividend_yield=_coerce_float(info.get("dividendYield")),
        currency=info.get("currency"),
        as_of=datetime.now(timezone.utc),
    )


async def fetch_moving_averages(ticker: str) -> MovingAverages:
    """Compute 50-, 100-, and 200-day simple moving averages.

    This is the primary callable for non-agent code. The LangChain
    :func:`get_moving_averages` tool delegates here.

    Args:
        ticker: Stock symbol (case-insensitive, uppercased internally).

    Raises:
        YFinanceFetchError: If the history fetch fails after all retries.
    """

    symbol = ticker.upper()
    history = await _fetch_history(symbol, period="1y")

    if history is None or history.empty or "Close" not in history.columns:
        return MovingAverages(
            ticker=symbol,
            sma_50=None,
            sma_100=None,
            sma_200=None,
            as_of=datetime.now(timezone.utc),
        )

    closes = history["Close"].dropna()
    return MovingAverages(
        ticker=symbol,
        sma_50=_sma_or_none(closes, 50),
        sma_100=_sma_or_none(closes, 100),
        sma_200=_sma_or_none(closes, 200),
        as_of=datetime.now(timezone.utc),
    )


async def fetch_volume_data(ticker: str) -> VolumeData:
    """Fetch current and average trading volume.

    This is the primary callable for non-agent code. The LangChain
    :func:`get_volume_data` tool delegates here. If both the info and history
    calls fail, raises :class:`YFinanceFetchError`. Otherwise degrades
    gracefully with ``None`` fields.

    Args:
        ticker: Stock symbol (case-insensitive, uppercased internally).

    Raises:
        YFinanceFetchError: If both the info and history calls fail.
    """

    symbol = ticker.upper()

    # Run both fetches concurrently — the two calls are independent.
    info_task: Awaitable[dict[str, Any]] = _fetch_info(symbol)
    history_task: Awaitable[pd.DataFrame] = _fetch_history(symbol, period="5d")

    info_result, history_result = await asyncio.gather(
        info_task, history_task, return_exceptions=True
    )

    # If BOTH failed we propagate — otherwise we degrade gracefully.
    if isinstance(info_result, Exception) and isinstance(history_result, Exception):
        raise YFinanceFetchError(
            ticker=symbol,
            operation="volume",
            original=info_result,  # type: ignore[arg-type]
        )

    info: dict[str, Any] = (
        info_result if not isinstance(info_result, Exception) else {}
    )
    history: pd.DataFrame | None = (
        history_result if not isinstance(history_result, Exception) else None
    )

    avg_volume = _coerce_int(
        _first_present(info, "averageVolume", "averageDailyVolume10Day")
    )

    current_volume: int | None = None
    if history is not None and not history.empty and "Volume" in history.columns:
        latest = history["Volume"].dropna()
        if not latest.empty:
            current_volume = _coerce_int(latest.iloc[-1])

    # Fallback: if history didn't yield a current volume, use info's regularMarketVolume.
    if current_volume is None:
        current_volume = _coerce_int(
            _first_present(info, "regularMarketVolume", "volume")
        )

    return VolumeData(
        ticker=symbol,
        current_volume=current_volume,
        avg_volume=avg_volume,
        as_of=datetime.now(timezone.utc),
    )


async def fetch_ticker_validation(ticker: str) -> TickerValidationResult:
    """Check whether ``ticker`` resolves to a real, tradeable instrument.

    This is the primary callable for non-agent code. The LangChain
    :func:`validate_ticker` tool delegates here.

    Never raises — failures are reported as ``valid=False`` with a ``reason``
    string.

    Args:
        ticker: Stock symbol to validate.
    """

    symbol = ticker.upper()

    try:
        info = await _fetch_info(symbol)
    except YFinanceFetchError as exc:
        return TickerValidationResult(
            ticker=symbol,
            valid=False,
            company_name=None,
            reason=f"lookup failed: {exc.operation}",
        )

    if not info:
        return TickerValidationResult(
            ticker=symbol,
            valid=False,
            company_name=None,
            reason="empty info response",
        )

    company_name = _first_present(info, "longName", "shortName")
    price = _first_present(
        info,
        "regularMarketPrice",
        "currentPrice",
        "previousClose",
    )

    if not company_name or price is None:
        return TickerValidationResult(
            ticker=symbol,
            valid=False,
            company_name=company_name,
            reason="no market data available",
        )

    return TickerValidationResult(
        ticker=symbol,
        valid=True,
        company_name=company_name,
        reason=None,
    )


# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------


@tool("get_ticker_fundamentals", args_schema=TickerArg)
async def get_ticker_fundamentals(ticker: str) -> Fundamentals:
    """Fetch fundamentals for a single ticker.

    LangChain tool wrapper around :func:`fetch_fundamentals`. Use that
    function directly when you don't need the tool metadata.

    Returns PE ratio, EPS, market cap, 52-week high/low, dividend yield,
    company name, and currency. Any field yfinance cannot populate is
    returned as ``None`` — callers must tolerate partial results.

    Args:
        ticker: Stock symbol (case-insensitive, uppercased internally).

    Returns:
        A :class:`~src.schemas.ticker_data.Fundamentals` instance.

    Raises:
        YFinanceFetchError: If the underlying yfinance call fails after
            all retries.
    """

    return await fetch_fundamentals(ticker)


@tool("get_moving_averages", args_schema=TickerArg)
async def get_moving_averages(ticker: str) -> MovingAverages:
    """Compute 50-, 100-, and 200-day simple moving averages.

    LangChain tool wrapper around :func:`fetch_moving_averages`. Use that
    function directly when you don't need the tool metadata.

    Args:
        ticker: Stock symbol (case-insensitive, uppercased internally).

    Returns:
        A :class:`~src.schemas.ticker_data.MovingAverages` instance.

    Raises:
        YFinanceFetchError: If the history fetch fails after all retries.
    """

    return await fetch_moving_averages(ticker)


@tool("get_volume_data", args_schema=TickerArg)
async def get_volume_data(ticker: str) -> VolumeData:
    """Fetch current and average trading volume.

    LangChain tool wrapper around :func:`fetch_volume_data`. Use that
    function directly when you don't need the tool metadata.

    Args:
        ticker: Stock symbol.

    Returns:
        A :class:`~src.schemas.ticker_data.VolumeData` instance.

    Raises:
        YFinanceFetchError: If both the info and history calls fail.
    """

    return await fetch_volume_data(ticker)


@tool("validate_ticker", args_schema=TickerArg)
async def validate_ticker(ticker: str) -> TickerValidationResult:
    """Check whether ``ticker`` resolves to a real, tradeable instrument.

    LangChain tool wrapper around :func:`fetch_ticker_validation`. Use that
    function directly when you don't need the tool metadata.

    Never raises — failures are reported as ``valid=False`` with a ``reason``
    string.

    Args:
        ticker: Stock symbol to validate.

    Returns:
        A :class:`~src.schemas.ticker_data.TickerValidationResult`.
    """

    return await fetch_ticker_validation(ticker)


__all__ = [
    "TickerArg",
    "YFinanceFetchError",
    "fetch_fundamentals",
    "fetch_moving_averages",
    "fetch_volume_data",
    "fetch_ticker_validation",
    "get_moving_averages",
    "get_ticker_fundamentals",
    "get_volume_data",
    "validate_ticker",
]
