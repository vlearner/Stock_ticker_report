"""Unit tests for src.tools.yfinance_tools.

Tests are grouped by concern:
  - YFinanceFetchError          — exception type, attributes, message
  - TickerArg                   — Pydantic arg validation
  - _coerce_float / _coerce_int — pure extraction helpers
  - _first_present              — dict key selection helper
  - _sma_or_none                — rolling-SMA helper
  - _run_with_retry             — timeout + retry orchestration (real threads,
                                  patched asyncio.wait_for for timeout cases)
  - get_ticker_fundamentals     — tool: full data, partial, fallback keys,
                                  fetch failure
  - get_moving_averages         — tool: full history, short history, empty /
                                  missing Close column, fetch failure
  - get_volume_data             — tool: both succeed, one-sided graceful
                                  degradation, both fail, history fallback
  - validate_ticker             — tool: valid, network error, empty info,
                                  missing name, missing price

All tool tests mock the async helpers _fetch_info / _fetch_history so no
real HTTP calls are made.  _run_with_retry tests use asyncio.to_thread with
real threads (fast lambdas) so they exercise the actual timeout/retry path.
"""

from __future__ import annotations

import asyncio
from datetime import timezone
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import settings
from src.config.yfinance_settings import yfinance_settings
from src.tools.yfinance_tools import (
    TickerArg,
    YFinanceFetchError,
    _coerce_float,
    _coerce_int,
    _first_present,
    _run_with_retry,
    _sma_or_none,
    get_moving_averages,
    get_ticker_fundamentals,
    get_volume_data,
    validate_ticker,
)

# Applied per-class below on async test classes only; sync classes must NOT
# carry the asyncio mark or pytest-asyncio (strict mode) will warn.

# Suppress a spurious Python 3.13 + AsyncMock interaction where contextlib's
# __exit__ introspects __doc__ on the mock and triggers "coroutine was never
# awaited" for asyncio.to_thread. All tests still exercise the real code paths.
pytestmark = pytest.mark.filterwarnings(
    "ignore:coroutine 'to_thread' was never awaited:RuntimeWarning"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_info() -> dict:
    """Realistic yfinance .info payload with all fields present."""
    return {
        "longName": "Apple Inc.",
        "shortName": "Apple",
        "trailingPE": 29.5,
        "forwardPE": 27.0,
        "trailingEps": 6.12,
        "forwardEps": 7.00,
        "marketCap": 3_000_000_000_000.0,
        "fiftyTwoWeekHigh": 230.0,
        "fiftyTwoWeekLow": 165.0,
        "dividendYield": 0.005,
        "currency": "USD",
        "regularMarketPrice": 210.0,
        "averageVolume": 80_000_000,
        "averageDailyVolume10Day": 75_000_000,
        "regularMarketVolume": 55_000_000,
    }


@pytest.fixture
def history_252() -> pd.DataFrame:
    """252-row daily-close DataFrame (enough for sma_200)."""
    closes = [float(150 + i * 0.1) for i in range(252)]
    return pd.DataFrame({"Close": closes, "Volume": [10_000_000] * 252})


@pytest.fixture
def history_60() -> pd.DataFrame:
    """60-row DataFrame — enough for sma_50 only."""
    closes = [float(150 + i * 0.1) for i in range(60)]
    return pd.DataFrame({"Close": closes, "Volume": [10_000_000] * 60})


# ---------------------------------------------------------------------------
# YFinanceFetchError
# ---------------------------------------------------------------------------


class TestYFinanceFetchError:
    def test_is_runtime_error(self):
        err = YFinanceFetchError("AAPL", "info", ValueError("oops"))
        assert isinstance(err, RuntimeError)

    def test_attributes_stored(self):
        original = ConnectionError("no network")
        err = YFinanceFetchError("tsla", "history", original)
        assert err.ticker == "tsla"
        assert err.operation == "history"
        assert err.original is original

    def test_message_includes_ticker_operation_and_type(self):
        original = TimeoutError("slow")
        err = YFinanceFetchError("MSFT", "volume", original)
        msg = str(err)
        assert "MSFT" in msg
        assert "volume" in msg
        assert "TimeoutError" in msg

    def test_message_includes_original_text(self):
        original = ValueError("rate limited")
        err = YFinanceFetchError("AAPL", "info", original)
        assert "rate limited" in str(err)


# ---------------------------------------------------------------------------
# TickerArg
# ---------------------------------------------------------------------------


class TestTickerArg:
    def test_valid_symbol(self):
        arg = TickerArg(ticker="AAPL")
        assert arg.ticker == "AAPL"

    def test_accepts_lowercase(self):
        # TickerArg does NOT uppercase — that's done inside the tool functions.
        arg = TickerArg(ticker="aapl")
        assert arg.ticker == "aapl"

    def test_accepts_exchange_suffix(self):
        arg = TickerArg(ticker="RIO.L")
        assert arg.ticker == "RIO.L"

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            TickerArg(ticker="")

    def test_too_long_rejected(self):
        with pytest.raises(ValidationError):
            TickerArg(ticker="A" * 16)

    def test_max_length_accepted(self):
        arg = TickerArg(ticker="A" * 15)
        assert len(arg.ticker) == 15


# ---------------------------------------------------------------------------
# _coerce_float
# ---------------------------------------------------------------------------


class TestCoerceFloat:
    def test_none_returns_none(self):
        assert _coerce_float(None) is None

    def test_nan_returns_none(self):
        assert _coerce_float(float("nan")) is None

    def test_valid_float(self):
        assert _coerce_float(3.14) == pytest.approx(3.14)

    def test_int_coerced_to_float(self):
        result = _coerce_float(42)
        assert result == pytest.approx(42.0)
        assert isinstance(result, float)

    def test_numeric_string(self):
        assert _coerce_float("29.5") == pytest.approx(29.5)

    def test_non_numeric_string_returns_none(self):
        assert _coerce_float("N/A") is None

    def test_zero_returns_zero(self):
        assert _coerce_float(0) == pytest.approx(0.0)

    def test_negative_value(self):
        assert _coerce_float(-5.5) == pytest.approx(-5.5)

    def test_list_returns_none(self):
        assert _coerce_float([1, 2, 3]) is None


# ---------------------------------------------------------------------------
# _coerce_int
# ---------------------------------------------------------------------------


class TestCoerceInt:
    def test_none_returns_none(self):
        assert _coerce_int(None) is None

    def test_nan_returns_none(self):
        assert _coerce_int(float("nan")) is None

    def test_int_value(self):
        assert _coerce_int(42) == 42

    def test_float_truncated(self):
        # int(3.9) == 3 — uses truncation via float → int
        assert _coerce_int(3.9) == 3

    def test_large_volume(self):
        assert _coerce_int(80_000_000) == 80_000_000

    def test_non_numeric_string_returns_none(self):
        assert _coerce_int("abc") is None

    def test_numeric_string(self):
        assert _coerce_int("100") == 100


# ---------------------------------------------------------------------------
# _first_present
# ---------------------------------------------------------------------------


class TestFirstPresent:
    def test_returns_first_key(self):
        assert _first_present({"a": 1, "b": 2}, "a", "b") == 1

    def test_skips_none_returns_second(self):
        assert _first_present({"a": None, "b": 2}, "a", "b") == 2

    def test_all_none_returns_none(self):
        assert _first_present({"a": None, "b": None}, "a", "b") is None

    def test_empty_dict_returns_none(self):
        assert _first_present({}, "a", "b") is None

    def test_missing_keys_returns_none(self):
        assert _first_present({"c": 3}, "a", "b") is None

    def test_zero_is_not_none(self):
        """Falsy but non-None values should be returned."""
        assert _first_present({"a": 0}, "a") == 0

    def test_false_is_not_none(self):
        assert _first_present({"a": False}, "a") is False

    def test_single_key_present(self):
        assert _first_present({"x": "hello"}, "x") == "hello"


# ---------------------------------------------------------------------------
# _sma_or_none
# ---------------------------------------------------------------------------


class TestSmaOrNone:
    def _series(self, n: int) -> pd.Series:
        return pd.Series([float(i) for i in range(1, n + 1)])

    def test_none_closes_returns_none(self):
        assert _sma_or_none(None, 50) is None  # type: ignore[arg-type]

    def test_fewer_than_window_returns_none(self):
        closes = self._series(49)
        assert _sma_or_none(closes, 50) is None

    def test_exactly_window_observations_returns_sma(self):
        closes = self._series(50)
        result = _sma_or_none(closes, 50)
        # SMA of 1..50 = 25.5
        assert result == pytest.approx(25.5)

    def test_more_than_window_observations(self):
        closes = self._series(100)
        result = _sma_or_none(closes, 50)
        # SMA of last 50 values (51..100) = 75.5
        assert result == pytest.approx(75.5)

    def test_sma_200_requires_200_rows(self):
        assert _sma_or_none(self._series(199), 200) is None
        assert _sma_or_none(self._series(200), 200) is not None

    def test_empty_series_returns_none(self):
        assert _sma_or_none(pd.Series([], dtype=float), 50) is None


# ---------------------------------------------------------------------------
# _run_with_retry
# ---------------------------------------------------------------------------


class TestRunWithRetry:
    pytestmark = pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        result = await _run_with_retry(lambda: "ok", operation="test", ticker="AAPL")
        assert result == "ok"

    async def test_returns_value_from_callable(self):
        result = await _run_with_retry(lambda: 42, operation="test", ticker="AAPL")
        assert result == 42

    async def test_success_after_one_transient_failure(self, monkeypatch):
        monkeypatch.setattr(yfinance_settings, "yfinance_retries", 1)
        attempts: list[int] = []

        def flaky() -> str:
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("transient")
            return "recovered"

        result = await _run_with_retry(flaky, operation="info", ticker="AAPL")
        assert result == "recovered"
        assert len(attempts) == 2

    async def test_all_attempts_fail_raises_yfinance_error(self, monkeypatch):
        monkeypatch.setattr(yfinance_settings, "yfinance_retries", 0)

        def always_fails():
            raise ValueError("boom")

        with pytest.raises(YFinanceFetchError) as exc_info:
            await _run_with_retry(always_fails, operation="info", ticker="TSLA")

        err = exc_info.value
        assert err.ticker == "TSLA"
        assert err.operation == "info"
        assert isinstance(err.original, ValueError)

    async def test_timeout_raises_yfinance_error(self, monkeypatch):
        monkeypatch.setattr(yfinance_settings, "yfinance_retries", 0)

        with patch(
            "src.tools.yfinance_tools.asyncio.wait_for",
            side_effect=asyncio.TimeoutError(),
        ):
            with pytest.raises(YFinanceFetchError) as exc_info:
                await _run_with_retry(lambda: None, operation="history", ticker="AAPL")

        err = exc_info.value
        assert err.operation == "history"
        assert err.ticker == "AAPL"
        assert isinstance(err.original, asyncio.TimeoutError)

    async def test_exhausted_retries_use_last_error(self, monkeypatch):
        monkeypatch.setattr(yfinance_settings, "yfinance_retries", 2)
        errors = [RuntimeError("first"), RuntimeError("second"), RuntimeError("third")]
        call_index = 0

        def always_fails():
            nonlocal call_index
            exc = errors[call_index]
            call_index += 1
            raise exc

        with pytest.raises(YFinanceFetchError) as exc_info:
            await _run_with_retry(always_fails, operation="test", ticker="X")

        # original should be the last exception raised
        assert exc_info.value.original is errors[2]
        assert call_index == 3  # 1 initial + 2 retries


# ---------------------------------------------------------------------------
# get_ticker_fundamentals
# ---------------------------------------------------------------------------


class TestGetTickerFundamentals:
    """Patches _fetch_info; calls the tool's underlying coroutine directly."""
    pytestmark = pytest.mark.asyncio

    async def test_full_info_all_fields_populated(self, full_info):
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=full_info),
        ):
            result = await get_ticker_fundamentals.coroutine(ticker="aapl")

        assert result.ticker == "AAPL"
        assert result.company_name == "Apple Inc."
        assert result.pe_ratio == pytest.approx(29.5)
        assert result.eps == pytest.approx(6.12)
        assert result.market_cap == pytest.approx(3e12)
        assert result.week_52_high == pytest.approx(230.0)
        assert result.week_52_low == pytest.approx(165.0)
        assert result.dividend_yield == pytest.approx(0.005)
        assert result.currency == "USD"

    async def test_ticker_uppercased(self, full_info):
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=full_info),
        ):
            result = await get_ticker_fundamentals.coroutine(ticker="msft")

        assert result.ticker == "MSFT"

    async def test_partial_info_graceful_degradation(self):
        partial = {"longName": "Fake Corp.", "currency": "USD"}
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=partial),
        ):
            result = await get_ticker_fundamentals.coroutine(ticker="FAKE")

        assert result.company_name == "Fake Corp."
        assert result.pe_ratio is None
        assert result.eps is None
        assert result.market_cap is None
        assert result.week_52_high is None
        assert result.dividend_yield is None

    async def test_falls_back_to_forward_pe(self):
        info = {"longName": "X Corp", "forwardPE": 22.0}
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=info),
        ):
            result = await get_ticker_fundamentals.coroutine(ticker="X")

        assert result.pe_ratio == pytest.approx(22.0)

    async def test_trailing_pe_preferred_over_forward(self):
        info = {"longName": "X Corp", "trailingPE": 30.0, "forwardPE": 22.0}
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=info),
        ):
            result = await get_ticker_fundamentals.coroutine(ticker="X")

        assert result.pe_ratio == pytest.approx(30.0)

    async def test_short_name_fallback(self):
        info = {"shortName": "Apple", "regularMarketPrice": 210.0}
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=info),
        ):
            result = await get_ticker_fundamentals.coroutine(ticker="AAPL")

        assert result.company_name == "Apple"

    async def test_empty_info_returns_all_none_fields(self):
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value={}),
        ):
            result = await get_ticker_fundamentals.coroutine(ticker="AAPL")

        assert result.pe_ratio is None
        assert result.company_name is None

    async def test_as_of_is_utc(self, full_info):
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=full_info),
        ):
            result = await get_ticker_fundamentals.coroutine(ticker="AAPL")

        assert result.as_of.tzinfo == timezone.utc

    async def test_fetch_error_propagates(self):
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(side_effect=YFinanceFetchError("AAPL", "info", IOError())),
        ):
            with pytest.raises(YFinanceFetchError):
                await get_ticker_fundamentals.coroutine(ticker="AAPL")


# ---------------------------------------------------------------------------
# get_moving_averages
# ---------------------------------------------------------------------------


class TestGetMovingAverages:
    pytestmark = pytest.mark.asyncio
    async def test_full_history_all_smas_computed(self, history_252):
        with patch(
            "src.tools.yfinance_tools._fetch_history",
            AsyncMock(return_value=history_252),
        ):
            result = await get_moving_averages.coroutine(ticker="aapl")

        assert result.ticker == "AAPL"
        assert result.sma_50 is not None
        assert result.sma_100 is not None
        assert result.sma_200 is not None

    async def test_sma_values_are_floats(self, history_252):
        with patch(
            "src.tools.yfinance_tools._fetch_history",
            AsyncMock(return_value=history_252),
        ):
            result = await get_moving_averages.coroutine(ticker="AAPL")

        assert isinstance(result.sma_50, float)
        assert isinstance(result.sma_200, float)

    async def test_short_history_only_sma_50(self, history_60):
        with patch(
            "src.tools.yfinance_tools._fetch_history",
            AsyncMock(return_value=history_60),
        ):
            result = await get_moving_averages.coroutine(ticker="AAPL")

        assert result.sma_50 is not None
        assert result.sma_100 is None
        assert result.sma_200 is None

    async def test_insufficient_history_all_none(self):
        df = pd.DataFrame({"Close": [float(i) for i in range(49)]})
        with patch(
            "src.tools.yfinance_tools._fetch_history",
            AsyncMock(return_value=df),
        ):
            result = await get_moving_averages.coroutine(ticker="AAPL")

        assert result.sma_50 is None
        assert result.sma_100 is None
        assert result.sma_200 is None

    async def test_empty_dataframe_returns_all_none(self):
        with patch(
            "src.tools.yfinance_tools._fetch_history",
            AsyncMock(return_value=pd.DataFrame()),
        ):
            result = await get_moving_averages.coroutine(ticker="AAPL")

        assert result.sma_50 is None
        assert result.sma_100 is None
        assert result.sma_200 is None

    async def test_missing_close_column_returns_all_none(self):
        df = pd.DataFrame({"Volume": [10_000_000] * 252})
        with patch(
            "src.tools.yfinance_tools._fetch_history",
            AsyncMock(return_value=df),
        ):
            result = await get_moving_averages.coroutine(ticker="AAPL")

        assert result.sma_50 is None

    async def test_ticker_uppercased(self, history_252):
        with patch(
            "src.tools.yfinance_tools._fetch_history",
            AsyncMock(return_value=history_252),
        ):
            result = await get_moving_averages.coroutine(ticker="tsla")

        assert result.ticker == "TSLA"

    async def test_as_of_utc(self, history_252):
        with patch(
            "src.tools.yfinance_tools._fetch_history",
            AsyncMock(return_value=history_252),
        ):
            result = await get_moving_averages.coroutine(ticker="AAPL")

        assert result.as_of.tzinfo == timezone.utc

    async def test_fetch_error_propagates(self):
        with patch(
            "src.tools.yfinance_tools._fetch_history",
            AsyncMock(side_effect=YFinanceFetchError("AAPL", "history", IOError())),
        ):
            with pytest.raises(YFinanceFetchError):
                await get_moving_averages.coroutine(ticker="AAPL")


# ---------------------------------------------------------------------------
# get_volume_data
# ---------------------------------------------------------------------------


class TestGetVolumeData:
    pytestmark = pytest.mark.asyncio
    async def test_both_succeed_returns_volumes(self, full_info, history_252):
        with (
            patch("src.tools.yfinance_tools._fetch_info", AsyncMock(return_value=full_info)),
            patch("src.tools.yfinance_tools._fetch_history", AsyncMock(return_value=history_252)),
        ):
            result = await get_volume_data.coroutine(ticker="aapl")

        assert result.ticker == "AAPL"
        assert result.avg_volume == 80_000_000
        assert result.current_volume == 10_000_000  # last row of fixture

    async def test_ticker_uppercased(self, full_info, history_252):
        with (
            patch("src.tools.yfinance_tools._fetch_info", AsyncMock(return_value=full_info)),
            patch("src.tools.yfinance_tools._fetch_history", AsyncMock(return_value=history_252)),
        ):
            result = await get_volume_data.coroutine(ticker="spy")

        assert result.ticker == "SPY"

    async def test_info_fails_history_succeeds_degrades_gracefully(self, history_252):
        """avg_volume becomes None; current_volume still extracted from history."""
        with (
            patch(
                "src.tools.yfinance_tools._fetch_info",
                AsyncMock(side_effect=YFinanceFetchError("AAPL", "info", IOError())),
            ),
            patch("src.tools.yfinance_tools._fetch_history", AsyncMock(return_value=history_252)),
        ):
            result = await get_volume_data.coroutine(ticker="AAPL")

        assert result.avg_volume is None
        assert result.current_volume == 10_000_000

    async def test_history_fails_info_succeeds_fallback_to_info_volume(self, full_info):
        """current_volume falls back to info's regularMarketVolume."""
        with (
            patch("src.tools.yfinance_tools._fetch_info", AsyncMock(return_value=full_info)),
            patch(
                "src.tools.yfinance_tools._fetch_history",
                AsyncMock(side_effect=YFinanceFetchError("AAPL", "history", IOError())),
            ),
        ):
            result = await get_volume_data.coroutine(ticker="AAPL")

        assert result.avg_volume == 80_000_000
        assert result.current_volume == 55_000_000  # regularMarketVolume from full_info

    async def test_both_fail_raises_yfinance_error(self):
        with (
            patch(
                "src.tools.yfinance_tools._fetch_info",
                AsyncMock(side_effect=YFinanceFetchError("AAPL", "info", IOError())),
            ),
            patch(
                "src.tools.yfinance_tools._fetch_history",
                AsyncMock(side_effect=YFinanceFetchError("AAPL", "history", IOError())),
            ),
        ):
            with pytest.raises(YFinanceFetchError) as exc_info:
                await get_volume_data.coroutine(ticker="AAPL")

        assert exc_info.value.ticker == "AAPL"
        assert exc_info.value.operation == "volume"

    async def test_history_no_volume_column_falls_back_to_info(self, full_info):
        df_no_vol = pd.DataFrame({"Close": [200.0, 201.0, 202.0]})
        with (
            patch("src.tools.yfinance_tools._fetch_info", AsyncMock(return_value=full_info)),
            patch("src.tools.yfinance_tools._fetch_history", AsyncMock(return_value=df_no_vol)),
        ):
            result = await get_volume_data.coroutine(ticker="AAPL")

        assert result.current_volume == 55_000_000  # regularMarketVolume from full_info

    async def test_avg_volume_fallback_to_10day(self):
        info_no_avg = {"averageDailyVolume10Day": 70_000_000, "regularMarketVolume": 50_000_000}
        with (
            patch("src.tools.yfinance_tools._fetch_info", AsyncMock(return_value=info_no_avg)),
            patch("src.tools.yfinance_tools._fetch_history", AsyncMock(return_value=pd.DataFrame())),
        ):
            result = await get_volume_data.coroutine(ticker="AAPL")

        assert result.avg_volume == 70_000_000

    async def test_as_of_utc(self, full_info, history_252):
        with (
            patch("src.tools.yfinance_tools._fetch_info", AsyncMock(return_value=full_info)),
            patch("src.tools.yfinance_tools._fetch_history", AsyncMock(return_value=history_252)),
        ):
            result = await get_volume_data.coroutine(ticker="AAPL")

        assert result.as_of.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# validate_ticker
# ---------------------------------------------------------------------------


class TestValidateTicker:
    pytestmark = pytest.mark.asyncio
    async def test_valid_ticker(self, full_info):
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=full_info),
        ):
            result = await validate_ticker.coroutine(ticker="aapl")

        assert result.ticker == "AAPL"
        assert result.valid is True
        assert result.company_name == "Apple Inc."
        assert result.reason is None

    async def test_ticker_uppercased(self, full_info):
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=full_info),
        ):
            result = await validate_ticker.coroutine(ticker="msft")

        assert result.ticker == "MSFT"

    async def test_fetch_error_returns_invalid(self):
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(side_effect=YFinanceFetchError("FAKE", "info", IOError())),
        ):
            result = await validate_ticker.coroutine(ticker="FAKE")

        assert result.valid is False
        assert "lookup failed" in result.reason
        assert result.company_name is None

    async def test_empty_info_returns_invalid(self):
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value={}),
        ):
            result = await validate_ticker.coroutine(ticker="FAKE")

        assert result.valid is False
        assert result.reason == "empty info response"

    async def test_no_company_name_returns_invalid(self):
        info = {"regularMarketPrice": 100.0}  # price present, name missing
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=info),
        ):
            result = await validate_ticker.coroutine(ticker="AAPL")

        assert result.valid is False
        assert result.reason == "no market data available"

    async def test_no_price_returns_invalid(self):
        info = {"longName": "Ghost Corp."}  # name present, price missing
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=info),
        ):
            result = await validate_ticker.coroutine(ticker="GHST")

        assert result.valid is False
        assert result.reason == "no market data available"

    async def test_short_name_accepted_for_validation(self):
        info = {"shortName": "Short Co.", "regularMarketPrice": 50.0}
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=info),
        ):
            result = await validate_ticker.coroutine(ticker="SC")

        assert result.valid is True
        assert result.company_name == "Short Co."

    async def test_price_fallback_to_current_price(self):
        info = {"longName": "Fallback Corp.", "currentPrice": 75.0}
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=info),
        ):
            result = await validate_ticker.coroutine(ticker="FB")

        assert result.valid is True

    async def test_price_fallback_to_previous_close(self):
        info = {"longName": "Stale Corp.", "previousClose": 60.0}
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(return_value=info),
        ):
            result = await validate_ticker.coroutine(ticker="STALE")

        assert result.valid is True

    async def test_never_raises(self):
        """validate_ticker must never propagate exceptions — always returns a result."""
        with patch(
            "src.tools.yfinance_tools._fetch_info",
            AsyncMock(side_effect=YFinanceFetchError("BAD", "info", RuntimeError("boom"))),
        ):
            result = await validate_ticker.coroutine(ticker="BAD")  # should not raise

        assert result.valid is False
