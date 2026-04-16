"""Unit tests for src.schemas.ticker_data.

Covers:
- Ticker uppercasing validator on every model that carries a ticker
- Optional-field graceful degradation (None values accepted)
- Field constraints (ge/le bounds, max_length)
- extra='forbid' rejects unknown keys
- validate_assignment catches bad updates
- Sentiment literal enforcement on NewsBundle
- TickerData completeness score bounds
- Factory defaults produce timezone-aware datetimes
"""

from __future__ import annotations

from datetime import timezone

import pytest
from pydantic import ValidationError

from src.schemas.ticker_data import (
    AnalystOutput,
    CritiqueResult,
    Fundamentals,
    GoalMetrics,
    MovingAverages,
    NewsBundle,
    NewsItem,
    TickerData,
    TickerValidationResult,
    VolumeData,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fundamentals(**kwargs) -> Fundamentals:
    return Fundamentals(ticker="AAPL", **kwargs)


def _make_moving_averages(**kwargs) -> MovingAverages:
    return MovingAverages(ticker="AAPL", **kwargs)


def _make_volume_data(**kwargs) -> VolumeData:
    return VolumeData(ticker="AAPL", **kwargs)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------


class TestFundamentals:
    def test_minimal_construction(self):
        f = _make_fundamentals()
        assert f.ticker == "AAPL"
        assert f.company_name is None
        assert f.pe_ratio is None
        assert f.eps is None
        assert f.market_cap is None
        assert f.week_52_high is None
        assert f.week_52_low is None
        assert f.dividend_yield is None
        assert f.currency is None

    def test_full_construction(self):
        f = Fundamentals(
            ticker="msft",
            company_name="Microsoft Corporation",
            pe_ratio=35.2,
            eps=11.45,
            market_cap=2_800_000_000_000.0,
            week_52_high=430.0,
            week_52_low=300.0,
            dividend_yield=0.008,
            currency="USD",
        )
        assert f.ticker == "MSFT"  # uppercased
        assert f.company_name == "Microsoft Corporation"
        assert f.pe_ratio == pytest.approx(35.2)
        assert f.currency == "USD"

    def test_ticker_uppercased(self):
        f = Fundamentals(ticker="tsla")
        assert f.ticker == "TSLA"

    def test_ticker_whitespace_stripped(self):
        f = Fundamentals(ticker="  AAPL  ")
        assert f.ticker == "AAPL"

    def test_as_of_is_utc(self):
        f = _make_fundamentals()
        assert f.as_of.tzinfo is not None
        assert f.as_of.tzinfo == timezone.utc

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            Fundamentals(ticker="AAPL", unknown_field=1)

    def test_validate_assignment(self):
        f = _make_fundamentals()
        f.ticker = "nvda"
        assert f.ticker == "NVDA"

    def test_all_optional_fields_none(self):
        f = Fundamentals(ticker="X", pe_ratio=None, eps=None, market_cap=None)
        assert f.pe_ratio is None


# ---------------------------------------------------------------------------
# MovingAverages
# ---------------------------------------------------------------------------


class TestMovingAverages:
    def test_minimal(self):
        ma = _make_moving_averages()
        assert ma.ticker == "AAPL"
        assert ma.sma_50 is None
        assert ma.sma_100 is None
        assert ma.sma_200 is None

    def test_full(self):
        ma = MovingAverages(ticker="goog", sma_50=170.0, sma_100=165.0, sma_200=155.0)
        assert ma.ticker == "GOOG"
        assert ma.sma_50 == pytest.approx(170.0)

    def test_ticker_uppercased(self):
        ma = MovingAverages(ticker="amzn")
        assert ma.ticker == "AMZN"

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            MovingAverages(ticker="AAPL", sma_7=999.0)

    def test_as_of_timezone_aware(self):
        ma = _make_moving_averages()
        assert ma.as_of.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# VolumeData
# ---------------------------------------------------------------------------


class TestVolumeData:
    def test_minimal(self):
        v = _make_volume_data()
        assert v.ticker == "AAPL"
        assert v.current_volume is None
        assert v.avg_volume is None

    def test_full(self):
        v = VolumeData(ticker="spy", current_volume=50_000_000, avg_volume=80_000_000)
        assert v.ticker == "SPY"
        assert v.current_volume == 50_000_000

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            VolumeData(ticker="AAPL", something_else=1)


# ---------------------------------------------------------------------------
# NewsItem
# ---------------------------------------------------------------------------


class TestNewsItem:
    def test_minimal(self):
        ni = NewsItem(title="Apple Beats Earnings", url="https://example.com/1")
        assert ni.title == "Apple Beats Earnings"
        assert ni.url == "https://example.com/1"
        assert ni.source is None
        assert ni.published is None
        assert ni.snippet is None

    def test_full(self):
        from datetime import datetime
        ni = NewsItem(
            title="AAPL Up 5%",
            url="https://example.com/2",
            source="Reuters",
            published=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            snippet="Apple shares rose...",
        )
        assert ni.source == "Reuters"
        assert ni.snippet == "Apple shares rose..."

    def test_title_stripped(self):
        ni = NewsItem(title="  AAPL  ", url="https://example.com")
        assert ni.title == "AAPL"

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            NewsItem(title="x", url="y", rating=5)


# ---------------------------------------------------------------------------
# NewsBundle
# ---------------------------------------------------------------------------


class TestNewsBundle:
    def test_minimal(self):
        nb = NewsBundle(ticker="aapl")
        assert nb.ticker == "AAPL"
        assert nb.items == []
        assert nb.sentiment == "unknown"

    def test_with_items(self):
        items = [NewsItem(title=f"Title {i}", url=f"https://example.com/{i}") for i in range(3)]
        nb = NewsBundle(ticker="TSLA", items=items, sentiment="positive")
        assert len(nb.items) == 3
        assert nb.sentiment == "positive"

    def test_valid_sentiments(self):
        for s in ("positive", "neutral", "negative", "unknown"):
            nb = NewsBundle(ticker="AAPL", sentiment=s)
            assert nb.sentiment == s

    def test_invalid_sentiment(self):
        with pytest.raises(ValidationError):
            NewsBundle(ticker="AAPL", sentiment="bullish")

    def test_ticker_uppercased(self):
        nb = NewsBundle(ticker="nvda")
        assert nb.ticker == "NVDA"

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            NewsBundle(ticker="AAPL", foo="bar")


# ---------------------------------------------------------------------------
# TickerData
# ---------------------------------------------------------------------------


class TestTickerData:
    def test_minimal(self):
        td = TickerData(ticker="AAPL")
        assert td.ticker == "AAPL"
        assert td.fundamentals is None
        assert td.moving_averages is None
        assert td.volume is None
        assert td.news is None
        assert td.warnings == []
        assert td.data_completeness_score == pytest.approx(0.0)

    def test_with_all_slices(self):
        f = Fundamentals(ticker="AAPL")
        ma = MovingAverages(ticker="AAPL")
        v = VolumeData(ticker="AAPL")
        nb = NewsBundle(ticker="AAPL")
        td = TickerData(
            ticker="AAPL",
            fundamentals=f,
            moving_averages=ma,
            volume=v,
            news=nb,
            warnings=["yfinance timeout"],
            data_completeness_score=0.75,
        )
        assert td.fundamentals is f
        assert td.moving_averages is ma
        assert len(td.warnings) == 1
        assert td.data_completeness_score == pytest.approx(0.75)

    def test_ticker_uppercased(self):
        td = TickerData(ticker="meta")
        assert td.ticker == "META"

    def test_completeness_score_lower_bound(self):
        with pytest.raises(ValidationError):
            TickerData(ticker="AAPL", data_completeness_score=-0.1)

    def test_completeness_score_upper_bound(self):
        with pytest.raises(ValidationError):
            TickerData(ticker="AAPL", data_completeness_score=1.1)

    def test_completeness_score_boundary_values(self):
        td_zero = TickerData(ticker="AAPL", data_completeness_score=0.0)
        td_one = TickerData(ticker="AAPL", data_completeness_score=1.0)
        assert td_zero.data_completeness_score == pytest.approx(0.0)
        assert td_one.data_completeness_score == pytest.approx(1.0)

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            TickerData(ticker="AAPL", source="manual")

    def test_partial_slices_allowed(self):
        """Only fundamentals present; other slices remain None — graceful degradation."""
        f = Fundamentals(ticker="AAPL", pe_ratio=28.0)
        td = TickerData(ticker="AAPL", fundamentals=f)
        assert td.fundamentals.pe_ratio == pytest.approx(28.0)
        assert td.moving_averages is None
        assert td.news is None


# ---------------------------------------------------------------------------
# TickerValidationResult
# ---------------------------------------------------------------------------


class TestTickerValidationResult:
    def test_valid_ticker(self):
        r = TickerValidationResult(ticker="aapl", valid=True, company_name="Apple Inc.", reason=None)
        assert r.ticker == "AAPL"
        assert r.valid is True
        assert r.company_name == "Apple Inc."

    def test_invalid_ticker(self):
        r = TickerValidationResult(ticker="FAKE999", valid=False, reason="No data found")
        assert r.valid is False
        assert r.reason == "No data found"
        assert r.company_name is None

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            TickerValidationResult(ticker="AAPL", valid=True, score=99)


# ---------------------------------------------------------------------------
# AnalystOutput
# ---------------------------------------------------------------------------


class TestAnalystOutput:
    def test_minimal(self):
        ao = AnalystOutput(summary="Apple is performing well.")
        assert ao.summary == "Apple is performing well."
        assert ao.key_points == []
        assert ao.iteration == 1

    def test_with_key_points(self):
        ao = AnalystOutput(
            summary="Strong earnings beat.",
            key_points=["Revenue up 10%", "EPS beat by $0.15"],
            iteration=2,
        )
        assert len(ao.key_points) == 2
        assert ao.iteration == 2

    def test_summary_max_length(self):
        long_summary = "x" * 501
        with pytest.raises(ValidationError):
            AnalystOutput(summary=long_summary)

    def test_summary_at_max_length(self):
        ao = AnalystOutput(summary="x" * 500)
        assert len(ao.summary) == 500

    def test_iteration_ge_1(self):
        with pytest.raises(ValidationError):
            AnalystOutput(summary="ok", iteration=0)

    def test_iteration_positive(self):
        ao = AnalystOutput(summary="ok", iteration=3)
        assert ao.iteration == 3

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            AnalystOutput(summary="ok", score=5)


# ---------------------------------------------------------------------------
# CritiqueResult
# ---------------------------------------------------------------------------


class TestCritiqueResult:
    def test_passed(self):
        cr = CritiqueResult(passed=True, grounded=True, no_advice=True, concise=True)
        assert cr.passed is True
        assert cr.issues == []

    def test_failed_with_issues(self):
        cr = CritiqueResult(
            passed=False,
            grounded=True,
            no_advice=False,
            concise=False,
            issues=["Contains investment advice", "Too verbose (620 chars)"],
        )
        assert cr.passed is False
        assert len(cr.issues) == 2
        assert "Too verbose" in cr.issues[1]

    def test_all_bool_combinations(self):
        for grounded in (True, False):
            for no_advice in (True, False):
                for concise in (True, False):
                    passed = grounded and no_advice and concise
                    cr = CritiqueResult(
                        passed=passed,
                        grounded=grounded,
                        no_advice=no_advice,
                        concise=concise,
                    )
                    assert cr.passed == passed

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            CritiqueResult(passed=True, grounded=True, no_advice=True, concise=True, rating=10)


# ---------------------------------------------------------------------------
# GoalMetrics
# ---------------------------------------------------------------------------


class TestGoalMetrics:
    def test_minimal(self):
        gm = GoalMetrics(data_fresh=True, completeness_score=1.0, total_latency_ms=500)
        assert gm.data_fresh is True
        assert gm.completeness_score == pytest.approx(1.0)
        assert gm.total_latency_ms == 500
        assert gm.warnings == []

    def test_with_warnings(self):
        gm = GoalMetrics(
            data_fresh=False,
            completeness_score=0.5,
            total_latency_ms=9000,
            warnings=["Brave rate-limited", "yfinance timeout on TSLA"],
        )
        assert len(gm.warnings) == 2
        assert gm.data_fresh is False

    def test_completeness_score_lower_bound(self):
        with pytest.raises(ValidationError):
            GoalMetrics(data_fresh=True, completeness_score=-0.01, total_latency_ms=100)

    def test_completeness_score_upper_bound(self):
        with pytest.raises(ValidationError):
            GoalMetrics(data_fresh=True, completeness_score=1.01, total_latency_ms=100)

    def test_latency_non_negative(self):
        with pytest.raises(ValidationError):
            GoalMetrics(data_fresh=True, completeness_score=1.0, total_latency_ms=-1)

    def test_zero_latency_allowed(self):
        gm = GoalMetrics(data_fresh=True, completeness_score=0.0, total_latency_ms=0)
        assert gm.total_latency_ms == 0

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            GoalMetrics(data_fresh=True, completeness_score=1.0, total_latency_ms=0, unknown=1)
