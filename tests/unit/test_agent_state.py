"""Unit tests for src.schemas.agent_state.

Covers:
- Route literal values are all present and exhaustive
- AgentState TypedDict can be instantiated with any subset of keys (total=False)
- Input, routing, result, memory, guardrail, and observability field types
- AgentState rejects no keys (empty dict is valid due to total=False)
- Integration: realistic state snapshots for each routing path
"""

from __future__ import annotations

import time
from typing import get_args

import pytest

from src.schemas.agent_state import AgentState, Route
from src.schemas.ticker_data import (
    AnalystOutput,
    CritiqueResult,
    Fundamentals,
    GoalMetrics,
    NewsBundle,
    TickerData,
)


# ---------------------------------------------------------------------------
# Route literal
# ---------------------------------------------------------------------------


class TestRoute:
    EXPECTED_ROUTES = {"single", "comparison", "news_only", "invalid", "rate_limited"}

    def test_all_expected_routes_exist(self):
        assert set(get_args(Route)) == self.EXPECTED_ROUTES

    def test_route_count(self):
        assert len(get_args(Route)) == 5

    @pytest.mark.parametrize("route", ["single", "comparison", "news_only", "invalid", "rate_limited"])
    def test_each_route_is_valid_literal(self, route):
        """Each route string must be a member of the Route Literal."""
        assert route in get_args(Route)


# ---------------------------------------------------------------------------
# AgentState TypedDict — construction
# ---------------------------------------------------------------------------


class TestAgentStateConstruction:
    def test_empty_state_is_valid(self):
        """total=False means an empty dict is a valid AgentState."""
        state: AgentState = {}
        assert state == {}

    def test_inputs_only(self):
        state: AgentState = {
            "user_id": "user_42",
            "chat_id": "chat_99",
            "user_message": "How is AAPL doing?",
        }
        assert state["user_id"] == "user_42"
        assert state["chat_id"] == "chat_99"
        assert state["user_message"] == "How is AAPL doing?"

    def test_routing_fields(self):
        state: AgentState = {
            "route": "single",
            "tickers": ["AAPL"],
        }
        assert state["route"] == "single"
        assert state["tickers"] == ["AAPL"]

    def test_comparison_route(self):
        state: AgentState = {
            "route": "comparison",
            "tickers": ["AAPL", "MSFT"],
        }
        assert len(state["tickers"]) == 2

    def test_data_by_ticker(self):
        td = TickerData(ticker="AAPL", data_completeness_score=0.8)
        state: AgentState = {"data_by_ticker": {"AAPL": td}}
        assert state["data_by_ticker"]["AAPL"].ticker == "AAPL"
        assert state["data_by_ticker"]["AAPL"].data_completeness_score == pytest.approx(0.8)

    def test_analyst_outputs(self):
        ao = AnalystOutput(summary="Apple earnings strong.", key_points=["EPS up"])
        state: AgentState = {"analyst_outputs": {"AAPL": ao}}
        assert state["analyst_outputs"]["AAPL"].summary == "Apple earnings strong."

    def test_critique_by_ticker(self):
        cr = CritiqueResult(passed=True, grounded=True, no_advice=True, concise=True)
        state: AgentState = {"critique_by_ticker": {"AAPL": cr}}
        assert state["critique_by_ticker"]["AAPL"].passed is True

    def test_iteration_count(self):
        state: AgentState = {"iteration_count": {"AAPL": 2, "MSFT": 1}}
        assert state["iteration_count"]["AAPL"] == 2

    def test_memory_fields(self):
        state: AgentState = {
            "recent_tickers": ["AAPL", "TSLA", "MSFT"],
            "preferred_format": "short",
        }
        assert len(state["recent_tickers"]) == 3
        assert state["preferred_format"] == "short"

    def test_preferred_format_none(self):
        state: AgentState = {"preferred_format": None}
        assert state["preferred_format"] is None

    def test_guardrail_fields(self):
        state: AgentState = {
            "rate_limit_remaining": 7,
            "blocked_reason": None,
        }
        assert state["rate_limit_remaining"] == 7
        assert state["blocked_reason"] is None

    def test_blocked_reason_set(self):
        state: AgentState = {
            "rate_limit_remaining": 0,
            "blocked_reason": "Rate limit exceeded: 10 queries/hour.",
        }
        assert state["rate_limit_remaining"] == 0
        assert "Rate limit" in state["blocked_reason"]

    def test_observability_fields(self):
        gm = GoalMetrics(data_fresh=True, completeness_score=1.0, total_latency_ms=800)
        state: AgentState = {
            "start_time": time.monotonic(),
            "metrics": gm,
            "final_message": "*AAPL* is up 2% today.",
            "errors": [],
        }
        assert state["final_message"] == "*AAPL* is up 2% today."
        assert state["errors"] == []
        assert state["metrics"].total_latency_ms == 800

    def test_errors_list(self):
        state: AgentState = {"errors": ["yfinance timeout on TSLA", "Brave rate-limited"]}
        assert len(state["errors"]) == 2


# ---------------------------------------------------------------------------
# AgentState — realistic pipeline snapshots
# ---------------------------------------------------------------------------


class TestAgentStateSnapshots:
    """End-to-end state snapshots mimicking what each pipeline stage would produce."""

    def _base_state(self) -> AgentState:
        return {
            "user_id": "tg_user_1",
            "chat_id": "tg_chat_100",
            "user_message": "Tell me about AAPL",
        }

    def test_post_orchestrator_single(self):
        state: AgentState = {
            **self._base_state(),
            "route": "single",
            "tickers": ["AAPL"],
            "rate_limit_remaining": 9,
            "blocked_reason": None,
        }
        assert state["route"] == "single"
        assert state["tickers"] == ["AAPL"]

    def test_post_orchestrator_invalid(self):
        state: AgentState = {
            **self._base_state(),
            "route": "invalid",
            "tickers": [],
            "blocked_reason": "Could not identify a valid ticker.",
        }
        assert state["route"] == "invalid"
        assert state["blocked_reason"] is not None

    def test_post_orchestrator_rate_limited(self):
        state: AgentState = {
            **self._base_state(),
            "route": "rate_limited",
            "tickers": [],
            "rate_limit_remaining": 0,
            "blocked_reason": "Hourly query limit reached.",
        }
        assert state["rate_limit_remaining"] == 0

    def test_post_data_fetcher(self):
        f = Fundamentals(ticker="AAPL", pe_ratio=29.1, market_cap=3e12)
        td = TickerData(ticker="AAPL", fundamentals=f, data_completeness_score=0.5)
        state: AgentState = {
            **self._base_state(),
            "route": "single",
            "tickers": ["AAPL"],
            "data_by_ticker": {"AAPL": td},
        }
        assert state["data_by_ticker"]["AAPL"].fundamentals.pe_ratio == pytest.approx(29.1)

    def test_post_news_fetcher(self):
        nb = NewsBundle(ticker="AAPL", sentiment="positive")
        td = TickerData(ticker="AAPL", news=nb, data_completeness_score=0.75)
        state: AgentState = {
            **self._base_state(),
            "route": "single",
            "tickers": ["AAPL"],
            "data_by_ticker": {"AAPL": td},
        }
        assert state["data_by_ticker"]["AAPL"].news.sentiment == "positive"

    def test_post_analyst(self):
        ao = AnalystOutput(
            summary="Apple shows strong revenue growth and solid fundamentals.",
            key_points=["Revenue up 8% YoY", "PE ratio 29x is fair value"],
            iteration=1,
        )
        state: AgentState = {
            **self._base_state(),
            "route": "single",
            "tickers": ["AAPL"],
            "analyst_outputs": {"AAPL": ao},
            "iteration_count": {"AAPL": 1},
        }
        assert len(state["analyst_outputs"]["AAPL"].key_points) == 2

    def test_post_critique_passed(self):
        cr = CritiqueResult(passed=True, grounded=True, no_advice=True, concise=True)
        state: AgentState = {
            **self._base_state(),
            "critique_by_ticker": {"AAPL": cr},
            "iteration_count": {"AAPL": 1},
        }
        assert state["critique_by_ticker"]["AAPL"].passed is True

    def test_post_critique_retry(self):
        cr = CritiqueResult(
            passed=False,
            grounded=True,
            no_advice=False,
            concise=True,
            issues=["Output contains investment advice."],
        )
        state: AgentState = {
            **self._base_state(),
            "critique_by_ticker": {"AAPL": cr},
            "iteration_count": {"AAPL": 2},
        }
        assert state["iteration_count"]["AAPL"] == 2
        assert not state["critique_by_ticker"]["AAPL"].passed

    def test_final_state(self):
        gm = GoalMetrics(data_fresh=True, completeness_score=1.0, total_latency_ms=1200)
        state: AgentState = {
            **self._base_state(),
            "route": "single",
            "tickers": ["AAPL"],
            "final_message": "*AAPL* — P/E: 29x | 52w high: $230",
            "metrics": gm,
            "errors": [],
        }
        assert state["final_message"].startswith("*AAPL*")
        assert state["metrics"].data_fresh is True

    def test_comparison_state_two_tickers(self):
        ao_aapl = AnalystOutput(summary="Apple solid.", key_points=[])
        ao_msft = AnalystOutput(summary="Microsoft dominates cloud.", key_points=[])
        state: AgentState = {
            **self._base_state(),
            "route": "comparison",
            "tickers": ["AAPL", "MSFT"],
            "analyst_outputs": {"AAPL": ao_aapl, "MSFT": ao_msft},
            "iteration_count": {"AAPL": 1, "MSFT": 1},
        }
        assert set(state["analyst_outputs"].keys()) == {"AAPL", "MSFT"}

    def test_recent_tickers_memory(self):
        state: AgentState = {
            **self._base_state(),
            "recent_tickers": ["TSLA", "NVDA", "AAPL", "GOOG", "META"],
        }
        assert len(state["recent_tickers"]) == 5
        assert "AAPL" in state["recent_tickers"]

    def test_errors_accumulated(self):
        state: AgentState = {
            **self._base_state(),
            "errors": [
                "yfinance: timeout fetching AAPL volume",
                "Brave: 429 rate-limited",
            ],
        }
        assert len(state["errors"]) == 2
