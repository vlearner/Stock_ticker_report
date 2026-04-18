"""Unit tests for the LangGraph pipeline skeleton (src.graph.pipeline).

These tests verify the graph structure, compilation, routing, and the
reflection loop — not the stub logic of individual nodes.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Ensure env vars are set before any import that triggers Settings construction.
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("BRAVE_API_KEY", "test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")

from src.graph.pipeline import (  # noqa: E402
    build_graph,
    compile_graph,
    route_after_critic,
    route_after_orchestrator,
)
from src.schemas.ticker_data import CritiqueResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _invoke(user_message: str, **extra) -> dict:
    """Compile the graph and invoke with a minimal valid state."""
    graph = compile_graph()
    state = {"user_message": user_message, "user_id": "u1", "chat_id": "c1", **extra}
    return await graph.ainvoke(state)


# ---------------------------------------------------------------------------
# 1. Graph compiles
# ---------------------------------------------------------------------------


class TestGraphCompiles:
    def test_compile_graph_returns_without_error(self):
        compiled = compile_graph()
        assert compiled is not None

    def test_compile_graph_with_no_checkpointer(self):
        compiled = compile_graph(checkpointer=None)
        assert compiled is not None

    def test_build_graph_returns_state_graph(self):
        from langgraph.graph import StateGraph

        graph = build_graph()
        assert isinstance(graph, StateGraph)


# ---------------------------------------------------------------------------
# 2. Single ticker route
# ---------------------------------------------------------------------------


class TestSingleTickerRoute:
    async def test_single_route_set(self):
        result = await _invoke("AAPL")
        assert result["route"] == "single"

    async def test_single_final_message_set(self):
        result = await _invoke("AAPL")
        assert result.get("final_message") is not None
        assert len(result["final_message"]) > 0

    async def test_single_tickers_contains_aapl(self):
        result = await _invoke("AAPL")
        assert "AAPL" in result["tickers"]

    async def test_single_data_by_ticker_populated(self):
        result = await _invoke("AAPL")
        assert "AAPL" in result.get("data_by_ticker", {})

    async def test_single_analyst_outputs_populated(self):
        result = await _invoke("AAPL")
        assert "AAPL" in result.get("analyst_outputs", {})


# ---------------------------------------------------------------------------
# 3. Comparison route
# ---------------------------------------------------------------------------


class TestComparisonRoute:
    async def test_comparison_route_with_vs(self):
        result = await _invoke("AAPL vs MSFT")
        assert result["route"] == "comparison"

    async def test_comparison_route_with_compare(self):
        result = await _invoke("compare AAPL MSFT")
        assert result["route"] == "comparison"

    async def test_comparison_both_tickers_present(self):
        # Use "compare" keyword; "vs" is alpha so the stub treats it as a ticker
        result = await _invoke("compare AAPL MSFT")
        tickers = result["tickers"]
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    async def test_comparison_final_message_set(self):
        result = await _invoke("AAPL vs MSFT")
        assert result.get("final_message") is not None

    async def test_comparison_data_by_ticker_has_both(self):
        result = await _invoke("compare AAPL MSFT")
        data = result.get("data_by_ticker", {})
        assert "AAPL" in data
        assert "MSFT" in data


# ---------------------------------------------------------------------------
# 4. News only route
# ---------------------------------------------------------------------------


class TestNewsOnlyRoute:
    async def test_news_only_route_set(self):
        result = await _invoke("news TSLA")
        assert result["route"] == "news_only"

    async def test_news_only_final_message_set(self):
        result = await _invoke("news TSLA")
        assert result.get("final_message") is not None

    async def test_news_only_tickers_set(self):
        result = await _invoke("news TSLA")
        assert "TSLA" in result["tickers"]

    async def test_news_only_no_fundamentals(self):
        """News-only route skips data_fetcher, so TickerData should have no
        fundamentals/moving_averages/volume — only news may be populated."""
        result = await _invoke("news TSLA")
        data = result.get("data_by_ticker", {})
        # news_fetcher creates TickerData with news only — no yfinance slices
        for td in data.values():
            assert td.fundamentals is None
            assert td.moving_averages is None
            assert td.volume is None


# ---------------------------------------------------------------------------
# 5. Invalid route
# ---------------------------------------------------------------------------


class TestInvalidRoute:
    async def test_invalid_route_set(self):
        result = await _invoke("!!!")
        assert result["route"] == "invalid"

    async def test_invalid_final_message_contains_couldnt(self):
        result = await _invoke("!!!")
        assert "couldn't identify" in result["final_message"].lower()

    async def test_invalid_errors_populated(self):
        result = await _invoke("!!!")
        errors = result.get("errors", [])
        assert any("route:invalid" in e for e in errors)

    async def test_invalid_no_tickers(self):
        result = await _invoke("!!!")
        assert result["tickers"] == []

    async def test_numeric_only_is_invalid(self):
        result = await _invoke("12345")
        assert result["route"] == "invalid"


# ---------------------------------------------------------------------------
# 6. Reflection loop respects max iterations
# ---------------------------------------------------------------------------


class TestReflectionLoop:
    def _make_failing_critic(self):
        """Return a critic_node replacement that always fails."""
        def _critic_always_fails(state):
            tickers = state.get("tickers", [])
            return {
                "critique_by_ticker": {
                    t: CritiqueResult(
                        passed=False,
                        grounded=False,
                        no_advice=True,
                        concise=True,
                        issues=["Always fails"],
                    )
                    for t in tickers
                }
            }
        return _critic_always_fails

    async def test_reflection_loop_limited_by_max_iterations(self):
        """With critic always failing, analyst should run exactly
        max_critic_iterations times, then the loop should exit."""
        from src.config import settings

        max_iter = settings.max_critic_iterations  # default = 2

        with patch("src.graph.pipeline.critic_node", self._make_failing_critic()):
            result = await _invoke("AAPL")

        # The analyst should have run max_iter times
        iteration_count = result.get("iteration_count", {})
        assert iteration_count.get("AAPL", 0) == max_iter

    async def test_reflection_loop_analyst_output_reflects_iterations(self):
        """Analyst output iteration field should match the final iteration."""
        from src.config import settings

        max_iter = settings.max_critic_iterations

        with patch("src.graph.pipeline.critic_node", self._make_failing_critic()):
            result = await _invoke("AAPL")

        analyst_out = result.get("analyst_outputs", {}).get("AAPL")
        assert analyst_out is not None
        assert analyst_out.iteration == max_iter

    async def test_normal_critic_passes_first_time(self):
        """Default stub critic passes, so analyst runs only once."""
        result = await _invoke("AAPL")
        iteration_count = result.get("iteration_count", {})
        assert iteration_count.get("AAPL") == 1


# ---------------------------------------------------------------------------
# 7. Graph has expected nodes
# ---------------------------------------------------------------------------


class TestGraphNodes:
    EXPECTED_NODES = {
        "orchestrator",
        "data_fetcher",
        "news_fetcher",
        "analyst",
        "critic",
        "formatter",
        "error_handler",
    }

    def test_all_expected_nodes_present(self):
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        # LangGraph may add __start__ / __end__ nodes internally, so check subset
        assert self.EXPECTED_NODES.issubset(node_names)

    def test_no_unexpected_application_nodes(self):
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        # Filter out LangGraph internal nodes (prefixed with __)
        app_nodes = {n for n in node_names if not n.startswith("__")}
        assert app_nodes == self.EXPECTED_NODES


# ---------------------------------------------------------------------------
# Conditional edge function unit tests
# ---------------------------------------------------------------------------


class TestRouteAfterOrchestrator:
    def test_invalid_routes_to_error(self):
        assert route_after_orchestrator({"route": "invalid"}) == "error"

    def test_rate_limited_routes_to_error(self):
        assert route_after_orchestrator({"route": "rate_limited"}) == "error"

    def test_news_only_routes_to_news_only(self):
        assert route_after_orchestrator({"route": "news_only"}) == "news_only"

    def test_single_routes_to_full_pipeline(self):
        assert route_after_orchestrator({"route": "single"}) == "full_pipeline"

    def test_comparison_routes_to_full_pipeline(self):
        assert route_after_orchestrator({"route": "comparison"}) == "full_pipeline"

    def test_missing_route_defaults_to_error(self):
        assert route_after_orchestrator({}) == "error"


class TestRouteAfterCritic:
    def test_all_passed_returns_pass(self):
        state = {
            "critique_by_ticker": {
                "AAPL": CritiqueResult(
                    passed=True, grounded=True, no_advice=True, concise=True
                )
            },
            "iteration_count": {"AAPL": 1},
        }
        assert route_after_critic(state) == "pass"

    def test_failed_under_limit_returns_revise(self):
        state = {
            "critique_by_ticker": {
                "AAPL": CritiqueResult(
                    passed=False, grounded=False, no_advice=True, concise=True
                )
            },
            "iteration_count": {"AAPL": 1},
        }
        assert route_after_critic(state) == "revise"

    def test_failed_at_limit_returns_pass(self):
        """When iteration_count >= max_critic_iterations, should pass even if
        the critique failed."""
        from src.config import settings

        state = {
            "critique_by_ticker": {
                "AAPL": CritiqueResult(
                    passed=False, grounded=False, no_advice=True, concise=True
                )
            },
            "iteration_count": {"AAPL": settings.max_critic_iterations},
        }
        assert route_after_critic(state) == "pass"

    def test_empty_critiques_returns_pass(self):
        assert route_after_critic({"critique_by_ticker": {}, "iteration_count": {}}) == "pass"
