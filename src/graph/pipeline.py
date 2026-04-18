"""LangGraph pipeline assembly — nodes, edges, conditional routing, checkpointer.

Step 3: Graph skeleton (routing, reflection loop, node inventory).
Step 4: Real agent implementations wired in (data_fetcher, news_fetcher,
        analyst, formatter, orchestrator).
Step 6: Orchestrator upgraded to LLM-based classification (ChatGroq
        llama-3.1-8b-instant) with heuristic fallback.
Security + Logging: correlation_id injected at pipeline entry; per-node latency
                    logging via @timed_node decorator.
"""

from __future__ import annotations

import uuid

from langgraph.graph import END, START, StateGraph

from src.agents import data_fetcher, formatter, news_fetcher, analyst, orchestrator
from src.config import settings
from src.graph.node_timer import timed_node
from src.schemas.agent_state import AgentState


# ---------------------------------------------------------------------------
# Node wrappers
# ---------------------------------------------------------------------------
# Each wrapper is a thin adapter that calls the agent's ``run()`` function.
# Keeping them as named functions (rather than lambdas) makes LangSmith
# traces readable and lets us patch individual nodes in tests.
#
# @timed_node("name") wraps each async node to:
#   • inject correlation_id + node name into the log context
#   • emit a structured "node_completed" log line with latency_ms


def _inject_correlation_id(state: AgentState) -> dict:
    """Entry node: assign a correlation_id if one is not already present.

    Every pipeline run gets a UUID that appears in all subsequent log
    records and error responses.  Callers (REST, Telegram) may pre-set the
    ID; if they do, it is preserved.
    """
    if state.get("correlation_id"):
        return {}
    return {"correlation_id": str(uuid.uuid4())}


@timed_node("orchestrator")
async def orchestrator_node(state: AgentState) -> dict:
    """Classify intent via LLM, extract tickers, and set route + format_style."""
    return await orchestrator.run(state)


@timed_node("data_fetcher")
async def data_fetcher_node(state: AgentState) -> dict:
    """Fetch fundamentals, moving averages, and volume from yfinance."""
    return await data_fetcher.run(state)


@timed_node("news_fetcher")
async def news_fetcher_node(state: AgentState) -> dict:
    """Fetch news from Brave Search (only when route or keywords require it)."""
    return await news_fetcher.run(state)


@timed_node("analyst")
async def analyst_node(state: AgentState) -> dict:
    """Generate structured AnalystOutput via ChatGroq (llama-3.3-70b)."""
    return await analyst.run(state)


@timed_node("critic")
async def critic_node(state: AgentState) -> dict:
    """Stub: evaluate analyst output against quality criteria (step 7).

    Always passes for now — real critique logic is added in step 7.
    """
    from src.schemas.ticker_data import CritiqueResult

    tickers = state.get("tickers", [])
    return {
        "critique_by_ticker": {
            t: CritiqueResult(passed=True, grounded=True, no_advice=True, concise=True)
            for t in tickers
        }
    }


@timed_node("formatter")
async def formatter_node(state: AgentState) -> dict:
    """Format pipeline results into a Telegram-ready message."""
    return formatter.run(state)


@timed_node("error_handler")
async def error_handler_node(state: AgentState) -> dict:
    """Return a user-friendly message for invalid or rate-limited routes."""
    route = state.get("route", "invalid")
    if route == "rate_limited":
        msg = "You've reached the query limit (10/hour). Please try again later."
    else:
        msg = (
            "I couldn't identify a valid ticker symbol. "
            "Please try something like 'AAPL' or 'MSFT'."
        )
    return {
        "final_message": msg,
        "errors": state.get("errors", []) + [f"route:{route}"],
    }


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def route_after_orchestrator(state: AgentState) -> str:
    """Route to appropriate subgraph based on orchestrator classification."""
    route = state.get("route", "invalid")
    if route in ("invalid", "rate_limited"):
        return "error"
    if route == "news_only":
        return "news_only"
    # "single" and "comparison" both go through the full pipeline
    return "full_pipeline"


def route_after_critic(state: AgentState) -> str:
    """Decide whether to revise (loop back to analyst) or proceed to formatter."""
    critiques = state.get("critique_by_ticker", {})
    iteration_count = state.get("iteration_count", {})

    for ticker, critique in critiques.items():
        if (
            not critique.passed
            and iteration_count.get(ticker, 1) < settings.max_critic_iterations
        ):
            return "revise"
    return "pass"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Assemble and return the (uncompiled) LangGraph StateGraph.

    Returns:
        An uncompiled ``StateGraph`` instance.  Call ``.compile()`` (or use
        :func:`compile_graph`) to obtain a runnable graph.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("entry", _inject_correlation_id)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("data_fetcher", data_fetcher_node)
    graph.add_node("news_fetcher", news_fetcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("formatter", formatter_node)
    graph.add_node("error_handler", error_handler_node)

    # Entry point: inject correlation_id, then classify
    graph.add_edge(START, "entry")
    graph.add_edge("entry", "orchestrator")

    # Orchestrator → subgraph selection
    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "full_pipeline": "data_fetcher",
            "news_only": "news_fetcher",
            "error": "error_handler",
        },
    )

    # Full pipeline flow
    graph.add_edge("data_fetcher", "news_fetcher")
    graph.add_edge("news_fetcher", "analyst")
    graph.add_edge("analyst", "critic")

    # Reflection loop (critic → analyst or formatter)
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "pass": "formatter",
            "revise": "analyst",
        },
    )

    # Terminal edges
    graph.add_edge("formatter", END)
    graph.add_edge("error_handler", END)

    return graph


def compile_graph(checkpointer=None):
    """Build and compile the graph with optional checkpointer.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver).
            If ``None``, no state persistence between invocations.

    Returns:
        Compiled graph ready for ``.invoke()`` / ``.ainvoke()``.
    """
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
