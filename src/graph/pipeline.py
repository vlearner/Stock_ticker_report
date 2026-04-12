"""LangGraph pipeline assembly — nodes, edges, conditional routing, checkpointer.

Implemented in Implementation Order step 3 (Graph skeleton) and beyond.
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.schemas.agent_state import AgentState


# ---------------------------------------------------------------------------
# Node implementations (stubs)
# ---------------------------------------------------------------------------


def orchestrator_node(state: AgentState) -> dict:
    """Stub: classifies intent and sets route + tickers.

    Real implementation will use LLM classification (step 6).
    For now, defaults to 'single' route with the first word as ticker.
    """
    message = state.get("user_message", "")
    words = message.strip().split()
    tickers = [w.upper() for w in words if w.isalpha() and 1 <= len(w) <= 5]

    if not tickers:
        return {"route": "invalid", "tickers": [], "start_time": time.monotonic()}

    # Simple heuristic routing stub
    if "vs" in message.lower() or "compare" in message.lower():
        route = "comparison"
    elif "news" in message.lower():
        route = "news_only"
    else:
        route = "single"

    return {"route": route, "tickers": tickers[:2], "start_time": time.monotonic()}


def data_fetcher_node(state: AgentState) -> dict:
    """Stub: will call yfinance tools and populate data_by_ticker (step 4).

    For now, sets empty TickerData for each ticker.
    """
    from src.schemas.ticker_data import TickerData

    tickers = state.get("tickers", [])
    return {"data_by_ticker": {t: TickerData(ticker=t) for t in tickers}}


def news_fetcher_node(state: AgentState) -> dict:
    """Stub: will call Brave Search and populate news in data_by_ticker (step 4).

    For now, returns state unchanged.
    """
    return {}


def analyst_node(state: AgentState) -> dict:
    """Stub: will generate plain-language summary using LLM (step 4).

    For now, sets a placeholder AnalystOutput.
    """
    from src.schemas.ticker_data import AnalystOutput

    tickers = state.get("tickers", [])
    iteration_count = state.get("iteration_count", {})
    outputs: dict = {}
    new_iterations: dict = {}
    for t in tickers:
        current = iteration_count.get(t, 0) + 1
        new_iterations[t] = current
        outputs[t] = AnalystOutput(
            summary=f"Placeholder analysis for {t} (iteration {current}).",
            key_points=[f"{t} data pending full implementation."],
            iteration=current,
        )
    return {"analyst_outputs": outputs, "iteration_count": new_iterations}


def critic_node(state: AgentState) -> dict:
    """Stub: will evaluate analyst output against quality criteria (step 7).

    For now, always passes.
    """
    from src.schemas.ticker_data import CritiqueResult

    tickers = state.get("tickers", [])
    return {
        "critique_by_ticker": {
            t: CritiqueResult(
                passed=True, grounded=True, no_advice=True, concise=True
            )
            for t in tickers
        }
    }


def formatter_node(state: AgentState) -> dict:
    """Stub: will format output for Telegram delivery (step 4).

    For now, concatenates analyst summaries.
    """
    outputs = state.get("analyst_outputs", {})
    parts = [f"**{t}**: {o.summary}" for t, o in outputs.items()]
    return {"final_message": "\n\n".join(parts) if parts else "No data available."}


def error_handler_node(state: AgentState) -> dict:
    """Handle invalid/rate-limited routes with a user-friendly message."""
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
    # "single" and "comparison" both go through full pipeline
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

    # Add nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("data_fetcher", data_fetcher_node)
    graph.add_node("news_fetcher", news_fetcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("formatter", formatter_node)
    graph.add_node("error_handler", error_handler_node)

    # Entry point
    graph.add_edge(START, "orchestrator")

    # Orchestrator routes
    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "full_pipeline": "data_fetcher",
            "news_only": "news_fetcher",
            "error": "error_handler",
        },
    )

    # Full pipeline edges
    graph.add_edge("data_fetcher", "news_fetcher")
    graph.add_edge("news_fetcher", "analyst")
    graph.add_edge("analyst", "critic")

    # Reflection loop
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
