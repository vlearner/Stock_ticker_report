"""LangGraph state definition.

The pipeline is a dict-like :class:`TypedDict` so LangGraph can merge node
outputs by key. All fields are ``total=False`` to let nodes emit partial
updates. Fields are keyed by ticker where the comparison route needs to
carry multiple parallel results.

Reducer-annotated fields (``Annotated[list, operator.add]`` etc.) are
deferred until the graph assembly step; plain ``TypedDict`` is sufficient
for the scaffolding milestone.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from src.schemas.ticker_data import (
    AnalystOutput,
    CritiqueResult,
    GoalMetrics,
    MAChartData,
    TickerData,
)

Route = Literal[
    "single",
    "comparison",
    "news_only",
    "chart",
    "invalid",
    "rate_limited",
]

FormatStyle = Literal["minimal", "rich", "news"]
"""Telegram message style set by the Orchestrator.

- ``"minimal"``  — ticker header + summary only (plain ticker query).
- ``"rich"``     — header + fundamentals snapshot + summary + key points
                   (detail keywords present, or comparison route).
- ``"news"``     — ticker header + news headlines (news_only route).
"""


class AgentState(TypedDict, total=False):
    """Shared state flowing through the LangGraph pipeline.

    Attributes:
        user_id: Stable user identifier from the messaging platform.
        chat_id: Platform chat/channel identifier (used for replies).
        user_message: Raw inbound text from the user.

        route: Chosen execution path picked by the Orchestrator.
        tickers: Normalized uppercase tickers extracted from the message.
        format_style: Formatter style chosen by the Orchestrator
            (``"minimal"``, ``"rich"``, or ``"news"``).

        data_by_ticker: Per-ticker :class:`TickerData` populated by the
            DataFetcher and NewsFetcher agents. Keyed by ticker to support
            comparison queries (Pattern 2 — Routing).
        analyst_outputs: Per-ticker Analyst outputs keyed by ticker.
        critique_by_ticker: Per-ticker Critic verdicts keyed by ticker
            (Pattern 5 — Reflection).
        iteration_count: Per-ticker retry counter for the reflection loop.

        recent_tickers: Last 5 tickers queried by this user — short-term
            memory (Pattern 7).
        preferred_format: User-scoped preference persisted in SQLite —
            long-term memory (Pattern 7).

        rate_limit_remaining: Remaining queries in the current hour window
            (Pattern 8 — Guardrails).
        blocked_reason: If the request was rejected by a guardrail, a
            human-readable reason for the refusal.

        correlation_id: Full UUID assigned at pipeline entry — uniquely
            identifies a single end-to-end run across all log records and
            error responses.
        run_id: Short UUID prefix (8 hex chars) generated in the orchestrator
            node — used to correlate log lines for a single pipeline run.
        start_time: ``time.monotonic()`` captured at pipeline entry — used
            to compute total latency for Pattern 9.
        metrics: Goal metrics evaluated at the end of the run.
        final_message: Fully formatted message ready for delivery.
        errors: Accumulated non-fatal error strings for LangSmith traces.
    """

    # --- Inputs -----------------------------------------------------------
    user_id: str
    chat_id: str
    user_message: str

    # --- Routing ----------------------------------------------------------
    route: Route
    tickers: list[str]
    format_style: FormatStyle

    # --- Pipeline results -------------------------------------------------
    chart_data: MAChartData | None
    data_by_ticker: dict[str, TickerData]
    analyst_outputs: dict[str, AnalystOutput]
    critique_by_ticker: dict[str, CritiqueResult]
    iteration_count: dict[str, int]

    # --- Memory -----------------------------------------------------------
    recent_tickers: list[str]
    preferred_format: str | None

    # --- Guardrails -------------------------------------------------------
    rate_limit_remaining: int
    blocked_reason: str | None

    # --- Observability ----------------------------------------------------
    correlation_id: str
    run_id: str
    start_time: float
    metrics: GoalMetrics | None
    final_message: str | None
    errors: list[str]
