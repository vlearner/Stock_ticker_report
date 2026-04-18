"""Per-node latency decorator for LangGraph pipeline nodes.

Usage::

    from src.graph.node_timer import timed_node

    @timed_node("analyst")
    async def analyst_node(state: AgentState) -> dict:
        ...

Each decorated node emits a structured log line on completion::

    node_completed node=analyst latency_ms=412.3 correlation_id=<uuid>

The decorator preserves the original function name (via ``functools.wraps``)
so LangSmith traces remain readable.  Works for both sync and async nodes.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def timed_node(name: str) -> Callable:
    """Return a decorator that wraps a LangGraph node with latency logging.

    Args:
        name: Human-readable node name used in log output and LangSmith traces.

    Returns:
        A decorator that wraps the node function.
    """

    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def _async_wrapper(state: Any) -> dict:
                t0 = time.monotonic()
                try:
                    return await fn(state)
                finally:
                    latency_ms = (time.monotonic() - t0) * 1000
                    correlation_id = state.get("correlation_id", "")
                    logger.info(
                        "node_completed node=%s latency_ms=%.1f correlation_id=%s",
                        name,
                        latency_ms,
                        correlation_id,
                    )

            return _async_wrapper

        else:

            @functools.wraps(fn)
            def _sync_wrapper(state: Any) -> dict:
                t0 = time.monotonic()
                try:
                    return fn(state)
                finally:
                    latency_ms = (time.monotonic() - t0) * 1000
                    correlation_id = state.get("correlation_id", "")
                    logger.info(
                        "node_completed node=%s latency_ms=%.1f correlation_id=%s",
                        name,
                        latency_ms,
                        correlation_id,
                    )

            return _sync_wrapper

    return decorator
