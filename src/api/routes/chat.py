"""Chat dispatcher endpoint — runs the full LangGraph pipeline.

The pipeline (orchestrator → data_fetcher → news_fetcher → analyst →
critic → formatter) classifies intent, fetches data, generates an LLM
analysis, and returns a Telegram-formatted reply to the web demo UI.

Public surface
--------------
- ``POST /api/v1/chat`` — body: :class:`ChatRequest`, response: :class:`ChatResponse`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from src.api.middleware import require_api_key, sanitize_message
from src.config import settings
from src.graph.pipeline import compile_graph

logger = logging.getLogger(__name__)

router = APIRouter()

# Compiled once at import time — thread-safe, reused across requests.
_pipeline = compile_graph()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field(..., min_length=1, max_length=64)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str
    route: str
    session_id: str
    session_query_limit: int


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message through the full LangGraph pipeline",
    description=(
        "Runs orchestrator → data_fetcher → news_fetcher → analyst → "
        "critic → formatter and returns the formatted reply."
    ),
    dependencies=[Depends(require_api_key)],
)
async def chat(request: ChatRequest) -> ChatResponse:
    # Sanitize before invoking the pipeline
    clean_message = sanitize_message(request.message)

    logger.info("Chat: sid=%s  message=%r", request.session_id, clean_message)

    state = await _pipeline.ainvoke({"user_message": clean_message})

    reply: str = state.get("final_message") or "Sorry, something went wrong."
    route: str = state.get("route") or "unknown"

    logger.info("Chat: sid=%s  route=%s  reply_len=%d", request.session_id, route, len(reply))

    return ChatResponse(
        reply=reply,
        route=route,
        session_id=request.session_id,
        session_query_limit=settings.demo_session_query_limit,
    )
