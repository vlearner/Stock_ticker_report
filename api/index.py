"""Vercel serverless entry point — self-contained demo FastAPI app.

No src.* imports. Uses only packages listed in api/requirements.txt.
The full LangGraph + Telegram pipeline runs locally via src/main.py.

Endpoint
--------
POST /api/v1/chat   body: {message, session_id}
                    response: {reply, route, session_id, session_query_limit}
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Optional

import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

_PUBLIC = Path(__file__).parent.parent / "public"

app = FastAPI(
    title="Stock Ticker Demo API",
    version="0.1.0",
    description="Demo API for the stock ticker report web UI.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SESSION_LIMIT = int(os.getenv("DEMO_SESSION_QUERY_LIMIT", "5"))

# ── Common English words to skip when scanning for ticker symbols ────────────
_SKIP = frozenset(
    "A AN AS AT BE BY DO GO IF IN IS IT MY NO OF ON OR SO TO UP US WE".split()
)
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")
_UNSAFE_RE = re.compile(r"[<>/{}\\[\]]")
_SPACE_RE = re.compile(r"[ \t\r\n]+")


# ── Request / response models ────────────────────────────────────────────────


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


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sanitize(text: str) -> str:
    text = _UNSAFE_RE.sub("", text).strip()
    return _SPACE_RE.sub(" ", text)[:200]


def _extract_ticker(message: str) -> Optional[str]:
    for m in _TICKER_RE.finditer(message.upper()):
        candidate = m.group(1)
        if candidate not in _SKIP:
            return candidate
    return None


def _fmt_price(v: Optional[float]) -> str:
    return f"${v:,.2f}" if v is not None else "N/A"


def _fmt_large(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    if v >= 1e12:
        return f"${v / 1e12:.2f}T"
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


def _build_reply(symbol: str) -> str:
    info = yf.Ticker(symbol).info
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price:
        return (
            f"I couldn't find current data for **{symbol}**. "
            "Please double-check the ticker symbol."
        )
    name = info.get("longName") or symbol
    lines = [
        f"**{name} ({symbol})**",
        f"Price: {_fmt_price(price)}",
        f"Market Cap: {_fmt_large(info.get('marketCap'))}",
        "P/E Ratio: " + (f"{info['trailingPE']:.1f}" if info.get("trailingPE") else "N/A"),
        f"52-Week Range: {_fmt_price(info.get('fiftyTwoWeekLow'))} – {_fmt_price(info.get('fiftyTwoWeekHigh'))}",
        "",
        "_Demo mode: live yfinance data. "
        "Full AI analysis runs in the local pipeline._",
    ]
    return "\n".join(lines)


# ── Static UI ────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> Response:
    return Response((_PUBLIC / "index.html").read_bytes(), media_type="text/html")


@app.get("/app.js")
async def app_js() -> Response:
    return Response((_PUBLIC / "app.js").read_bytes(), media_type="application/javascript")


@app.get("/styles.css")
async def styles_css() -> Response:
    return Response((_PUBLIC / "styles.css").read_bytes(), media_type="text/css")


# ── Chat endpoint ─────────────────────────────────────────────────────────────


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    message = _sanitize(request.message)
    ticker = _extract_ticker(message)

    if ticker:
        try:
            reply = await asyncio.to_thread(_build_reply, ticker)
        except Exception:
            reply = (
                f"Unable to fetch data for **{ticker}** right now. "
                "Please try again shortly."
            )
        route = "ticker_lookup"
    else:
        reply = (
            "Hi! This is the **Stock Ticker Demo**. "
            "Ask me about a stock by its ticker symbol — "
            "for example: *What is the price of AAPL?* or *Tell me about TSLA.*"
        )
        route = "greeting"

    return ChatResponse(
        reply=reply,
        route=route,
        session_id=request.session_id,
        session_query_limit=_SESSION_LIMIT,
    )
