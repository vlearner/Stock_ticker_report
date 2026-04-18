"""FastAPI application — exposes stock tools as REST endpoints.

Each feature area has its own route module so components stay independent.
Only the route modules you include here need their service keys configured.

Run locally
-----------
::

    # From the project root (with .env present):
    python -m src.api.app

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
Web demo:    http://localhost:8000/  (when ``settings.serve_static_ui`` is on)

Adding future routes
--------------------
1. Create ``src/api/routes/<feature>.py`` with a ``router = APIRouter()``.
2. Add ``from src.api.routes import <feature>`` below.
3. Add ``app.include_router(<feature>.router, prefix="/api/v1", tags=["<feature>"])``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import chat, news, ticker
from src.config import settings

app = FastAPI(
    title="Stock Ticker Report API",
    version="0.1.0",
    description=(
        "REST interface for the stock ticker report tools. "
        "Each endpoint maps to a standalone tool — only the relevant "
        "API key is required per route."
    ),
)

# ---------------------------------------------------------------------------
# CORS — permissive by default; restrict via ``cors_allowed_origins`` in prod.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(news.router, prefix="/api/v1", tags=["news"])
app.include_router(ticker.router, prefix="/api/v1", tags=["ticker"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

# Future routes follow the same pattern:
# app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])


# ---------------------------------------------------------------------------
# Static UI — mounted only for local dev; on Vercel the /public/ folder is
# served by the CDN, not the Python function.
# ---------------------------------------------------------------------------

_PUBLIC_DIR = Path(__file__).resolve().parents[2] / "public"
if settings.serve_static_ui and _PUBLIC_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(_PUBLIC_DIR), html=True),
        name="public",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("src.api.app:app", host="localhost", port=8000, reload=True)
