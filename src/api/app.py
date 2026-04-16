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

Adding future routes
--------------------
1. Create ``src/api/routes/<feature>.py`` with a ``router = APIRouter()``.
2. Add ``from src.api.routes import <feature>`` below.
3. Add ``app.include_router(<feature>.router, prefix="/api/v1", tags=["<feature>"])``.
"""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import news, ticker

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
# Routes
# ---------------------------------------------------------------------------

app.include_router(news.router, prefix="/api/v1", tags=["news"])
app.include_router(ticker.router, prefix="/api/v1", tags=["ticker"])

# Future routes follow the same pattern:
# app.include_router(analyze.router, prefix="/api/v1", tags=["analyze"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
