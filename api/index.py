"""Vercel Python entry point.

The ``@vercel/python`` runtime discovers ``app`` via ASGI; importing our
FastAPI app here is enough. The demo path uses only the tool functions
in ``src/tools/`` — no LangGraph, no Groq, no Telegram on this serverless
bundle (see ``api/requirements.txt`` and ``.vercelignore``).
"""

from src.api.app import app  # noqa: F401  (imported for Vercel's runtime)
