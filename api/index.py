"""Vercel Python entry point.

The ``@vercel/python`` runtime discovers ``app`` via ASGI; importing our
FastAPI app here is enough. The demo path uses only the tool functions
in ``src/tools/`` — no LangGraph, no Groq, no Telegram on this serverless
bundle (see ``api/requirements.txt`` and ``.vercelignore``).
"""

import os
import sys
import types

# Vercel deploys src/ contents to the Lambda root WITHOUT the src/ prefix
# (e.g. src/config/__init__.py → /config/__init__.py).
# src/api/* IS compiled into the binary with the src/ prefix intact.
# Extend (or create) the src package's __path__ to include the function
# root so that imports like `from src.config import settings` resolve.
_fn_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "src" in sys.modules:
    _src = sys.modules["src"]
    if hasattr(_src, "__path__") and _fn_root not in list(_src.__path__):
        _src.__path__.append(_fn_root)
else:
    _src = types.ModuleType("src")
    _src.__path__ = [_fn_root]
    _src.__package__ = "src"
    sys.modules["src"] = _src

from src.api.app import app  # noqa: F401  (imported for Vercel's runtime)
