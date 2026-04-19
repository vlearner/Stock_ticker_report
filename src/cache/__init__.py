"""Caching primitives for expensive pipeline steps.

Current caches
--------------
- :class:`AnalystCache` — stores generated :class:`AnalystOutput` keyed by
  ``ticker:YYYY-MM-DD`` so repeat queries within the same UTC day can skip
  the Groq LLM call entirely.

Each cache uses its own SQLite table in ``settings.sqlite_db_path`` (the
same DB that backs the rate limiter and LangGraph checkpointer).
"""

from src.cache.analyst_cache import AnalystCache, analyst_cache

__all__ = ["AnalystCache", "analyst_cache"]
