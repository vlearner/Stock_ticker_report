"""Brave Search settings — standalone, requires only BRAVE_API_KEY.

Import ``brave_settings`` here directly when working with Brave Search tools
or the news API route. No other service keys are required.

Example
-------
::

    from src.config.brave import brave_settings

    print(brave_settings.brave_api_key)
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BraveSettings(BaseSettings):
    """Settings for the Brave Search integration.

    Only ``BRAVE_API_KEY`` is required. All other fields have sensible
    defaults so this class can be instantiated with a minimal ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    brave_api_key: str = Field(..., description="Brave Search API key.")
    brave_search_timeout_s: float = Field(
        default=10.0,
        gt=0.0,
        description="HTTP timeout for Brave Search requests (seconds).",
    )


brave_settings = BraveSettings()  # type: ignore[call-arg]
"""Module-level singleton — import this directly, never construct BraveSettings()."""

__all__ = ["BraveSettings", "brave_settings"]
