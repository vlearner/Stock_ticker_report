"""yfinance settings — standalone, no API key required.

Import ``yfinance_settings`` here directly when working with yfinance tools
or the ticker API routes. No LLM, Brave, or Telegram keys are required.

Example
-------
::

    from src.config.yfinance_settings import yfinance_settings

    print(yfinance_settings.yfinance_timeout_s)
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class YFinanceSettings(BaseSettings):
    """Settings for the yfinance integration.

    No API key is required — yfinance fetches public market data directly.
    All fields have sensible defaults so this class can be instantiated with
    an empty (or absent) ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    yfinance_timeout_s: float = Field(
        default=8.0,
        gt=0.0,
        description="Per-call yfinance timeout (seconds).",
    )
    yfinance_retries: int = Field(
        default=1,
        ge=0,
        description="Retries on transient yfinance failures.",
    )


yfinance_settings = YFinanceSettings()
"""Module-level singleton — import this directly, never construct YFinanceSettings()."""

__all__ = ["YFinanceSettings", "yfinance_settings"]
