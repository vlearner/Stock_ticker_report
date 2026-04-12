"""Runtime configuration loaded from environment / ``.env``.

All configuration is centralized here in a single
:class:`pydantic_settings.BaseSettings` subclass and exported as the module-
level ``settings`` singleton. Every other module imports ``settings`` from
here — do not re-read environment variables anywhere else.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Environment variables are resolved case-insensitively. A ``.env`` file
    at the project root is loaded automatically if present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM (Groq) -------------------------------------------------------
    groq_api_key: str = Field(..., description="Groq API key.")
    groq_fast_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Cheap/fast model for classification, validation, critique.",
    )
    groq_analysis_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Larger model used by the Analyst agent.",
    )

    # --- News / Search ----------------------------------------------------
    brave_api_key: str = Field(..., description="Brave Search API key.")
    brave_search_timeout_s: float = Field(
        default=10.0, gt=0.0, description="Brave Search HTTP timeout (seconds)."
    )

    # --- Messaging --------------------------------------------------------
    telegram_bot_token: str = Field(..., description="Telegram bot token.")
    messaging_adapter: Literal["telegram"] = Field(
        default="telegram",
        description="Which MessagingAdapter to instantiate in src/main.py.",
    )

    # --- Observability (LangSmith) ---------------------------------------
    langchain_tracing_v2: bool = Field(
        default=True,
        description="Enable LangSmith tracing — required from day one.",
    )
    langchain_api_key: str | None = Field(
        default=None, description="LangSmith API key."
    )
    langchain_project: str = Field(
        default="stock-ticker-report",
        description="LangSmith project name for traces.",
    )

    # --- Persistence ------------------------------------------------------
    sqlite_db_path: str = Field(
        default="./data/state.db",
        description=(
            "SQLite file backing both the LangGraph checkpointer and the "
            "rate-limit / user-preference tables."
        ),
    )

    # --- yfinance tuning --------------------------------------------------
    yfinance_timeout_s: float = Field(
        default=8.0, gt=0.0, description="Per-call yfinance timeout (seconds)."
    )
    yfinance_retries: int = Field(
        default=1, ge=0, description="Retries on transient yfinance failures."
    )

    # --- Guardrails / pipeline -------------------------------------------
    rate_limit_per_hour: int = Field(
        default=10, ge=1, description="Max queries per user per rolling hour."
    )
    max_critic_iterations: int = Field(
        default=2, ge=1, description="Max Analyst → Critic revision cycles."
    )
    pipeline_warning_threshold_s: float = Field(
        default=8.0,
        gt=0.0,
        description="Emit a latency warning if total pipeline exceeds this.",
    )


settings = Settings()  # type: ignore[call-arg]
"""Module-level singleton. Import this — never construct ``Settings()`` directly."""
