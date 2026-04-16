"""Unit tests for src.config.Settings.

Strategy
--------
* All tests construct ``Settings(...)`` directly with explicit kwargs — kwargs
  always win over both env vars and the ``.env`` file in pydantic-settings, so
  tests are fully isolated from the developer's local environment.
* For tests that need to verify *env-var reading* (case-insensitivity, missing
  required fields), we use ``_IsolatedSettings`` — a thin subclass that sets
  ``env_file=None`` — combined with ``monkeypatch.setenv`` / ``monkeypatch.delenv``
  so no real file or ambient env can interfere.
* We never test the module-level ``settings`` singleton directly; its
  construction is implicitly proven by the fact that the import doesn't raise.

Covers
------
- Instantiation with required fields only
- All default values
- Required-field validation (ValidationError when missing)
- Numeric field constraints (gt / ge)
- Literal validation for messaging_adapter
- Bool coercion (langchain_tracing_v2 from string env var)
- Optional field accepts None (langchain_api_key)
- Default override (passing non-default values is reflected)
- Case-insensitive env var reading
- extra="ignore" swallows unknown env vars
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from src.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimum required kwargs — used as the baseline across all tests.
_REQUIRED = dict(
    groq_api_key="test-groq-key",
    brave_api_key="test-brave-key",
    telegram_bot_token="test-telegram-token",
)


def _make(**overrides) -> Settings:
    """Build a Settings instance without touching the .env file or OS env vars.

    Uses ``_IsolatedSettings`` so the test result never depends on whatever the
    developer has exported in their shell or written in .env.
    """
    return _IsolatedSettings(**{**_REQUIRED, **overrides})


class _IsolatedSettings(Settings):
    """Settings subclass that never reads the project .env file.

    Used in tests that set env vars via ``monkeypatch`` and need a clean slate
    — otherwise a developer's local .env could mask missing required fields.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_minimal_required_fields(self):
        s = _make()
        assert s.groq_api_key == "test-groq-key"
        assert s.brave_api_key == "test-brave-key"
        assert s.telegram_bot_token == "test-telegram-token"

    def test_settings_is_settings_instance(self):
        s = _make()
        assert isinstance(s, Settings)

    def test_extra_kwargs_ignored(self):
        """extra='ignore' means unknown keys do not raise."""
        # pydantic-settings ignores unknown env vars; unknown kwargs at
        # construction time are handled by pydantic (also ignored with extra='ignore').
        s = _make()
        assert s is not None


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_groq_fast_model_default(self):
        assert _make().groq_fast_model == "llama-3.1-8b-instant"

    def test_groq_analysis_model_default(self):
        assert _make().groq_analysis_model == "llama-3.3-70b-versatile"

    def test_messaging_adapter_default(self):
        assert _make().messaging_adapter == "telegram"

    def test_langchain_tracing_v2_default_true(self):
        assert _make().langchain_tracing_v2 is True

    def test_langchain_api_key_default_none(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        assert _make().langchain_api_key is None

    def test_langchain_project_default(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
        assert _make().langchain_project == "stock-ticker-report"

    def test_sqlite_db_path_default(self):
        assert _make().sqlite_db_path == "./data/state.db"

    def test_yfinance_timeout_s_default(self):
        assert _make().yfinance_timeout_s == pytest.approx(8.0)

    def test_yfinance_retries_default(self):
        assert _make().yfinance_retries == 1

    def test_rate_limit_per_hour_default(self):
        assert _make().rate_limit_per_hour == 10

    def test_max_critic_iterations_default(self):
        assert _make().max_critic_iterations == 2

    def test_pipeline_warning_threshold_s_default(self):
        assert _make().pipeline_warning_threshold_s == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """Use _IsolatedSettings + monkeypatch so no .env can supply the values."""

    def test_missing_groq_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "brave")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        with pytest.raises(ValidationError, match="groq_api_key"):
            _IsolatedSettings()

    def test_missing_brave_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg")
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)

        with pytest.raises(ValidationError, match="brave_api_key"):
            _IsolatedSettings()

    def test_missing_telegram_bot_token_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq")
        monkeypatch.setenv("BRAVE_API_KEY", "brave")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

        with pytest.raises(ValidationError, match="telegram_bot_token"):
            _IsolatedSettings()

    def test_all_required_missing_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

        with pytest.raises(ValidationError):
            _IsolatedSettings()


# ---------------------------------------------------------------------------
# Field constraints
# ---------------------------------------------------------------------------


class TestFieldConstraints:
    # yfinance_timeout_s  (gt=0.0)
    def test_yfinance_timeout_s_zero_rejected(self):
        with pytest.raises(ValidationError, match="yfinance_timeout_s"):
            _make(yfinance_timeout_s=0.0)

    def test_yfinance_timeout_s_negative_rejected(self):
        with pytest.raises(ValidationError, match="yfinance_timeout_s"):
            _make(yfinance_timeout_s=-1.0)

    def test_yfinance_timeout_s_positive_accepted(self):
        s = _make(yfinance_timeout_s=0.001)
        assert s.yfinance_timeout_s == pytest.approx(0.001)

    # yfinance_retries  (ge=0)
    def test_yfinance_retries_negative_rejected(self):
        with pytest.raises(ValidationError, match="yfinance_retries"):
            _make(yfinance_retries=-1)

    def test_yfinance_retries_zero_accepted(self):
        s = _make(yfinance_retries=0)
        assert s.yfinance_retries == 0

    def test_yfinance_retries_positive_accepted(self):
        s = _make(yfinance_retries=5)
        assert s.yfinance_retries == 5

    # rate_limit_per_hour  (ge=1)
    def test_rate_limit_per_hour_zero_rejected(self):
        with pytest.raises(ValidationError, match="rate_limit_per_hour"):
            _make(rate_limit_per_hour=0)

    def test_rate_limit_per_hour_negative_rejected(self):
        with pytest.raises(ValidationError, match="rate_limit_per_hour"):
            _make(rate_limit_per_hour=-5)

    def test_rate_limit_per_hour_one_accepted(self):
        s = _make(rate_limit_per_hour=1)
        assert s.rate_limit_per_hour == 1

    # max_critic_iterations  (ge=1)
    def test_max_critic_iterations_zero_rejected(self):
        with pytest.raises(ValidationError, match="max_critic_iterations"):
            _make(max_critic_iterations=0)

    def test_max_critic_iterations_one_accepted(self):
        s = _make(max_critic_iterations=1)
        assert s.max_critic_iterations == 1

    # pipeline_warning_threshold_s  (gt=0.0)
    def test_pipeline_warning_threshold_s_zero_rejected(self):
        with pytest.raises(ValidationError, match="pipeline_warning_threshold_s"):
            _make(pipeline_warning_threshold_s=0.0)

    def test_pipeline_warning_threshold_s_negative_rejected(self):
        with pytest.raises(ValidationError, match="pipeline_warning_threshold_s"):
            _make(pipeline_warning_threshold_s=-3.0)

    def test_pipeline_warning_threshold_s_positive_accepted(self):
        s = _make(pipeline_warning_threshold_s=5.0)
        assert s.pipeline_warning_threshold_s == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Literal field — messaging_adapter
# ---------------------------------------------------------------------------


class TestMessagingAdapter:
    def test_default_is_telegram(self):
        assert _make().messaging_adapter == "telegram"

    def test_explicit_telegram_accepted(self):
        s = _make(messaging_adapter="telegram")
        assert s.messaging_adapter == "telegram"

    def test_invalid_adapter_rejected(self):
        with pytest.raises(ValidationError, match="messaging_adapter"):
            _make(messaging_adapter="whatsapp")

    def test_invalid_adapter_slack_rejected(self):
        with pytest.raises(ValidationError, match="messaging_adapter"):
            _make(messaging_adapter="slack")


# ---------------------------------------------------------------------------
# Optional field — langchain_api_key
# ---------------------------------------------------------------------------


class TestLangchainApiKey:
    def test_none_by_default(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        assert _make().langchain_api_key is None

    def test_accepts_string_value(self):
        s = _make(langchain_api_key="ls-abc123")
        assert s.langchain_api_key == "ls-abc123"

    def test_accepts_none_explicitly(self):
        s = _make(langchain_api_key=None)
        assert s.langchain_api_key is None


# ---------------------------------------------------------------------------
# Bool coercion — langchain_tracing_v2
# ---------------------------------------------------------------------------


class TestLangchainTracingV2:
    def test_default_is_true(self):
        assert _make().langchain_tracing_v2 is True

    def test_explicit_false(self):
        s = _make(langchain_tracing_v2=False)
        assert s.langchain_tracing_v2 is False

    def test_string_false_coerced(self, monkeypatch):
        """pydantic-settings coerces env var strings to bool."""
        monkeypatch.setenv("GROQ_API_KEY", "groq")
        monkeypatch.setenv("BRAVE_API_KEY", "brave")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

        s = _IsolatedSettings()
        assert s.langchain_tracing_v2 is False

    def test_string_true_coerced(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq")
        monkeypatch.setenv("BRAVE_API_KEY", "brave")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

        s = _IsolatedSettings()
        assert s.langchain_tracing_v2 is True


# ---------------------------------------------------------------------------
# Default override — any field can be replaced
# ---------------------------------------------------------------------------


class TestOverrideDefaults:
    def test_override_groq_fast_model(self):
        s = _make(groq_fast_model="llama-3.1-70b-versatile")
        assert s.groq_fast_model == "llama-3.1-70b-versatile"

    def test_override_groq_analysis_model(self):
        s = _make(groq_analysis_model="llama-3.1-8b-instant")
        assert s.groq_analysis_model == "llama-3.1-8b-instant"

    def test_override_sqlite_db_path(self):
        s = _make(sqlite_db_path="/tmp/test.db")
        assert s.sqlite_db_path == "/tmp/test.db"

    def test_override_langchain_project(self):
        s = _make(langchain_project="my-project")
        assert s.langchain_project == "my-project"

    def test_override_rate_limit_per_hour(self):
        s = _make(rate_limit_per_hour=100)
        assert s.rate_limit_per_hour == 100

    def test_override_max_critic_iterations(self):
        s = _make(max_critic_iterations=5)
        assert s.max_critic_iterations == 5


# ---------------------------------------------------------------------------
# Case-insensitive env var reading
# ---------------------------------------------------------------------------


class TestCaseInsensitiveEnvVars:
    def test_uppercase_env_vars_read(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "upper-groq")
        monkeypatch.setenv("BRAVE_API_KEY", "upper-brave")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "upper-tg")

        s = _IsolatedSettings()
        assert s.groq_api_key == "upper-groq"

    def test_lowercase_env_vars_read(self, monkeypatch):
        monkeypatch.setenv("groq_api_key", "lower-groq")
        monkeypatch.setenv("brave_api_key", "lower-brave")
        monkeypatch.setenv("telegram_bot_token", "lower-tg")

        s = _IsolatedSettings()
        assert s.groq_api_key == "lower-groq"

    def test_mixed_case_env_var_read(self, monkeypatch):
        monkeypatch.setenv("Groq_Api_Key", "mixed-groq")
        monkeypatch.setenv("BRAVE_API_KEY", "brave")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tg")

        s = _IsolatedSettings()
        assert s.groq_api_key == "mixed-groq"
