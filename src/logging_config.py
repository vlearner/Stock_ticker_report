"""Application-wide logging configuration.

Call :func:`configure_logging` once at startup (before any loggers are used).
It selects between human-readable text output and structured JSON output
based on the ``JSON_LOGS`` environment variable or ``settings.json_logs``.

JSON mode requires ``python-json-logger`` (listed in requirements.txt).
Falls back to text logging gracefully if the package is missing.
"""

from __future__ import annotations

import logging


def configure_logging(json_logs: bool | None = None) -> None:
    """Configure the root logger.

    Args:
        json_logs: Override flag.  When ``None``, reads ``settings.json_logs``.
            Pass ``True`` to force JSON, ``False`` to force text.
    """
    if json_logs is None:
        try:
            from src.config import settings as _settings

            json_logs = _settings.json_logs
        except Exception:
            json_logs = False

    if json_logs:
        try:
            from pythonjsonlogger.json import JsonFormatter  # type: ignore[import-untyped]

            handler = logging.StreamHandler()
            handler.setFormatter(
                JsonFormatter(
                    fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                    rename_fields={
                        "asctime": "ts",
                        "levelname": "level",
                        "name": "logger",
                    },
                )
            )
            root = logging.getLogger()
            root.handlers.clear()
            root.addHandler(handler)
            root.setLevel(logging.INFO)
            return
        except ImportError:
            logging.getLogger(__name__).warning(
                "python-json-logger not installed — falling back to text logging"
            )

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        level=logging.INFO,
    )
