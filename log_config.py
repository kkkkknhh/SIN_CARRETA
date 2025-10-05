"""Centralised logging configuration for the MINIMINIMOON project."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for log records."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03d"

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """Return the record as a JSON string."""
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_record["stack"] = self.formatStack(record.stack_info)

        return json.dumps(log_record, ensure_ascii=False)


def configure_logging(
    log_level: Optional[str] = None,
    json_format: Optional[bool] = None,
) -> None:
    """Configure logging once for the entire application.

    Respects LOG_LEVEL and LOG_FORMAT env vars unless explicit args are given.
    Subsequent calls update handler levels/formatters without duplicating handlers.
    """
    level_string = log_level or os.getenv("LOG_LEVEL", "INFO")
    level = get_log_level(level_string)

    json_requested = (
        json_format
        if json_format is not None
        else os.getenv("LOG_FORMAT", "").strip().lower() == "json"
    )

    formatter: logging.Formatter = (
        JsonFormatter()
        if json_requested
        else logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)

    root_logger.setLevel(level)

    LOGGER.debug(
        "Logging configured with level: %s (%s), json_format=%s",
        logging.getLevelName(level),
        level,
        json_requested,
    )

    if level_string.upper() != logging.getLevelName(level):
        LOGGER.warning(
            "Invalid LOG_LEVEL value '%s' - falling back to %s",
            level_string,
            logging.getLevelName(level),
        )


def get_log_level(level_string: str) -> int:
    """Map a log-level string to a logging module constant, defaulting to INFO."""
    level_string = level_string.upper()
    valid_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,  # alias
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,  # alias
    }
    return valid_levels.get(level_string, logging.INFO)


# Configure logging on module import
configure_logging()
