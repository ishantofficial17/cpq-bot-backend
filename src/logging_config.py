"""
logging_config.py — Structured JSON logging for the CPQ RAG system.

Produces machine-readable JSON logs that work with ELK, CloudWatch, Datadog,
Grafana Loki, etc.  Human-readable plain-text mode is available for local dev
by setting LOG_FORMAT=text in .env.

Usage
-----
    from src.logging_config import setup_logging
    setup_logging()          # call ONCE at application startup
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra fields attached via `log.info("...", extra={...})`
        for key in ("request_id", "session_id", "duration_ms", "step",
                     "status_code", "method", "path", "error_type"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s  %(levelname)-8s  [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )


def setup_logging():
    """
    Configure the root logger.

    Set LOG_FORMAT=text in .env for human-readable output (default for dev).
    Set LOG_FORMAT=json for structured JSON output (recommended for prod).
    Set LOG_LEVEL to DEBUG/INFO/WARNING/ERROR as needed.
    """
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub",
                   "sentence_transformers", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
