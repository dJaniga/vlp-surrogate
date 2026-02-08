from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import tomllib

_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
}


class ExtraFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_KEYS and not key.startswith("_")
        }
        if extras:
            extra_pairs = " ".join(f"{key}={value}" for key, value in extras.items())
            record.extra = f": {extra_pairs}"
        else:
            record.extra = ""
        return super().format(record)


def setup_logging() -> None:
    config_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    logging_config = config.get("tool", {}).get("vlp_surrogate", {}).get("logging")
    if not logging_config:
        logging.basicConfig(level=logging.INFO)
        return
    logging.config.dictConfig(logging_config)
    logging.getLogger(__name__).info(
        "Logging configured", extra={"config": "pyproject"}
    )
