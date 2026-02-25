from __future__ import annotations

import logging
import logging.config
import tomllib
from pathlib import Path

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
    config_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    logging_config = config["tool"]["vlp_surrogate"]["logging"]

    Path("logs").mkdir(parents=True, exist_ok=True)

    try:
        logging.config.dictConfig(logging_config)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise

    logging.getLogger(__name__).info(
        "Logging configured", extra={"config": "pyproject"}
    )
