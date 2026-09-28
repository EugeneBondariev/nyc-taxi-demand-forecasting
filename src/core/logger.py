import logging
import logging.handlers
from pathlib import Path

import colorlog

from .context import correlation_id_var

_LOG_FORMAT = "%(asctime)s [%(correlation_id)s] %(name)s %(levelname)s %(message)s"
_LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get("-")  # type: ignore[attr-defined]
        return True


def setup_logging(log_file: str = "app.log") -> None:
    cid_filter = CorrelationIdFilter()

    console = colorlog.StreamHandler()
    console.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(correlation_id)s] %(name)s %(levelname)s%(reset)s %(message)s"
        )
    )
    console.addFilter(cid_filter)

    _LOGS_DIR.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        _LOGS_DIR / log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    file_handler.addFilter(cid_filter)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(console)
    logging.root.addHandler(file_handler)
