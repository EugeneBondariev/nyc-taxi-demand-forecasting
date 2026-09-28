import logging

import colorlog

from .context import correlation_id_var


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get("-")  # type: ignore[attr-defined]
        return True


def setup_logging() -> None:
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(correlation_id)s] %(name)s %(levelname)s%(reset)s %(message)s"
        )
    )
    handler.addFilter(CorrelationIdFilter())
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(handler)
