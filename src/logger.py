import logging
import colorlog


def setup_logging():
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(name)s %(levelname)s%(reset)s %(message)s"
        )
    )
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(handler)
