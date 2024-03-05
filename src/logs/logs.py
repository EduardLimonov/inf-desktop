import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from settings import settings


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s-[%(levelname)s]-%(name)s-(%(filename)s).%(funcName)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    default_handler = logging.StreamHandler(sys.stdout)
    default_handler.setLevel(logging.INFO)
    default_handler.setFormatter(formatter)
    logger.addHandler(default_handler)

    if settings.WITH_FILE_LOGGER:
        os.makedirs(os.path.dirname(settings.FILE_LOGGER_PATH), exist_ok=True)
        file_handler = TimedRotatingFileHandler(settings.FILE_LOGGER_PATH, when="D", backupCount=5, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger():
    return setup_logger("main")


logger = get_logger()
