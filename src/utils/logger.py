import logging
from logging.handlers import RotatingFileHandler
from config.config import settings
from config.configuration import LOG_LEVEL
from config.constants import LOG_FILE_NAME, LOG_FORMAT

def get_logger(name: str = "MedicalAIPlatform") -> logging.Logger:
    logger = logging.getLogger(name)

    # hasHandlers() also checks the root logger. Streamlit and tensorflow add
    # handlers there, so it returned True and our file handler was never added.
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # without this every line is printed twice when the root logger has a handler
    logger.propagate = False

    log_file = settings.logs_dir / LOG_FILE_NAME
    formatter = logging.Formatter(LOG_FORMAT)

    # rotate the file so the log does not grow forever
    file_handler = RotatingFileHandler(
        log_file, maxBytes = 5 * 1024 * 1024, backupCount = 3, encoding = "utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = get_logger()
