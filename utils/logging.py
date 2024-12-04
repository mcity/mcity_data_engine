import logging
import os
from datetime import datetime

from colorlog import ColoredFormatter


def configure_logging():
    """
    Configures the logging for the application.

    This function sets up logging to both the console and a file. Console logs
    are color-coded based on the log level, while file logs are plain text.
    Log files are stored in a "logs" directory, and each log file is named
    with a timestamp to ensure uniqueness.

    The log format includes the timestamp, log level, and the log message.

    Log levels and their corresponding colors:
    - DEBUG: cyan
    - INFO: default
    - WARNING: yellow
    - ERROR: red
    - CRITICAL: red with white background

    If the "logs" directory does not exist, it will be created.

    Raises:
        OSError: If the directory creation fails.
    """
    os.makedirs("logs/dataengine", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/dataengine/{timestamp}_dataengine.log"

    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
        handlers=[
            handler,  # Log to console with colors
            logging.FileHandler(log_filename),
        ],
        force=True
    )