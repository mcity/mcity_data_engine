import logging
import os
from datetime import datetime
from logging import StreamHandler

from colorlog import ColoredFormatter


def configure_logging():
    """Configures the logging for the application."""
    os.makedirs("logs/dataengine", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/dataengine/{timestamp}_dataengine.log"

    colored_formatter = ColoredFormatter(
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

    # Set pymongo logger to only show WARN and above
    logging.getLogger("pymongo").setLevel(logging.WARNING)

    # Create console handler with INFO level
    console_handler = StreamHandler()  # Changed from ColoredFormatter to StreamHandler
    console_handler.setFormatter(colored_formatter)
    console_handler.setLevel(logging.INFO)

    # Create file handler with DEBUG level
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG,  # Set root logger to DEBUG to capture all logs
        format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
        handlers=[
            console_handler,  # Log INFO and above to console with colors
            file_handler,  # Log DEBUG and above to file
        ],
        force=True,
    )
