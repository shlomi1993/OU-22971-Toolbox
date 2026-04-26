"""
Centralized logging configuration for the Ray-based TLC replay system.
"""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels.
    """
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    LEVEL_NAMES = {
        "DEBUG": "DEBUG",
        "INFO": "INFO ",
        "WARNING": "WARN ",
        "ERROR": "ERROR",
        "CRITICAL": "CRIT ",
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with date, colored level, and message.

        Args:
            record (logging.LogRecord): LogRecord instance, typically passed by the logging framework.

        Returns:
            str: Formatted log message
        """
        level_name = self.LEVEL_NAMES.get(record.levelname, record.levelname)
        level_color = self.COLORS.get(record.levelname, "")
        colored_level = f"{level_color}{level_name}{self.RESET}"
        timestamp = self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S")
        message = record.getMessage()
        log = f"{timestamp} :: {colored_level} :: {message}"

        # Add exception info if present
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            log = f"{log}\n{record.exc_text}"

        return log  # Example: "2024-06-01 12:00:00 :: INFO  :: This is an info message"


# Configure and export a single logger instance
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(ColoredFormatter())
logger.addHandler(handler)
