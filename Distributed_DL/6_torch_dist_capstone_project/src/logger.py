"""
Centralized logging configuration for the distributed SimCLR training system.
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
            record (logging.LogRecord): LogRecord instance.

        Returns:
            str: Formatted log message.
        """
        timestamp = self.formatTime(record, datefmt="%Y-%m-%d %H:%M:%S")
        level_name = self.LEVEL_NAMES.get(record.levelname, record.levelname)
        level_color = self.COLORS.get(record.levelname, "")
        colored_level = f"{level_color}{level_name}{self.RESET}"
        message = record.getMessage()
        log = f"{timestamp}  {colored_level} {message}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            log = f"{log}\n{record.exc_text}"

        return log


# Configure and export a single logger instance
g_logger = logging.getLogger()
g_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(ColoredFormatter())
g_logger.addHandler(handler)
