"""
utils/logger.py – Enterprise-grade rotating file + console logger factory.

Usage::

    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello from %s", __name__)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import get_settings

settings = get_settings()

# ── Formatter ─────────────────────────────────────────────────────────────────

_CONSOLE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | "
    "%(funcName)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _ensure_log_dir() -> Path:
    """Create the log directory if it does not exist and return its path."""
    log_dir = settings.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with both console and rotating-file handlers.

    Calling this function multiple times with the same *name* returns the
    same logger (idempotent – Python's logging module guarantees this).

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A fully configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the module is reimported
    if logger.handlers:
        return logger

    numeric_level = logging.getLevelName(settings.log_level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logger.setLevel(numeric_level)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(
        logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)
    )
    logger.addHandler(console_handler)

    # ── Rotating file handler ─────────────────────────────────────────────────
    log_dir = _ensure_log_dir()
    log_file = log_dir / "face_service.log"
    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(
        logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
    )
    logger.addHandler(file_handler)

    # Prevent log records from being passed to the root logger
    logger.propagate = False

    return logger
