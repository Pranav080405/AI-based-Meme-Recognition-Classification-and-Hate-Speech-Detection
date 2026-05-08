"""
utils/logger.py — Centralized logging with loguru
"""

import os
import sys
from loguru import logger
from config import LOG_DIR, LOG_FILE, LOG_LEVEL


def setup_logger() -> None:
    """Configure loguru for file + console output."""
    os.makedirs(LOG_DIR, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler — human-friendly
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — full detail, rotation daily
    logger.add(
        LOG_FILE,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        rotation="1 day",
        retention="7 days",
        compression="zip",
    )


# Auto-setup on import
setup_logger()

__all__ = ["logger"]