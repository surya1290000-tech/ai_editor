"""
core/logger.py

Centralised logging setup using loguru.
All modules import get_logger() from here.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: Path | None = None, verbose: bool = False) -> None:
    """
    Configure loguru for the application.
    Called once at application startup.

    Args:
        log_dir: Directory to write log files. If None, no file logging.
        verbose: If True, set console level to DEBUG.
    """
    # Remove the default handler
    logger.remove()

    console_level = "DEBUG" if verbose else "INFO"

    # Console handler — colourised, human-readable
    logger.add(
        sys.stderr,
        level=console_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=verbose,
    )

    # File handler — full detail, JSON-friendly for debugging
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "agent1_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{line} | {message}",
            rotation="50 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )


def get_logger(name: str):
    """
    Return a loguru logger bound to the given module name.
    Usage: logger = get_logger(__name__)
    """
    return logger.bind(name=name)
