"""Lightweight module-level logger factory (no external dependencies)."""
from __future__ import annotations

import logging

_CONFIGURED: dict = {}


def get_logger(name: str = "helix") -> logging.Logger:
    """Return a logger that emits to stderr with a consistent format.

    The handler is only attached once per logger name to avoid duplicate
    log lines when called repeatedly.
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    _CONFIGURED[name] = True
    return logger
