"""Logging centralizzato per FractalNova."""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str = "fractalnova") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root = logging.getLogger("fractalnova")
        root.addHandler(handler)
        root.setLevel(os.getenv("FRACTALNOVA_LOG_LEVEL", "INFO").upper())
        root.propagate = False
        _CONFIGURED = True
    return logger
