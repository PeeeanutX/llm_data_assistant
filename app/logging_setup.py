"""Centralised logging configuration.

Keeping this in one place means every module just calls
``logging.getLogger(__name__)`` and gets a sensible, consistent format.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger exactly once.

    Streamlit reruns the script on every interaction, so we guard against
    installing duplicate handlers.
    """
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    # Replace any pre-existing handlers so format is consistent.
    root.handlers = [handler]

    # Tame noisy third-party loggers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # transformers spams a "Accessing `__path__` from ..." deprecation warning
    # for every image-processing submodule when Streamlit's file watcher walks
    # loaded modules. These are harmless; mute everything below ERROR.
    logging.getLogger("transformers").setLevel(logging.ERROR)

    _configured = True
