"""Central logging configuration (Rich handler for readable terminal logs)."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(level: str = "WARNING") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(name)s: %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    # Third-party HTTP clients are noisy at INFO.
    for noisy in ("httpx", "httpcore", "openai", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
