"""Central logging configuration (Rich handler for readable terminal logs)."""

from __future__ import annotations

import logging
import warnings

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
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Known upstream noise: langchain-openai's structured output stores the parsed Pydantic
    # object in a field typed `None`, which makes Pydantic warn on every call. Harmless.
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
