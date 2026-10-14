"""Current date/time tool (timezone-aware via the IANA tz database)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


def now_string(timezone: str = "local") -> str:
    if timezone.strip().lower() in ("", "local"):
        now = datetime.now().astimezone()
    else:
        now = datetime.now(ZoneInfo(timezone.strip()))
    return now.strftime("%A, %d %B %Y, %H:%M:%S %Z (UTC%z)")


@tool
def get_current_time(timezone: str = "local") -> str:
    """Get the current date and time. `timezone` is an IANA name such as 'Asia/Tokyo',
    'Europe/London', 'America/New_York', 'Asia/Kolkata', or 'local' for this machine."""
    try:
        return now_string(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return (
            f"Error: unknown timezone '{timezone}'. Use an IANA name like 'Asia/Tokyo' "
            "(Region/City)."
        )
