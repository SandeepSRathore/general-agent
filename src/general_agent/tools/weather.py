"""Weather tool backed by Open-Meteo (free, no API key): geocode the place, then fetch
current conditions plus a short daily forecast."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from langchain_core.tools import tool

from general_agent.config import get_settings

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes (subset used by Open-Meteo).
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}  # fmt: skip


FORECAST_DAYS = 3

# Small models pass placeholders when they don't know the place; geocoding them can match
# real towns (there is a place called "Unknown"), producing confidently wrong weather.
_PLACEHOLDER = re.compile(
    r"^\s*((the\s+)?(unknown|n/?a|none|null|tbd|here|current|my|user'?s?|your)"
    r"(\s+(city|location|place|town|area))?|where\s+i\s+live|location|city)?\s*$",
    re.IGNORECASE,
)


class WeatherError(RuntimeError):
    pass


def _client() -> httpx.Client:
    # transport retries cover connection errors; timeout bounds every request.
    return httpx.Client(
        timeout=get_settings().http_timeout_seconds,
        transport=httpx.HTTPTransport(retries=2),
    )


def fetch_weather(location: str, client: httpx.Client | None = None) -> str:
    owns_client = client is None
    client = client or _client()
    try:
        place = _geocode(client, location)

        forecast = client.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max",
                "timezone": "auto",
                # Ask for 4 days: around local midnight Open-Meteo's daily series can
                # still start at yesterday, which is filtered out in _format.
                "forecast_days": FORECAST_DAYS + 1,
            },
        )
        forecast.raise_for_status()
        return _format(place, forecast.json())
    except httpx.HTTPError as exc:
        logger.warning("weather lookup failed: %s", exc)
        raise WeatherError(f"Weather service unavailable: {exc.__class__.__name__}") from exc
    finally:
        if owns_client:
            client.close()


def _geocode(client: httpx.Client, location: str) -> dict[str, Any]:
    # Open-Meteo matches place *names* only, so "Paris, France" finds nothing; fall back
    # to the part before the first comma.
    for name in dict.fromkeys([location.strip(), location.split(",")[0].strip()]):
        geo = client.get(GEOCODE_URL, params={"name": name, "count": 1, "format": "json"})
        geo.raise_for_status()
        if results := geo.json().get("results"):
            return dict(results[0])
    raise WeatherError(f"Could not find a place called '{location}'.")


def _format(place: dict[str, Any], data: dict[str, Any]) -> str:
    name = ", ".join(
        str(p) for p in (place.get("name"), place.get("admin1"), place.get("country")) if p
    )
    cur = data["current"]
    units = data.get("current_units", {})
    lines = [
        f"Weather for {name} (local time {cur.get('time')}):",
        f"- Now: {WEATHER_CODES.get(cur.get('weather_code'), 'unknown')}, "
        f"{cur.get('temperature_2m')}{units.get('temperature_2m', '°C')} "
        f"(feels like {cur.get('apparent_temperature')}{units.get('apparent_temperature', '°C')}),"
        f" humidity {cur.get('relative_humidity_2m')}%, "
        f"wind {cur.get('wind_speed_10m')} {units.get('wind_speed_10m', 'km/h')}",
    ]
    daily = data.get("daily", {})
    today = str(cur.get("time", ""))[:10]  # local date, e.g. "2026-10-15"
    labels = ["today", "tomorrow"]
    upcoming = [(i, day) for i, day in enumerate(daily.get("time", [])) if day >= today]
    for n, (i, day) in enumerate(upcoming[:FORECAST_DAYS]):
        label = f" ({labels[n]})" if n < len(labels) else ""
        lines.append(
            f"- {day}{label}: {WEATHER_CODES.get(daily['weather_code'][i], 'unknown')}, "
            f"{daily['temperature_2m_min'][i]} to {daily['temperature_2m_max'][i]}°C, "
            f"rain chance {daily['precipitation_probability_max'][i]}%"
        )
    return "\n".join(lines)


@tool
def get_weather(location: str) -> str:
    """Get current weather and a 3-day forecast for a real city or place name, e.g. 'Pune'
    or 'Paris, France'. Never pass a placeholder; if the place is unknown, use ask_user."""
    if _PLACEHOLDER.match(location):
        return (
            f"Error: '{location}' is not a real place name. Do not guess; call ask_user to ask "
            "the user which city they mean."
        )
    try:
        return fetch_weather(location)
    except WeatherError as exc:
        return f"Error: {exc}"
