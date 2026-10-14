"""Weather tool backed by Open-Meteo (free, no API key): geocode the place, then fetch
current conditions plus a short daily forecast."""

from __future__ import annotations

import logging
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
        geo = client.get(GEOCODE_URL, params={"name": location, "count": 1, "format": "json"})
        geo.raise_for_status()
        results = geo.json().get("results") or []
        if not results:
            raise WeatherError(f"Could not find a place called '{location}'.")
        place: dict[str, Any] = results[0]

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
                "forecast_days": 3,
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


def _format(place: dict[str, Any], data: dict[str, Any]) -> str:
    name = ", ".join(str(p) for p in (place.get("name"), place.get("admin1"), place.get("country"))
                     if p)
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
    for i, day in enumerate(daily.get("time", [])):
        lines.append(
            f"- {day}: {WEATHER_CODES.get(daily['weather_code'][i], 'unknown')}, "
            f"{daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]}°C, "
            f"rain chance {daily['precipitation_probability_max'][i]}%"
        )
    return "\n".join(lines)


@tool
def get_weather(location: str) -> str:
    """Get current weather and a 3-day forecast for a city or place name, e.g. 'Pune' or
    'Paris, France'."""
    try:
        return fetch_weather(location)
    except WeatherError as exc:
        return f"Error: {exc}"
