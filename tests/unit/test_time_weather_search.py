from typing import Any

import httpx
import pytest
import respx

from general_agent.tools import search_tools as web_search_module
from general_agent.tools.time_tools import get_current_time
from general_agent.tools.weather import FORECAST_URL, GEOCODE_URL, get_weather


def test_time_known_timezone() -> None:
    out = get_current_time.invoke({"timezone": "Asia/Tokyo"})
    assert "JST" in out


def test_time_local_default() -> None:
    assert "UTC" in get_current_time.invoke({})


def test_time_unknown_timezone_returns_error() -> None:
    assert get_current_time.invoke({"timezone": "Mars/Base"}).startswith("Error")


FORECAST_JSON = {
    "current_units": {"temperature_2m": "°C", "apparent_temperature": "°C",
                      "wind_speed_10m": "km/h"},
    "current": {"time": "2026-10-14T12:00", "temperature_2m": 29.1, "apparent_temperature": 31.0,
                "relative_humidity_2m": 60, "weather_code": 2, "wind_speed_10m": 11.2},
    "daily": {"time": ["2026-10-14"], "weather_code": [61], "temperature_2m_max": [30.0],
              "temperature_2m_min": [21.0], "precipitation_probability_max": [40]},
}  # fmt: skip


@respx.mock
def test_weather_happy_path() -> None:
    respx.get(GEOCODE_URL).mock(
        return_value=httpx.Response(200, json={"results": [
            {"name": "Pune", "admin1": "Maharashtra", "country": "India",
             "latitude": 18.5, "longitude": 73.8}]})  # fmt: skip
    )
    respx.get(FORECAST_URL).mock(return_value=httpx.Response(200, json=FORECAST_JSON))
    out = get_weather.invoke({"location": "Pune"})
    assert "Pune, Maharashtra, India" in out
    assert "partly cloudy" in out
    assert "slight rain" in out


@respx.mock
def test_weather_unknown_place() -> None:
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json={}))
    assert "Could not find" in get_weather.invoke({"location": "Nowhereville"})


@respx.mock
def test_weather_service_down() -> None:
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(503))
    assert get_weather.invoke({"location": "Pune"}).startswith("Error: Weather service")


def test_web_search_formats_and_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDDGS:
        def __init__(self, **_: Any) -> None: ...

        def text(self, query: str, max_results: int) -> list[dict[str, str]]:
            if query == "boom":
                raise RuntimeError("rate limited")
            return [{"title": "LangGraph", "href": "https://x.dev", "body": "Graphs for agents"}]

    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    out = web_search_module.web_search.invoke({"query": "langgraph"})
    assert "https://x.dev" in out and "<untrusted_data>" in out
    assert web_search_module.web_search.invoke({"query": "boom"}).startswith("Error")
