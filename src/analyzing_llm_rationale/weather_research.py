"""Read-only, source-aware evidence for prediction-market weather contracts.

The venue's settlement rule remains authoritative.  This module never turns a
forecast or observation into a settlement value; it retrieves official NWS
inputs only for contracts whose supplied rules name NWS, and describes other
official sources honestly when they are not integrated.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from analyzing_llm_rationale.weather_markets import classify_weather_market

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("foresea.weather_research")
meter = metrics.get_meter("foresea.weather_research")

weather_research_requests = meter.create_counter(
    "weather_research.requests",
    unit="1",
    description="Read-only weather market research requests by source and outcome",
)
weather_research_duration = meter.create_histogram(
    "weather_research.duration",
    unit="s",
    description="Duration of source-aware weather market research",
)

_NWS_API = "https://api.weather.gov"
_REQUEST_TIMEOUT_S = 8
_HEADERS = {
    "User-Agent": "Foresea-weather-research/1.0 (support@foresea.ink)",
    "Accept": "application/geo+json",
}


def _value(props: Mapping[str, Any], name: str) -> Optional[float]:
    raw = props.get(name)
    if isinstance(raw, Mapping):
        raw = raw.get("value")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _celsius_to_fahrenheit(value: Optional[float]) -> Optional[float]:
    return round((value * 9 / 5) + 32, 1) if value is not None else None


def _json_get(http_get: Callable[..., Any], url: str) -> Mapping[str, Any]:
    response = http_get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT_S)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("weather source returned an invalid payload")
    return payload


def _forecast_periods(http_get: Callable[..., Any], geometry: Mapping[str, Any]) -> List[Dict[str, Any]]:
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return []
    try:
        lon, lat = float(coordinates[0]), float(coordinates[1])
    except (TypeError, ValueError):
        return []
    point = _json_get(http_get, f"{_NWS_API}/points/{lat:.4f},{lon:.4f}")
    hourly_url = (point.get("properties") or {}).get("forecastHourly")
    if not isinstance(hourly_url, str) or not hourly_url.startswith("https://"):
        return []
    hourly = _json_get(http_get, hourly_url)
    periods = (hourly.get("properties") or {}).get("periods") or []
    result: List[Dict[str, Any]] = []
    for period in periods[:8]:
        if not isinstance(period, Mapping):
            continue
        precip = period.get("probabilityOfPrecipitation") or {}
        result.append({
            "start_time": period.get("startTime"),
            "temperature_f": period.get("temperature"),
            "short_forecast": period.get("shortForecast"),
            "precipitation_probability": precip.get("value") if isinstance(precip, Mapping) else None,
        })
    return result


def research_weather_market(
    quote: Mapping[str, Any],
    *,
    http_get: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Return bounded weather evidence with explicit source authority.

    For an NWS-settled contract with a named ICAO station, the output includes
    the current NWS station observation and nearby NWS hourly forecast.  The
    daily climate report remains the settlement authority and can differ from
    preliminary observations.  Weather Company contracts are not queried
    because Foresea has no licensed Weather Company data-feed integration.
    """
    started = time.perf_counter()
    with tracer.start_as_current_span("weather_research.market") as span:
        try:
            brief = classify_weather_market(quote)
            span.set_attributes({
                "weather.is_weather": brief.is_weather,
                "weather.market_type": brief.market_type,
                "weather.settlement_source": brief.settlement_source,
            })
            if not brief.is_weather:
                result = {
                    "weather_market": brief.as_dict(),
                    "source_status": "not_applicable",
                    "observations": [],
                    "forecast_periods": [],
                    "notice": "This contract is not classified as weather.",
                }
                outcome = "not_applicable"
            elif brief.settlement_source == "weather_company":
                result = {
                    "weather_market": brief.as_dict(),
                    "source_status": "official_source_not_integrated",
                    "observations": [],
                    "forecast_periods": [],
                    "notice": (
                        "The Weather Company is the named settlement source. Foresea has no licensed "
                        "Weather Company feed, so no proxy is represented as official settlement data."
                    ),
                }
                outcome = "not_integrated"
            elif brief.settlement_source != "nws_daily_climate_report" or not brief.station:
                reason = "missing_station" if brief.settlement_source == "nws_daily_climate_report" else "unverified_source"
                result = {
                    "weather_market": brief.as_dict(),
                    "source_status": reason,
                    "observations": [],
                    "forecast_periods": [],
                    "notice": "No source-matched NWS lookup was performed; the contract data is incomplete for this research pass.",
                }
                outcome = reason
            else:
                if http_get is None:
                    import requests
                    http_get = requests.get
                payload = _json_get(http_get, f"{_NWS_API}/stations/{brief.station}/observations/latest")
                props = payload.get("properties") or {}
                geometry = payload.get("geometry") or {}
                observation = {
                    "source": "NWS station observation",
                    "station": brief.station,
                    "timestamp": props.get("timestamp"),
                    "temperature_f": _celsius_to_fahrenheit(_value(props, "temperature")),
                    "dewpoint_f": _celsius_to_fahrenheit(_value(props, "dewpoint")),
                    "wind_speed_mps": _value(props, "windSpeed"),
                    "precipitation_last_hour_mm": _value(props, "precipitationLastHour"),
                    "authority": "preliminary_observation_not_final_daily_settlement",
                }
                periods = _forecast_periods(http_get, geometry if isinstance(geometry, Mapping) else {})
                result = {
                    "weather_market": brief.as_dict(),
                    "source_status": "nws_observation_available",
                    "observations": [observation],
                    "forecast_periods": periods,
                    "notice": (
                        "NWS observations and forecast are source-matched research inputs. The final NWS Daily "
                        "Climate Report named in the contract remains the settlement authority."
                    ),
                }
                outcome = "available"
            span.set_attributes({"weather.source_status": result["source_status"], "outcome": outcome})
            weather_research_requests.add(1, {
                "settlement_source": brief.settlement_source,
                "outcome": outcome,
            })
            logger.info(
                "weather market research source=%s status=%s",
                brief.settlement_source,
                result["source_status"],
            )
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            weather_research_requests.add(1, {"settlement_source": "unknown", "outcome": "failure"})
            logger.warning("weather market research failed", exc_info=True)
            raise
        finally:
            weather_research_duration.record(time.perf_counter() - started)
