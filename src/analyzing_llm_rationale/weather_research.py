"""Read-only, source-aware evidence for prediction-market weather contracts.

The venue's settlement rule remains authoritative.  This module never turns a
forecast or observation into a settlement value; it retrieves official NWS
inputs only for contracts whose supplied rules name NWS, and describes other
official sources honestly when they are not integrated.
"""
from __future__ import annotations

import copy
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

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
_CACHE_TTL_S = max(30, min(900, int(os.environ.get("FORESEA_WEATHER_RESEARCH_CACHE_TTL_S", "300"))))
_CACHE_MAX_ENTRIES = max(1, min(128, int(os.environ.get("FORESEA_WEATHER_RESEARCH_CACHE_MAX_ENTRIES", "32"))))
_NWS_FAILURES_BEFORE_CIRCUIT = max(
    1, min(10, int(os.environ.get("FORESEA_WEATHER_RESEARCH_NWS_FAILURES_BEFORE_CIRCUIT", "3")))
)
_NWS_CIRCUIT_COOLDOWN_S = max(
    15, min(900, int(os.environ.get("FORESEA_WEATHER_RESEARCH_NWS_CIRCUIT_COOLDOWN_S", "120")))
)
_HEADERS = {
    "User-Agent": "Foresea-weather-research/1.0 (support@foresea.ink)",
    "Accept": "application/geo+json",
}
_runtime_lock = threading.RLock()
_nws_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_nws_consecutive_failures = 0
_nws_circuit_open_until = 0.0


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


_STATION_COORDINATES: Dict[str, Tuple[float, float]] = {
    "KNYC": (40.7829, -73.9654),  # New York Central Park
    "KMDW": (41.7868, -87.7522),  # Chicago Midway
    "KORD": (41.9742, -87.9073),  # Chicago O'Hare
    "KMIA": (25.7959, -80.2870),  # Miami International
    "KAUS": (30.1975, -97.6664),  # Austin Bergstrom
    "KDEN": (39.8561, -104.6737), # Denver International
    "KPHL": (39.8721, -75.2411), # Philadelphia International
    "KSFO": (37.6213, -122.3790), # San Francisco
    "KLAX": (33.9416, -118.4085), # Los Angeles
    "KBOS": (42.3656, -71.0096),  # Boston Logan
    "KDFW": (32.8998, -97.0403),  # Dallas/Fort Worth
    "KATL": (33.6407, -84.4277),  # Atlanta Hartsfield
    "KSEA": (47.4502, -122.3088), # Seattle Tacoma
    "KDCA": (38.8512, -77.0402),  # Washington Reagan
}
_OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"


def _extract_coordinates(
    geometry: Mapping[str, Any], station: Optional[str] = None
) -> Optional[Tuple[float, float]]:
    coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
    if isinstance(coordinates, list) and len(coordinates) >= 2:
        try:
            lon, lat = float(coordinates[0]), float(coordinates[1])
            return lat, lon
        except (TypeError, ValueError):
            pass
    if station and station.upper() in _STATION_COORDINATES:
        return _STATION_COORDINATES[station.upper()]
    return None


def _fetch_model_forecast(
    http_get: Callable[..., Any], lat: float, lon: float
) -> Optional[Dict[str, Any]]:
    """Fetch high-resolution global numerical / AI forecast via Open-Meteo for comparison.

    Returns projected diurnal high/low and sampled hourly curve. Returns None on network error.
    """
    try:
        url = (
            f"{_OPEN_METEO_API}?latitude={lat:.4f}&longitude={lon:.4f}&"
            f"hourly=temperature_2m&temperature_unit=fahrenheit&forecast_days=2"
        )
        response = http_get(url, timeout=_REQUEST_TIMEOUT_S)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        payload = response.json() if hasattr(response, "json") else None
        if not isinstance(payload, Mapping):
            return None
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        if not times or not temps or len(times) != len(temps):
            return None
        valid_pairs = [(t, float(temp)) for t, temp in zip(times[:24], temps[:24]) if temp is not None]
        if not valid_pairs:
            return None
        valid_temps = [p[1] for p in valid_pairs]
        return {
            "provider": "open-meteo",
            "model": "high_res_multi_model",
            "forecast_start": valid_pairs[0][0],
            "forecast_end": valid_pairs[-1][0],
            "projected_high_f": round(max(valid_temps), 1),
            "projected_low_f": round(min(valid_temps), 1),
            "hourly_curve": [
                {"time": t, "temperature_f": temp}
                for t, temp in valid_pairs[::2][:8]
            ],
            "notice": (
                "Predictive multi-model numerical/AI forecast reference. "
                "Official settlement is governed strictly by the contract's named settlement source."
            ),
        }
    except Exception as exc:
        logger.debug("Model forecast fetch unavailable: %s", exc)
        return None


def _forecast_periods(
    http_get: Callable[..., Any], geometry: Mapping[str, Any], station: Optional[str] = None
) -> List[Dict[str, Any]]:
    coords = _extract_coordinates(geometry, station)
    if not coords:
        return []
    lat, lon = coords
    try:
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
    except Exception as exc:
        logger.debug("NWS forecast periods unavailable: %s", exc)
        return []


def _cached_nws_research(station: str) -> Optional[Dict[str, Any]]:
    now = time.monotonic()
    with _runtime_lock:
        entry = _nws_cache.get(station)
        if entry is None or entry[0] <= now:
            if entry is not None:
                _nws_cache.pop(station, None)
            return None
        result = copy.deepcopy(entry[1])
    result["research_cached"] = True
    return result


def _cache_nws_research(station: str, result: Dict[str, Any]) -> None:
    with _runtime_lock:
        if len(_nws_cache) >= _CACHE_MAX_ENTRIES:
            oldest = min(_nws_cache, key=lambda key: _nws_cache[key][0])
            _nws_cache.pop(oldest, None)
        stored = copy.deepcopy(result)
        stored["research_cached"] = False
        _nws_cache[station] = (time.monotonic() + _CACHE_TTL_S, stored)


def _nws_circuit_is_open() -> bool:
    with _runtime_lock:
        return _nws_circuit_open_until > time.monotonic()


def _record_nws_success() -> None:
    global _nws_consecutive_failures, _nws_circuit_open_until
    with _runtime_lock:
        _nws_consecutive_failures = 0
        _nws_circuit_open_until = 0.0


def _record_nws_failure() -> bool:
    """Return whether this failure has opened the bounded NWS circuit."""
    global _nws_consecutive_failures, _nws_circuit_open_until
    with _runtime_lock:
        _nws_consecutive_failures += 1
        if _nws_consecutive_failures >= _NWS_FAILURES_BEFORE_CIRCUIT:
            _nws_circuit_open_until = time.monotonic() + _NWS_CIRCUIT_COOLDOWN_S
        return _nws_circuit_open_until > time.monotonic()


def _reset_runtime_state_for_test() -> None:
    """Reset only in-process cache/circuit state for deterministic unit tests."""
    global _nws_consecutive_failures, _nws_circuit_open_until
    with _runtime_lock:
        _nws_cache.clear()
        _nws_consecutive_failures = 0
        _nws_circuit_open_until = 0.0


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
                    "model_forecast": None,
                    "notice": "This contract is not classified as weather.",
                }
                outcome = "not_applicable"
            elif brief.settlement_source == "weather_company":
                result = {
                    "weather_market": brief.as_dict(),
                    "source_status": "official_source_not_integrated",
                    "observations": [],
                    "forecast_periods": [],
                    "model_forecast": None,
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
                    "model_forecast": None,
                    "notice": "No source-matched NWS lookup was performed; the contract data is incomplete for this research pass.",
                }
                outcome = reason
            else:
                cached = _cached_nws_research(brief.station)
                if cached is not None:
                    span.set_attributes({"weather.source_status": cached["source_status"], "outcome": "cache_hit"})
                    weather_research_requests.add(1, {
                        "settlement_source": brief.settlement_source,
                        "outcome": "cache_hit",
                    })
                    return cached
                if _nws_circuit_is_open():
                    result = {
                        "weather_market": brief.as_dict(),
                        "source_status": "nws_circuit_open",
                        "observations": [],
                        "forecast_periods": [],
                        "model_forecast": None,
                        "research_cached": False,
                        "notice": "NWS research is temporarily paused after repeated upstream failures; retry later.",
                    }
                    outcome = "circuit_open"
                    span.set_attributes({"weather.source_status": result["source_status"], "outcome": outcome})
                    weather_research_requests.add(1, {
                        "settlement_source": brief.settlement_source,
                        "outcome": outcome,
                    })
                    return result
                if http_get is None:
                    import requests
                    http_get = requests.get
                try:
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
                    periods = _forecast_periods(http_get, geometry if isinstance(geometry, Mapping) else {}, brief.station)
                    coords = _extract_coordinates(geometry if isinstance(geometry, Mapping) else {}, brief.station)
                    model_forecast = (
                        _fetch_model_forecast(http_get, coords[0], coords[1])
                        if coords
                        else None
                    )
                    result = {
                        "weather_market": brief.as_dict(),
                        "source_status": "nws_observation_available",
                        "observations": [observation],
                        "forecast_periods": periods,
                        "model_forecast": model_forecast,
                        "research_cached": False,
                        "notice": (
                            "NWS observations and forecast are source-matched research inputs. The final NWS Daily "
                            "Climate Report named in the contract remains the settlement authority."
                        ),
                    }
                    _record_nws_success()
                    _cache_nws_research(brief.station, result)
                    outcome = "available"
                except Exception as exc:
                    circuit_opened = _record_nws_failure()
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR))
                    result = {
                        "weather_market": brief.as_dict(),
                        "source_status": "nws_temporarily_unavailable",
                        "observations": [],
                        "forecast_periods": [],
                        "model_forecast": None,
                        "research_cached": False,
                        "notice": (
                            "NWS source-matched research is temporarily unavailable; no proxy data was used. "
                            "The contract's final NWS Daily Climate Report remains authoritative."
                        ),
                    }
                    outcome = "circuit_opened" if circuit_opened else "upstream_failure"
                    logger.warning("NWS weather market research unavailable", exc_info=True)
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
