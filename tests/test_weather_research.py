from __future__ import annotations

import os
import unittest

from analyzing_llm_rationale.weather_research import (
    calculate_bracket_probability,
    parse_market_strike,
    research_weather_market,
)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class WeatherResearchTests(unittest.TestCase):
    def setUp(self):
        from analyzing_llm_rationale import weather_research

        weather_research._reset_runtime_state_for_test()

    def tearDown(self):
        from analyzing_llm_rationale import weather_research

        weather_research._reset_runtime_state_for_test()

    def test_nws_contract_returns_observation_and_hourly_forecast(self):
        quote = {
            "question": "What will the highest temperature in Chicago be today?",
            "category": "Weather",
            "resolution_criteria": "NWS Daily Climate Report, station KORD.",
        }
        responses = {
            "https://api.weather.gov/stations/KORD/observations/latest": {
                "properties": {
                    "timestamp": "2026-08-29T10:00:00+00:00",
                    "temperature": {"value": 20.0},
                    "dewpoint": {"value": 10.0},
                    "windSpeed": {"value": 4.0},
                    "precipitationLastHour": {"value": 0.0},
                },
                "geometry": {"coordinates": [-87.9, 41.9]},
            },
            "https://api.weather.gov/points/41.9000,-87.9000": {
                "properties": {"forecastHourly": "https://example.test/hourly"},
            },
            "https://example.test/hourly": {
                "properties": {"periods": [{
                    "startTime": "2026-08-29T11:00:00+00:00",
                    "temperature": 69,
                    "shortForecast": "Partly Cloudy",
                    "probabilityOfPrecipitation": {"value": 10},
                }]},
            },
        }

        def get(url, **_kwargs):
            return _Response(responses[url])

        result = research_weather_market(quote, http_get=get)

        self.assertEqual(result["source_status"], "nws_observation_available")
        self.assertEqual(result["observations"][0]["temperature_f"], 68.0)
        self.assertEqual(result["forecast_periods"][0]["temperature_f"], 69)
        self.assertIn("final NWS Daily Climate Report", result["notice"])

    def test_weather_company_contract_never_substitutes_a_proxy(self):
        quote = {
            "question": "Will Chicago temperature at 5 PM EDT exceed 80F?",
            "category": "Weather",
            "resolution_criteria": "The Weather Company reports station KORD.",
        }
        result = research_weather_market(quote, http_get=lambda *_args, **_kwargs: self.fail("must not fetch"))

        self.assertEqual(result["source_status"], "official_source_not_integrated")
        self.assertEqual(result["observations"], [])
        self.assertIn("no licensed Weather Company feed", result["notice"])

    def test_nws_contract_without_station_does_not_guess_location(self):
        quote = {
            "question": "What will the highest temperature in Chicago be today?",
            "category": "Weather",
            "resolution_criteria": "NWS Daily Climate Report.",
        }
        result = research_weather_market(quote, http_get=lambda *_args, **_kwargs: self.fail("must not fetch"))

        self.assertEqual(result["source_status"], "missing_station")
        self.assertEqual(result["observations"], [])

    def test_nws_result_is_cached_without_representing_it_as_final_settlement(self):
        quote = {
            "question": "What will the highest temperature in Chicago be today?",
            "category": "Weather",
            "resolution_criteria": "NWS Daily Climate Report, station KORD.",
        }
        calls = []

        def get(url, **_kwargs):
            calls.append(url)
            if url.endswith("/observations/latest"):
                return _Response({
                    "properties": {"temperature": {"value": 20.0}},
                    "geometry": {"coordinates": [-87.9, 41.9]},
                })
            if "/points/" in url:
                return _Response({"properties": {"forecastHourly": "https://example.test/hourly"}})
            return _Response({"properties": {"periods": []}})

        first = research_weather_market(quote, http_get=get)
        second = research_weather_market(quote, http_get=get)

        self.assertEqual(first["source_status"], "nws_observation_available")
        self.assertFalse(first["research_cached"])
        self.assertTrue(second["research_cached"])
        self.assertEqual(len(calls), 4)
        self.assertIn("final NWS Daily Climate Report", second["notice"])

    def test_repeated_nws_failures_open_a_circuit_without_proxy_data(self):
        quote = {
            "question": "What will the highest temperature in Chicago be today?",
            "category": "Weather",
            "resolution_criteria": "NWS Daily Climate Report, station KORD.",
        }
        calls = []

        def failing_get(url, **_kwargs):
            calls.append(url)
            raise RuntimeError("NWS unavailable")

        results = [research_weather_market(quote, http_get=failing_get) for _ in range(4)]

        self.assertEqual(results[0]["source_status"], "nws_temporarily_unavailable")
        self.assertEqual(results[2]["source_status"], "nws_temporarily_unavailable")
        self.assertEqual(results[3]["source_status"], "nws_circuit_open")
        self.assertEqual(len(calls), 3)

    def test_nws_contract_returns_model_forecast(self):
        quote = {
            "question": "What will the highest temperature in New York City be today?",
            "category": "Weather",
            "resolution_criteria": "NWS Daily Climate Report, station KNYC.",
        }
        responses = {
            "https://api.weather.gov/stations/KNYC/observations/latest": {
                "properties": {
                    "timestamp": "2026-09-07T10:00:00+00:00",
                    "temperature": {"value": 20.0},
                    "dewpoint": {"value": 10.0},
                    "windSpeed": {"value": 3.0},
                    "precipitationLastHour": {"value": 0.0},
                },
                "geometry": {"coordinates": [-73.9654, 40.7829]},
            },
            "https://api.weather.gov/points/40.7829,-73.9654": {
                "properties": {"forecastHourly": "https://example.test/hourly"},
            },
            "https://example.test/hourly": {
                "properties": {"periods": []},
            },
            "https://api.open-meteo.com/v1/forecast?latitude=40.7829&longitude=-73.9654&hourly=temperature_2m&temperature_unit=fahrenheit&forecast_days=2": {
                "hourly": {
                    "time": ["2026-09-07T12:00", "2026-09-07T13:00", "2026-09-07T14:00"],
                    "temperature_2m": [72.0, 78.5, 75.0],
                },
            },
        }

        def get(url, **_kwargs):
            return _Response(responses[url])

        result = research_weather_market(quote, http_get=get)

        self.assertIsNotNone(result.get("model_forecast"))
        # Calibrated with KNYC bias (-0.5)
        self.assertEqual(result["model_forecast"]["raw_projected_high_f"], 78.5)
        self.assertEqual(result["model_forecast"]["projected_high_f"], 78.0)
        self.assertEqual(result["model_forecast"]["station_bias_offset_f"], -0.5)
        self.assertEqual(result["model_forecast"]["provider"], "open-meteo")

    def test_google_weather_forecast_and_bracket_probability(self):
        quote = {
            "question": "Will the high temperature in Chicago Midway be 80° or above?",
            "subtitle": "80° or above",
            "category": "Weather",
            "resolution_criteria": "NWS Daily Climate Report, station KMDW.",
            "price": 0.40,
        }
        responses = {
            "https://api.weather.gov/stations/KMDW/observations/latest": {
                "properties": {
                    "timestamp": "2026-09-07T10:00:00+00:00",
                    "temperature": {"value": 22.0},
                    "dewpoint": {"value": 12.0},
                    "windSpeed": {"value": 5.0},
                    "precipitationLastHour": {"value": 0.0},
                },
                "geometry": {"coordinates": [-87.7522, 41.7868]},
            },
            "https://api.weather.gov/points/41.7868,-87.7522": {
                "properties": {"forecastHourly": "https://example.test/hourly"},
            },
            "https://example.test/hourly": {
                "properties": {"periods": []},
            },
            "https://weather.googleapis.com/v1/forecast/hours:lookup?location.latitude=41.7868&location.longitude=-87.7522&hours=24&unitsSystem=IMPERIAL&key=test-gmp-key": {
                "forecastHours": [
                    {
                        "interval": {"startTime": "2026-09-07T12:00:00Z"},
                        "temperature": {"unit": "FAHRENHEIT", "degrees": 75.0},
                    },
                    {
                        "interval": {"startTime": "2026-09-07T16:00:00Z"},
                        "temperature": {"unit": "FAHRENHEIT", "degrees": 81.0},
                    },
                ]
            },
        }

        def get(url, **_kwargs):
            return _Response(responses[url])

        try:
            os.environ["GOOGLE_WEATHER_API_KEY"] = "test-gmp-key"
            result = research_weather_market(quote, http_get=get)
            self.assertEqual(result["source_status"], "nws_observation_available")
            mf = result["model_forecast"]
            self.assertEqual(mf["provider"], "google_maps_weather")
            self.assertEqual(mf["model"], "google_deepmind_weathernext_metnet")
            self.assertEqual(mf["raw_projected_high_f"], 81.0)
            # KMDW bias is +1.5°F (urban heat island)
            self.assertEqual(mf["station_bias_offset_f"], 1.5)
            self.assertEqual(mf["projected_high_f"], 82.5)

            bp = result["bracket_probability"]
            self.assertIsNotNone(bp)
            self.assertEqual(bp["strike_spec"]["strike_type"], "greater_than_or_equal")
            self.assertEqual(bp["strike_spec"]["strike_f"], 80.0)
            self.assertEqual(bp["model_mean_high_f"], 82.5)
            # When mean is 82.5 and strike is 80, prob of >= 80 is very high (> 0.9)
            self.assertGreater(bp["model_probability"], 0.90)
            self.assertEqual(bp["market_implied_probability"], 0.40)
            self.assertGreater(bp["model_edge"], 0.50)
        finally:
            os.environ.pop("GOOGLE_WEATHER_API_KEY", None)

    def test_bracket_probability_calculations(self):
        # Less than
        spec1 = parse_market_strike({"subtitle": "Below 75.5°"})
        self.assertEqual(spec1["strike_type"], "less_than")
        self.assertEqual(spec1["strike_f"], 75.5)
        p1 = calculate_bracket_probability(75.5, spec1, uncertainty_std_f=2.0)
        self.assertEqual(p1["model_probability"], 0.50)

        # Between
        spec2 = parse_market_strike({"question": "Will temperature be 70 to 80 degrees?"})
        self.assertEqual(spec2["strike_type"], "between")
        self.assertEqual(spec2["strike_low_f"], 70.0)
        self.assertEqual(spec2["strike_high_f"], 80.0)
        p2 = calculate_bracket_probability(75.0, spec2, uncertainty_std_f=2.0)
        self.assertGreater(p2["model_probability"], 0.98)


if __name__ == "__main__":
    unittest.main()
