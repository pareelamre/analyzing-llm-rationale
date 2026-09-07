from __future__ import annotations

import unittest

from analyzing_llm_rationale.weather_research import research_weather_market


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
        self.assertEqual(result["model_forecast"]["projected_high_f"], 78.5)
        self.assertEqual(result["model_forecast"]["projected_low_f"], 72.0)
        self.assertEqual(result["model_forecast"]["provider"], "open-meteo")


if __name__ == "__main__":
    unittest.main()
