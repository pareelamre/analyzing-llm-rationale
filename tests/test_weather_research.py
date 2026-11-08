from __future__ import annotations

import os
import unittest

from analyzing_llm_rationale.weather_research import (
    calculate_bracket_probability,
    contract_day,
    contract_probabilities,
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
        # 4 NWS calls + 1 ensemble call on the first pass; the second pass is
        # served entirely from cache.
        self.assertEqual(len(calls), 5)
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
            "https://api.open-meteo.com/v1/forecast?latitude=40.7829&longitude=-73.9654&hourly=temperature_2m&temperature_unit=fahrenheit&forecast_days=2&timezone=auto": {
                "hourly": {
                    "time": ["2026-09-07T12:00", "2026-09-07T13:00", "2026-09-07T14:00",
                             "2026-09-07T15:00", "2026-09-07T16:00", "2026-09-07T17:00",
                             "2026-09-07T18:00", "2026-09-07T19:00", "2026-09-07T20:00",
                             "2026-09-07T21:00", "2026-09-07T22:00", "2026-09-07T23:00",
                             "2026-09-08T00:00", "2026-09-08T01:00", "2026-09-08T02:00",
                             "2026-09-08T03:00", "2026-09-08T04:00", "2026-09-08T05:00",
                             "2026-09-08T06:00", "2026-09-08T07:00", "2026-09-08T08:00",
                             "2026-09-08T09:00", "2026-09-08T10:00", "2026-09-08T11:00"],
                    "temperature_2m": [72.0, 78.5, 75.0, 74.0, 73.0, 72.0, 71.0, 70.0,
                                       69.0, 68.0, 67.0, 66.0, 65.0, 64.0, 63.0, 62.0,
                                       61.0, 60.0, 59.0, 58.0, 57.0, 56.0, 55.0, 54.0],
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


def _station_get(day_highs, periods=(), google_hours=None):
    """A fake HTTP get for KORD: Open-Meteo hours per local day (one at each
    day's high, the rest 10°F lower), NWS hourly periods, and optionally
    Google forecast hours."""
    def get(url, **_kwargs):
        if url.endswith("/observations/latest"):
            return _Response({"properties": {"temperature": {"value": 10.0}}, "geometry": {"coordinates": [-87.9, 41.9]}})
        if "/points/" in url:
            return _Response({"properties": {"forecastHourly": "https://example.test/hourly"}})
        if url == "https://example.test/hourly":
            return _Response({"properties": {"periods": list(periods)}})
        if "weather.googleapis.com" in url:
            return _Response({"forecastHours": google_hours or []})
        if "ensemble-api" in url:
            return _Response({"hourly": {"time": []}})
        times, temps = [], []
        for day, high in day_highs:
            times += [f"{day}T{h:02d}:00" for h in range(24)]
            temps += [high - 10.0] * 14 + [high] + [high - 10.0] * 9
        return _Response({"hourly": {"time": times, "temperature_2m": temps}})
    return get


class WeatherContractDayTests(unittest.TestCase):
    def setUp(self):
        from analyzing_llm_rationale import weather_research

        weather_research._reset_runtime_state_for_test()
        self._env = {k: os.environ.pop(k) for k in ("GOOGLE_WEATHER_API_KEY", "GMP_API_KEY") if k in os.environ}

    def tearDown(self):
        from analyzing_llm_rationale import weather_research

        weather_research._reset_runtime_state_for_test()
        os.environ.pop("GOOGLE_WEATHER_API_KEY", None)
        os.environ.update(self._env)

    @staticmethod
    def _quote(subject):
        return {"question": f"Highest temperature in Chicago on Nov 5, 2026? {subject}", "category": "Weather",
                "resolution_criteria": "NWS Daily Climate Report, station KORD."}

    def test_contract_day_reads_the_question_or_the_kalshi_ticker(self):
        self.assertEqual(contract_day({"question": "Will it rain in Tampa on Nov 5, 2026?"}), "2026-11-05")
        self.assertEqual(contract_day({"question": "Rain?", "ticker": "KXRAIN-26NOV05-TPA"}), "2026-11-05")
        self.assertIsNone(contract_day({"question": "Will it rain in Tampa today?"}))

    def test_an_events_brackets_are_priced_in_whole_degrees(self):
        # Read literally, "56° to 57°" and "58° to 59°" left a 57.3°F high in
        # neither bracket, so a 57°F ensemble put 10% on 58-59.
        research = {
            "weather_market": {"station": "KORD"},
            "ensemble_forecast": {"member_maxima_f": [55.2, 56.4, 57.3, 57.6, 58.1, 59.4] * 2 + [61.0] * 3},
            "model_forecast": {"projected_high_f": 57.0, "projected_low_f": 40.0},
        }
        options = ["55° or below", "56° to 57°", "58° to 59°", "60° or above"]

        chances = contract_probabilities(research, options)

        self.assertEqual([round(chances[o] * 15) for o in options], [2, 4, 6, 3])
        self.assertAlmostEqual(sum(chances.values()), 1.0)
        low = contract_probabilities(research, ["Will the low be <40° on Nov 5, 2026?"], daily_low=True)
        self.assertAlmostEqual(low["Will the low be <40° on Nov 5, 2026?"], 0.39, places=2)

    def test_cached_station_data_prices_each_bracket_itself(self):
        # Cached per station, the whole result gave every Denver bracket the
        # first one's strike and probability.
        get = _station_get([("2026-11-05", 59.0)])
        first = research_weather_market(self._quote("56° to 57°"), http_get=get)
        second = research_weather_market(self._quote("59° to 60°"), http_get=get)

        self.assertTrue(second["research_cached"])
        self.assertEqual(second["bracket_probability"]["strike_spec"]["strike_low_f"], 59.0)
        self.assertGreater(second["bracket_probability"]["model_probability"],
                           first["bracket_probability"]["model_probability"])

    def test_a_contract_day_selects_that_days_forecasts(self):
        # Late on Nov 4 the first full local day was the 4th, so tomorrow's
        # contract was priced on today's high.
        periods = [{"startTime": f"{day}T{h}:00:00-06:00", "temperature": 50, "shortForecast": "Rain",
                    "probabilityOfPrecipitation": {"value": 40}} for day in ("2026-11-04", "2026-11-05") for h in (21, 22)]
        get = _station_get([("2026-11-04", 70.0), ("2026-11-05", 60.0)], periods)
        quote = self._quote("59° to 60°")

        result = research_weather_market(quote, http_get=get, target_day="2026-11-05")
        default = research_weather_market(quote, http_get=get)

        self.assertEqual(result["model_forecast"]["forecast_day_local"], "2026-11-05")
        self.assertEqual(result["model_forecast"]["projected_high_f"], 60.0)
        self.assertEqual([p["start_time"][:10] for p in result["forecast_periods"]], ["2026-11-05"] * 2)
        self.assertEqual(default["model_forecast"]["forecast_day_local"], "2026-11-04")
        self.assertEqual(len(default["forecast_periods"]), 4)

    def test_google_forecast_for_a_contract_day_spans_that_local_day(self):
        hours = [{"interval": {"startTime": f"2026-11-0{5 + (h + 6) // 24}T{(h + 6) % 24:02d}:00:00Z"},
                  "displayDateTime": {"year": 2026, "month": 11, "day": 5, "hours": h},
                  "temperature": {"unit": "FAHRENHEIT", "degrees": 62.0 if h == 15 else 50.0}} for h in range(24)]
        hours.append({"interval": {"startTime": "2026-11-06T06:00:00Z"},
                      "displayDateTime": {"year": 2026, "month": 11, "day": 6, "hours": 0},
                      "temperature": {"unit": "FAHRENHEIT", "degrees": 75.0}})
        os.environ["GOOGLE_WEATHER_API_KEY"] = "test-gmp-key"

        result = research_weather_market(
            self._quote("61° to 62°"), http_get=_station_get([], google_hours=hours), target_day="2026-11-05"
        )
        model = result["model_forecast"]
        self.assertEqual((model["provider"], model["forecast_day_local"]), ("google_maps_weather", "2026-11-05"))
        self.assertEqual(model["projected_high_f"], 62.0)

        # Without local dates, the day can't be checked: Open-Meteo's is used.
        for hour in hours:
            hour.pop("displayDateTime")
        from analyzing_llm_rationale import weather_research
        weather_research._reset_runtime_state_for_test()
        result = research_weather_market(
            self._quote("61° to 62°"), http_get=_station_get([("2026-11-05", 58.0)], google_hours=hours),
            target_day="2026-11-05",
        )
        self.assertEqual(result["model_forecast"]["provider"], "open-meteo")
        self.assertEqual(result["model_forecast"]["projected_high_f"], 58.0)


if __name__ == "__main__":
    unittest.main()
