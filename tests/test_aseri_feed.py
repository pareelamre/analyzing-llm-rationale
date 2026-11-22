"""aseri's daily signals as evidence: topic routing, the evidence text and credit, caching,
failures, and how the pipeline and the agent keep the item (aseri_feed, news_pipeline)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import aseri_feed  # noqa: E402
from analyzing_llm_rationale import news_pipeline as np_mod  # noqa: E402
from analyzing_llm_rationale.news_pipeline import NewsPipeline  # noqa: E402

HORMUZ = "Will Iran close the Strait of Hormuz by December 31?"


def _series(sid, title, unit, baseline, steps, licence="IMF Copyright and Usage",
            credit="Sources: UN Global Platform; IMF PortWatch (portwatch.imf.org).", **extra):
    return {"series_id": sid, "title": title, "unit": unit, "baseline": baseline,
            "baseline_label": "its 365-day mean", "interval_steps": steps, "source_status": "ok",
            "stale_since": None, "last_observation_date": "2026-11-15",
            "source": {"title": "IMF PortWatch, Daily Chokepoints Data"},
            "license": {"name": licence, "attribution": credit},
            "links": {"latest": f"series/{sid}/latest.json", "actuals": f"series/{sid}/actuals.json",
                      "scores": f"series/{sid}/scores.json"}, **extra}


CATALOG = {
    "license": {"name": "CC BY-NC 4.0", "attribution": "Forecasts: aseri (model m1, Timestral-209M by Mistral AI)."},
    "series": [
        _series("hormuz_transits", "Strait of Hormuz", "ships a day", 22.0, [3, 7, 14]),
        _series("gpr_daily", "Geopolitical risk", "index (1985-2019 average = 100)", 100, [7, 14],
                licence="CC BY 4.0", credit="Caldara, D. and M. Iacoviello (2022)."),
    ],
}


def _latest(sid, lo, mid, hi, n=14):
    return {"series_id": sid, "reference_date": "2026-11-15", "forecast_date": "2026-11-21",
            "target_end_date": [f"2026-11-{16 + i:02d}" for i in range(n)],
            "quantiles": {"0.1": [lo] * n, "0.5": [mid] * n, "0.9": [hi] * n}}


RESPONSES = {
    "series.json": CATALOG,
    "series/hormuz_transits/latest.json": _latest("hormuz_transits", 0.0, 2.2, 5.9),
    "series/hormuz_transits/actuals.json": {"date": [f"2026-11-{d:02d}" for d in range(1, 16)],
                                            "observation": [8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 2, 3, 2, None, 2]},
    "series/hormuz_transits/scores.json": {"min_n_to_display": 30, "live": {"n": 0},
                                           "retrospective": {"buckets": {"1-7": {"coverage_80": {"rate": 0.889}}}}},
    "series/gpr_daily/latest.json": _latest("gpr_daily", 106, 158, 229),
    "series/gpr_daily/actuals.json": {"date": ["2026-11-14", "2026-11-15"], "observation": [100, 190]},
}


class FakeHttp:
    def __init__(self, responses=RESPONSES, fail=()):
        self.responses, self.fail, self.calls = responses, set(fail), []

    def __call__(self, url, timeout, headers):
        path = url.removeprefix(aseri_feed.API + "/")
        self.calls.append((path, timeout, headers["User-Agent"]))
        if path in self.fail or path not in self.responses:
            raise OSError(f"{path} down")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: self.responses[path])


class RoutingTests(unittest.TestCase):
    def test_questions_route_to_their_series_most_specific_first(self):
        self.assertEqual(aseri_feed.series_for(HORMUZ), ["hormuz_transits", "gpr_daily"])
        self.assertEqual(aseri_feed.series_for("Will Russia and Ukraine agree to a ceasefire in 2026?"),
                         ["ua_air_raid_hours", "gpr_daily"])
        self.assertEqual(aseri_feed.series_for("Will China blockade Taiwan before 2027?"),
                         ["taiwan_strait_transits", "gpr_daily"])
        self.assertEqual(aseri_feed.series_for("Will the Houthis strike a ship in the Red Sea?"),
                         ["bab_el_mandeb_transits", "suez_transits", "gpr_daily"])
        self.assertIn("bab_el_mandeb_transits", aseri_feed.series_for("Will traffic through Bab el-Mandeb recover?"))

    def test_unrelated_questions_and_partial_words_get_nothing(self):
        for question in ("Will the Fed cut rates in December?",
                         "Will a hurricane make landfall on the Gulf of Mexico coast?",
                         "Will a new Star Wars film be released in 2027?",
                         "Will the Warriors win the NBA title?",
                         "Will Brent oil close above $100?"):
            self.assertEqual(aseri_feed.series_for(question), [], question)

    def test_at_most_four_series(self):
        everything = "Iran Houthi Suez Taiwan Ukraine war"
        self.assertEqual(len(aseri_feed.series_for(everything)), aseri_feed.MAX_SERIES)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        aseri_feed.clear_cache()

    def test_one_item_with_level_normal_range_record_and_credit(self):
        http = FakeHttp()
        [item] = aseri_feed.evidence(HORMUZ, http_get=http)
        self.assertEqual(item["source_channel"], "aseri")
        self.assertEqual(item["url"], aseri_feed.SITE)
        self.assertEqual(item["publish_date"], "2026-11-15")
        self.assertEqual(item["title"], "aseri daily signals: Strait of Hormuz, Geopolitical risk")
        text = item["text"]
        # The 7-day average skips the missing day: (1+2+3+2+3+2+2)/7 = 2.1, 90% below a normal of 22.
        self.assertIn("2 ships a day on Nov 15; 7-day average 2.1, 90% below normal (normal 22", text)
        self.assertIn("Last 30 days ranged 1-8.", text)
        self.assertIn("Nov 18: likely 0-5.9, middle 2.2", text)   # step 3
        self.assertIn("Nov 29: likely 0-5.9", text)               # step 14
        self.assertIn("live record 0 scored forecasts (shown from 30)", text)
        self.assertIn("the range held 89% of the time", text)
        self.assertIn("Geopolitical risk (", text)
        self.assertIn("190 on Nov 15", text)                      # an index has no unit after the number
        self.assertIn("IMF PortWatch (portwatch.imf.org).", text)
        self.assertIn("Caldara, D. and M. Iacoviello (2022).", text)
        self.assertIn("Forecasts: aseri (model m1, Timestral-209M by Mistral AI).", text)
        self.assertIn("CC BY-NC 4.0; used non-commercially", text)
        self.assertLessEqual(len(item["summary"]), 250)
        self.assertTrue(item["summary"].startswith("aseri: Strait of Hormuz 2.1 ships a day (7-day avg), 90% below normal"))
        self.assertTrue(all(t == aseri_feed.TIMEOUT_S and "foresea.ink" in ua for _, t, ua in http.calls))

    def test_cached_for_an_hour(self):
        http = FakeHttp()
        aseri_feed.evidence(HORMUZ, http_get=http)
        first = len(http.calls)
        aseri_feed.evidence(HORMUZ, http_get=http)
        self.assertEqual(len(http.calls), first)
        with mock.patch.object(aseri_feed.time, "monotonic", return_value=aseri_feed.time.monotonic() + 3601):
            aseri_feed.evidence(HORMUZ, http_get=http)
        self.assertGreater(len(http.calls), first)

    def test_a_failure_is_remembered_for_ten_minutes(self):
        http = FakeHttp(fail={"series.json"})
        self.assertEqual(aseri_feed.evidence(HORMUZ, http_get=http), [])
        self.assertEqual(aseri_feed.evidence(HORMUZ, http_get=http), [])
        self.assertEqual(len(http.calls), 1)
        with mock.patch.object(aseri_feed.time, "monotonic", return_value=aseri_feed.time.monotonic() + 601):
            aseri_feed.evidence(HORMUZ, http_get=http)
        self.assertEqual(len(http.calls), 2)

    def test_failures_never_raise(self):
        self.assertEqual(aseri_feed.evidence(HORMUZ, http_get=FakeHttp(fail={"series.json"})), [])
        aseri_feed.clear_cache()
        [item] = aseri_feed.evidence(HORMUZ, http_get=FakeHttp(fail={"series/hormuz_transits/latest.json"}))
        self.assertEqual(item["title"], "aseri daily signals: Geopolitical risk")
        aseri_feed.clear_cache()
        [item] = aseri_feed.evidence(HORMUZ, http_get=FakeHttp(fail={"series/hormuz_transits/scores.json"}))
        self.assertNotIn("Track record", item["text"].split("\n")[1])
        self.assertEqual(aseri_feed.evidence("Will the Fed cut rates?", http_get=FakeHttp()), [])

    def test_a_stale_source_says_so(self):
        catalog = {**CATALOG, "series": [dict(CATALOG["series"][0], source_status="stale", stale_since="2026-11-10")]}
        [item] = aseri_feed.evidence("Will Iran reopen Hormuz?", http_get=FakeHttp({**RESPONSES, "series.json": catalog}))
        self.assertIn("Source status stale, stale since 2026-11-10", item["text"])


class PipelineTests(unittest.TestCase):
    def test_on_by_default_and_ranked_as_query_specific(self):
        self.assertIn("aseri", np_mod.DEFAULT_FETCH_SOURCES)
        self.assertIn("aseri", np_mod.SOURCE_DIVERSITY_ORDER)
        self.assertIn("aseri", np_mod._QUERY_SPECIFIC_CHANNELS)
        self.assertNotIn("aseri", np_mod._QUERY_AGNOSTIC_CHANNELS)

    def test_fetched_only_when_enabled_and_on_topic(self):
        item = {"title": "aseri daily signals", "url": aseri_feed.SITE, "source_channel": "aseri"}
        on = NewsPipeline.__new__(NewsPipeline)
        on._newsapi_key, on._fetch_sources = None, ("aseri",)
        with mock.patch.object(aseri_feed, "evidence", return_value=[item]) as fetch:
            self.assertEqual(on.fetch(HORMUZ), [item])
            fetch.assert_called_once_with(HORMUZ)
        with mock.patch.object(aseri_feed, "evidence") as fetch:
            self.assertEqual(on.fetch("Will the Fed cut rates in December?"), [])
            fetch.assert_not_called()
        off = NewsPipeline.__new__(NewsPipeline)
        off._newsapi_key, off._fetch_sources = None, ("web",)
        off._fetch_web = lambda query, limit: []
        with mock.patch.object(aseri_feed, "evidence") as fetch:
            off.fetch(HORMUZ)
            fetch.assert_not_called()

    def test_kept_below_the_relevance_floor(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._fetch_sources, pipeline._min_relevance = ("web", "aseri"), 0.25
        ranked = [{"title": "News", "url": "https://example.com/a", "source_channel": "web", "relevance": 0.6},
                  {"title": "aseri daily signals", "url": aseri_feed.SITE, "source_channel": "aseri", "relevance": 0.05},
                  {"title": "Off topic", "url": "https://example.com/b", "source_channel": "web", "relevance": 0.05}]
        kept = pipeline.select_diverse_sources(ranked, top_k=3)
        self.assertEqual([a["title"] for a in kept], ["News", "aseri daily signals"])


if __name__ == "__main__":
    unittest.main()
