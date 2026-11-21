"""MCP tools are labelled for agents, and the index, pairs and fund tools summarise their feeds."""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import mcp_server as ms  # noqa: E402

try:
    import mcp  # noqa: F401
    HAVE_MCP = True
except ImportError:  # the SDK is an optional extra
    HAVE_MCP = False


@unittest.skipUnless(HAVE_MCP, "mcp SDK not installed")
class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.server = ms.create_mcp_server(base_url="https://example.invalid")
        self.tools = {t.name: t for t in asyncio.run(self.server.list_tools())}

    def test_every_tool_says_whether_it_only_reads(self):
        self.assertTrue(all(t.annotations is not None for t in self.tools.values()))
        acting = {n for n, t in self.tools.items() if not t.annotations.readOnlyHint}
        self.assertEqual(acting, set(ms.TOOL_ANNOTATIONS))
        self.assertTrue(self.tools["foresea_place_order"].annotations.destructiveHint)
        self.assertFalse(self.tools["foresea_place_order"].annotations.idempotentHint)
        self.assertTrue(self.tools["foresea_forecast"].annotations.readOnlyHint)

    def test_the_catalog_lists_the_new_tools_and_resources(self):
        cat = asyncio.run(ms.catalog(self.server))
        names = [t["name"] for t in cat["tools"]]
        for name in ("foresea_index_forecasts", "foresea_commodity_pairs", "foresea_fund_board"):
            self.assertIn(name, names)
        self.assertIn("foresea://fund/board", [r["uri"] for r in cat["resources"]])
        index_tool = next(t for t in cat["tools"] if t["name"] == "foresea_index_forecasts")
        self.assertEqual(set(index_tool["parameters"]), {"index", "horizon"})
        self.assertFalse(index_tool["parameters"]["index"]["required"])

    def test_weather_tools_warn_that_tests_found_no_edge(self):
        for name in ("foresea_weather_radar", "foresea_weather_forecast"):
            self.assertIn("no profitable weather edge", " ".join(self.tools[name].description.split()))


INDEX = {
    "as_of": "2026-11-20", "generated_at": "x", "source": "Upstox",
    "indices": [
        {"name": "Nifty 50", "close": 100.0, "as_of": "2026-11-20", "horizons": [
            {"h": 1, "target_date": "2026-11-23", "quantiles": [95, 96, 98, 100, 102, 104, 105], "p_up": 0.52},
            {"h": 5, "target_date": "2026-11-27", "quantiles": [90, 92, 96, 100, 104, 108, 110], "p_up": 0.55}]},
        {"name": "Nifty Bank", "close": 200.0, "as_of": "2026-11-20", "horizons": []}],
    "track_record": {"by_horizon": [{"h": 1, "n": 70}], "by_index": {"Nifty 50": [{"h": 1, "n": 7}]},
                     "first_forecast": "2026-11-06"},
    "backtest": {"window": "2016-2026", "by_horizon": [{"h": 1, "n": 4962}],
                 "by_index": {"Nifty 50": [{"h": 1, "n": 537}]}},
}


class SummaryTests(unittest.TestCase):
    def test_index_ranges_read_the_right_quantiles_and_one_index_gets_its_own_record(self):
        out = ms.summarize_index_forecast(INDEX, index="nifty 50", horizon=5)
        self.assertEqual([ix["name"] for ix in out["indices"]], ["Nifty 50"])
        row = out["indices"][0]["horizons"][0]
        self.assertEqual((row["range_80"], row["range_90"], row["median"]), ([92, 108], [90, 110], 100))
        self.assertEqual(out["track_record"]["live"], [{"h": 1, "n": 7}])
        self.assertEqual(out["track_record"]["backtest"], [{"h": 1, "n": 537}])
        everything = ms.summarize_index_forecast(INDEX)
        self.assertEqual(everything["track_record"]["index"], "all ten indices")
        self.assertEqual(everything["track_record"]["backtest"], [{"h": 1, "n": 4962}])
        self.assertEqual(everything["available_indices"], ["Nifty 50", "Nifty Bank"])

    def test_pairs_keep_the_long_run_label_and_filter_by_name(self):
        raw = {"pairs": [{"key": "copper-zinc", "label": "Copper / Zinc (MCX)", "z": -3.4, "stretched": True,
                          "long_run": {"z": 0.6, "percentile": 0.7, "tide": "away",
                                       "story": {"driver": "mixed", "cause": "c", "sources": ["s"], "from": "x"}}},
                         {"key": "brent-wti", "label": "Brent / WTI", "z": 0.1}],
               "long_run": {"tide": {"away_size": 0.5}}, "rule": {"entry_z": 2.0}}
        out = ms.summarize_commodity_pairs(raw, pair="zinc")
        self.assertEqual([p["key"] for p in out["pairs"]], ["copper-zinc"])
        self.assertEqual(out["pairs"][0]["long_run"]["tide"], "away")
        self.assertNotIn("from", out["pairs"][0]["long_run"]["story"])
        self.assertEqual(out["label_record"], {"away_size": 0.5})

    def test_the_fund_is_marked_paper_and_trades_are_capped(self):
        raw = {"nav_usd": 10100.0, "cash_usd": 5000.0, "positions": [{"label": "Copper", "side": "LONG", "x": 1}],
               "trades": [{"ts": str(i), "action": "open", "label": "t"} for i in range(5)], "retired": []}
        out = ms.summarize_fund_board(raw, trades=2)
        self.assertTrue(out["paper"])
        self.assertEqual(len(out["recent_trades"]), 2)
        self.assertNotIn("x", out["positions"][0])


if __name__ == "__main__":
    unittest.main()
