"""Daily usage counts: what is counted, by which kind of client, and that counts survive."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import usage_counter as uc  # noqa: E402

CHROME = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
GPTBOT = "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; GPTBot/1.2; +https://openai.com/gptbot)"
CLAUDEBOT = "Mozilla/5.0 (compatible; ClaudeBot/1.0; +claudebot@anthropic.com)"


class ClassifyTests(unittest.TestCase):
    def test_clients(self):
        self.assertEqual(uc.client_class(CHROME), "browser")
        self.assertEqual(uc.client_class(GPTBOT), "crawler")
        self.assertEqual(uc.client_class(CLAUDEBOT), "crawler")
        self.assertEqual(uc.client_class("python-httpx/0.28.1"), "agent")
        self.assertEqual(uc.client_class("curl/8.7.1"), "agent")
        self.assertEqual(uc.client_class(""), "unknown")
        self.assertEqual(uc.client_class("SomethingElse/1.0"), "other")

    def test_requests_fall_into_groups_and_assets_are_left_out(self):
        self.assertEqual(uc.classify("GET", "/llms.txt", None, "text/plain"), ("discovery", "/llms.txt"))
        self.assertEqual(uc.classify("GET", "/.well-known/mcp/server.json", None, "application/json"),
                         ("discovery", "/.well-known/mcp/server.json"))
        self.assertEqual(uc.classify("POST", "/mcp/", None, "application/json"), ("mcp", "POST"))
        self.assertEqual(uc.classify("GET", "/edge/indices", None, "text/html; charset=utf-8"),
                         ("page", "/edge/indices"))
        self.assertEqual(uc.classify("GET", "/chat/conv_1793841013888", None, "text/html"), ("page", "/chat/*"))
        self.assertEqual(uc.classify("GET", "/agent/runs/abc", "/agent/runs/{run_id}", "application/json"),
                         ("api", "GET /agent/runs/{run_id}"))
        for path in ("/static/fund_live.json", "/assets/app.js", "/favicon.ico", "/analytics/visit", "/health"):
            self.assertIsNone(uc.classify("GET", path, None, "application/json"), path)

    def test_only_other_sites_count_as_referrers(self):
        own = {"foresea.ink", "www.foresea.ink"}
        self.assertEqual(uc.referrer_host("https://www.google.com/search?q=x", own), "google.com")
        self.assertEqual(uc.referrer_host("https://chatgpt.com/", own), "chatgpt.com")
        self.assertIsNone(uc.referrer_host("https://foresea.ink/edge", own))
        self.assertIsNone(uc.referrer_host("", own))


class CounterTests(unittest.TestCase):
    def test_counts_wait_in_memory_and_flush_about_once_a_minute(self):
        now = [1_795_000_000.0]
        store = mock.Mock()
        counter = uc.UsageCounter(store, clock=lambda: now[0])
        counter.add("api", "GET /edge-board", "browser")
        counter.add("api", "GET /edge-board", "browser")
        self.assertFalse(counter.due())
        now[0] += 61
        self.assertTrue(counter.due())
        self.assertEqual(counter.flush(), 1)
        [batch] = store.add.call_args[0]
        self.assertEqual(list(batch.values()), [2])
        self.assertEqual(counter.pending(), {})

    def test_a_failed_save_keeps_the_counts_for_next_time(self):
        store = mock.Mock()
        store.add.side_effect = RuntimeError("datastore down")
        counter = uc.UsageCounter(store)
        counter.add("mcp_tool", "foresea_forecast", "tool")
        with self.assertRaises(RuntimeError):
            counter.flush()
        self.assertEqual(sum(counter.pending().values()), 1)

    def test_the_summary_groups_counts_by_day_client_and_name(self):
        counter = uc.UsageCounter(None)
        for _ in range(3):
            counter.add("discovery", "/llms.txt", "crawler")
        counter.add("discovery", "/llms.txt", "agent")
        counter.add("mcp_tool", "foresea_track_record", "tool")
        counter.add("page", "/", "browser")
        counter.add("referrer", "chatgpt.com")
        out = counter.summary(7)
        self.assertEqual(out["discovery"], {"/llms.txt": {"crawler": 3, "agent": 1}})
        self.assertEqual(out["mcp_tools"], [{"name": "foresea_track_record", "count": 1}])
        self.assertEqual(out["referrers"], [{"name": "chatgpt.com", "count": 1}])
        self.assertEqual(out["totals"]["discovery"], 4)
        self.assertEqual(out["clients"]["page"], {"browser": 1})
        self.assertEqual(len(out["by_day"]), 1)


class StoreTests(unittest.TestCase):
    def test_counts_add_up_across_flushes_and_survive_a_new_client(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {"FORESEA_DATASTORE_BACKEND": "sql"}):
            from analyzing_llm_rationale import datastore_sql as ds
            path = str(Path(directory) / "usage.sqlite3")
            counter = uc.UsageCounter(uc.DatastoreUsageStore(ds.Client(path=path)))
            counter.add("api", "POST /predict", "agent")
            counter.flush()
            counter.add("api", "POST /predict", "agent")
            counter.add("page", "/", "browser")
            counter.flush()
            fresh = uc.DatastoreUsageStore(ds.Client(path=path)).load(30)
            self.assertEqual(sorted(fresh.values()), [1, 2])
            self.assertEqual(uc.UsageCounter(uc.DatastoreUsageStore(ds.Client(path=path))).summary(30)["totals"],
                             {"api": 2, "page": 1})


if __name__ == "__main__":
    unittest.main()


class ServerUsageTests(unittest.TestCase):
    def test_requests_are_counted_and_the_summary_reports_them(self):
        from fastapi.testclient import TestClient

        from analyzing_llm_rationale import server
        with mock.patch.object(server, "_USAGE", uc.UsageCounter(None)), \
                mock.patch.object(server, "_get_datastore", return_value=None):
            client = TestClient(server.app)
            client.get("/llms.txt", headers={"User-Agent": GPTBOT})
            client.get("/robots.txt", headers={"User-Agent": "python-httpx/0.28"})
            client.get("/", headers={"User-Agent": CHROME, "Referer": "https://chatgpt.com/c/abc"})
            client.get("/health")
            out = client.get("/analytics/usage?days=7").json()
        self.assertEqual(out["discovery"]["/llms.txt"], {"crawler": 1})
        self.assertEqual(out["discovery"]["/robots.txt"], {"agent": 1})
        self.assertEqual(out["clients"]["page"], {"browser": 1})
        self.assertEqual(out["referrers"], [{"name": "chatgpt.com", "count": 1}])
        self.assertNotIn("/health", str(out["api"]))
