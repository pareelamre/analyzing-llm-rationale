"""An evidence source that stops working must not look like one with nothing to say.

Every fetcher in news_pipeline ended in `except Exception: return []`, and
the module had no logger. TAVILY_API_KEY and SERPER_API_KEY are set on the
live service: had either expired or run out of quota, every forecast would
have quietly lost its web evidence, indistinguishable from a quiet news
day. The server reports an evidence error only when every source comes
back empty, and even then cannot say which one failed.

Behaviour is unchanged -- a failing fetcher still returns [] -- it now
says so, once per source per five minutes. Fetchers run on every
forecast, so an outage logged per request would bury the line worth
reading; the throttle is tested as deliberately as the warning.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import news_pipeline  # noqa: E402
from analyzing_llm_rationale.news_pipeline import NewsPipeline  # noqa: E402


def pipeline():
    return NewsPipeline(
        use_query_planner=False, summarize_articles=False, use_embeddings=False
    )


class SourceFailureTests(unittest.TestCase):
    def setUp(self):
        news_pipeline._last_source_failure.clear()
        self.addCleanup(news_pipeline._last_source_failure.clear)

    def _failing_post(self):
        return mock.patch("requests.post", side_effect=RuntimeError("401 key expired"))

    def test_a_failing_source_still_returns_nothing(self):
        with self._failing_post(), self.assertLogs("foresea.news", "WARNING"):
            self.assertEqual(pipeline()._web_tavily("election", limit=3), [])

    def test_the_warning_names_the_source_and_carries_the_traceback(self):
        with self._failing_post(), self.assertLogs("foresea.news", "WARNING") as logs:
            pipeline()._web_tavily("election", limit=3)
        self.assertEqual(len(logs.records), 1)
        record = logs.records[0]
        self.assertIn("source=_web_tavily", record.getMessage())
        self.assertIsNotNone(record.exc_info, "the cause must be attached")
        self.assertIn("401 key expired", str(record.exc_info[1]))

    def test_an_outage_is_reported_once_per_window_not_per_request(self):
        with self._failing_post(), self.assertLogs("foresea.news", "WARNING") as logs:
            p = pipeline()
            for _ in range(25):
                p._web_tavily("election", limit=3)
        self.assertEqual(len(logs.records), 1)

    def test_it_reports_again_once_the_window_has_passed(self):
        clock = [1000.0]
        with (
            self._failing_post(),
            mock.patch.object(news_pipeline.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertLogs("foresea.news", "WARNING") as logs,
        ):
            p = pipeline()
            p._web_tavily("q", limit=3)
            clock[0] += news_pipeline._SOURCE_FAILURE_LOG_INTERVAL_S - 1
            p._web_tavily("q", limit=3)
            clock[0] += 2
            p._web_tavily("q", limit=3)
        self.assertEqual(len(logs.records), 2)

    def test_each_source_has_its_own_window(self):
        """One source failing must not silence a second one."""
        with (
            mock.patch("requests.post", side_effect=RuntimeError("tavily down")),
            self.assertLogs("foresea.news", "WARNING") as logs,
        ):
            p = pipeline()
            p._web_tavily("q", limit=3)
            try:
                raise RuntimeError("serper down")
            except RuntimeError:
                news_pipeline._note_source_failure("_web_serper")
        sources = sorted(r.getMessage().split("source=")[1].split(";")[0] for r in logs.records)
        self.assertEqual(sources, ["_web_serper", "_web_tavily"])

    def test_a_working_source_says_nothing(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": [{"title": "t", "url": "https://a.test/x", "content": "c"}]}
        with mock.patch("requests.post", return_value=response):
            with self.assertNoLogs("foresea.news", "WARNING"):
                found = pipeline()._web_tavily("q", limit=3)
        self.assertEqual(len(found), 1)


class EveryFetcherReportsTests(unittest.TestCase):
    """Read against the source: a fetcher that misses the call is silent."""

    def test_no_fetcher_swallows_a_failure_silently(self):
        import ast

        source = Path(news_pipeline.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        silent = []
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            if not (fn.name.startswith("_web_") or fn.name.startswith("_fetch_")):
                continue
            for handler in ast.walk(fn):
                if not isinstance(handler, ast.ExceptHandler):
                    continue
                if not (isinstance(handler.type, ast.Name) and handler.type.id == "Exception"):
                    continue
                reports = any(
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "id", "") == "_note_source_failure"
                    for stmt in handler.body
                )
                nested_try = isinstance(handler.body[0], ast.Try)
                if not reports and not nested_try:
                    silent.append(f"{fn.name}:{handler.lineno}")
        self.assertEqual(silent, [], f"these fetchers swallow failures silently: {silent}")


if __name__ == "__main__":
    unittest.main()
