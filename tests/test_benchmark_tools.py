from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import benchmark_tools  # noqa: E402


class BenchmarkToolTests(unittest.TestCase):
    def test_manage_notes_add_search_edit_delete_with_limits(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "notes.json"
            ctx = benchmark_tools.ToolContext(agent_id="model-a")

            added = benchmark_tools.manage_notes(
                {"action": "add", "text": "Watch the FOMC date.", "tags": ["rates"]},
                ctx,
                path=path,
            )
            self.assertTrue(added["ok"])
            note_id = added["note"]["id"]

            found = benchmark_tools.manage_notes(
                {"action": "search", "query": "fomc"},
                ctx,
                path=path,
            )
            self.assertEqual([n["id"] for n in found["notes"]], [note_id])

            edited = benchmark_tools.manage_notes(
                {"action": "edit", "id": note_id, "text": "Watch CPI before FOMC."},
                ctx,
                path=path,
            )
            self.assertEqual(edited["note"]["text"], "Watch CPI before FOMC.")

            deleted = benchmark_tools.manage_notes(
                {"action": "delete", "id": note_id},
                ctx,
                path=path,
            )
            self.assertEqual(deleted["deleted"], 1)

            too_long = benchmark_tools.manage_notes(
                {"action": "add", "text": "x" * (benchmark_tools.MAX_NOTE_CHARS + 1)},
                ctx,
                path=path,
            )
            self.assertFalse(too_long["ok"])

    def test_web_search_filters_blacklisted_citations(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output": [{
                "content": [{
                    "text": "Search summary.",
                    "annotations": [
                        {"title": "Bad", "url": "https://coinmarketcap.com/currencies/x"},
                        {"title": "Good", "url": "https://example.com/story"},
                    ],
                }]
            }]
        }

        with (
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
            mock.patch.object(benchmark_tools.requests, "post", return_value=response) as post,
        ):
            result = benchmark_tools.web_search({"query": "test market"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"], "Search summary.")
        self.assertEqual(result["sources"], [{"title": "Good", "url": "https://example.com/story"}])
        self.assertEqual(result["blocked_results"], 1)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["tools"][0]["type"], "web_search")
        self.assertIn("coinmarketcap.com", payload["instructions"])

    def test_place_trade_defaults_to_shadow_kalshi_buy(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        result = benchmark_tools.place_trade(
            {"ticker": "KXTEST", "side": "yes", "price": 0.42, "quantity": 2},
            ctx,
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["submitted"])
        self.assertEqual(result["mode"], "shadow")
        self.assertEqual(result["normalized_order"]["platform"], "kalshi")
        self.assertEqual(result["normalized_order"]["action"], "buy")
        self.assertEqual(result["normalized_order"]["outcome"], "yes")


if __name__ == "__main__":
    unittest.main()
