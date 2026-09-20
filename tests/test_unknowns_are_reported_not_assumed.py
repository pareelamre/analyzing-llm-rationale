"""What the code could not determine must not read as a clean result.

An audit of the agent-trading path for handlers that swallow a failure and
carry on found four places where an unknown was published as a fact:

  - the bid/ask spread cap records ``clears: true`` when no bid exists, so a
    guard that ran on 27 of 263 published trades reads as if it ran on all;
  - a retired model's unreadable archive month was dropped from the manifest,
    publishing a history with a silent hole;
  - a model with no notes file was given whatever the default notes path
    held, which belongs to another run.

A fourth candidate, a model whose cycle age cannot be parsed reading as
fresh, turned out to be deliberate: StaleMaskedByFailedAttemptTests pins it,
so it is left alone.

The row-metadata half of the same audit is held in test_agent_trading_stats.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_agent_trading_audit  # noqa: E402
import build_agent_trading_board  # noqa: E402

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402


def _quote(**fields):
    base = {
        "platform": "Kalshi", "ident": "KXSPREAD", "question": "Q?",
        "yes_ask": 0.60, "close_time": "2027-01-01T00:00:00Z",
    }
    return {**base, **fields}


class SpreadCheckReportsWhatItCouldNotMeasureTests(unittest.TestCase):
    def check(self, quote):
        with mock.patch.object(market_data, "fetch_kalshi", return_value=quote):
            return benchmark_tools._resolve_shadow_marketability("KXSPREAD", "yes", 0.60)

    def test_a_measured_spread_is_reported_as_checked(self):
        spread = self.check(_quote(yes_bid=0.58))["spread_check"]
        self.assertEqual(spread["status"], "checked")
        self.assertTrue(spread["clears"])
        self.assertAlmostEqual(spread["spread"], 0.02)

    def test_a_wide_spread_still_fails_the_check(self):
        spread = self.check(_quote(yes_bid=0.10))["spread_check"]
        self.assertEqual(spread["status"], "checked")
        self.assertFalse(spread["clears"])

    def test_no_bid_at_all_is_reported_as_unmeasured(self):
        """The usual case in the published audit, and the one that read clean."""
        spread = self.check(_quote())["spread_check"]
        self.assertEqual(spread["status"], "unknown_no_bid")
        self.assertIsNone(spread["spread"])

    def test_a_zero_bid_is_unmeasured_too(self):
        # A market with nothing resting is the most illiquid case, not a
        # tight one.
        self.assertEqual(self.check(_quote(yes_bid=0.0))["spread_check"]["status"], "unknown_no_bid")

    def test_the_trade_audit_records_whether_the_cap_ran(self):
        for quote, expected in ((_quote(yes_bid=0.58), "checked"), (_quote(), "unknown_no_bid")):
            with self.subTest(expected=expected):
                audit = benchmark_tools._trade_audit_context(
                    requested_price=0.60,
                    requested_quantity=10,
                    market_check=self.check(quote),
                    sizing={},
                    guard={"allowed": True},
                )
                self.assertEqual(audit["quote"]["spread_status"], expected)


class ArchiveManifestKeepsUnreadableMonthsTests(unittest.TestCase):
    def test_an_unreadable_retired_month_is_indexed_not_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "archives"
            model_root = root / "retired-model"
            model_root.mkdir(parents=True)
            (model_root / "2026-08.json").write_text(
                json.dumps({"items": [{"action_id": "a1"}]}), encoding="utf-8",
            )
            (model_root / "2026-09.json").write_text("{ truncated", encoding="utf-8")

            periods, records = build_agent_trading_audit._read_existing_archive_periods(
                root, "retired-model",
            )

        by_month = {p["month"]: p for p in periods}
        self.assertEqual(sorted(by_month), ["2026-08", "2026-09"])
        self.assertEqual(by_month["2026-09"]["status"], "unreadable")
        self.assertEqual(by_month["2026-09"]["records"], 0)
        self.assertNotIn("status", by_month["2026-08"])
        # Its records are genuinely unavailable, so nothing is invented.
        self.assertEqual([r["action_id"] for r in records], ["a1"])

    def test_a_month_without_an_items_list_is_indexed_too(self):
        with tempfile.TemporaryDirectory() as td:
            model_root = Path(td) / "archives" / "retired-model"
            model_root.mkdir(parents=True)
            (model_root / "2026-07.json").write_text(json.dumps({"items": "nope"}), encoding="utf-8")
            periods, _ = build_agent_trading_audit._read_existing_archive_periods(
                Path(td) / "archives", "retired-model",
            )
        self.assertEqual([p["status"] for p in periods], ["unreadable"])


class BoardDoesNotGuessTests(unittest.TestCase):
    def test_a_model_with_no_notes_file_is_not_given_another_runs_notes(self):
        with tempfile.TemporaryDirectory() as td:
            other = Path(td) / "someone-elses-notes.json"
            other.write_text(json.dumps({"model-a": [{"text": "not this model's"}]}), encoding="utf-8")
            with (
                mock.patch.object(build_agent_trading_board, "STORE_DIR", Path(td) / "store"),
                mock.patch.dict("os.environ", {"FORESEA_AGENT_NOTES_PATH": str(other)}, clear=False),
            ):
                self.assertEqual(build_agent_trading_board._load_model_notes("model-a"), {})

    def test_a_model_with_its_own_notes_file_still_gets_them(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "store" / "model-a"
            store.mkdir(parents=True)
            (store / "notes.json").write_text(
                json.dumps({"model-a": [{"text": "mine"}]}), encoding="utf-8",
            )
            with mock.patch.object(build_agent_trading_board, "STORE_DIR", Path(td) / "store"):
                notes = build_agent_trading_board._load_model_notes("model-a")
        self.assertEqual([n["text"] for n in notes["model-a"]], ["mine"])


if __name__ == "__main__":
    unittest.main()
