from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from analyzing_llm_rationale import agent_trading_stats, benchmark_tools  # noqa: E402

_spec = importlib.util.spec_from_file_location("seed_on_device_agent", _ROOT / "scripts" / "seed_on_device_agent.py")
seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed)


def _retired_store(root: Path, model: str, positions: list[tuple]) -> None:
    (root / model).mkdir(parents=True)
    conn = sqlite3.connect(root / model / "store.sqlite")
    benchmark_tools._ensure_account_schema(conn)
    for platform, ticker, side, qty, entry in positions:
        conn.execute("INSERT INTO agent_positions VALUES (?, ?, ?, ?, ?, ?, ?, 'x')",
                     (model, platform, ticker, side, qty, qty * entry, entry))
    conn.commit()
    conn.close()


QUOTES = {
    "WIN": {"question": "Will A happen?", "no_bid": 0.84, "no_ask": 0.86, "yes_bid": 0.14, "yes_ask": 0.16},
    "LOSE-BELIEVED": {"question": "Will B happen?", "yes_bid": 0.06, "yes_ask": 0.08, "no_bid": 0.92, "no_ask": 0.94},
    "LOSE-DOUBTED": {"question": "Will C happen?", "no_bid": 0.05, "no_ask": 0.07, "yes_bid": 0.93, "yes_ask": 0.95},
    "CLOSED": {"question": "Will D happen?"},
}


def _quote(platform, ticker):
    return {**QUOTES[ticker], "platform": platform, "ident": ticker}


class SeedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _retired_store(self.tmp, "glm-5-3", [("kalshi", "WIN", "NO", 200.0, 0.47), ("kalshi", "LOSE-DOUBTED", "NO", 100.0, 0.16)])
        _retired_store(self.tmp, "qwen3-8-27b", [("kalshi", "WIN", "NO", 100.0, 0.34), ("kalshi", "LOSE-BELIEVED", "YES", 50.0, 0.09),
                                                 ("kalshi", "CLOSED", "NO", 10.0, 0.9)])
        # The model still backs B (YES at 20% against a 6% price) but not C (NO at 3% against a 5% price).
        self.asked = []

        def judge(q, side, bid):
            self.asked.append(q["ident"])
            return {"LOSE-BELIEVED": 0.20, "LOSE-DOUBTED": 0.97}[q["ident"]]

        self.decisions = seed.plan_transfer(seed.retired_positions(self.tmp), quote=_quote, judge=judge)

    def test_winners_move_and_losers_move_only_when_the_model_still_backs_them(self):
        by = {d["ticker"]: d for d in self.decisions}
        self.assertTrue(by["WIN"]["keep"])
        self.assertEqual(by["WIN"]["quantity"], 300.0)
        self.assertEqual([h["agent"] for h in by["WIN"]["holders"]], ["glm-5-3", "qwen3-8-27b"])
        self.assertTrue(by["LOSE-BELIEVED"]["keep"])
        self.assertFalse(by["LOSE-DOUBTED"]["keep"])
        self.assertFalse(by["CLOSED"]["keep"])
        # Only losing positions are put to the model.
        self.assertEqual(sorted(self.asked), ["LOSE-BELIEVED", "LOSE-DOUBTED"])

    def test_the_new_account_is_worth_exactly_ten_thousand_at_todays_bids(self):
        store = self.tmp / "on-device.sqlite"
        summary = seed.book_transfer(store, self.decisions)
        conn = sqlite3.connect(store)
        conn.row_factory = sqlite3.Row
        account = conn.execute("SELECT * FROM agent_accounts").fetchone()
        positions = conn.execute("SELECT * FROM agent_positions ORDER BY ticker").fetchall()
        value = account["cash"] + sum(p["cost_basis"] for p in positions)
        self.assertAlmostEqual(value, seed.ACCOUNT_VALUE)
        self.assertEqual(account["starting_cash"], seed.ACCOUNT_VALUE)
        self.assertEqual(summary["positions"], 2)
        self.assertEqual([p["ticker"] for p in positions], ["LOSE-BELIEVED", "WIN"])
        self.assertAlmostEqual(positions[1]["avg_entry_price"], 0.84)
        actions = conn.execute("SELECT * FROM agent_actions").fetchall()
        self.assertEqual({(a["action_type"], a["mode"], a["fee"]) for a in actions}, {("trade", "transfer", 0.0)})
        # The equity curve treats the transfers like opening trades: the value stays at 10,000.
        curve = agent_trading_stats.agent_equity_curve(conn, seed.ON_DEVICE)
        self.assertAlmostEqual(curve["value_curve"][-1]["account_value"], seed.ACCOUNT_VALUE, places=4)
        conn.close()
        with self.assertRaises(SystemExit):
            seed.book_transfer(store, self.decisions)

    def test_the_probability_is_read_after_any_reasoning(self):
        self.assertEqual(seed._probability("<think>maybe 0.9 or 1</think>\n0.35"), 0.35)
        self.assertEqual(seed._probability("Probability: .2"), 0.2)
        self.assertIsNone(seed._probability("I cannot say."))


if __name__ == "__main__":
    unittest.main()
