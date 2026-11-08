from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import fund_ledger as ledger  # noqa: E402

GOLD_MCX = {"id": "mcx:GOLDPETAL", "label": "Gold (MCX)", "venue": "mcx", "asset_class": "commodity",
            "underlying": "gold", "currency": "INR", "multiplier": 1.0, "key": "MCX_FO|1"}
GOLD_PERP = {"id": "kalshi_perp:KXGOLDPERP", "label": "Gold perp", "venue": "kalshi_perp", "asset_class": "commodity",
             "underlying": "gold", "currency": "USD", "multiplier": 1.0, "key": "KXGOLDPERP"}
EVENT = {"id": "kalshi:KXFED-26DEC", "label": "Fed cut", "venue": "kalshi", "asset_class": "event",
         "underlying": "kalshi:KXFED-26DEC", "currency": "USD", "multiplier": 1.0, "key": "KXFED-26DEC"}


class FundLedgerTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        ledger.ensure_schema(self.conn)
        ledger.open_fund(self.conn, "fund", 10_000.0)

    def test_a_long_rupee_future_is_worth_its_collateral_plus_profit_in_dollars(self):
        # 10 g of gold at Rs 12,000/g with USD/INR 100: $1,200 set aside.
        ledger.buy(self.conn, "fund", GOLD_MCX, "LONG", 10, 12_000.0, fx=100.0, attribution={"pairs_model": 1.0})
        snap = ledger.nav(self.conn, "fund", {"mcx:GOLDPETAL": {"bid": 12_300.0, "ask": 12_310.0}}, {"INR": 100.0})
        self.assertAlmostEqual(snap["cash_usd"], 8_800.0)
        # Rs 300/g x 10 g = Rs 3,000 = $30 at the long's bid.
        self.assertAlmostEqual(snap["positions"][0]["pnl_usd"], 30.0)
        self.assertAlmostEqual(snap["nav_usd"], 10_030.0)
        self.assertAlmostEqual(snap["unrealized_pnl_by_analyst"]["pairs_model"], 30.0)

    def test_a_short_hedge_offsets_the_long_and_nets_the_underlying(self):
        ledger.buy(self.conn, "fund", GOLD_MCX, "LONG", 10, 12_000.0, fx=100.0, attribution={"pairs_model": 1.0})
        ledger.buy(self.conn, "fund", GOLD_PERP, "SHORT", 300, 4.0, attribution={"orchestrator": 1.0}, purpose="hedge")
        quotes = {"mcx:GOLDPETAL": {"bid": 11_800.0, "ask": 11_810.0}, "kalshi_perp:KXGOLDPERP": {"bid": 3.92, "ask": 3.93}}
        snap = ledger.nav(self.conn, "fund", quotes, {"INR": 100.0})
        pnl = {p["instrument_id"]: p["pnl_usd"] for p in snap["positions"]}
        self.assertAlmostEqual(pnl["mcx:GOLDPETAL"], -20.0)
        # The short closes at the ask: (4.00 - 3.93) x 300 = $21.
        self.assertAlmostEqual(pnl["kalshi_perp:KXGOLDPERP"], 21.0)
        self.assertAlmostEqual(snap["net_exposure_by_underlying"]["gold"], 0.0)

    def test_selling_realizes_profit_and_returns_collateral(self):
        ledger.buy(self.conn, "fund", GOLD_PERP, "LONG", 100, 4.0, attribution={"a": 0.5, "b": 0.5})
        ledger.sell(self.conn, "fund", GOLD_PERP["id"], "LONG", 40, 4.5)
        account = self.conn.execute("SELECT cash, realized_pnl FROM fund_account").fetchone()
        self.assertAlmostEqual(account["realized_pnl"], 20.0)
        self.assertAlmostEqual(account["cash"], 10_000.0 - 400.0 + 160.0 + 20.0)
        held = ledger.positions(self.conn, "fund")[0]
        self.assertAlmostEqual(held["quantity"], 60)
        self.assertAlmostEqual(held["collateral_usd"], 240.0)
        snap = ledger.nav(self.conn, "fund", {}, {})
        self.assertEqual(snap["realized_pnl_by_analyst"], {"a": 10.0, "b": 10.0})

    def test_a_binary_side_is_marked_at_its_own_bid(self):
        ledger.buy(self.conn, "fund", EVENT, "NO", 100, 0.90)
        snap = ledger.nav(self.conn, "fund", {EVENT["id"]: {"yes_bid": 0.05, "no_bid": 0.93}}, {})
        self.assertAlmostEqual(snap["positions"][0]["pnl_usd"], 3.0)
        self.assertAlmostEqual(snap["nav_usd"], 10_003.0)

    def test_adding_merges_attribution_by_quantity(self):
        ledger.buy(self.conn, "fund", GOLD_PERP, "LONG", 100, 4.0, attribution={"index_model": 1.0})
        ledger.buy(self.conn, "fund", GOLD_PERP, "LONG", 300, 4.0, attribution={"finbert": 1.0})
        held = ledger.positions(self.conn, "fund")[0]
        self.assertIn('"finbert": 0.75', held["attribution_json"])
        self.assertIn('"index_model": 0.25', held["attribution_json"])

    def test_a_fill_the_cash_cannot_cover_changes_nothing(self):
        with self.assertRaises(ledger.LedgerError):
            ledger.buy(self.conn, "fund", GOLD_PERP, "LONG", 10_000, 4.0)
        self.assertEqual(ledger.positions(self.conn, "fund"), [])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM fund_trades").fetchone()[0], 0)

    def test_an_unquoted_position_is_held_at_its_collateral(self):
        ledger.buy(self.conn, "fund", GOLD_PERP, "SHORT", 100, 4.0)
        snap = ledger.nav(self.conn, "fund", {}, {})
        self.assertAlmostEqual(snap["nav_usd"], 10_000.0)
        self.assertFalse(snap["positions"][0]["marked"])

    def test_notes_and_evidence_are_kept_with_the_fill(self):
        ledger.buy(self.conn, "fund", GOLD_PERP, "LONG", 10, 4.0, note="Gold parity premium at 2.1 sd.",
                   note_source="model", evidence=[{"source": "pairs_model", "figure": "z=2.1"}])
        row = self.conn.execute("SELECT note, note_source, evidence_json, purpose FROM fund_trades").fetchone()
        self.assertEqual(row["note_source"], "model")
        self.assertIn("z=2.1", row["evidence_json"])
        self.assertEqual(row["purpose"], "view")


if __name__ == "__main__":
    unittest.main()
