"""The credibility audit read four keys the edge board does not publish.

It looked for `volume`, `volume_usd`, `open_interest` and `evidence`.
Across a live 26-row board those appear on **0** rows. What the board does
publish -- `market_volume`, `market_liquidity`, `evidence_count` -- appears
on **26**.

So the liquidity assessment never executed, and the evidence check always
took its penalty branch. Every row scored exactly the same:

    1.00  start
   -0.15  sparse_retrieved_evidence   (evidence never present)
   +0.05  verifiable_resolution_criteria
   ----
    0.90  grade A, for all 26 rows

The published board agrees: credibility_score was 0.9 on 26 of 26. A
grader that returns a constant is not grading. KXNFLRETIRE-MSTAFFORD9-2627
carried grade A with market_volume 0.0 and market_bid 0.0, and the
portfolio optimizer put its largest allocation there.

Fixing the key names does not make anything grade below A -- the +0.10
evidence bonus outweighs the -0.20 liquidity penalty. That is a weighting
question, deliberately left alone here. What changes is that the flags
become true: six rows now carry low_liquidity_spread_risk.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.edge_credibility import (  # noqa: E402
    _evidence_count,
    _first_present,
    audit_edge_opportunity,
)


def _board_row(**over):
    """A row shaped the way the edge board actually publishes them."""
    row = {
        "question": "Will X happen before the 2027 season?",
        "model_probability": 0.03,
        "market_probability": 0.465,
        "resolution_criteria": "R" * 60,
        "market_volume": 0.0,
        "market_liquidity": None,
        "evidence_count": 3,
    }
    row.update(over)
    return row


class FirstPresentTests(unittest.TestCase):
    def test_a_real_zero_is_not_skipped(self):
        """`or` would fall through 0.0 -- the case the penalty is for."""
        self.assertEqual(_first_present({"market_volume": 0.0}, "volume", "market_volume"), 0.0)

    def test_earlier_keys_win(self):
        self.assertEqual(_first_present({"volume": 7, "market_volume": 9}, "volume", "market_volume"), 7)

    def test_absent_everywhere_is_none(self):
        self.assertIsNone(_first_present({}, "volume", "market_volume"))

    def test_none_is_treated_as_absent(self):
        self.assertEqual(_first_present({"volume": None, "market_volume": 3}, "volume", "market_volume"), 3)


class EvidenceCountTests(unittest.TestCase):
    def test_a_list_is_measured(self):
        self.assertEqual(_evidence_count({"evidence": [1, 2, 3]}), 3)

    def test_a_count_is_read_when_there_is_no_list(self):
        self.assertEqual(_evidence_count({"evidence_count": 4}), 4)

    def test_neither_is_zero(self):
        self.assertEqual(_evidence_count({}), 0)

    def test_junk_is_zero_not_an_exception(self):
        for raw in ("many", None, [], {}):
            with self.subTest(raw=raw):
                self.assertEqual(_evidence_count({"evidence_count": raw}), 0)


class TheAuditSeesBoardRowsTests(unittest.TestCase):
    def test_a_zero_volume_market_is_flagged(self):
        audit = audit_edge_opportunity(_board_row())
        self.assertIn("low_liquidity_spread_risk", audit["credibility_flags"])

    def test_a_deep_market_is_credited(self):
        audit = audit_edge_opportunity(_board_row(market_volume=250_000.0))
        self.assertIn("healthy_market_liquidity", audit["credibility_flags"])

    def test_evidence_count_is_believed(self):
        audit = audit_edge_opportunity(_board_row())
        self.assertIn("grounded_3_evidence_items", audit["credibility_flags"])
        self.assertNotIn("sparse_retrieved_evidence", audit["credibility_flags"])

    def test_a_row_with_no_evidence_still_takes_the_penalty(self):
        audit = audit_edge_opportunity(_board_row(evidence_count=0))
        self.assertIn("sparse_retrieved_evidence", audit["credibility_flags"])

    def test_the_grader_is_no_longer_constant(self):
        """The defect's signature: every input produced the same score."""
        scores = {
            audit_edge_opportunity(_board_row(market_volume=v, evidence_count=e))["credibility_score"]
            for v, e in ((0.0, 0), (0.0, 3), (250_000.0, 3), (250_000.0, 0))
        }
        self.assertGreater(len(scores), 1)

    def test_the_legacy_key_names_still_work(self):
        """Callers that do supply `volume`/`evidence` are unaffected."""
        audit = audit_edge_opportunity({
            "question": "Q", "model_probability": 0.5, "market_probability": 0.4,
            "resolution_criteria": "R" * 60, "volume": 10.0, "evidence": [1, 2],
        })
        self.assertIn("low_liquidity_spread_risk", audit["credibility_flags"])
        self.assertIn("grounded_2_evidence_items", audit["credibility_flags"])


if __name__ == "__main__":
    unittest.main()
