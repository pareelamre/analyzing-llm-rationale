"""The published track record showed 128 zeros as if they were losses.

/track-record serialises a `profit_edge` for every bet. A bet sized at
zero edge that lost computes

    0.0 * -1.0 - 0.0   ==   -0.0

which is exactly zero -- `-0.0 == 0.0` is True and `-0.0 < 0` is False --
but json.dumps writes it as "-0.0". On the live endpoint 128 of the
bets read that way: a column of apparent tiny losses that were in fact
no position taken at all.

Nothing was wrong with the arithmetic, which is why nothing caught it.
It is a presentation defect in a public artifact, and the only test that
can see it is one that looks at the sign bit -- assertEqual(0.0, -0.0)
passes, so the obvious assertion is blind to exactly this.

Scope note: skill_ci_high also publishes -0.0 and is deliberately left
alone. That value is a genuinely negative bound of around -1e-5 rounded
to four places, so its sign is information -- it says the confidence
interval does not reach zero. Collapsing it would assert the opposite.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.track_record_live import (  # noqa: E402
    _published_round,
    paper_pnl,
)


def is_negative(value: float) -> bool:
    """True for -0.0 as well as for ordinary negatives."""
    return math.copysign(1.0, value) < 0


class PublishedRoundTests(unittest.TestCase):
    def test_a_negative_zero_loses_its_sign(self):
        result = _published_round(0.0 * -1.0 - 0.0)
        self.assertFalse(is_negative(result))
        self.assertEqual(json.dumps(result), "0.0")

    def test_a_real_negative_keeps_its_sign(self):
        result = _published_round(-1.0 - 0.02)
        self.assertTrue(is_negative(result))
        self.assertAlmostEqual(result, -1.02)

    def test_ordinary_values_are_unchanged(self):
        for value in (0.3645, -0.1234, 1.5, 0.0, 12.0):
            self.assertEqual(_published_round(value), round(value, 4))

    def test_it_still_rounds(self):
        self.assertEqual(_published_round(0.123456), 0.1235)
        self.assertEqual(_published_round(0.123456, 2), 0.12)

    def test_a_small_negative_is_not_rescued_into_zero(self):
        """Only the sign of an exact zero is dropped, never a real value."""
        self.assertTrue(is_negative(_published_round(-0.0001)))


class PaperPnlTests(unittest.TestCase):
    """The path that actually reaches /track-record."""

    @staticmethod
    def _row(model_p, market_p, outcome, ident):
        return {
            "model": "council", "platform": "kalshi", "ident": ident,
            "question": "Q " + ident, "domain": "other",
            "model_probability": model_p, "market_probability": market_p,
            "outcome": outcome,
            "snapshot_ts": "2026-09-01T00:00:00Z",
            "resolved_ts": "2026-09-02T00:00:00Z",
        }

    def setUp(self):
        self.result = paper_pnl([
            self._row(0.60, 0.60, 0, "ZERO-EDGE-LOSS"),
            self._row(0.70, 0.40, 1, "REAL-EDGE-WIN"),
        ])
        self.assertIsNotNone(self.result)
        self.bets = {bet["ident"]: bet for bet in self.result["bets"]}

    def test_a_bet_with_no_edge_that_lost_publishes_a_plain_zero(self):
        bet = self.bets["ZERO-EDGE-LOSS"]
        self.assertEqual(bet["stake_edge"], 0.0, "no edge means no position")
        self.assertFalse(
            is_negative(bet["profit_edge"]),
            "a position never taken cannot have lost money",
        )

    def test_nothing_in_the_payload_serialises_as_negative_zero(self):
        self.assertNotIn("-0.0", json.dumps(self.result["bets"]))

    def test_a_real_win_is_still_reported(self):
        """The collapse must not flatten genuine figures."""
        self.assertGreater(self.bets["REAL-EDGE-WIN"]["profit_edge"], 0.0)

    def test_a_real_loss_is_still_negative(self):
        result = paper_pnl([self._row(0.70, 0.40, 0, "REAL-EDGE-LOSS")])
        bet = result["bets"][0]
        self.assertTrue(is_negative(bet["profit_edge"]))


if __name__ == "__main__":
    unittest.main()
