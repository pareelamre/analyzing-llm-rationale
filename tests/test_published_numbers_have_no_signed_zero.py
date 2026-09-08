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

Scanning every committed static/*.json for a signed zero found four
producers, and only one of them is this bug:

    profit_edge     128 + 67   the sign of a zero          FIXED
    edge                    1   a real small negative       left
    skill_ci_high           1   a real small negative       left
    open_exposure           2   a real zero, but see below  left

The distinction is the whole point, and it is not visible from the
value: every one of them prints as "-0.0". It has to be recovered from
where the number came from.

edge and skill_ci_high are genuinely negative, just smaller than their
own rounding. Collapsing those would not tidy them, it would state
something false. edge is the sharper case: the same row publishes
stance "model_below_market", which is computed from the unrounded
`signed < 0`. Publish edge as 0.0 and the row contradicts itself.

open_exposure is a true zero, like profit_edge -- a residual of about
-1e-15 left by adding each stake back at settlement. Fixing it is
correct but it changes docs/autonomous-twin/REPLAY_BASELINE.json, a
documented reproducibility baseline whose value is that it does not
drift. Rewriting that hash for two cosmetic zeros is not a trade to
make quietly, so it is left for a decision rather than folded in here.
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


class TheOtherSignedZerosAreRealTests(unittest.TestCase):
    """Guard the scope: these must NOT be collapsed.

    Without this, the obvious follow-up is to apply _published_round
    everywhere a -0.0 was seen, which would be wrong three times out of
    four.
    """

    def test_a_small_negative_edge_is_not_the_same_as_agreement(self):
        """The board publishes edge and stance from the same number.

        stance reads the unrounded value, so a row with a genuinely
        negative edge says "model_below_market" while the rounded edge
        prints as -0.0. Collapsing the edge would leave the row
        disagreeing with itself.
        """
        signed = -0.0004
        stance = (
            "model_above_market" if signed > 0
            else "model_below_market" if signed < 0 else "agree"
        )
        self.assertEqual(stance, "model_below_market")
        self.assertTrue(
            is_negative(round(signed, 3)),
            "rounding must keep the sign of a real negative",
        )
        self.assertFalse(
            is_negative(_published_round(signed, 3)),
            "which is exactly why _published_round is not used here",
        )

    def test_the_sign_of_a_zero_is_the_only_thing_being_dropped(self):
        """A true zero and a small negative both print -0.0 at 4dp."""
        artefact = 0.0 * -1.0
        genuine = -0.00001
        self.assertEqual(json.dumps(round(artefact, 4)), "-0.0")
        self.assertEqual(json.dumps(round(genuine, 4)), "-0.0")
        self.assertEqual(artefact, 0.0)
        self.assertNotEqual(genuine, 0.0)


if __name__ == "__main__":
    unittest.main()
