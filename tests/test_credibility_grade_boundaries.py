"""The score at which a market becomes tradable.

audit_edge_opportunity grades on two thresholds and sets is_credible from
them. The optimizer filters on that score (min_credibility_score=0.60,
"Grade A & B"), so these two comparisons decide which markets receive
capital.

Both were unpinned: changing `>= 0.80` to `> 0.80` and `>= 0.60` to
`> 0.60` passed the whole suite. That is not a measure-zero edge case --
final_score is `round(score, 2)`, so it lands on a 0.01 grid and hits the
boundaries exactly with ordinary inputs.

The inputs below were found by searching the input space for scores that
land exactly on each threshold, not chosen to be convenient.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.edge_credibility import (  # noqa: E402
    audit_edge_opportunity,
)


def _opp(model_p, market_p, *, volume, evidence, criteria="", horizon="30d"):
    return {
        "question": "Will X happen?",
        "model_probability": model_p,
        "market_probability": market_p,
        "resolution_criteria": criteria,
        "market_volume": volume,
        "evidence": [{"i": i} for i in range(evidence)],
        "horizon": horizon,
    }


class GradeBoundariesAreInclusiveTests(unittest.TestCase):
    def test_exactly_eighty_is_an_A(self):
        result = audit_edge_opportunity(
            _opp(0.51, 0.50, volume=0, evidence=1, horizon=""),
        )
        self.assertEqual(result["credibility_score"], 0.80)
        self.assertEqual(result["credibility_grade"], "A")
        self.assertTrue(result["is_credible"])

    def test_just_below_eighty_is_a_B(self):
        result = audit_edge_opportunity(
            _opp(0.51, 0.50, volume=None, evidence=0, horizon=""),
        )
        self.assertEqual(result["credibility_score"], 0.75)
        self.assertEqual(result["credibility_grade"], "B")

    def test_exactly_sixty_is_a_B_and_still_tradable(self):
        """0.60 is the optimizer's own min_credibility_score.

        If this boundary became exclusive, a market scoring exactly the
        documented minimum would be graded C and filtered out.
        """
        result = audit_edge_opportunity(
            _opp(0.01, 0.52, volume=5000, evidence=0),
        )
        self.assertEqual(result["credibility_score"], 0.60)
        self.assertEqual(result["credibility_grade"], "B")
        self.assertTrue(result["is_credible"])

    def test_just_below_sixty_is_a_C_and_not_tradable(self):
        result = audit_edge_opportunity(
            _opp(0.51, 0.50, volume=0, evidence=0, horizon=""),
        )
        self.assertEqual(result["credibility_score"], 0.55)
        self.assertEqual(result["credibility_grade"], "C")
        self.assertFalse(result["is_credible"])


class TheScoreLandsOnTheBoundaryTests(unittest.TestCase):
    def test_the_score_is_rounded_to_a_hundredth(self):
        """Why the boundary is reachable rather than measure-zero.

        final_score = max(0.0, min(1.0, round(score, 2))), so every score is
        a multiple of 0.01 and 0.60 and 0.80 are values it actually takes.
        """
        for opp in (
            _opp(0.51, 0.50, volume=0, evidence=1, horizon=""),
            _opp(0.01, 0.52, volume=5000, evidence=0),
            _opp(0.51, 0.50, volume=0, evidence=0, horizon=""),
        ):
            score = audit_edge_opportunity(opp)["credibility_score"]
            with self.subTest(score=score):
                self.assertAlmostEqual(score, round(score, 2), places=10)

    def test_grade_and_credibility_never_disagree(self):
        """is_credible is what the optimizer filters on; the grade is what a
        human reads. A market cannot be shown as C and funded, or A and
        withheld."""
        for opp in (
            _opp(0.51, 0.50, volume=0, evidence=1, horizon=""),
            _opp(0.51, 0.50, volume=None, evidence=0, horizon=""),
            _opp(0.01, 0.52, volume=5000, evidence=0),
            _opp(0.51, 0.50, volume=0, evidence=0, horizon=""),
        ):
            result = audit_edge_opportunity(opp)
            with self.subTest(grade=result["credibility_grade"]):
                self.assertEqual(
                    result["is_credible"],
                    result["credibility_grade"] in ("A", "B"),
                )


if __name__ == "__main__":
    unittest.main()
