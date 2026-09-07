"""The two thresholds that decide whether an arbitrage is reported.

scan_cross_venue_arbitrage gates on `overlap >= min_overlap` and then
`spread >= min_spread`. Both were unpinned: making either exclusive passed
the suite, and a market pair sitting exactly on a threshold is the one a
scanner is most likely to be asked about.

The fixtures here are chosen, not convenient. A spread of "three cents"
usually is not 0.03: of the 9,801 cent pairs, 152 differ by three cents
yet compare strictly greater than 0.03 in floating point, and only 40 are
exactly equal. A test built on 0.02 vs 0.05 therefore passes whether the
comparison is `>=` or `>` and pins nothing. These use pairs from the 40.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.arbitrage_scanner import (  # noqa: E402
    _compute_keyword_overlap,
    scan_cross_venue_arbitrage,
)

#: Seven shared tokens, six unique to A, seven unique to B: 7/20 = 0.35.
_SHARED = "alpha bravo charlie delta echo foxtrot golf"
_TITLE_A = _SHARED + " hotel india juliet kilo lima mike"
_TITLE_B = _SHARED + " november oscar papa quebec romeo sierra tango"


def _scan(p_prob, k_prob, *, title_a=None, title_b=None, **kwargs):
    return scan_cross_venue_arbitrage(
        polymarket_markets=[{"question": title_a or _TITLE_A, "probability": p_prob}],
        kalshi_markets=[{"question": title_b or _TITLE_A, "probability": k_prob}],
        **kwargs,
    )


class SpreadThresholdTests(unittest.TestCase):
    def test_a_spread_exactly_at_the_minimum_is_reported(self):
        self.assertEqual(abs(0.08 - 0.05), 0.03)  # exact, unlike most cent pairs
        found = _scan(0.05, 0.08, min_spread=0.03)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["spread"], 0.03)

    def test_a_spread_below_the_minimum_is_not(self):
        self.assertEqual(len(_scan(0.05, 0.07, min_spread=0.03)), 0)

    def test_the_fixture_really_sits_on_the_boundary(self):
        """Guard the fixture itself.

        If this ever stops being exactly equal, the boundary test above
        silently becomes a test of 'comfortably above the threshold'.
        """
        self.assertEqual(abs(0.08 - 0.05), 0.03)
        self.assertNotEqual(abs(0.05 - 0.02), 0.03)  # the naive choice


class OverlapThresholdTests(unittest.TestCase):
    def test_an_overlap_exactly_at_the_minimum_is_scanned(self):
        self.assertEqual(_compute_keyword_overlap(_TITLE_A, _TITLE_B), 0.35)
        found = _scan(0.05, 0.08, title_b=_TITLE_B, min_overlap=0.35, min_spread=0.03)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["overlap_score"], 0.35)

    def test_an_overlap_below_the_minimum_is_skipped(self):
        weaker = _SHARED + " november oscar papa quebec romeo sierra tango uniform"
        self.assertLess(_compute_keyword_overlap(_TITLE_A, weaker), 0.35)
        found = _scan(0.05, 0.08, title_b=weaker, min_overlap=0.35, min_spread=0.03)
        self.assertEqual(len(found), 0)

    def test_the_overlap_fixture_is_exactly_seven_twentieths(self):
        self.assertEqual(_compute_keyword_overlap(_TITLE_A, _TITLE_B), 7 / 20)


class DirectionTests(unittest.TestCase):
    def test_the_cheaper_venue_is_the_one_bought(self):
        [found] = _scan(0.05, 0.08, min_spread=0.03)
        self.assertEqual(found["executable_action"]["long_venue"], "Polymarket")
        self.assertAlmostEqual(found["executable_action"]["long_price"], 0.05)

        [other] = _scan(0.08, 0.05, min_spread=0.03)
        self.assertEqual(other["executable_action"]["long_venue"], "Kalshi")
        self.assertAlmostEqual(other["executable_action"]["long_price"], 0.05)


if __name__ == "__main__":
    unittest.main()
