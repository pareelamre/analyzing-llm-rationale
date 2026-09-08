"""Every resolved snapshot lands in exactly one edge bucket.

edge_calibration slices the track record by disagreement size and reports
skill per slice. It is the evidence behind statements about which edges
have paid -- including the 20pp+ bucket's measured skill.

The slicing is `lo <= _edge(r) < hi`. Making the lower bound exclusive
passed the suite, and it does not merely reshuffle rows: with `lo <`, a
snapshot whose edge is exactly 0.20 leaves 20pp+ and cannot enter 10-20pp
either, because that bucket ends below 0.20. It is dropped from the
report entirely, and the remaining counts still look reasonable.

Exact boundary values are not rare. Probabilities are published rounded,
so 0.70 against 0.50 gives exactly 0.20, and a model agreeing with the
market gives exactly 0.0.

These tests assert the partition rather than the boundaries one at a
time: every row appears once, and the bucket counts sum to the input.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.track_record_live import (  # noqa: E402
    _EDGE_BUCKETS,
    _edge,
    _edge_label,
    edge_calibration,
)


def _row(model_p, market_p, *, correct=True, model_brier=0.20, market_brier=0.25):
    return {
        "model_probability": model_p,
        "market_probability": market_p,
        "model_brier": model_brier,
        "market_brier": market_brier,
        "model_correct": correct,
    }


#: One row per bucket boundary, plus interior points.
_EDGES = [0.0, 0.01, 0.049, 0.05, 0.07, 0.099, 0.10, 0.15, 0.199, 0.20, 0.35, 0.90]


class ThePartitionTests(unittest.TestCase):
    def _rows(self):
        return [_row(round(0.50 + e, 4), 0.50) for e in _EDGES]

    def test_the_bucket_counts_sum_to_the_input(self):
        rows = self._rows()
        report = edge_calibration(rows)
        self.assertEqual(sum(b["n"] for b in report), len(rows))

    def test_no_snapshot_falls_between_two_buckets(self):
        """The failure the exclusive bound produces: a row in no bucket."""
        for edge in _EDGES:
            with self.subTest(edge=edge):
                matches = [
                    label for label, lo, hi in _EDGE_BUCKETS if lo <= edge < hi
                ]
                self.assertEqual(len(matches), 1, f"edge {edge} matched {matches}")

    def test_the_boundaries_belong_to_the_upper_bucket(self):
        self.assertEqual(_edge_label(0.0), "0-5pp")
        self.assertEqual(_edge_label(0.05), "5-10pp")
        self.assertEqual(_edge_label(0.10), "10-20pp")
        self.assertEqual(_edge_label(0.20), "20pp+")

    def test_just_below_a_boundary_stays_in_the_lower_bucket(self):
        self.assertEqual(_edge_label(0.0499), "0-5pp")
        self.assertEqual(_edge_label(0.0999), "5-10pp")
        self.assertEqual(_edge_label(0.1999), "10-20pp")

    def test_a_row_at_a_boundary_is_still_reported_somewhere(self):
        """It is counted. Which side it falls on is a separate question.

        0.70 against 0.50 is 0.19999999999999996, not 0.20, so it lands in
        10-20pp rather than 20pp+. Of the 158 two-decimal pairs exactly
        twenty cents apart, 74 fall short of the boundary this way and 84
        clear it -- so pairs a reader would call identical are split between
        buckets by float representation alone.

        That is recorded rather than corrected here: rounding the edge before
        bucketing would move rows between published skill figures, and the
        20pp+ bucket is one people are actively reasoning about. The property
        that matters for correctness -- nothing is lost -- holds either way,
        and is what this asserts.
        """
        report = edge_calibration([_row(0.70, 0.50)])
        self.assertEqual(sum(b["n"] for b in report), 1)
        self.assertAlmostEqual(_edge(_row(0.70, 0.50)), 0.20, places=9)
        self.assertLess(_edge(_row(0.70, 0.50)), 0.20)

    def test_pairs_that_look_identical_can_land_in_different_buckets(self):
        """The float effect, stated so it is not rediscovered as a surprise."""
        twenty_apart = [(a / 100.0, b / 100.0)
                        for a in range(1, 100) for b in range(1, 100)
                        if abs(a - b) == 20]
        short = [p for p in twenty_apart if abs(p[0] - p[1]) < 0.20]
        self.assertGreater(len(short), 0)
        self.assertLess(len(short), len(twenty_apart))

    def test_a_model_agreeing_with_the_market_is_still_counted(self):
        """Edge exactly 0.0. Dropped entirely under an exclusive bound."""
        report = edge_calibration([_row(0.50, 0.50)])
        self.assertEqual(sum(b["n"] for b in report), 1)
        self.assertEqual([b["edge_bucket"] for b in report], ["0-5pp"])


class TheBucketsAreContiguousTests(unittest.TestCase):
    def test_each_bucket_starts_where_the_next_ends(self):
        """Structural: no gap and no overlap, whatever the numbers become."""
        ordered = sorted(_EDGE_BUCKETS, key=lambda b: b[1])
        self.assertEqual(ordered[0][1], 0.0)
        for lower, upper in zip(ordered, ordered[1:]):
            self.assertEqual(lower[2], upper[1], f"{lower[0]} does not meet {upper[0]}")
        self.assertEqual(ordered[-1][2], float("inf"))


class TheEdgeItselfTests(unittest.TestCase):
    def test_disagreement_is_symmetric(self):
        """Being 20 points under the market is the same size gap as over."""
        self.assertAlmostEqual(_edge(_row(0.70, 0.50)), _edge(_row(0.30, 0.50)))

    def test_a_missing_probability_reads_as_zero_not_an_error(self):
        self.assertAlmostEqual(_edge({"model_probability": None,
                                      "market_probability": 0.50}), 0.50)


if __name__ == "__main__":
    unittest.main()
