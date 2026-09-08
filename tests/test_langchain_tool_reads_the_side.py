"""The edge-board tool told an agent to BUY every opportunity.

ForeseaEdgeBoardTool rendered `o.get("recommendation", "BUY")`. No row
carries `recommendation`. Checked against the live board on 2026-09-07:

    recommendation   present on  0/27 rows
    side             present on 27/27 rows
    side values      Counter({'NO': 15, 'YES': 11, None: 1})

So every line read "Action: BUY", including the fifteen the board rates
NO -- a majority -- and the reader is an LLM agent choosing a trade. The
fallback looked like a default and behaved like an answer.

Same shape as #533 (optimizer reading `ticker`, which the board does not
publish) and #558 (whale flow reading `price`/`size`, which the venue
tapes do not publish).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.langchain_tools import _recommendation_of  # noqa: E402

#: The keys a real edge-board row carries, trimmed to what this tool reads.
BOARD_ROW = {
    "question": "Will the Fed cut rates before July 31, 2026?",
    "platform": "Kalshi",
    "ident": "KXFED-26JUL-H",
    "edge": -0.12,
    "market_probability": 0.62,
    "model_probability": 0.50,
    "side": "NO",
    "stance": "fade",
}


class RecommendationFollowsTheSideTests(unittest.TestCase):
    def test_a_no_row_is_not_rendered_as_buy(self):
        self.assertEqual(_recommendation_of(BOARD_ROW), "BUY NO")

    def test_a_yes_row_says_yes(self):
        self.assertEqual(_recommendation_of(dict(BOARD_ROW, side="YES")), "BUY YES")

    def test_the_side_is_read_case_and_space_insensitively(self):
        for raw in ("no", " No ", "NO"):
            with self.subTest(raw=raw):
                self.assertEqual(_recommendation_of(dict(BOARD_ROW, side=raw)), "BUY NO")

    def test_a_row_without_a_side_does_not_invent_one(self):
        """One live row had side None. Guessing BUY there is the bug."""
        row = {k: v for k, v in BOARD_ROW.items() if k != "side"}
        self.assertNotIn("BUY", _recommendation_of(row))

    def test_it_falls_back_to_the_stance_before_giving_up(self):
        row = {k: v for k, v in BOARD_ROW.items() if k != "side"}
        self.assertEqual(_recommendation_of(row), "FADE")

    def test_with_neither_it_asks_for_review_rather_than_acting(self):
        row = {k: v for k, v in BOARD_ROW.items() if k not in ("side", "stance")}
        self.assertEqual(_recommendation_of(row), "REVIEW")

    def test_the_absent_key_is_not_consulted(self):
        """`recommendation` is never published; honouring it would re-hide
        the bug behind a key that only appears in tests."""
        row = dict(BOARD_ROW, recommendation="BUY")
        self.assertEqual(_recommendation_of(row), "BUY NO")


class TheOldFallbackWasNotADefaultTests(unittest.TestCase):
    def test_the_live_distribution_makes_it_wrong_more_often_than_right(self):
        """Recorded so the cost is legible: 15 NO against 11 YES."""
        rows = [dict(BOARD_ROW, side="NO") for _ in range(15)]
        rows += [dict(BOARD_ROW, side="YES") for _ in range(11)]
        wrong = sum(1 for r in rows if _recommendation_of(r) != "BUY YES")
        self.assertEqual(wrong, 15)
        self.assertGreater(wrong, len(rows) / 2)


if __name__ == "__main__":
    unittest.main()


class TheToolRendersTheSideTests(unittest.TestCase):
    """Pin the call site, not only the helper.

    Restoring `o.get("recommendation", "BUY")` inside _run survived the tests
    above, because they exercised _recommendation_of directly. That is the
    same gap #539 found: the helper was held, the place that uses it was not.
    """

    def setUp(self):
        # The module degrades by binding the tool names to None rather than
        # leaving them undefined, so the import succeeds without langchain and
        # only the value says so. Catching ImportError here never fired, and
        # CI -- which has no langchain -- got `'NoneType' object is not
        # callable` instead of a skip.
        from analyzing_llm_rationale.langchain_tools import ForeseaEdgeBoardTool

        if ForeseaEdgeBoardTool is None:
            raise unittest.SkipTest("langchain-core is not available")
        self.tool_cls = ForeseaEdgeBoardTool

    def _render(self, rows):
        class _Client:
            def get_edge_board(self, min_edge=0.05, limit=5):
                return rows

        return self.tool_cls(client=_Client())._run()

    def test_a_no_row_renders_as_buy_no(self):
        out = self._render([dict(BOARD_ROW, side="NO")])
        self.assertIn("BUY NO", out)

    def test_a_no_row_does_not_render_a_bare_buy(self):
        out = self._render([dict(BOARD_ROW, side="NO")])
        self.assertNotIn("Action: BUY |", out)

    def test_a_yes_row_renders_as_buy_yes(self):
        self.assertIn("BUY YES", self._render([dict(BOARD_ROW, side="YES")]))

    def test_the_live_mix_renders_both_actions(self):
        out = self._render([
            dict(BOARD_ROW, side="NO"), dict(BOARD_ROW, side="YES"),
        ])
        self.assertIn("BUY NO", out)
        self.assertIn("BUY YES", out)
