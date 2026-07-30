import unittest
from pathlib import Path


class FrontendCopyTests(unittest.TestCase):
    def test_edge_board_omits_paper_return_language(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        renderer = index.split("function renderEdgeBoard(d) {", 1)[1].split(
            "function _fmtDate", 1
        )[0]

        self.assertNotIn("paper pnl", renderer.lower())
        self.assertNotIn("paper trading", renderer.lower())
        self.assertNotIn("paper roi", renderer.lower())
        self.assertIn("resolved forecast quality", renderer.lower())
        self.assertIn("resolved strategy performance", renderer.lower())
        self.assertIn('id="eb-equity-svg"', renderer)

    def test_edge_board_mark_to_market_chart_tracks_scads_models(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("const _MTM_MODEL_COLORS", index)
        self.assertIn("'scads-alias-code':", index)
        self.assertIn("'scads-alias-reasoning':", index)
        self.assertIn("'kimi-k2.7-code':", index)
        self.assertIn('id="eb-mtm-svg"', index)
        self.assertIn('id="eb-mtm-table-body"', index)
        self.assertIn("head: _ebModelHead(m.model)", index)
        self.assertIn("raw.includes('kimi-k2.7')", index)
        self.assertIn("const padL = 60, padR = 34, padT = 26", index)
        self.assertIn("legendRowH = 17", index)
        self.assertIn("_equitySvg(curves, 760, 180", index)
        self.assertIn('id="eb-model-comparison-svg"', index)
        self.assertIn("d.mark_to_market_by_model", index)
        self.assertIn("data-eq-id", index)
        self.assertIn("const _equityChartStates = new Map()", index)
        self.assertIn("_attachEdgeBoardChartHovers(host)", index)
        self.assertIn("'#eb-model-comparison-svg', '#eb-mtm-svg', '#eb-equity-svg'", index)

    def test_edge_board_growth_curve_normalizes_multiplier_to_bankroll(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        normalizer = index.split("function _ebNormalizeBankrollCurve(s) {", 1)[1].split(
            "function _ebPnlCurve", 1
        )[0]

        self.assertIn("maxAbs <= 10", normalizer)
        self.assertIn("v * EB_STARTING_BANKROLL", normalizer)
        self.assertIn("v * (EB_STARTING_BANKROLL / 100)", normalizer)

    def test_edge_board_equity_chart_insets_points_from_edges(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        renderer = index.split("function _equitySvg(", 1)[1].split(
            "function _attachEquityHover", 1
        )[0]

        self.assertIn("const pointInsetX", renderer)
        self.assertIn("const innerW", renderer)
        self.assertIn("padL + pointInsetX + ((t - tMin) / tSpan) * innerW", renderer)
        self.assertIn("pointInsetX, innerW", renderer)


if __name__ == "__main__":
    unittest.main()
