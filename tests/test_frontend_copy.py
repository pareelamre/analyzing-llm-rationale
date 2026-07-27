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


if __name__ == "__main__":
    unittest.main()
