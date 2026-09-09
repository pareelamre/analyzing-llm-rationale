import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TwinOperatorUiTests(unittest.TestCase):
    def test_operator_desk_has_accessible_state_and_control_surfaces(self):
        page = (ROOT / "frontend" / "trade.html").read_text(encoding="utf-8")
        script = (ROOT / "frontend" / "src" / "twin-desk.ts").read_text(encoding="utf-8")

        for identity in (
            "twinOperatorDesk", "twinConnectionNotice", "twinPortfolioList",
            "twinBlockerList", "twinDecisionList", "twinCommandList",
            "twinMandateForm", "twinApproveBtn", "twinRevokeBtn", "twinPauseBtn",
        ):
            self.assertIn(f'id="{identity}"', page)
        for endpoint in (
            "/twin/status", "/twin/portfolio", "/twin/readiness",
            "/twin/decisions", "/twin/commands", "/twin/mandates", "/twin/pause",
        ):
            self.assertIn(endpoint, script)
        self.assertIn("ACTIVATE ${activeMandate.live ? \"LIVE\" : \"SHADOW\"}", script)
        self.assertIn("REVOKE AUTONOMY", script)
        self.assertIn("submission_unknown", script)
        self.assertIn("partially_filled", script)
        self.assertNotIn("connection_ref", script)
        self.assertNotIn("venue_account_ref", script)

    def test_built_trade_page_references_the_compiled_operator_asset(self):
        built = (ROOT / "static" / "trade.html").read_text(encoding="utf-8")
        match = re.search(r'src="/static/assets/(trade-[A-Za-z0-9_-]+\.js)"', built)
        self.assertIsNotNone(match)
        asset = ROOT / "static" / "assets" / match.group(1)
        self.assertTrue(asset.is_file())
        compiled = asset.read_text(encoding="utf-8")
        self.assertIn("/twin/status", compiled)
        self.assertIn("REVOKE AUTONOMY", compiled)


if __name__ == "__main__":
    unittest.main()
