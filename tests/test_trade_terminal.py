from __future__ import annotations

import unittest
from pathlib import Path


class TradeTerminalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trade_html = (
            Path(__file__).resolve().parents[1] / "static" / "trade.html"
        ).read_text(encoding="utf-8")
        cls.frontend_trade_html = (
            Path(__file__).resolve().parents[1] / "frontend" / "trade.html"
        ).read_text(encoding="utf-8")

    def test_chat_handoff_requires_a_fresh_terminal_quote_and_manual_order_details(self) -> None:
        self.assertIn("function configureResearchHandoff", self.trade_html)
        self.assertIn("function renderVenueQuote", self.trade_html)
        self.assertIn("function setSuggestedLimitPrice", self.trade_html)
        self.assertIn("The chat only hands the terminal research context.", self.trade_html)
        self.assertIn("must set a current limit order", self.trade_html)
        self.assertIn("function consumeResearchHandoff", self.trade_html)
        self.assertIn("sessionStorage.removeItem(key)", self.trade_html)
        self.assertIn("body.expected_edge = researchContext.modelProbability - researchContext.marketProbability", self.trade_html)

    def test_terminal_does_not_render_fabricated_market_data(self) -> None:
        self.assertNotIn("loadMockOrderBook", self.trade_html)
        self.assertNotIn("marketOdds + 0.12", self.trade_html)
        self.assertIn("No executable venue quote is available", self.trade_html)
        self.assertIn("Refresh reconciliation before treating it as filled", self.trade_html)

    def test_terminal_uses_secure_connections_and_exposes_reconciliation(self) -> None:
        self.assertIn("/trading/connections", self.trade_html)
        self.assertNotIn("getStoredCreds", self.trade_html)
        self.assertNotIn("allVenueCreds", self.trade_html)
        self.assertIn("/trading/runs", self.trade_html)
        self.assertIn("TRADE RUN SUBMITTED", self.trade_html)
        self.assertIn("Portfolio reconciliation", self.trade_html)
        self.assertIn("/trading/portfolio?platform=", self.trade_html)
        self.assertIn("CANCEL OPEN ORDER", self.trade_html)
        self.assertIn("AWAITING RECONCILIATION", self.trade_html)

    def test_terminal_exposes_server_enforced_risk_controls(self) -> None:
        self.assertIn("Real-money risk controls", self.trade_html)
        self.assertIn("/trading/guardrails", self.trade_html)
        self.assertIn("function toggleTradingPause", self.trade_html)
        self.assertIn("Save lower limits", self.trade_html)
        self.assertIn("fresh quote, available balance, current exposure", self.trade_html)
        self.assertIn("platform_kill_switch", self.trade_html)

    def test_frontend_source_terminal_cannot_reintroduce_browser_credential_storage(self) -> None:
        self.assertIn("/trading/connections", self.frontend_trade_html)
        self.assertNotIn("getStoredCreds", self.frontend_trade_html)
        self.assertNotIn("allVenueCreds", self.frontend_trade_html)
        self.assertNotIn("foresea_venue_creds", self.frontend_trade_html)
        self.assertNotIn("venue_credentials: ", self.frontend_trade_html)


if __name__ == "__main__":
    unittest.main()
