from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import pr_agent  # noqa: E402


class PRAgentTests(unittest.TestCase):
    def test_packet_includes_agent_to_agent_install_metadata(self):
        packet = pr_agent.build_pr_agent_packet(audience="mcp", canonical="https://foresea.test/")

        self.assertEqual(packet["name"], "Foresea PR Agent")
        self.assertEqual(packet["audience"], "mcp")
        self.assertIn("no_spam_policy", packet)
        self.assertEqual(packet["links"]["mcp"], "https://foresea.test/mcp/")
        self.assertIn("foresea_pr_agent", packet["mcp"]["tools"])
        self.assertIn("claude mcp add", packet["mcp"]["install_command"])
        self.assertIn("live market track record is still accumulating", packet["message"])

    def test_unknown_audience_falls_back_to_agent(self):
        packet = pr_agent.build_pr_agent_packet(audience="unknown")

        self.assertEqual(packet["audience"], "agent")
        self.assertIn("prediction-market intelligence", packet["intro"])


if __name__ == "__main__":
    unittest.main()
