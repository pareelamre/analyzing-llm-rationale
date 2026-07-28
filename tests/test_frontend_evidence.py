import unittest
from pathlib import Path


class FrontendEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

    def test_empty_evidence_has_a_visible_status(self):
        renderer = self.index.split("function sourceFeedHtml(", 1)[1].split(
            "function makeAIBubble", 1
        )[0]

        self.assertIn("evidenceError", renderer)
        self.assertIn("Evidence status", renderer)
        self.assertIn("This forecast is evidence-limited.", renderer)
        self.assertIn('href="${escAttr(s.url)}"', renderer)
        self.assertIn("sourceFeedHtml(sources, data.evidence_error)", self.index)

    def test_agent_report_preserves_evidence_metadata(self):
        adapter = self.index.split("function _agentReportData(", 1)[1].split(
            "function _agentReportToMarkdown", 1
        )[0]

        self.assertIn("_agentReportToMarkdown(d, { includeSources: false })", adapter)
        self.assertIn("evidence_sources: d.evidence_sources || []", adapter)
        self.assertIn("evidence_error: d.evidence_error || ''", adapter)
        self.assertGreaterEqual(self.index.count("_agentReportData(report)"), 2)
        self.assertIn("streamedEvidenceSources = data.evidence_sources", self.index)
        self.assertIn("evidence_sources: streamedEvidenceSources", self.index)
        self.assertIn("const includeSources = options.includeSources !== false;", self.index)


if __name__ == "__main__":
    unittest.main()
