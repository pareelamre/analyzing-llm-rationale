import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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

    def test_report_to_markdown_never_renders_raw_grounding_note(self):
        # d.grounding is internal self-calibration API metadata, not answer
        # text -- it must never be concatenated into the visible chat bubble
        # markdown, regardless of which model produced the report.
        renderer = self.index.split("function _agentReportToMarkdown(", 1)[1].split(
            "function _isDiscoveryRequest", 1
        )[0]

        self.assertNotIn("d.grounding", renderer)

    def test_report_to_markdown_places_edge_prompt_last(self):
        # The edge-prompt caption should read as an invitation to keep
        # chatting, so it must be the last thing appended -- after skills
        # and sources, not spliced in the middle of the reply.
        renderer = self.index.split("function _agentReportToMarkdown(", 1)[1].split(
            "function _isDiscoveryRequest", 1
        )[0]

        self.assertIn("Curious if there's an edge?", renderer)
        skills_pos = renderer.index("(d.skills || [])")
        sources_pos = renderer.index("includeSources && !isChatReply")
        prompt_pos = renderer.index("Curious if there's an edge?")
        self.assertLess(skills_pos, prompt_pos)
        self.assertLess(sources_pos, prompt_pos)
        # Must be the last md += before the function returns.
        return_pos = renderer.index("return md;")
        self.assertLess(prompt_pos, return_pos)
        after_prompt = renderer[renderer.index("md += `\\n\\n*Curious"):return_pos]
        self.assertEqual(after_prompt.count("md +="), 1)


if __name__ == "__main__":
    unittest.main()
