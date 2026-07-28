from __future__ import annotations

import unittest
from pathlib import Path


class ChatTurnUxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_html = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

    def test_forecast_wait_state_is_accessible_and_server_driven(self) -> None:
        self.assertIn('class="msg-ai forecast-wait"', self.index_html)
        self.assertIn(
            'role="status" aria-live="polite" aria-atomic="true"',
            self.index_html,
        )
        self.assertIn(
            "typingEl._setForecastProgress?.(data.status, data);",
            self.index_html,
        )
        self.assertNotIn("const phases = ['Gathering evidence", self.index_html)

    def test_wait_state_respects_reduced_motion(self) -> None:
        self.assertIn("@media (prefers-reduced-motion: reduce)", self.index_html)
        self.assertIn(".forecast-wait-orbit", self.index_html)
        self.assertIn(".forecast-wait-scan::after", self.index_html)

    def test_wait_experience_emits_bounded_product_events(self) -> None:
        self.assertIn("trackEvent('forecast_wait_started', { surface });", self.index_html)
        self.assertIn("trackEvent('forecast_wait_phase'", self.index_html)
        self.assertIn("trackEvent('forecast_wait_finished'", self.index_html)
        self.assertIn("duration_bucket: _waitDurationBucket", self.index_html)

    def test_streamed_turn_settles_without_redrawing_the_thread(self) -> None:
        self.assertIn("settlePromptActions(request.userMessageId);", self.index_html)
        finally_block = self.index_html.split(
            "} finally {\n    if (activeRequest === request) {",
            1,
        )[1].split("\n  }\n}", 1)[0]
        self.assertNotIn("renderMessages();", finally_block)

    def test_stop_cancels_reader_and_preserves_partial_answer(self) -> None:
        self.assertIn("activeRequest.cancelStream?.();", self.index_html)
        self.assertIn("reader.cancel().catch(() => {});", self.index_html)
        self.assertIn("if (streamCancelled || signal.aborted)", self.index_html)
        self.assertIn("if (text) finish(null, 'cancelled');", self.index_html)
        self.assertIn("data.generation_stopped = true;", self.index_html)

    def test_response_is_scoped_to_its_originating_conversation(self) -> None:
        self.assertIn(
            "appendMessage(request.convId, { role: 'assistant'",
            self.index_html,
        )
        self.assertIn("activeId !== convId", self.index_html)
        self.assertIn(
            "if (activeRequest?.convId === id) stopGeneration();",
            self.index_html,
        )

    def test_agent_progress_advances_from_server_events(self) -> None:
        self.assertIn("typingEl._setAgentProgress?.(label);", self.index_html)
        self.assertIn("div._setAgentProgress = label =>", self.index_html)
        self.assertNotIn("}, 1100);", self.index_html)


if __name__ == "__main__":
    unittest.main()
