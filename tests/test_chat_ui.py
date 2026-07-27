from __future__ import annotations

import unittest
from pathlib import Path


class ChatUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_html = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

    def test_composer_uses_the_shell_as_its_focus_indicator(self) -> None:
        self.assertIn(
            ".input-box #questionInput:focus-visible {\n      outline: none;",
            self.index_html,
        )

    def test_hidden_live_status_does_not_leave_an_empty_pill(self) -> None:
        self.assertIn(
            ".chat-live-status {\n      display: none !important;",
            self.index_html,
        )
        self.assertIn(
            'class="chat-live-status" aria-hidden="true"',
            self.index_html,
        )

    def test_chat_content_stays_in_the_shared_reading_column(self) -> None:
        self.assertIn(
            "div.className = 'max-w-2xl mx-auto w-full flex justify-end';",
            self.index_html,
        )
        self.assertIn("lg:max-w-[640px]", self.index_html)


if __name__ == "__main__":
    unittest.main()
