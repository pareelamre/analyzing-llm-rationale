"""A failed notes read used to wipe every agent's notebook.

``_load_notes`` reported any read failure as an empty notebook, and
``manage_notes`` saves back whatever it loaded. So one unreadable read --
a file lock, a half-finished GCS sync, a truncated write -- turned the next
``add`` into a single-note file written over all 50 notes, for every agent
in the file. Nothing said anything: the agent saw ``ok: true``.

It was caught by a test failing once in a full local run, then passing on
its own: the notebook the test had just filled came back empty.

Now a read that failed raises, so nothing saves over the file, and content
that cannot be parsed is moved to a ``.corrupt-`` name instead of being
dropped.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyzing_llm_rationale import benchmark_tools  # noqa: E402

CTX = benchmark_tools.ToolContext(agent_id="model-a")
EXISTING = {
    "model-a": [{"id": "n1", "text": "Fed Sep 2026: model 45% vs market 42%.", "tags": ["fed"]}],
    "model-b": [{"id": "n2", "text": "CPI print lands Thursday.", "tags": []}],
}


class NotesReadFailureTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "notes.json"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")

    def add(self, text="new observation"):
        return benchmark_tools.manage_notes({"action": "add", "text": text}, CTX, path=self.path)

    def corrupt_files(self):
        return [p for p in self.dir.iterdir() if ".corrupt-" in p.name]

    def test_a_transient_read_failure_does_not_wipe_the_notebook(self):
        self.write(json.dumps(EXISTING))
        before = self.path.read_bytes()
        with mock.patch.object(Path, "read_text", side_effect=OSError("file is locked")):
            result = self.add()
        self.assertFalse(result["ok"], result)
        self.assertEqual(self.path.read_bytes(), before, "the notebook must be untouched")
        self.assertEqual(self.corrupt_files(), [], "a locked file is not corrupt")

    def test_a_transient_read_failure_is_raised_not_reported_as_empty(self):
        self.write(json.dumps(EXISTING))
        with mock.patch.object(Path, "read_text", side_effect=OSError("file is locked")):
            with self.assertRaises(benchmark_tools.NotesUnreadableError):
                benchmark_tools._load_notes(self.path)

    def test_unparseable_content_is_kept_aside_rather_than_overwritten(self):
        self.write('{"model-a": [{"text": "half a writ')
        original = self.path.read_bytes()

        result = self.add("a fresh note")

        self.assertTrue(result["ok"], result)
        kept = self.corrupt_files()
        self.assertEqual(len(kept), 1, "the unreadable bytes must be preserved")
        self.assertEqual(kept[0].read_bytes(), original)
        # The agent keeps working: the rewritten file holds the new note.
        self.assertEqual(
            [n["text"] for n in json.loads(self.path.read_text())["model-a"]], ["a fresh note"],
        )

    def test_content_of_the_wrong_shape_is_kept_aside_too(self):
        self.write(json.dumps(["not", "a", "notebook"]))
        original = self.path.read_bytes()
        self.assertTrue(self.add()["ok"])
        kept = self.corrupt_files()
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_bytes(), original)

    def test_nothing_is_set_aside_when_the_bytes_cannot_be_preserved(self):
        self.write("{ broken")
        before = self.path.read_bytes()
        with mock.patch.object(Path, "replace", side_effect=OSError("cannot rename")):
            result = self.add()
        self.assertFalse(result["ok"], result)
        self.assertEqual(self.path.read_bytes(), before)

    def test_a_missing_notebook_is_still_simply_empty(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(benchmark_tools._load_notes(self.path), {})
        self.assertTrue(self.add("first note")["ok"])
        self.assertEqual(self.corrupt_files(), [])

    def test_a_readable_notebook_is_unchanged_in_behaviour(self):
        self.write(json.dumps(EXISTING))
        self.assertTrue(self.add("another note")["ok"])
        saved = json.loads(self.path.read_text())
        self.assertEqual([n["text"] for n in saved["model-b"]], ["CPI print lands Thursday."])
        self.assertEqual(len(saved["model-a"]), 2)
        self.assertEqual(self.corrupt_files(), [])


class ReadersThatOnlyDisplayNotesTests(unittest.TestCase):
    """Callers that never save may carry on without notes; they must not crash."""

    def test_the_cycle_prompt_omits_notes_it_cannot_read(self):
        import agent_trading_tick

        with mock.patch.object(
            benchmark_tools, "_load_notes",
            side_effect=benchmark_tools.NotesUnreadableError("locked"),
        ):
            self.assertEqual(agent_trading_tick._recalled_notes_block("model-a"), "")

    def test_the_board_publishes_without_the_notes_it_cannot_read(self):
        import build_agent_trading_board

        with mock.patch.object(
            benchmark_tools, "_load_notes",
            side_effect=benchmark_tools.NotesUnreadableError("locked"),
        ):
            self.assertEqual(build_agent_trading_board._load_model_notes("model-a"), {})


if __name__ == "__main__":
    unittest.main()
