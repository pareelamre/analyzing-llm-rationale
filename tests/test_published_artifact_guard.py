"""The check that stops a branch rewinding published data.

On 2026-09-07 static/agent_trading_live.json was published at 20:21 UTC.
PR #546, opened at 21:11 and merged at 21:14, carried the 18:23 copy and
put it back. For the next 25 minutes the live board served a three-hour-old
payload while its own audit index -- written by the same run -- still read
20:21, so the board and its audit disagreed by two hours.

Nothing failed. The file is valid JSON either way and no test reads it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from check_published_artifacts import (  # noqa: E402
    OK,
    PUBLISHED_ARTIFACTS,
    REWOUND,
    _generated_at,
    check,
)

ARTIFACT = "static/agent_trading_live.json"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git",) + args, cwd=repo, capture_output=True, text=True, check=True,
    ).stdout


class _Repo:
    """A throwaway git repo with one artifact, to exercise the real check."""

    def __init__(self, stack):
        self.path = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        _git(self.path, "init", "-q", "-b", "main")
        _git(self.path, "config", "user.email", "t@example.com")
        _git(self.path, "config", "user.name", "t")
        (self.path / "static").mkdir()

    def publish(self, generated_at: str, message: str) -> str:
        target = self.path / ARTIFACT
        target.write_text(
            json.dumps({"generated_at": generated_at, "leaderboard": []}),
            encoding="utf-8",
        )
        _git(self.path, "add", "-A")
        _git(self.path, "commit", "-q", "-m", message)
        return _git(self.path, "rev-parse", "HEAD").strip()


class RewindDetectionTests(unittest.TestCase):
    def setUp(self):
        import contextlib

        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.repo = _Repo(self.stack)
        self._cwd = Path.cwd()
        import os

        os.chdir(self.repo.path)
        self.addCleanup(lambda: os.chdir(self._cwd))

    def test_the_real_incident_is_caught(self):
        """The exact timestamps from 2026-09-07."""
        base = self.repo.publish("2026-09-07T20:21:00.723088+00:00", "publish 20:21")
        self.repo.publish("2026-09-07T18:23:10.144676+00:00", "a branch puts back 18:23")

        status, messages = check(base, [ARTIFACT])

        self.assertEqual(status, REWOUND)
        joined = "\n".join(messages)
        self.assertIn(ARTIFACT, joined)
        self.assertIn("1:57:50", joined)

    def test_publishing_forward_is_fine(self):
        base = self.repo.publish("2026-09-07T18:23:10+00:00", "publish 18:23")
        self.repo.publish("2026-09-07T20:21:00+00:00", "publish 20:21")

        status, _ = check(base, [ARTIFACT])
        self.assertEqual(status, OK)

    def test_a_branch_that_leaves_it_alone_passes(self):
        base = self.repo.publish("2026-09-07T20:21:00+00:00", "publish")
        (self.repo.path / "unrelated.py").write_text("x = 1\n", encoding="utf-8")
        _git(self.repo.path, "add", "-A")
        _git(self.repo.path, "commit", "-q", "-m", "unrelated change")

        status, messages = check(base, [ARTIFACT])
        self.assertEqual(status, OK)
        self.assertEqual(messages, [])

    def test_an_artifact_absent_from_the_base_is_not_a_rewind(self):
        base = self.repo.publish("2026-09-07T20:21:00+00:00", "publish")
        status, _ = check(base, ["static/not_yet_published.json"])
        self.assertEqual(status, OK)

    def test_an_equal_timestamp_is_not_a_rewind(self):
        """Two branches off the same publish must not fail each other."""
        base = self.repo.publish("2026-09-07T20:21:00+00:00", "publish")
        (self.repo.path / "unrelated.py").write_text("x = 1\n", encoding="utf-8")
        _git(self.repo.path, "add", "-A")
        _git(self.repo.path, "commit", "-q", "-m", "same artifact, other change")

        status, _ = check(base, [ARTIFACT])
        self.assertEqual(status, OK)


class TimestampParsingTests(unittest.TestCase):
    def test_it_reads_the_field(self):
        blob = json.dumps({"generated_at": "2026-09-07T20:21:00+00:00"})
        self.assertIsNotNone(_generated_at(blob))

    def test_unusable_input_is_none_rather_than_an_exception(self):
        for blob in (None, "", "not json", "[]", json.dumps({}),
                     json.dumps({"generated_at": 17}),
                     json.dumps({"generated_at": "not a date"})):
            with self.subTest(blob=blob):
                self.assertIsNone(_generated_at(blob))


class TheGuardedListTests(unittest.TestCase):
    def test_every_named_artifact_exists(self):
        """A path that has been renamed would be checked silently forever."""
        for path in PUBLISHED_ARTIFACTS:
            with self.subTest(path=path):
                self.assertTrue((_ROOT / path).is_file(), f"{path} is missing")

    def test_every_named_artifact_carries_the_field_it_is_checked_on(self):
        for path in PUBLISHED_ARTIFACTS:
            with self.subTest(path=path):
                blob = (_ROOT / path).read_text(encoding="utf-8")
                self.assertIsNotNone(
                    _generated_at(blob),
                    f"{path} has no readable generated_at, so it cannot be guarded",
                )


if __name__ == "__main__":
    unittest.main()
