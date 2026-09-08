"""The sweep must not report findings it invented.

mutation_check.py answers "would this test notice if line 42 were
wrong". The sweep asks that of every line, so its own failure mode is
different: not a wrong answer about one line, but a shortlist padded
with lines that were never really mutated.

The way that happens is string contents. A regex over raw text happily
turns `max` into `min` inside a docstring, or `>=` into `>` inside a log
message. mutation_check would then apply a change that alters no
behaviour, the test would pass, and the sweep would report a survivor
that is pure noise. At the scale a sweep runs at, that noise is the
difference between a usable shortlist and an unreadable one.

So the line selection is what is under test here, not the subprocess
plumbing, which mutation_check's own tests already cover.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mutation_sweep import (  # noqa: E402
    RULES,
    candidates,
    code_lines,
    has_local_changes,
)


def _module(source):
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    )
    handle.write(source)
    handle.close()
    return Path(handle.name)


class CodeLineTests(unittest.TestCase):
    def setUp(self):
        self.path = _module(
            'def f(a, b):\n'
            '    """Return max(a, b) when a >= b."""\n'
            '    # prefer max() here\n'
            '    label = "a >= b and more"\n'
            '    if a >= b:\n'
            '        return max(a, b)\n'
            '    return label\n'
        )
        self.addCleanup(self.path.unlink)

    def test_operators_in_code_are_found(self):
        lines = code_lines(self.path)
        self.assertIn(5, lines, "the real comparison")
        self.assertIn(6, lines, "the real max()")

    def test_a_docstring_is_not_a_finding(self):
        self.assertNotIn(2, code_lines(self.path))

    def test_a_comment_is_not_a_finding(self):
        self.assertNotIn(3, code_lines(self.path))

    def test_a_string_literal_is_not_a_finding(self):
        self.assertNotIn(4, code_lines(self.path))

    def test_a_file_that_does_not_parse_yields_nothing(self):
        broken = _module("def f(:\n")
        self.addCleanup(broken.unlink)
        self.assertEqual(code_lines(broken), set())


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.path = _module(
            "def f(a, b):\n"
            "    if a >= b:\n"
            "        return max(a, b)\n"
            "    return min(a, b)\n"
        )
        self.addCleanup(self.path.unlink)

    def _sweep(self, **kw):
        options = {"rules": list(RULES), "start": 1, "end": 10**9}
        options.update(kw)
        return list(candidates(self.path, **options))

    def test_one_mutation_per_line(self):
        numbers = [number for number, _rule, _old, _new in self._sweep()]
        self.assertEqual(numbers, sorted(set(numbers)))

    def test_the_mutation_actually_changes_the_line(self):
        for _number, _rule, old, new in self._sweep():
            self.assertNotEqual(old, new)

    def test_a_line_range_is_respected(self):
        numbers = [n for n, _r, _o, _e in self._sweep(start=3, end=3)]
        self.assertEqual(numbers, [3])

    def test_restricting_the_rules_restricts_the_findings(self):
        found = self._sweep(rules=["min-max"])
        self.assertEqual([rule for _n, rule, _o, _e in found], ["min-max"])
        self.assertEqual([n for n, _r, _o, _e in found], [4])

    def test_every_rule_is_a_real_substitution(self):
        """A rule that cannot change anything would pad every sweep."""
        for name, (pattern, replacement) in RULES.items():
            self.assertNotEqual(pattern, replacement, name)


class InterruptionGuardTests(unittest.TestCase):
    """A killed sweep must not leave a mutation behind.

    SIGKILL cannot be caught, so the restore alone is not enough. The
    refusal to start on a dirty target is the second half: an interrupted
    run leaves the file looking exactly that way, so the next sweep stops
    and asks rather than restoring to the mutated state.
    """

    def _repo(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)

        def run(*args):
            return subprocess.run(args, cwd=directory, capture_output=True, text=True)

        run("git", "init", "-q")
        run("git", "config", "user.email", "t@example.com")
        run("git", "config", "user.name", "t")
        (directory / "mod.py").write_text("x = 1\n", encoding="utf-8")
        run("git", "add", "-A")
        run("git", "commit", "-qm", "initial")
        return directory

    def test_a_clean_target_is_not_reported_dirty(self):
        self.assertFalse(has_local_changes(self._repo(), "mod.py"))

    def test_a_modified_target_is_reported_dirty(self):
        directory = self._repo()
        (directory / "mod.py").write_text("x = 2\n", encoding="utf-8")
        self.assertTrue(has_local_changes(directory, "mod.py"))

    def test_an_unrelated_modification_does_not_count(self):
        """The check is per-file, so other work in progress does not block."""
        directory = self._repo()
        (directory / "other.py").write_text("y = 1\n", encoding="utf-8")
        self.assertFalse(has_local_changes(directory, "mod.py"))

    def test_somewhere_without_git_is_not_treated_as_dirty(self):
        """No checkout means nothing to compare, not a refusal to run."""
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        (directory / "mod.py").write_text("x = 1\n", encoding="utf-8")
        self.assertFalse(has_local_changes(directory, "mod.py"))

    def test_no_git_binary_at_all_is_not_treated_as_dirty(self):
        """Missing git is not evidence of a mutation, so it must not block.

        The temp-directory case above exits non-zero; this is the other
        branch, where the executable is absent and the call raises.
        """
        import scripts.mutation_sweep as sweep

        def explode(*_args, **_kwargs):
            raise FileNotFoundError("git")

        original = sweep.subprocess.run
        sweep.subprocess.run = explode
        self.addCleanup(setattr, sweep.subprocess, "run", original)
        self.assertFalse(has_local_changes(Path("."), "mod.py"))

    def test_the_file_is_put_back_when_the_process_exits(self):
        """guard_the_file restores through atexit, not only on success."""
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        target = directory / "mod.py"
        target.write_text("original\n", encoding="utf-8")

        repo_root = str(Path(__file__).resolve().parents[1])
        script = "\n".join([
            "import sys",
            "sys.path.insert(0, %r)" % repo_root,
            "from pathlib import Path",
            "from scripts.mutation_sweep import guard_the_file",
            "p = Path(%r)" % str(target),
            "guard_the_file(p)",
            "p.write_text('MUTATED', encoding='utf-8')",
            "raise SystemExit(1)",
        ])
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(target.read_text(encoding="utf-8"), "original\n")


if __name__ == "__main__":
    unittest.main()
