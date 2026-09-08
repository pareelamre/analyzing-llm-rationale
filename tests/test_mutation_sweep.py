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
    has_local_changes,
    literal_spans,
    parses,
)


def _module(source):
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    )
    handle.write(source)
    handle.close()
    return Path(handle.name)


class LiteralTextIsNeverMutatedTests(unittest.TestCase):
    """The contract is per-column, not per-line.

    A line can hold both code and prose. `label = "a >= b"` is a real
    assignment, so any line-level filter keeps it -- and then a regex
    finds the >= inside the quotes. Python 3.12 makes this sharper: an
    f-string is no longer one STRING token, so its literal text arrives
    as FSTRING_MIDDLE with real code around it. A line-level filter kept
    those and reported the prose `+` in

        f"BUY YES @ {p:.1f}% + BUY NO @ {k:.1f}%"

    as a surviving mutation. Two of those came out of an
    arbitrage_scanner sweep, which is what prompted working in columns.
    """

    def _sweep(self, source):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        )
        handle.write(source)
        handle.close()
        path = Path(handle.name)
        self.addCleanup(path.unlink)
        found = list(candidates(path, rules=list(RULES), start=1, end=10**9))
        return found, path

    def test_an_operator_inside_a_plain_string_is_not_a_candidate(self):
        found, _ = self._sweep('label = "a >= b and more"\n')
        self.assertEqual(found, [])

    def test_an_operator_inside_f_string_text_is_not_a_candidate(self):
        """The 3.12 case: prose and code share the line."""
        found, _ = self._sweep('msg = f"BUY YES @ {p:.1f}% + BUY NO @ {k:.1f}%"\n')
        self.assertEqual(found, [])

    def test_an_operator_inside_an_f_string_expression(self):
        """How precise this is depends on the interpreter, safely.

        From 3.12 an f-string is tokenised in pieces, so the code inside
        the braces is visible as code and is a candidate. Before that the
        whole f-string is a single STRING token and the sweep skips all
        of it.

        The direction of the difference is the point: the older
        behaviour loses a candidate, it does not invent one. A sweep on
        3.11 is less thorough inside f-strings and never noisier, which
        is the safe way round for a tool whose output is a shortlist to
        read by hand. CI runs 3.11 and this was written on 3.12, so the
        two genuinely disagree and both are correct.
        """
        found, _ = self._sweep('msg = f"total {a + b}"\n')
        if sys.version_info >= (3, 12):
            self.assertEqual(len(found), 1)
            self.assertIn("a - b", found[0][3])
        else:
            self.assertEqual(found, [])

    def test_a_docstring_is_not_a_candidate(self):
        found, _ = self._sweep(
            'def f(a, b):\n    """Return max(a, b) when a >= b."""\n    return 1\n'
        )
        self.assertEqual(found, [])

    def test_a_comment_is_not_a_candidate(self):
        found, _ = self._sweep("x = 1\n# prefer max() here\n")
        self.assertEqual(found, [])

    def test_real_code_on_the_same_line_as_a_string_is_still_found(self):
        """Skipping the whole line would lose this."""
        found, _ = self._sweep('if a >= b: label = "note >= here"\n')
        self.assertEqual(len(found), 1)
        self.assertIn("if a > b", found[0][3])
        self.assertIn('"note >= here"', found[0][3])

    def test_a_file_that_does_not_parse_yields_nothing(self):
        found, path = self._sweep("def f(:\n")
        self.assertEqual(found, [])
        self.assertFalse(parses(path))

    def test_a_triple_quoted_block_is_prose_on_every_line_it_covers(self):
        source = "x = " + '"""' + "\nmax(a, b) and a >= b\n" + '"""' + "\n"
        found, path = self._sweep(source)
        self.assertEqual(found, [])
        self.assertIn(2, literal_spans(path))


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
