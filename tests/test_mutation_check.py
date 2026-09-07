"""The mutation checker must refuse to report a result it did not earn.

Its whole value is telling you whether a test would notice a broken line.
That claim is worthless if the mutation quietly failed to apply, which is
how this repo's two real misreadings happened: a substring anchor that
matched a differently-indented line, and stale bytecode serving the old
code after a same-tick rewrite.

So the failure modes are the thing under test here, not the happy path.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.mutation_check import (  # noqa: E402
    CAUGHT,
    NOT_APPLIED,
    SURVIVED,
    apply_mutation,
)


class ApplyMutationTests(unittest.TestCase):
    _LINES = [
        "def f(x):\n",
        "    return x > 10\n",
        "    # unreachable\n",
    ]

    def test_it_replaces_on_the_named_line(self):
        out = apply_mutation(self._LINES, 2, "> 10", ">= 10")
        self.assertEqual(out[1], "    return x >= 10\n")

    def test_other_lines_are_untouched(self):
        out = apply_mutation(self._LINES, 2, "> 10", ">= 10")
        self.assertEqual(out[0], self._LINES[0])
        self.assertEqual(out[2], self._LINES[2])

    def test_the_input_is_not_mutated_in_place(self):
        before = list(self._LINES)
        apply_mutation(self._LINES, 2, "> 10", ">= 10")
        self.assertEqual(self._LINES, before)

    def test_a_line_out_of_range_raises(self):
        for lineno in (0, 4, 99, -1):
            with self.subTest(lineno=lineno), self.assertRaises(ValueError):
                apply_mutation(self._LINES, lineno, "x", "y")

    def test_an_anchor_absent_from_that_line_raises(self):
        """The substring exists in the file, just not on the line named."""
        with self.assertRaises(ValueError) as ctx:
            apply_mutation(self._LINES, 1, "> 10", ">= 10")
        self.assertIn("does not contain", str(ctx.exception))

    def test_a_no_op_mutation_raises(self):
        """Replacing something with itself would report a false SURVIVED."""
        with self.assertRaises(ValueError) as ctx:
            apply_mutation(self._LINES, 2, "> 10", "> 10")
        self.assertIn("no-op", str(ctx.exception))

    def test_the_error_names_the_line_it_looked_at(self):
        with self.assertRaises(ValueError) as ctx:
            apply_mutation(self._LINES, 3, "return", "pass")
        self.assertIn("unreachable", str(ctx.exception))


class ExitStatusTests(unittest.TestCase):
    def test_the_three_outcomes_are_distinct(self):
        """A caller must be able to tell "not applied" from "survived"."""
        self.assertEqual(len({CAUGHT, SURVIVED, NOT_APPLIED}), 3)

    def test_caught_is_the_success_status(self):
        self.assertEqual(CAUGHT, 0)

    def test_not_applied_is_not_confusable_with_survived(self):
        self.assertNotEqual(NOT_APPLIED, SURVIVED)


if __name__ == "__main__":
    unittest.main()
