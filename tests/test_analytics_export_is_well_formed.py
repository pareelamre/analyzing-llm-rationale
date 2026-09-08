"""The operator CSV export is built from attacker-controlled strings.

/analytics/export was assembled with f-strings. Two of the fields in it
are set by whoever is being exported:

  * a display name is only .strip()ed on the way in -- any character, up
    to 120 of them, and registration is self-service;
  * the model label comes from the analytics event's `metadata`, a
    free-form dict from the browser with only a size limit on it.

The user rows were wrapped in quotes but never escaped, so a quote in a
display name closed the field early. The event and model rows were not
quoted at all, so a comma shifted every column after it. Both corrupt
the export an operator is reading to make decisions.

Separately, quoting does not help against formula injection: a
spreadsheet decides a cell is a formula from its first character after
unquoting, so a name of `=HYPERLINK(...)` runs on open.

These tests read the export back with a real CSV parser rather than
matching substrings, which is the only way the escaping is actually
checked.
"""

import csv
import io
import unittest

from analyzing_llm_rationale.server import _csv_cell


class CsvCellTests(unittest.TestCase):
    def test_a_leading_formula_character_is_neutralised(self):
        for lead in ("=", "+", "-", "@", "\t", "\r"):
            payload = lead + 'HYPERLINK("http://evil","click")'
            self.assertEqual(_csv_cell(payload), "'" + payload, lead)

    def test_ordinary_text_is_left_exactly_alone(self):
        for text in ("Ada Lovelace", "gpt-oss-120b", "2026-09-08", "a=b", ""):
            self.assertEqual(_csv_cell(text), text)

    def test_none_becomes_empty_rather_than_the_word_none(self):
        self.assertEqual(_csv_cell(None), "")

    def test_separators_are_left_to_the_csv_writer(self):
        """_csv_cell must not double-escape; csv.writer owns quoting."""
        self.assertEqual(_csv_cell('a,b"c'), 'a,b"c')


class ExportRoundTripTests(unittest.TestCase):
    """A row with every hostile character must survive a real parser."""

    HOSTILE_NAME = 'Robert"); DROP TABLE users;--, and, more'

    def _rows(self, name):
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["User ID", "Email", "Name"])
        writer.writerow([_csv_cell("u1"), _csv_cell("a@b.com"), _csv_cell(name)])
        return list(csv.reader(io.StringIO(buf.getvalue())))

    def test_quotes_and_commas_survive_a_round_trip(self):
        header, row = self._rows(self.HOSTILE_NAME)
        self.assertEqual(header, ["User ID", "Email", "Name"])
        self.assertEqual(len(row), 3, "the name must not spill into new columns")
        self.assertEqual(row[2], self.HOSTILE_NAME)

    def test_a_newline_in_a_name_does_not_start_a_new_record(self):
        rows = self._rows("line one\nline two")
        self.assertEqual(len(rows), 2, "an embedded newline must stay inside its cell")
        self.assertEqual(rows[1][2], "line one\nline two")

    def test_a_formula_name_round_trips_as_inert_text(self):
        _, row = self._rows("=1+1")
        self.assertEqual(row[2], "'=1+1")
        self.assertFalse(row[2].startswith("="))


if __name__ == "__main__":
    unittest.main()
