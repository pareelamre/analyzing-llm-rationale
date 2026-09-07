"""_get_price_history / _get_forecast_history must not fail silently.

Both try DuckDB, then the document store, then give up and return []. The
give-up was a bare `except Exception: pass`, so a total storage failure
was indistinguishable from a market that genuinely has no history.

That distinction matters here more than in a read endpoint. These two feed
`market_price_history` and `forecast_history` into the context each
snapshot's forecast is made with. If both stores are down the model still
forecasts, and that forecast is still scored and published into the track
record -- just made blind, with nothing saying so.

Returning [] is still right: one unreadable market should not abort a tick
over hundreds. Only the silence is wrong.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import track_record_live as trl  # noqa: E402

_LOGGER = "analyzing_llm_rationale.track_record_live"


class _Query:
    def __init__(self, rows=None, raises=None):
        self._rows, self._raises = rows or [], raises

    def add_filter(self, *a, **k):
        return None

    def fetch(self):
        if self._raises:
            raise self._raises
        return self._rows


class _DocStore:
    """A document-store client: no `_con`, so the DuckDB branch is skipped."""

    def __init__(self, rows=None, raises=None):
        self._rows, self._raises = rows or [], raises

    def query(self, kind=None, **k):
        return _Query(self._rows, self._raises)


class _BrokenDuckDB(_DocStore):
    """Has `_con`, and it explodes -- so both paths are exercised."""

    class _Con:
        def execute(self, *a, **k):
            raise RuntimeError("duckdb file is locked")

    def __init__(self, **kw):
        super().__init__(**kw)
        self._con = self._Con()


class PriceHistoryTests(unittest.TestCase):
    def test_a_total_failure_returns_empty_rather_than_raising(self):
        store = _DocStore(raises=ConnectionError("datastore unreachable"))
        with self.assertLogs(_LOGGER, level="WARNING"):
            self.assertEqual(trl._get_price_history(store, "mkt-1"), [])

    def test_a_total_failure_names_the_market_and_the_consequence(self):
        store = _DocStore(raises=ConnectionError("datastore unreachable"))
        with self.assertLogs(_LOGGER, level="WARNING") as caught:
            trl._get_price_history(store, "mkt-1")
        joined = "\n".join(caught.output)
        self.assertIn("mkt-1", joined)
        self.assertIn("ConnectionError", joined)
        self.assertIn("without price context", joined)

    def test_falling_back_from_duckdb_to_the_document_store_is_quiet(self):
        """A working fallback is not a problem, so it must not warn."""
        store = _BrokenDuckDB(rows=[{"ts": 1, "market_probability": 0.4}])
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            rows = trl._get_price_history(store, "mkt-1")
        self.assertEqual(rows[0]["probability"], 0.4)

    def test_a_market_with_genuinely_no_history_does_not_warn(self):
        """Empty is not an error; conflating them is what this guards."""
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            self.assertEqual(trl._get_price_history(_DocStore(rows=[]), "mkt-1"), [])


class ForecastHistoryTests(unittest.TestCase):
    def test_a_total_failure_returns_empty_and_warns(self):
        store = _DocStore(raises=ConnectionError("datastore unreachable"))
        with self.assertLogs(_LOGGER, level="WARNING") as caught:
            self.assertEqual(trl._get_forecast_history(store, "polymarket", "mkt-2", "m"), [])
        joined = "\n".join(caught.output)
        self.assertIn("mkt-2", joined)
        self.assertIn("without its own prior forecasts", joined)

    def test_a_market_with_no_prior_forecasts_does_not_warn(self):
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            self.assertEqual(
                trl._get_forecast_history(_DocStore(rows=[]), "polymarket", "mkt-2", "m"), []
            )


if __name__ == "__main__":
    unittest.main()
