"""/feed/latest must not pretend it filtered by credibility when it did not.

The route audits the edge board, then filters:

    is_cred = item.get("is_credible", True) if cred_score is None else ...
    if edge >= min_edge and is_cred:

An unaudited row carries no credibility_score and no is_credible, so it
takes the `True` default and passes. That is fine while the audit works.
It stops being fine when the audit raises, because the handler swallowed
the exception and served the raw rows -- silently turning the documented
min_credibility parameter into a no-op for that call.

Not hypothetical: audit_edge_board raises on the aggregate mapping, which
is exactly the mistake two other callers made (fixed earlier). Here it
would degrade instead of fail, and nothing said so.

This does not change what is filtered -- whether an unassessed row should
be admitted is a product decision. It makes the degradation visible.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from analyzing_llm_rationale import server as srv  # noqa: E402

_BOARD = {
    "edge_board": [
        {"question": "Q1", "model_probability": 0.8, "market_probability": 0.5,
         "edge": 0.3, "credibility_score": 0.9, "is_credible": True},
        {"question": "Q2", "model_probability": 0.7, "market_probability": 0.4,
         "edge": 0.3, "credibility_score": 0.1, "is_credible": False},
    ]
}


class FeedCredibilityAuditTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(srv.app)

    def _get(self, **params):
        with (
            mock.patch.object(srv, "_read_edge_board_record", return_value=dict(_BOARD)),
            mock.patch.object(srv, "_read_agent_trading_board", return_value={}),
        ):
            return self.client.get("/feed/latest", params=params)

    def test_a_healthy_audit_reports_that_it_audited(self):
        resp = self._get(min_credibility=0.6)
        self.assertEqual(resp.status_code, 200)
        self.assertIs(resp.json()["credibility_audited"], True)

    def test_a_failed_audit_is_reported_not_hidden(self):
        with mock.patch(
            "analyzing_llm_rationale.edge_credibility.audit_edge_board",
            side_effect=ValueError("dictionary update sequence element #0 has length 1"),
        ):
            resp = self._get(min_credibility=0.6)
        self.assertEqual(resp.status_code, 200)
        self.assertIs(resp.json()["credibility_audited"], False)

    def test_a_failed_audit_still_serves_signals(self):
        """Degrading beats 500ing; the point is saying so."""
        with mock.patch(
            "analyzing_llm_rationale.edge_credibility.audit_edge_board",
            side_effect=ValueError("boom"),
        ):
            resp = self._get(min_credibility=0.6, min_edge=0.05)
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(len(resp.json()["market_edge_signals"]), 1)

    def test_a_failed_audit_is_logged_with_the_consequence(self):
        with mock.patch(
            "analyzing_llm_rationale.edge_credibility.audit_edge_board",
            side_effect=ValueError("boom"),
        ):
            with self.assertLogs(srv.logger, level="WARNING") as caught:
                self._get(min_credibility=0.6)
        joined = "\n".join(caught.output)
        self.assertIn("min_credibility", joined)
        self.assertIn("not filter", joined)

    def test_a_healthy_audit_logs_nothing(self):
        with self.assertNoLogs(srv.logger, level="WARNING"):
            self._get(min_credibility=0.6)

    def test_the_flag_is_always_present(self):
        """A caller should not have to infer it from absence."""
        self.assertIn("credibility_audited", self._get().json())


if __name__ == "__main__":
    unittest.main()
