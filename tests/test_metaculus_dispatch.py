"""Dispatch contract with synthetic credentials and no external writes."""

import unittest
from unittest.mock import Mock, patch

import requests

from analyzing_llm_rationale.metaculus_dispatch import DispatchError, dispatch, main


class MetaculusDispatchTests(unittest.TestCase):
    def test_idle_workflow_dispatches_live_main_once(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"workflow_runs": []}
        session.post.return_value.status_code = 204
        self.assertEqual(dispatch("synthetic-test-credential", session), "dispatched")
        session.post.assert_called_once()
        call = session.post.call_args
        self.assertEqual(call.kwargs["json"], {"ref": "main", "inputs": {"submit": "true"}})
        self.assertFalse(call.kwargs["allow_redirects"])
        self.assertTrue(call.args[0].endswith("/metaculus-futureeval.yml/dispatches"))
        self.assertEqual(session.get.call_count, 5)
        self.assertEqual(
            [c.kwargs["params"] for c in session.get.call_args_list],
            [{"branch": "main", "status": s, "per_page": 1} for s in ("queued", "in_progress", "waiting", "pending", "requested")],
        )
        for get_call in session.get.call_args_list:
            self.assertEqual(get_call.args[0], call.args[0].removesuffix("dispatches") + "runs")
            self.assertFalse(get_call.kwargs["allow_redirects"])
            self.assertEqual(get_call.kwargs["timeout"], 10)

    def test_every_active_status_prevents_additional_dispatch(self):
        for position in range(5):
            with self.subTest(position=position):
                session = Mock()
                replies = []
                for index in range(position + 1):
                    reply = Mock(status_code=200)
                    reply.json.return_value = {"workflow_runs": [{"id": 1}] if index == position else []}
                    replies.append(reply)
                session.get.side_effect = replies
                self.assertEqual(dispatch("synthetic-test-credential", session), "skipped")
                self.assertEqual(session.get.call_count, position + 1)
                session.post.assert_not_called()

    def test_missing_or_whitespace_credentials_do_not_call_github(self):
        for token in ("", " ", "test\ncredential"):
            with self.subTest(token=repr(token)):
                session = Mock()
                with self.assertRaises(DispatchError):
                    dispatch(token, session)
                session.get.assert_not_called()
                session.post.assert_not_called()

    def test_failed_or_malformed_run_checks_never_dispatch(self):
        for status, payload in ((401, {}), (302, {}), (503, {}), (200, []), (200, {})):
            with self.subTest(status=status, payload=payload):
                session = Mock()
                session.get.return_value.status_code = status
                session.get.return_value.json.return_value = payload
                with self.assertRaises(DispatchError):
                    dispatch("synthetic-test-credential", session)
                session.post.assert_not_called()
                self.assertFalse(session.get.call_args.kwargs["allow_redirects"])

    def test_uncertain_post_is_not_retried_or_logged(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"workflow_runs": []}
        session.post.side_effect = requests.Timeout("synthetic-test-credential")
        with self.assertLogs("analyzing_llm_rationale.metaculus_dispatch", "INFO") as logs:
            with self.assertRaises(DispatchError) as error:
                dispatch("synthetic-test-credential", session)
        session.post.assert_called_once()
        self.assertNotIn("synthetic-test-credential", str(error.exception) + str(logs.output))

    def test_nonaccepted_dispatch_fails_closed(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"workflow_runs": []}
        for status in (200, 302, 401, 429, 503):
            with self.subTest(status=status):
                session.post.return_value.status_code = status
                with self.assertRaises(DispatchError):
                    dispatch("synthetic-test-credential", session)

    def test_cli_without_secret_exits_nonzero(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "analyzing_llm_rationale.observability.init_observability"
        ), patch("logging.basicConfig"), self.assertLogs("analyzing_llm_rationale.metaculus_dispatch", "ERROR"):
            self.assertEqual(main(), 1)

    def test_invalid_json_never_dispatches(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.side_effect = requests.exceptions.JSONDecodeError("invalid", "no JSON", 0)
        with self.assertRaises(DispatchError):
            dispatch("synthetic-test-credential", session)
        session.post.assert_not_called()

    def test_idle_snapshots_are_best_effort_not_an_atomic_lock(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"workflow_runs": []}
        session.post.return_value.status_code = 204
        # Two stale idle snapshots can enqueue two events. Forecast concurrency
        # and authoritative prior-forecast checks, not this probe, protect writes.
        self.assertEqual(dispatch("synthetic-test-credential", session), "dispatched")
        self.assertEqual(dispatch("synthetic-test-credential", session), "dispatched")
        self.assertEqual(session.post.call_count, 2)
