from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import pr_outreach  # noqa: E402


class FakeResponse:
    status_code = 202
    text = "accepted"

    def json(self):
        return {"ok": True}


class FakeJsonRpcErrorResponse:
    status_code = 200
    text = '{"error":{"message":"bad request"}}'

    def json(self):
        return {"error": {"message": "bad request"}}


class FakeCodeErrorResponse:
    status_code = 200
    text = '{"code":400,"message":"invalid project"}'

    def json(self):
        return {"code": 400, "message": "invalid project"}


class FakeSession:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or FakeResponse()

    def post(self, endpoint, headers=None, json=None, data=None, timeout=None):
        self.calls.append({
            "endpoint": endpoint,
            "headers": headers,
            "json": json,
            "data": data,
            "timeout": timeout,
        })
        return self.response


class PROutreachTests(unittest.TestCase):
    def test_load_targets_accepts_targets_object(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "targets.json"
            path.write_text(json.dumps({
                "targets": [{
                    "name": "Catalog",
                    "endpoint": "https://agents.example/inbox",
                    "audience": "catalog",
                    "headers": {"X-Test": "1"},
                }]
            }))

            targets = pr_outreach.load_targets(path)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].name, "Catalog")
        self.assertEqual(targets[0].audience, "catalog")
        self.assertEqual(targets[0].headers, {"X-Test": "1"})

    def test_load_targets_rejects_non_http_endpoint(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "targets.json"
            path.write_text(json.dumps([{"name": "Bad", "endpoint": "mailto:x@example.com"}]))

            with self.assertRaises(ValueError):
                pr_outreach.load_targets(path)

    def test_load_targets_expands_env_headers(self):
        old = os.environ.get("PR_AGENT_TEST_AUTH")
        os.environ["PR_AGENT_TEST_AUTH"] = "Bearer secret"
        try:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "targets.json"
                path.write_text(json.dumps([{
                    "name": "Agent",
                    "endpoint": "https://agent.example/inbox",
                    "headers": {"Authorization": "$PR_AGENT_TEST_AUTH"},
                }]))

                targets = pr_outreach.load_targets(path)
        finally:
            if old is None:
                os.environ.pop("PR_AGENT_TEST_AUTH", None)
            else:
                os.environ["PR_AGENT_TEST_AUTH"] = old

        self.assertEqual(targets[0].headers, {"Authorization": "Bearer secret"})

    def test_load_targets_accepts_form_transport(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "targets.json"
            path.write_text(json.dumps([{
                "name": "AgentNDX",
                "endpoint": "https://agentndx.ai/api/submit",
                "transport": "form",
                "form": {
                    "name": "Foresea",
                    "github_url": "https://github.com/pareelamre/analyzing-llm-rationale",
                    "protocols": ["MCP"],
                },
            }]))

            targets = pr_outreach.load_targets(path)

        self.assertEqual(targets[0].transport, "form")
        self.assertEqual(targets[0].form["protocols"], ["MCP"])

    def test_load_targets_accepts_json_transport(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "targets.json"
            path.write_text(json.dumps([{
                "name": "MCP Directory",
                "endpoint": "https://mcp.directory/api/submit-server",
                "transport": "json",
                "body": {
                    "githubUrl": "https://github.com/pareelamre/analyzing-llm-rationale",
                    "description": "Remote MCP server for prediction-market forecasting.",
                },
            }]))

            targets = pr_outreach.load_targets(path)

        self.assertEqual(targets[0].transport, "json")
        self.assertEqual(
            targets[0].body["githubUrl"],
            "https://github.com/pareelamre/analyzing-llm-rationale",
        )

    def test_filter_unsent_and_mark_sent(self):
        target = pr_outreach.OutreachTarget(name="Agent", endpoint="https://agent.example/inbox")
        state = {"sent": {}}
        self.assertEqual(pr_outreach.filter_unsent([target], state), [target])

        pr_outreach.mark_sent(state, [{
            "target": "Agent",
            "endpoint": "https://agent.example/inbox",
            "status": "sent",
            "status_code": 202,
        }])

        self.assertEqual(pr_outreach.filter_unsent([target], state), [])
        self.assertEqual(pr_outreach.filter_unsent([target], state, resend=True), [target])

    def test_empty_state_file_is_fresh_state(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            path.write_text("")

            state = pr_outreach.load_state(path)

        self.assertEqual(state, {"sent": {}})

    def test_dry_run_does_not_post(self):
        target = pr_outreach.OutreachTarget(name="Agent", endpoint="https://agent.example/inbox")
        session = FakeSession()

        results = pr_outreach.send_outreach([target], send=False, session=session)

        self.assertEqual(results[0]["status"], "dry_run")
        self.assertEqual(session.calls, [])
        self.assertIn("Foresea PR Agent", results[0]["payload"]["from"])

    def test_dry_run_json_target_includes_body(self):
        target = pr_outreach.OutreachTarget(
            name="MCP Directory",
            endpoint="https://mcp.directory/api/submit-server",
            transport="json",
            body={"githubUrl": "https://github.com/pareelamre/analyzing-llm-rationale"},
        )
        session = FakeSession()

        results = pr_outreach.send_outreach([target], send=False, session=session)

        self.assertEqual(results[0]["status"], "dry_run")
        self.assertEqual(results[0]["transport"], "json")
        self.assertEqual(
            results[0]["body"]["githubUrl"],
            "https://github.com/pareelamre/analyzing-llm-rationale",
        )
        self.assertEqual(session.calls, [])

    def test_send_posts_payload(self):
        target = pr_outreach.OutreachTarget(
            name="Agent",
            endpoint="https://agent.example/inbox",
            audience="mcp",
            headers={"Authorization": "Bearer token"},
        )
        session = FakeSession()

        results = pr_outreach.send_outreach([target], send=True, session=session, pause_s=0)

        self.assertEqual(results[0]["status"], "sent")
        self.assertEqual(session.calls[0]["endpoint"], "https://agent.example/inbox")
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer token")
        self.assertEqual(session.calls[0]["json"]["type"], "foresea.pr_agent.outreach")
        self.assertEqual(session.calls[0]["json"]["audience"], "mcp")

    def test_send_form_target_posts_form_data(self):
        target = pr_outreach.OutreachTarget(
            name="AgentNDX",
            endpoint="https://agentndx.ai/api/submit",
            audience="catalog",
            transport="form",
            form={"name": "Foresea", "github_url": "https://github.com/pareelamre/analyzing-llm-rationale"},
        )
        session = FakeSession()

        results = pr_outreach.send_outreach([target], send=True, session=session, pause_s=0)

        self.assertEqual(results[0]["status"], "sent")
        self.assertIsNone(session.calls[0]["json"])
        self.assertEqual(session.calls[0]["data"]["name"], "Foresea")
        self.assertNotIn("Content-Type", session.calls[0]["headers"])

    def test_send_json_target_posts_body(self):
        target = pr_outreach.OutreachTarget(
            name="mcpub",
            endpoint="https://mcpub.dev/mcp",
            audience="catalog",
            transport="json",
            body={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "submit", "arguments": {"url": "https://foresea.ink"}},
            },
        )
        session = FakeSession()

        results = pr_outreach.send_outreach([target], send=True, session=session, pause_s=0)

        self.assertEqual(results[0]["status"], "sent")
        self.assertIsNone(session.calls[0]["data"])
        self.assertEqual(session.calls[0]["headers"]["Content-Type"], "application/json")
        self.assertEqual(session.calls[0]["json"]["method"], "tools/call")

    def test_send_json_target_treats_jsonrpc_error_as_failed(self):
        target = pr_outreach.OutreachTarget(
            name="mcpub",
            endpoint="https://mcpub.dev/mcp",
            transport="json",
            body={"jsonrpc": "2.0", "method": "tools/call"},
        )
        session = FakeSession(response=FakeJsonRpcErrorResponse())

        results = pr_outreach.send_outreach([target], send=True, session=session, pause_s=0)

        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[0]["status_code"], 200)

    def test_send_json_target_treats_nonzero_code_as_failed(self):
        target = pr_outreach.OutreachTarget(
            name="mcp.so",
            endpoint="https://mcp.so/api/submit-project",
            transport="json",
            body={"name": "Foresea", "type": "server", "url": "https://foresea.ink"},
        )
        session = FakeSession(response=FakeCodeErrorResponse())

        results = pr_outreach.send_outreach([target], send=True, session=session, pause_s=0)

        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[0]["status_code"], 200)


if __name__ == "__main__":
    unittest.main()
