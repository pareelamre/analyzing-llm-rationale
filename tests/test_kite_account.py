"""Behavior contracts for the private Kite Personal integration (synthetic data)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import requests
from fastapi.testclient import TestClient

from analyzing_llm_rationale import kite_account, server

CREDS = {"api_key": "fixturekey", "access_token": "fixturetoken"}


class KiteClientTests(unittest.TestCase):
    def test_only_free_account_reads_use_fixed_host_and_auth(self):
        # covers: R2
        for resource, path, data in (
            ("profile", "/user/profile", {"user_id": "TEST01"}),
            ("holdings", "/portfolio/holdings", [{"tradingsymbol": "TEST"}]),
            ("positions", "/portfolio/positions", {"net": [], "day": []}),
            ("funds", "/user/margins", {"equity": {"net": 10}}),
            ("orders", "/orders", [{"order_id": "test-order"}]),
        ):
            with self.subTest(resource=resource), mock.patch.object(kite_account.requests, "get") as get:
                get.return_value.status_code = 200
                get.return_value.json.return_value = {"status": "success", "data": data}
                self.assertEqual(kite_account.read_account(CREDS, resource), data)
                self.assertEqual(get.call_args.args, ("https://api.kite.trade" + path,))
                self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "token fixturekey:fixturetoken")
                self.assertFalse(get.call_args.kwargs["allow_redirects"])
                self.assertEqual(get.call_args.kwargs["timeout"], (3.05, 10))
        with mock.patch.object(kite_account.requests, "get") as get:
            for resource in ("quote", "historical", "../orders", "place_order"):
                with self.assertRaises(kite_account.KiteAccountError):
                    kite_account.read_account(CREDS, resource)
            get.assert_not_called()

    def test_failures_never_expose_upstream_secrets(self):
        # covers: R3
        for status in (302, 401, 403, 429, 500):
            with self.subTest(status=status), mock.patch.object(kite_account.requests, "get") as get:
                get.return_value.status_code = status
                get.return_value.json.return_value = {"message": "fixturetoken", "status": "error"}
                with self.assertRaises(kite_account.KiteAccountError) as raised:
                    kite_account.read_account(CREDS, "holdings")
                self.assertNotIn("fixturetoken", str(raised.exception))
        with mock.patch.object(kite_account.requests, "get", side_effect=requests.Timeout("fixturetoken")):
            with self.assertRaises(kite_account.KiteAccountError) as raised:
                kite_account.read_account(CREDS, "funds")
            self.assertNotIn("fixturetoken", str(raised.exception))

    def test_malformed_credentials_and_payload_fail_closed(self):
        for payload in ({}, {**CREDS, "api_secret": "unused"}, {**CREDS, "api_key": "key:injected"}, {**CREDS, "access_token": "bad\r\nheader"}):
            with self.assertRaises(kite_account.KiteAccountError):
                kite_account.validate_credentials(payload)
        for payload in ({"status": "success"}, {"status": "success", "data": "fixturetoken"}, {"status": "error", "message": "fixturetoken"}):
            with mock.patch.object(kite_account.requests, "get") as get:
                get.return_value.status_code = 200
                get.return_value.json.return_value = payload
                with self.assertRaises(kite_account.KiteAccountError):
                    kite_account.read_account(CREDS, "holdings")


class KiteRouteTests(unittest.TestCase):
    def setUp(self):
        enabled = mock.patch.dict(os.environ, {"FORESEA_ENABLE_KITE_PERSONAL": "true"})
        enabled.start()
        self.addCleanup(enabled.stop)
        self.client = TestClient(server.app)
        self.headers = {"Authorization": "Bearer " + server._issue_session("kite-owner", "test@example.invalid", "Test", "")}
        self.other_headers = {"Authorization": "Bearer " + server._issue_session("kite-other", "other@example.invalid", "Other", "")}
        self.records = {}
        self.stack = __import__("contextlib").ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(server, "_check_rate_limit"))
        self.stack.enter_context(mock.patch.object(server, "_secure_connection_configured", return_value=True))
        self.stack.enter_context(mock.patch.object(server, "_get_datastore", return_value=object()))
        self.stack.enter_context(mock.patch.object(server, "_connection_status", side_effect=lambda user, venue: server.TradingConnectionStatus(platform=venue, connected=(user, venue) in self.records)))
        self.stack.enter_context(mock.patch.object(server, "_put_trading_connection", side_effect=self.save))
        self.stack.enter_context(mock.patch.object(server, "_kite_connection_mutation", side_effect=self.mutate))
        self.stack.enter_context(mock.patch.object(server, "_stored_trading_credentials", side_effect=lambda user, venue: self.records.get((user, venue))))
        self.stack.enter_context(mock.patch.object(server, "_delete_trading_connection", side_effect=lambda user, venue: self.records.pop((user, venue), None)))
        self.read = self.stack.enter_context(mock.patch.object(kite_account, "read_account", return_value=[]))

    def save(self, user, venue, credentials):
        self.records[user, venue] = credentials
        return server.TradingConnectionStatus(platform=venue, connected=True)

    def mutate(self, user, action, credentials=None, intent=None):
        if action == "begin":
            return "fixture-intent"
        if action == "save":
            return self.save(user, "kite", credentials).model_dump()
        self.records.pop((user, "kite"), None)
        return {"platform": "kite", "connected": False}

    def test_bounded_body_and_rate_limit_fail_before_external_work(self):
        response = self.client.put("/integrations/kite", headers=self.headers, content=b"x" * 2049)
        self.assertEqual(response.status_code, 413)
        self.read.assert_not_called()
        from fastapi import HTTPException

        with mock.patch.object(server, "_check_rate_limit", side_effect=HTTPException(429, "Rate limited")) as limit:
            for method, path in (("get", "/integrations/kite"), ("put", "/integrations/kite"), ("delete", "/integrations/kite"), ("get", "/integrations/kite/orders")):
                response = getattr(self.client, method)(path, headers=self.headers)
                self.assertEqual(response.status_code, 429)
            self.assertEqual(limit.call_count, 4)
        self.read.assert_not_called()

    def test_decryption_failure_is_safe_and_generic_trading_rejects_kite(self):
        with mock.patch.object(server, "_stored_trading_credentials", side_effect=server.SecureTradingConnectionError("fixturetoken")):
            response = self.client.get("/integrations/kite/holdings", headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("fixturetoken", response.text)
        response = self.client.delete("/trading/connections/kite", headers=self.headers)
        self.assertEqual(response.status_code, 422)

    def test_disabled_access_keeps_status_and_disconnect_available(self):
        with mock.patch.dict(os.environ, {"FORESEA_ENABLE_KITE_PERSONAL": "false"}):
            self.assertEqual(self.client.put("/integrations/kite", headers=self.headers, json=CREDS).status_code, 503)
            self.assertEqual(self.client.get("/integrations/kite/orders", headers=self.headers).status_code, 503)
            status = self.client.get("/integrations/kite", headers=self.headers)
            self.assertFalse(status.json()["enabled"])
            self.assertEqual(self.client.delete("/integrations/kite", headers=self.headers).status_code, 200)
        self.read.assert_not_called()

    def test_auth_required_for_every_operation(self):
        # covers: R1
        for method, path in (("get", "/integrations/kite"), ("put", "/integrations/kite"), ("delete", "/integrations/kite"), *(("get", "/integrations/kite/" + r) for r in ("holdings", "positions", "funds", "orders"))):
            response = getattr(self.client, method)(path)
            self.assertEqual(response.status_code, 401, response.text)
        self.read.assert_not_called()

    def test_owner_reads_and_disconnects_only_their_connection(self):
        # covers: R1, R2
        connected = self.client.put("/integrations/kite", headers=self.headers, json=CREDS)
        self.assertEqual(connected.status_code, 200, connected.text)
        self.assertNotIn("fixturetoken", connected.text)
        self.assertEqual(self.read.call_args.args, (CREDS, "profile"))
        self.assertFalse(self.client.get("/integrations/kite", headers=self.other_headers).json()["connected"])
        self.assertEqual(self.client.get("/integrations/kite/holdings", headers=self.other_headers).status_code, 409)
        self.client.delete("/integrations/kite", headers=self.other_headers)
        for resource in ("holdings", "positions", "funds", "orders"):
            response = self.client.get("/integrations/kite/" + resource, headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("no-store", response.headers["cache-control"])
            self.assertEqual(response.json()["data"], [])
        self.client.delete("/integrations/kite", headers=self.headers)
        self.assertEqual(self.client.get("/integrations/kite/orders", headers=self.headers).status_code, 409)
        self.assertEqual(self.client.post("/integrations/kite/orders", headers=self.headers, json={}).status_code, 405)

    def test_invalid_or_expired_credentials_are_safe_and_not_saved(self):
        # covers: R3
        response = self.client.put("/integrations/kite", headers=self.headers, json={**CREDS, "access_token": "fixturetoken\r\n"})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("fixturetoken", response.text)
        self.read.side_effect = kite_account.KiteAccountError("Kite session expired. Reconnect.", 409)
        response = self.client.put("/integrations/kite", headers=self.headers, json=CREDS)
        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.records)

    def test_storage_and_kms_fail_closed(self):
        with mock.patch.object(server, "_get_datastore", return_value=None):
            response = self.client.put("/integrations/kite", headers=self.headers, json=CREDS)
        self.assertEqual(response.status_code, 503)
        self.read.assert_not_called()
        self.assertFalse(self.records)
        with mock.patch.object(server, "_secure_connection_configured", return_value=False):
            response = self.client.put("/integrations/kite", headers=self.headers, json=CREDS)
        self.assertEqual(response.status_code, 503)
        self.assertFalse(self.records)

class KiteEncryptionTests(unittest.TestCase):
    def test_sql_transaction_rejects_superseded_connection(self):
        from fastapi import HTTPException

        from analyzing_llm_rationale import datastore_sql

        client = datastore_sql.Client(path=":memory:")
        self.addCleanup(client.close)
        with mock.patch.dict(os.environ, {"FORESEA_DATASTORE_BACKEND": "sql"}), mock.patch.object(server, "_get_datastore", return_value=client), mock.patch.object(server, "_put_trading_connection") as save:
            intent = server._kite_connection_mutation("owner", "begin")
            server._kite_connection_mutation("owner", "disconnect")
            with self.assertRaises(HTTPException) as raised:
                server._kite_connection_mutation("owner", "save", CREDS, intent)
            self.assertEqual(raised.exception.status_code, 409)
            save.assert_not_called()

    def test_real_envelope_storage_is_bound_to_user_and_kite(self):
        """Exercise real encryption/decryption with synthetic KMS and datastore."""
        import os
        from types import SimpleNamespace

        from google.cloud.datastore import Key

        class MemoryStore:
            def __init__(self):
                import threading

                self.records = {}
                self.lock = threading.RLock()

            def transaction(self):
                return self.lock

            def key(self, *parts):
                return Key(*parts, project="test-project")

            def put(self, entity):
                with self.lock:
                    self.records[entity.key] = entity

            def get(self, key):
                with self.lock:
                    return self.records.get(key)

            def delete(self, key):
                with self.lock:
                    self.records.pop(key, None)

        class FixtureKms:
            def encrypt(self, *, request):
                self.key = request["plaintext"]
                self.aad = request["additional_authenticated_data"]
                return SimpleNamespace(ciphertext=b"wrapped-fixture-dek", name="fixture-version")

            def decrypt(self, *, request):
                if self.aad != request["additional_authenticated_data"]:
                    raise ValueError("wrong owner or venue")
                return SimpleNamespace(plaintext=self.key)

        store, kms = MemoryStore(), FixtureKms()
        with mock.patch.dict(os.environ, {"FORESEA_ENABLE_KITE_PERSONAL": "true", "FORESEA_TRADING_KMS_KEY_NAME": "projects/test-project/locations/global/keyRings/foresea/cryptoKeys/trading"}), mock.patch.object(server, "_get_datastore", return_value=store), mock.patch.object(server, "_get_trading_kms_client", return_value=kms), mock.patch.object(kite_account, "read_account", return_value={"user_id": "TEST01"}):
            result = server._kite_operation("owner", "connect", CREDS)
            self.assertTrue(result["connected"])
            self.assertNotIn("fixturetoken", str(store.records))
            self.assertNotIn("fixturekey", str(store.records))
            self.assertEqual(server._stored_trading_credentials("owner", "kite"), CREDS)
            self.assertIsNone(server._stored_trading_credentials("other", "kite"))
            self.assertIsNone(server._stored_trading_credentials("owner", "kalshi"))
            # Even copying a ciphertext record to another owner's key cannot decrypt it.
            copied = dict(next(record for record in store.records.values() if "encrypted_credentials" in record))
            store.records[store.key("User", "other", server._TRADING_CONNECTION_KIND, "kite")] = copied
            with self.assertRaises(server.SecureTradingConnectionError):
                server._stored_trading_credentials("other", "kite")
            server._kite_operation("owner", "disconnect")
            self.assertIsNone(server._stored_trading_credentials("owner", "kite"))

            # A delayed profile request must not resurrect a disconnected session.
            import threading
            from concurrent.futures import ThreadPoolExecutor

            from fastapi import HTTPException

            entered, resume = threading.Event(), threading.Event()

            def delayed_profile(*args):
                entered.set()
                self.assertTrue(resume.wait(5))
                return {"user_id": "TEST01"}

            with mock.patch.object(kite_account, "read_account", side_effect=delayed_profile), ThreadPoolExecutor(1) as executor:
                pending = executor.submit(server._kite_operation, "owner", "connect", CREDS)
                try:
                    self.assertTrue(entered.wait(5))
                    server._kite_operation("owner", "disconnect")
                finally:
                    resume.set()
                with self.assertRaises(HTTPException) as raised:
                    pending.result(timeout=5)
                self.assertEqual(raised.exception.status_code, 409)
            self.assertIsNone(server._stored_trading_credentials("owner", "kite"))

class KiteUiTests(unittest.TestCase):
    def test_positions_segments_escaping_stale_session_and_missing_dom(self):
        import subprocess
        from pathlib import Path

        script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('frontend/index.html', 'utf8');
const code = html.slice(html.indexOf('// Private Kite data stays'), html.indexOf('// ── BYO Model Providers Manager'));
class Element {
  constructor() { this.value = ''; this._text = ''; this.children = []; this.style = {}; }
  set textContent(value) { this._text = value; this.children = []; }
  get textContent() { return this._text; }
  set innerHTML(value) { throw Error('private broker text must never become HTML'); }
  replaceChildren() { this.children = []; }
  append(value) { this.children.push(value); }
  createTHead() { const head = new Element(); this.append(head); return head; }
  createTBody() { const body = new Element(); this.append(body); return body; }
  insertRow() { const row = new Element(); this.append(row); return row; }
  insertCell() { const cell = new Element(); this.append(cell); return cell; }
}
const ids = ['kiteApiKey', 'kiteAccessToken', 'kiteAccountData', 'kiteAccountMessage', 'kiteChip', 'kiteResource'];
const elements = new Map(ids.map(id => [id, new Element()]));
let session = 'Bearer first-session', reply;
const context = vm.createContext({
  document: {getElementById: id => elements.get(id) || null, createElement: () => new Element()},
  authHeaders: () => ({Authorization: session}),
  fetch: () => new Promise(resolve => { reply = resolve; }),
  console,
});
vm.runInContext(code, context);
context.renderKiteAccount('positions', {net: [{tradingsymbol: 'NET'}], day: [{tradingsymbol: '<img onerror=attack()>'}]});
const body = elements.get('kiteAccountData').children[0].children[1];
assert.equal(body.children.length, 2);
assert.equal(body.children[0].children[0].textContent, 'Net');
assert.equal(body.children[1].children[0].textContent, 'Day');
assert.equal(body.children[1].children[1].textContent, '<img onerror=attack()>');
(async () => {
  elements.get('kiteResource').value = 'holdings';
  const pending = context.kiteAccountAction('read');
  session = 'Bearer other-user';
  reply({ok: true, json: async () => ({data: [{tradingsymbol: 'PRIVATE'}], fetched_at: '2026-10-26T12:00:00Z'})});
  await pending;
  assert.equal(elements.get('kiteAccountData').children.length, 0);
  elements.get('kiteAccessToken').value = 'fixturetoken';
  context.clearKiteAccount();
  assert.equal(elements.get('kiteAccessToken').value, '');
  assert.equal(elements.get('kiteAccountData').textContent, '');
  elements.clear();
  context.clearKiteAccount();
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
        result = subprocess.run(["node", "-e", script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
