"""Real optional SDK signing; synthetic HTTP and SQL, never venue traffic."""
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_offline_probe():
    blocked_network = []

    def deny_network(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            blocked_network.append(event)
            raise AssertionError("SDK contract probe denies network")

    sys.addaudithook(deny_network)
    import base64
    from dataclasses import replace
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal
    from importlib.metadata import version
    from tempfile import TemporaryDirectory
    from unittest import mock

    from eth_account import Account
    from eth_account.messages import encode_typed_data
    from py_clob_client_v2.clob_types import OrderArgs, OrderPayload, PartialCreateOrderOptions
    from py_clob_client_v2.endpoints import (
        CANCEL,
        GET_CLOB_MARKET,
        GET_MARKET_BY_TOKEN,
        GET_ORDER,
        GET_TICK_SIZE,
        POST_ORDER,
        VERSION,
    )
    from py_clob_client_v2.order_utils.exchange_order_builder_v2 import ExchangeOrderBuilderV2
    from py_clob_client_v2.order_utils.model.order_data_v2 import OrderV2

    from analyzing_llm_rationale import trading
    from analyzing_llm_rationale.datastore_sql import Client
    from analyzing_llm_rationale.twin.models import AccountScope, TradeIntent
    from analyzing_llm_rationale.twin.store import DatastoreTwinStore

    # Public Anvil test vector; never read a wallet from the environment.
    dummy_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    signer_address = Account.from_key(dummy_key).address
    host = "https://offline.invalid"
    credentials = {
        "polymarket_private_key": dummy_key,
        "polymarket_api_key": "fixture-api-key",
        "polymarket_api_secret": base64.urlsafe_b64encode(b"fixture-only").decode(),
        "polymarket_api_passphrase": "fixture-passphrase",
        "polymarket_chain_id": "137", "polymarket_clob_host": host,
    }
    cases = 0
    unknown_reads = []
    first_client = None
    for signature_type in (0, 1, 2):
        funder = signer_address if signature_type == 0 else "0x" + "22" * 20
        creds = {**credentials, "polymarket_signature_type": str(signature_type),
                 "polymarket_funder_address": funder}
        for negative_risk in (False, True):
            for style in ("limit", "market"):
                for side in ("BUY", "SELL"):
                    client = trading._polymarket_client(creds)
                    captured = []

                    def get(endpoint, headers=None, params=None, *, negative_risk=negative_risk):
                        if endpoint == host + VERSION:
                            return {"version": 2}
                        if endpoint == host + GET_TICK_SIZE:
                            assert params == {"token_id": "123"}
                            return {"minimum_tick_size": "0.01"}
                        if endpoint == host + GET_MARKET_BY_TOKEN + "123":
                            return {"condition_id": "fixture-condition"}
                        if endpoint == host + GET_CLOB_MARKET + "fixture-condition":
                            return {"t": [{"t": "123"}], "mts": "0.01",
                                    "nr": negative_risk, "fd": {"r": 0, "e": 0}}
                        unknown_reads.append(endpoint)
                        raise AssertionError("unexpected SDK read")

                    def post(endpoint, headers=None, data=None, params=None, *,
                             side=side, funder=funder, signature_type=signature_type,
                             negative_risk=negative_risk, client=client, captured=captured):
                        assert endpoint == host + POST_ORDER
                        body = json.loads(data)
                        wire = body["order"]
                        assert body["owner"] == "fixture-api-key"
                        assert body["orderType"] == "FAK"
                        assert body["postOnly"] is False and body["deferExec"] is False
                        assert wire["tokenId"] == "123" and wire["side"] == side
                        assert wire["maker"].lower() == funder.lower()
                        assert wire["signer"].lower() == signer_address.lower()
                        assert wire["signatureType"] == signature_type
                        amounts = (wire["makerAmount"], wire["takerAmount"])
                        assert amounts == (("4200000", "10000000") if side == "BUY"
                                           else ("10000000", "4200000"))
                        assert headers["POLY_API_KEY"] == "fixture-api-key"
                        unsigned = OrderV2(**{
                            **{k: v for k, v in wire.items() if k != "signature"},
                            "salt": str(wire["salt"]), "side": 0 if side == "BUY" else 1,
                        })
                        exchange = (
                            "0xe2222d279d744050d28e00520010520000310F59" if negative_risk else
                            "0xE111180000d2663C0091e4f400237545B87B996B"
                        )
                        builder = ExchangeOrderBuilderV2(exchange, 137, client.signer)
                        typed = builder.build_order_typed_data(unsigned)
                        assert typed["domain"]["name"] == "Polymarket CTF Exchange"
                        assert typed["domain"]["version"] == "2"
                        recovered = Account.recover_message(
                            encode_typed_data(full_message=typed), signature=wire["signature"],
                        )
                        assert recovered == signer_address
                        order_hash = builder.build_order_hash(typed)
                        assert len(order_hash) == 66 and order_hash.startswith("0x")
                        captured.append(order_hash)
                        return {"success": True, "orderID": order_hash, "status": "delayed",
                                "tradeIDs": []}

                    normalized = {"order_type": style, "exchange_order": {
                        "token_id": "123", "tick_size": "0.01", "neg_risk": negative_risk,
                        "side": side, "price": 0.42, "size": 10.0,
                        "amount": 4.2 if side == "BUY" else 10.0, "order_type": "FAK",
                    }}
                    with mock.patch.object(trading, "_polymarket_client", return_value=client), \
                         mock.patch.object(client, "_get", side_effect=get), \
                         mock.patch.object(client, "_post", side_effect=post):
                        result = trading._place_polymarket_order(normalized, creds)
                    assert len(captured) == 1
                    ack = result["acknowledgement"]
                    assert ack["venue_order_id"] == captured[0]
                    assert ack["status"] == "acknowledged" and ack["trade_ids"] == []
                    cases += 1
                    if first_client is None:
                        first_client = client
    assert not unknown_reads

    # Demonstrate SDK preparation and ID storage separately from the current
    # sign-and-post helper. This is NOT production adapter wiring or a real lookup.
    signed = first_client.create_order(
        OrderArgs(token_id="123", price=0.42, size=10.0, side="BUY"),
        PartialCreateOrderOptions(tick_size="0.01", neg_risk=False),
    )
    builder = ExchangeOrderBuilderV2(
        "0xE111180000d2663C0091e4f400237545B87B996B", 137, first_client.signer,
    )
    order_hash = builder.build_order_hash(builder.build_order_typed_data(signed))
    now = datetime(2026, 10, 19, tzinfo=timezone.utc)
    scope = AccountScope("sdk-fixture", "fixture-owner", "polymarket", "fixture-account",
                         "live", "USD", "fixture-connection", 1, now)
    intent = TradeIntent(
        "sdk-intent", scope.id, 1, "polymarket:live:fixture-condition", "BUY_YES",
        Decimal("10"), Decimal("0.42"), "FAK", "fixture-forecast", None,
        "fixture-policy", "fixture-strategy", "fixture-market", Decimal("0"), Decimal("0"),
        now + timedelta(minutes=1), now,
    )
    with TemporaryDirectory() as directory:
        database = str(Path(directory) / "commands.sqlite3")
        client = Client(path=database)
        try:
            store = DatastoreTwinStore(client)
            store.register_account(scope, venue_available_cash=Decimal("10"), loss_limit=Decimal("10"))
            store.reserve_intent(intent, cash=Decimal("4.2"), max_loss=Decimal("4.2"),
                                 now=now, client_order_id=order_hash)
        finally:
            client.close()
        client = Client(path=database)
        try:
            persisted = DatastoreTwinStore(client).command_for_intent(intent).client_order_id
            assert persisted == order_hash
        finally:
            client.close()
    reads = []

    def lookup(endpoint, headers=None, params=None):
        assert endpoint == host + GET_ORDER + persisted
        assert headers["POLY_API_KEY"] == "fixture-api-key"
        reads.append(endpoint)
        return {"id": persisted, "status": "LIVE"}

    def cancel(endpoint, headers=None, data=None, params=None):
        assert endpoint == host + CANCEL
        assert json.loads(data) == {"orderID": persisted}
        return {"canceled": [persisted], "not_canceled": {}}

    with mock.patch.object(first_client, "_get", side_effect=lookup), \
         mock.patch.object(first_client, "_delete", side_effect=cancel):
        assert first_client.get_order(persisted)["id"] == persisted
        assert first_client.cancel_order(OrderPayload(orderID=persisted))["canceled"] == [persisted]
    assert len(reads) == 1
    # The test's creation of an SDK hash must not be confused with a persisted
    # signed body: the SQL store retains only the identity/fingerprint here.
    assert replace(intent, quantity=Decimal("9")).intent_hash != intent.intent_hash
    assert not blocked_network, "SDK attempted an unmocked connection"
    print(json.dumps({"sdk_version": version("py-clob-client-v2"), "signed_cases": cases,
                      "sql_identity_and_lookup": True, "network_requests": 0}))


class PolymarketSdkContractTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("py_clob_client_v2"),
                         "optional trading SDK absent; SDK contract evidence unavailable")
    def test_real_sdk_signing_amounts_and_persisted_identity_offline(self):
        env = {key: value for key, value in os.environ.items() if key.upper() in {
            "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG",
            "APPDATA", "LOCALAPPDATA", "USERPROFILE",
        }}
        env.update(PYTHONPATH=str(ROOT / "src"), FORESEA_DATASTORE_BACKEND="sql")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--offline-probe"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["signed_cases"], 24)
        self.assertTrue(report["sql_identity_and_lookup"])
        self.assertEqual(report["network_requests"], 0)


if __name__ == "__main__":
    if "--offline-probe" in sys.argv:
        run_offline_probe()
    else:
        unittest.main()
