from __future__ import annotations

import base64
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import trading  # noqa: E402

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "twin" / "venues"


def _fixture(name: str):
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))

_TRADING_ENV = [
    "FORESEA_ENABLE_TRADING",
    "FORESEA_ALLOW_MARKET_ORDERS",
    "FORESEA_MAX_ORDER_NOTIONAL",
    "KALSHI_API_KEY_ID",
    "KALSHI_ACCESS_KEY_ID",
    "KALSHI_PRIVATE_KEY",
    "KALSHI_PRIVATE_KEY_FILE",
    "KALSHI_BASE_URL",
    "POLYMARKET_PRIVATE_KEY",
    "POLYMARKET_API_KEY",
    "POLYMARKET_API_SECRET",
    "POLYMARKET_API_PASSPHRASE",
]


def _rsa_private_key_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


class FakeResponse:
    def __init__(self, payload, status=201):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload


class TradingTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = {name: os.environ.get(name) for name in _TRADING_ENV}
        for name in _TRADING_ENV:
            os.environ.pop(name, None)

    def tearDown(self):
        for name in _TRADING_ENV:
            os.environ.pop(name, None)
            if self._saved_env[name] is not None:
                os.environ[name] = self._saved_env[name]

    def test_account_status_does_not_expose_secret_values(self):
        os.environ.update({
            "KALSHI_API_KEY_ID": "key-id",
            "KALSHI_PRIVATE_KEY": "secret-private-key",
            "POLYMARKET_PRIVATE_KEY": "pm-secret",
            "POLYMARKET_API_KEY": "pm-key",
            "POLYMARKET_API_SECRET": "pm-secret",
            "POLYMARKET_API_PASSPHRASE": "pm-passphrase",
        })

        status = trading.account_status()

        self.assertTrue(status["venues"]["kalshi"]["configured"])
        rendered = str(status)
        self.assertNotIn("secret-private-key", rendered)
        self.assertNotIn("pm-passphrase", rendered)

    def test_kalshi_preview_builds_v2_order_payload(self):
        preview = trading.preview_order({
            "platform": "kalshi",
            "ticker": "kx-test",
            "action": "buy",
            "outcome": "no",
            "price": 0.31,
            "quantity": 3,
        })

        order = preview["normalized_order"]["exchange_order"]
        self.assertEqual(order["ticker"], "KX-TEST")
        self.assertEqual(order["side"], "ask")
        self.assertEqual(order["count"], "3.00")
        self.assertEqual(order["price"], "0.6900")
        self.assertEqual(order["time_in_force"], "good_till_canceled")
        self.assertEqual(order["self_trade_prevention_type"], "taker_at_cross")
        self.assertFalse(preview["would_execute"])

    def test_kalshi_v2_maps_all_selected_outcome_actions_to_yes_book(self):
        cases = {
            ("buy", "yes"): ("bid", "0.3000"),
            ("sell", "yes"): ("ask", "0.3000"),
            ("buy", "no"): ("ask", "0.7000"),
            ("sell", "no"): ("bid", "0.7000"),
        }
        for (action, outcome), expected in cases.items():
            with self.subTest(action=action, outcome=outcome):
                preview = trading.preview_order({
                    "platform": "kalshi",
                    "ticker": "KX-TEST",
                    "action": action,
                    "outcome": outcome,
                    "price": "0.30",
                    "quantity": "3",
                })
                self.assertEqual(
                    (preview["normalized_order"]["exchange_order"]["side"],
                     preview["normalized_order"]["exchange_order"]["price"]),
                    expected,
                )
                self.assertEqual(
                    preview["estimated_notional"], 0.9 if action == "buy" else 2.1
                )

    def test_kalshi_rejects_unsupported_self_trade_prevention_type(self):
        with self.assertRaises(trading.TradingValidationError):
            trading.preview_order({
                "platform": "kalshi",
                "ticker": "KX-TEST",
                "price": "0.30",
                "quantity": "1",
                "self_trade_prevention_type": "cancel_all",
            })

    def test_kalshi_requires_contract_fixed_point_price_precision(self):
        with self.assertRaises(trading.TradingValidationError):
            trading.preview_order({
                "platform": "kalshi", "ticker": "KX-TEST", "price": "0.30001", "quantity": "1"
            })

    def test_preview_rejects_order_above_notional_cap(self):
        os.environ["FORESEA_MAX_ORDER_NOTIONAL"] = "5"

        with self.assertRaises(trading.TradingValidationError):
            trading.preview_order({
                "platform": "kalshi",
                "ticker": "KXTEST",
                "price": 0.60,
                "quantity": 10,
            })

    def test_polymarket_preview_accepts_direct_token_id_without_sdk(self):
        preview = trading.preview_order({
            "platform": "polymarket",
            "token_id": "123",
            "action": "buy",
            "outcome": "yes",
            "price": 0.42,
            "quantity": 5,
        })

        order = preview["normalized_order"]["exchange_order"]
        self.assertEqual(order["token_id"], "123")
        self.assertEqual(order["side"], "BUY")
        self.assertEqual(order["size"], 5.0)

    def test_polymarket_preview_requires_price_on_market_tick_and_minimum_size(self):
        with self.assertRaises(trading.TradingValidationError):
            trading.preview_order({
                "platform": "polymarket", "token_id": "123", "price": "0.421",
                "quantity": "5", "tick_size": "0.01",
            })
        with self.assertRaises(trading.TradingValidationError):
            trading.preview_order({
                "platform": "polymarket", "token_id": "123", "price": "0.42",
                "quantity": "4", "min_size": "5",
            })

    def test_polymarket_market_buy_uses_bounded_usd_and_sell_uses_shares(self):
        buy = trading.preview_order({
            "platform": "polymarket", "token_id": "123", "action": "buy",
            "price": "0.42", "quantity": "10", "order_type": "market", "max_cost": "4.20",
        })["normalized_order"]["exchange_order"]
        sell = trading.preview_order({
            "platform": "polymarket", "token_id": "123", "action": "sell",
            "price": "0.42", "quantity": "10", "order_type": "market",
        })["normalized_order"]["exchange_order"]
        self.assertEqual((buy["amount"], buy["amount_type"]), (4.2, "usd"))
        self.assertEqual((sell["amount"], sell["amount_type"]), (10.0, "shares"))

    def test_submission_acknowledgements_keep_unfilled_and_partial_states_distinct(self):
        kalshi = _fixture("kalshi_create_order_v2.json")
        ack = trading._normalise_kalshi_acknowledgement(
            kalshi, expected_client_order_id="foresea-fixture-order"
        )
        self.assertEqual(ack["status"], "acknowledged")
        self.assertEqual(ack["fill_quantity"], 0.0)
        partial = {**kalshi, "fill_count": "1.00", "remaining_count": "2.00"}
        self.assertEqual(
            trading._normalise_kalshi_acknowledgement(
                partial, expected_client_order_id="foresea-fixture-order"
            )["status"],
            "partially_filled",
        )
        poly = trading._normalise_polymarket_acknowledgement(
            _fixture("polymarket_market_order_ack.json")
        )
        self.assertEqual(poly["status"], "matched")
        self.assertEqual(poly["trade_ids"], ["trade-001"])

    def test_submission_acknowledgements_fail_closed_on_bad_or_rejected_shapes(self):
        with self.assertRaises(trading.TradingExecutionError):
            trading._normalise_kalshi_acknowledgement(
                {"order_id": "x", "client_order_id": "expected"},
                expected_client_order_id="expected",
            )
        with self.assertRaises(trading.TradingExecutionError):
            trading._normalise_polymarket_acknowledgement(
                {"success": False, "errorMsg": "insufficient allowance"}
            )
        with self.assertRaises(trading.TradingExecutionError):
            trading._normalise_polymarket_acknowledgement(
                {"success": True, "orderID": "x", "status": "mystery"}
            )
        with self.assertRaises(trading.TradingExecutionError):
            trading._normalise_polymarket_acknowledgement({
                "success": True, "orderID": "x", "status": "matched", "tradeIDs": ["t", "t"],
            })
        with self.assertRaises(trading.TradingExecutionError):
            trading._normalise_polymarket_acknowledgement(
                {"success": True, "orderID": "x", "status": "matched", "tradeIDs": ["t"]}
            )
        with self.assertRaises(trading.TradingExecutionError):
            trading._normalise_kalshi_acknowledgement(
                {"order_id": "x", "client_order_id": "expected", "fill_count": "0", "remaining_count": "1", "extra": "x" * 64_001},
                expected_client_order_id="expected",
            )

    def test_kalshi_auth_headers_sign_path_without_query(self):
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        pem = _rsa_private_key_pem()
        key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
        os.environ["KALSHI_API_KEY_ID"] = "key-id"
        os.environ["KALSHI_PRIVATE_KEY"] = pem

        headers = trading._kalshi_auth_headers(
            "POST",
            "/trade-api/v2/portfolio/events/orders?ignored=true",
            timestamp_ms="123",
        )

        signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
        key.public_key().verify(
            signature,
            b"123POST/trade-api/v2/portfolio/events/orders",
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        self.assertEqual(headers["KALSHI-ACCESS-KEY"], "key-id")
        self.assertEqual(headers["KALSHI-ACCESS-TIMESTAMP"], "123")

    def test_live_order_requires_enable_flag(self):
        with self.assertRaises(trading.TradingDisabledError):
            trading.place_order(
                {
                    "platform": "kalshi",
                    "ticker": "KXTEST",
                    "price": 0.50,
                    "quantity": 1,
                    "execute": True,
                    "confirmation": trading.CONFIRMATION_PHRASE,
                },
                user_id="user-1",
            )

    def test_live_kalshi_order_posts_after_confirmation(self):
        os.environ.update({
            "FORESEA_ENABLE_TRADING": "true",
            "KALSHI_API_KEY_ID": "key-id",
            "KALSHI_PRIVATE_KEY": _rsa_private_key_pem(),
            "KALSHI_BASE_URL": "https://external-api.demo.kalshi.co/trade-api/v2",
        })
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(url=url, headers=headers, json=json, timeout=timeout)
            return FakeResponse({
                "order_id": "ord_123",
                "client_order_id": json["client_order_id"],
                "fill_count": "0.00",
                "remaining_count": "2.00",
            })

        with mock.patch("requests.post", fake_post):
            result = trading.place_order(
                {
                    "platform": "kalshi",
                    "ticker": "KXTEST",
                    "action": "buy",
                    "outcome": "yes",
                    "price": 0.56,
                    "quantity": 2,
                    "execute": True,
                    "confirmation": trading.CONFIRMATION_PHRASE,
                },
                user_id="user-1",
            )

        self.assertTrue(result["submitted"])
        self.assertEqual(captured["url"], "https://external-api.demo.kalshi.co/trade-api/v2/portfolio/events/orders")
        self.assertEqual(captured["json"]["side"], "bid")
        self.assertEqual(captured["json"]["price"], "0.5600")
        self.assertIn("KALSHI-ACCESS-SIGNATURE", captured["headers"])
        self.assertEqual(result["venue_response"]["acknowledgement"]["status"], "acknowledged")


if __name__ == "__main__":
    unittest.main()
