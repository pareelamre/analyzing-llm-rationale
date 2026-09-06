"""Bring-your-own-account (per-request credential) trading tests.

The shared server account is covered by test_trading.py. These cover the path
where a signed-in user supplies their OWN Kalshi/Polymarket credentials per
request: env is empty, credentials flow through `creds`, the separate
FORESEA_ENABLE_BYO_TRADING gate applies, and nothing is persisted/echoed.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import trading  # noqa: E402

_TRADING_ENV = [
    "FORESEA_ENABLE_TRADING",
    "FORESEA_ENABLE_BYO_TRADING",
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


class ByoTradingTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = {name: os.environ.get(name) for name in _TRADING_ENV}
        for name in _TRADING_ENV:
            os.environ.pop(name, None)

    def tearDown(self):
        for name in _TRADING_ENV:
            os.environ.pop(name, None)
            if self._saved_env[name] is not None:
                os.environ[name] = self._saved_env[name]

    def test_account_status_reports_request_source_with_creds(self):
        # No env at all; readiness comes purely from request creds.
        creds = {
            "kalshi_api_key_id": "user-key",
            "kalshi_private_key": "user-private-key",
        }
        status = trading.account_status(creds)
        self.assertEqual(status["credential_source"], "request")
        self.assertTrue(status["venues"]["kalshi"]["configured"])
        # Secrets never leak into the status payload.
        self.assertNotIn("user-private-key", str(status))

    def test_account_status_without_creds_is_server_source(self):
        status = trading.account_status()
        self.assertEqual(status["credential_source"], "server_environment")
        self.assertFalse(status["venues"]["kalshi"]["configured"])

    def test_byo_preview_succeeds_without_env_and_caps_notional(self):
        creds = {"kalshi_api_key_id": "k", "kalshi_private_key": "pk"}
        preview = trading.preview_order(
            {"platform": "kalshi", "ticker": "KXTEST", "price": 0.40, "quantity": 2},
            creds,
        )
        self.assertFalse(preview["would_execute"])
        self.assertTrue(
            any("FORESEA_ENABLE_BYO_TRADING" in w for w in preview["warnings"])
        )

        os.environ["FORESEA_MAX_ORDER_NOTIONAL"] = "5"
        with self.assertRaises(trading.TradingValidationError):
            trading.preview_order(
                {"platform": "kalshi", "ticker": "KXTEST", "price": 0.60, "quantity": 10},
                creds,
            )

    def test_byo_order_blocked_until_byo_flag_set(self):
        # The shared-account flag must NOT enable the BYO path.
        os.environ["FORESEA_ENABLE_TRADING"] = "true"
        creds = {"kalshi_api_key_id": "k", "kalshi_private_key": _rsa_private_key_pem()}
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
                creds=creds,
            )

    def test_byo_kalshi_order_signs_with_request_key(self):
        os.environ["FORESEA_ENABLE_BYO_TRADING"] = "true"  # note: no FORESEA_ENABLE_TRADING
        creds = {
            "kalshi_api_key_id": "byo-key-id",
            "kalshi_private_key": _rsa_private_key_pem(),
            "kalshi_base_url": "https://external-api.demo.kalshi.co/trade-api/v2",
        }
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(url=url, headers=headers, json=json)
            return FakeResponse({
                "order_id": "ord_byo",
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
                creds=creds,
            )

        self.assertTrue(result["submitted"])
        self.assertEqual(
            captured["url"],
            "https://external-api.demo.kalshi.co/trade-api/v2/portfolio/events/orders",
        )
        self.assertEqual(captured["headers"]["KALSHI-ACCESS-KEY"], "byo-key-id")
        self.assertEqual(result["venue_response"]["acknowledgement"]["status"], "acknowledged")
        # Response echo must not contain the supplied secret.
        self.assertNotIn(creds["kalshi_private_key"], str(result))

    def test_byo_confirmation_phrase_still_required(self):
        os.environ["FORESEA_ENABLE_BYO_TRADING"] = "true"
        creds = {"kalshi_api_key_id": "k", "kalshi_private_key": _rsa_private_key_pem()}
        with self.assertRaises(trading.TradingValidationError):
            trading.place_order(
                {
                    "platform": "kalshi",
                    "ticker": "KXTEST",
                    "price": 0.50,
                    "quantity": 1,
                    "execute": True,
                    "confirmation": "nope",
                },
                user_id="user-1",
                creds=creds,
            )


if __name__ == "__main__":
    unittest.main()
