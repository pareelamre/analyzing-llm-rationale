"""Contract, isolation and reconnect regressions for additional venue APIs."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import market_data as md  # noqa: E402
from analyzing_llm_rationale import trading, venue_streams  # noqa: E402
from analyzing_llm_rationale import venue_api as api


class VenueReadTests(unittest.TestCase):
    def test_scope_and_path_injection_rejected_before_network(self):
        with patch.object(md, "_get_json") as get:
            for platform, op, params in [("kalshi", "historical_fills", {}),
                                         ("polymarket", "cancel_all", {}),
                                         ("kalshi", "historical_market", {"ticker": "../portfolio"})]:
                with self.assertRaises(md.MarketDataInputError):
                    api.read(platform, op, params)
            get.assert_not_called()

    def test_batch_post_keeps_token_body_and_fee_rate_uses_path(self):
        response = MagicMock()
        response.json.return_value = {"123": "0.04"}
        with patch.object(api.requests, "post", return_value=response) as post:
            self.assertEqual(api.read("polymarket", "spreads", body=[{"token_id": "123"}])["data"], {"123": "0.04"})
            self.assertEqual(post.call_args.args, ("https://clob.polymarket.com/spreads",))
            self.assertEqual(post.call_args.kwargs["json"], [{"token_id": "123"}])
        with patch.object(md, "_get_json", return_value={"base_fee": 0}) as get:
            api.read("polymarket", "fee_rate", {"token_id": "123"})
            self.assertTrue(get.call_args.args[0].endswith("/fee-rate/123"))

    def test_signed_batch_books_serialize_tickers_and_history_preserves_cursor(self):
        with patch.object(trading, "_kalshi_request", return_value={"orderbooks": []}) as get:
            api.read("kalshi", "orderbooks", {"tickers": ["A", "B"]}, access="account", creds={"key": "test"})
            self.assertEqual(get.call_args.kwargs["params"], {"tickers": ["A", "B"]})
        with patch.object(trading, "_kalshi_request", return_value={"settlements": [], "cursor": "page2"}) as get:
            result = api.read("kalshi", "settlements", {"cursor": "page1", "limit": 10}, access="account", creds={"key": "test"})
            self.assertEqual(result["next_cursor"], "page2")
            self.assertEqual(get.call_args.kwargs["params"]["cursor"], "page1")

    def test_input_bounds_and_required_identifiers(self):
        for operation, params in [("holders", {}), ("holders", {"market": ["bad"]}),
                                   ("live_volume", {"id": 0})]:
            with self.assertRaises(md.MarketDataInputError):
                api.read("polymarket", operation, params)
        with self.assertRaises(md.MarketDataInputError):
            api.read("kalshi", "candlesticks", {"market_tickers": "A", "start_ts": 10, "end_ts": 9, "period_interval": 60})

    def test_account_address_cannot_be_overridden_and_offset_is_explicit(self):
        address = "0x" + "a" * 40
        with patch.object(trading, "_polymarket_client"), patch.object(trading, "_polymarket_account_address", return_value=address), patch.object(md, "_get_json", return_value=[{}, {}]) as get:
            result = api.read("polymarket", "activity", {"limit": 2, "offset": 4}, access="account", creds={"key": "test"})
            self.assertEqual(result["next_offset"], 6)
            self.assertEqual(get.call_args.kwargs["params"]["user"], address)
            with self.assertRaises(md.MarketDataInputError):
                api.read("polymarket", "activity", {"user": "another-user"}, access="account", creds={"key": "test"})

    def test_clob_cursor_survives_sdk_transport(self):
        client = MagicMock(host="https://clob.polymarket.com")
        client._get.return_value = {"data": [], "next_cursor": "LTE="}
        with patch.object(trading, "_polymarket_client", return_value=client):
            result = api.read("polymarket", "orders", {"next_cursor": "abc"}, access="account", creds={"key": "test"})
        self.assertIsNone(result["next_cursor"])
        self.assertEqual(client._get.call_args.kwargs["params"], {"next_cursor": "abc"})

    def test_resolution_falls_back_only_for_not_found(self):
        with patch.object(md, "_get_json", side_effect=[md.MarketDataNotFound(), {"market": {"status": "settled", "result": "yes"}}]) as get:
            self.assertEqual(md.resolve_kalshi("OLD-MARKET"), 1)
            self.assertTrue(get.call_args.args[0].endswith("/historical/markets/OLD-MARKET"))
        with patch.object(md, "_get_json", side_effect=md.MarketDataError("503")) as get:
            with self.assertRaises(md.MarketDataError):
                md.resolve_kalshi("OLD-MARKET")
            self.assertEqual(get.call_count, 1)

    def test_series_prefixed_ticker_is_normalised_before_the_archive_fallback(self):
        # Two independent fixes met on these exact lines and each is easy to
        # lose in a merge: normalising a "series/ticker" ident left by an older
        # ident_from_url (#337), and falling back to the historical endpoint
        # for archived markets (#412). Order matters -- _kalshi_market_detail
        # percent-encodes with safe="", so a surviving "/" would go upstream as
        # "%2F" and match nothing on either endpoint. Pin both together.
        with patch.object(
            md, "_get_json",
            side_effect=[md.MarketDataNotFound(), {"market": {"status": "settled", "result": "yes"}}],
        ) as get:
            self.assertEqual(md.resolve_kalshi("KXSERIES/KXSERIES-26SEP"), 1)
            urls = [c.args[0] for c in get.call_args_list]
        # The series prefix is gone, and was never percent-encoded instead.
        for url in urls:
            self.assertNotIn("%2F", url)
            self.assertNotIn("KXSERIES/KXSERIES", url)
        self.assertTrue(urls[0].endswith("/KXSERIES-26SEP"))
        # And the archive fallback still fired for the archived ticker.
        self.assertIn("/historical/markets/KXSERIES-26SEP", urls[1])

    def test_fetch_kalshi_also_normalises_before_the_archive_fallback(self):
        # The same pair of fixes collided in the other function too.
        with patch.object(
            md, "_get_json",
            side_effect=[md.MarketDataNotFound(),
                         {"market": {"ticker": "KXSERIES-26SEP", "status": "active"}}],
        ) as get:
            md.fetch_kalshi("kxseries/kxseries-26sep")
            urls = [c.args[0] for c in get.call_args_list]
        self.assertNotIn("%2F", urls[0])
        self.assertTrue(urls[0].endswith("/KXSERIES-26SEP"))
        self.assertIn("/historical/markets/KXSERIES-26SEP", urls[1])

    def test_empty_live_candles_can_use_archive(self):
        with patch.object(md, "_get_json", side_effect=[{"candlesticks": []}, {"candlesticks": [{"end_period_ts": 100}]}]) as get:
            rows = md.fetch_kalshi_candlesticks("OLD-MARKET", start_ts=1, end_ts=100)
        self.assertEqual(len(rows), 1)
        self.assertIn("/historical/markets/", get.call_args.args[0])


class VenueActionTests(unittest.TestCase):
    def test_enablement_confirmation_and_reduction_validation(self):
        with patch.object(trading, "_kalshi_request") as send, patch.dict(os.environ, {"FORESEA_ENABLE_BYO_TRADING": "false"}):
            with self.assertRaises(trading.TradingValidationError):
                api.action("kalshi", "cancel_all", creds={"key": "test"})
            with self.assertRaises(trading.TradingDisabledError):
                api.action("kalshi", "cancel_all", execute=True, confirmation="MANAGE REAL ORDERS", creds={"key": "test"})
            send.assert_not_called()
        with patch.object(trading, "_require_execution_enabled"):
            with self.assertRaises(trading.TradingValidationError):
                api.action("kalshi", "decrease_order", {"order_id": "id"}, {"reduce_by": "1.00", "reduce_to": "2.00"}, execute=True, confirmation="MANAGE REAL ORDERS", creds={"key": "test"})

    def test_batch_cancel_uses_delete_json_and_no_retries(self):
        body = {"orders": [{"order_id": "id", "subaccount": 2, "exchange_index": 1}]}
        with patch.object(trading, "_require_execution_enabled"), patch.object(trading, "_kalshi_request", return_value={"orders": []}) as send:
            api.action("kalshi", "cancel_orders", body=body, execute=True, confirmation="MANAGE REAL ORDERS", creds={"key": "test"})
            self.assertEqual(send.call_args.args, ("DELETE", "/portfolio/events/orders/batched"))
            self.assertEqual(send.call_args.kwargs["json_body"], body)
        with patch.object(trading, "_require_execution_enabled"), patch.object(trading, "_kalshi_request", side_effect=trading.TradingExecutionError("timeout")) as send:
            with self.assertRaises(trading.TradingExecutionError):
                api.action("kalshi", "cancel_orders", body=body, execute=True, confirmation="MANAGE REAL ORDERS", creds={"key": "test"})
            self.assertEqual(send.call_count, 1)

    def test_poly_cancel_and_heartbeat_use_sdk_auth_methods(self):
        client = MagicMock()
        sdk_types = SimpleNamespace(OrderMarketCancelParams=SimpleNamespace)
        with patch.dict(sys.modules, {"py_clob_client_v2.clob_types": sdk_types}), patch.object(trading, "_require_execution_enabled"), patch.object(trading, "_polymarket_client", return_value=client):
            for op, body in [("cancel_all", {}), ("cancel_market_orders", {"market": "condition", "asset_id": "123"}), ("heartbeat", {"heartbeat_id": "previous-id"})]:
                api.action("polymarket", op, body=body, execute=True, confirmation="MANAGE REAL ORDERS", creds={"key": "test"})
        client.cancel_all.assert_called_once_with()
        self.assertEqual(client.cancel_market_orders.call_args.args[0].asset_id, "123")
        client.post_heartbeat.assert_called_once_with("previous-id")

    def test_scoped_cancel_cannot_turn_into_cancel_all(self):
        with patch.object(trading, "_require_execution_enabled"), patch.object(trading, "_polymarket_client") as client:
            with self.assertRaises(trading.TradingValidationError):
                api.action("polymarket", "cancel_market_orders", body={"market": "", "asset_id": ""}, execute=True, confirmation="MANAGE REAL ORDERS", creds={"key": "test"})
            client.assert_not_called()

    def test_group_limits_require_consistent_positive_values_and_use_put(self):
        with patch.object(trading, "_require_execution_enabled"), patch.object(trading, "_kalshi_request", return_value={}) as send:
            for body in ({}, {"contracts_limit_fp": "0.00"}, {"contracts_limit": 2, "contracts_limit_fp": "3.00"}):
                with self.assertRaises(trading.TradingValidationError):
                    api.action("kalshi", "limit_order_group", {"order_group_id": "group"}, body, execute=True, confirmation="MANAGE REAL ORDERS", creds={"key": "test"})
            send.assert_not_called()
            api.action("kalshi", "limit_order_group", {"order_group_id": "group", "subaccount": 2}, {"contracts_limit": 2}, execute=True, confirmation="MANAGE REAL ORDERS", creds={"key": "test"})
            self.assertEqual(send.call_args.args, ("PUT", "/portfolio/order_groups/group/limit"))
            self.assertEqual(send.call_args.kwargs["params"], {"subaccount": 2})

    def test_signed_request_sends_json_and_signs_path_without_query(self):
        response = MagicMock(status_code=200)
        response.json.return_value = {"orders": []}
        with patch.object(trading, "_kalshi_auth_headers", return_value={}) as sign, patch("requests.request", return_value=response) as request:
            trading._kalshi_request("DELETE", "/portfolio/events/orders/batched", creds={"kalshi_base_url": "https://demo-api.kalshi.co/trade-api/v2"}, params={"subaccount": 2}, json_body={"orders": [{"order_id": "id"}]})
        self.assertEqual(sign.call_args.args, ("DELETE", "/trade-api/v2/portfolio/events/orders/batched"))
        self.assertEqual(request.call_args.kwargs["json"], {"orders": [{"order_id": "id"}]})


class VenueStreamTests(unittest.IsolatedAsyncioTestCase):
    def test_subscription_auth_and_environment(self):
        with patch.object(trading, "_kalshi_auth_headers", return_value={"signed": "value"}) as sign:
            url, headers, frame = venue_streams.subscription("kalshi", "user", [], {"kalshi_base_url": "https://demo-api.kalshi.co/trade-api/v2"})
            self.assertIn("external-api-ws.demo.kalshi.co", url)
            self.assertEqual(frame["params"]["channels"], ["fill", "user_orders"])
            self.assertEqual(sign.call_args.args, ("GET", "/trade-api/ws/v2"))
            self.assertEqual(headers, {"signed": "value"})
        with self.assertRaises(trading.TradingNotConfiguredError):
            venue_streams.subscription("polymarket", "user", [], None)
        with self.assertRaises(md.MarketDataInputError):
            venue_streams.subscription("polymarket", "market", ["bad-token"], None)

    async def test_poly_stream_sends_native_subscription_redacts_keys_and_closes(self):
        connection = MagicMock()
        connection.__aenter__ = AsyncMock(return_value=connection)
        connection.__aexit__ = AsyncMock(return_value=False)
        connection.send = AsyncMock()
        connection.__aiter__.return_value = ['PONG', json.dumps({"event_type": "book", "owner": "secret-api-key", "bids": []})]
        with patch("websockets.asyncio.client.connect", new=AsyncMock(return_value=connection)):
            stream = venue_streams.stream("polymarket", "market", ["123"])
            self.assertEqual((await anext(stream))["type"], "stream_reset")
            event = await anext(stream)
            self.assertNotIn("owner", event["data"])
            self.assertEqual(event["data"]["event_type"], "book")
            await stream.aclose()
        self.assertEqual(json.loads(connection.send.call_args.args[0]), {"type": "market", "assets_ids": ["123"]})
        connection.__aexit__.assert_awaited_once()

    async def test_sequence_gap_triggers_reset_before_new_data(self):
        connection = MagicMock()
        connection.__aenter__ = AsyncMock(return_value=connection)
        connection.__aexit__ = AsyncMock(return_value=False)
        connection.send = AsyncMock()
        connection.__aiter__.return_value = [json.dumps({"sid": 1, "seq": 1}), json.dumps({"sid": 1, "seq": 3})]
        with patch.object(venue_streams, "subscription", return_value=("wss://test", {}, {})), patch("websockets.asyncio.client.connect", new=AsyncMock(return_value=connection)):
            stream = venue_streams.stream("kalshi", "market", ["A"])
            await anext(stream)
            self.assertEqual((await anext(stream))["data"]["seq"], 1)
            self.assertEqual((await anext(stream))["type"], "stream_reconnecting")
            await stream.aclose()


class VenueRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from starlette.testclient import TestClient

        from analyzing_llm_rationale import server
        cls.server = server
        cls.client = TestClient(server.app)

    def test_public_route_cannot_read_accounts_or_execute(self):
        with patch.object(self.server, "_check_rate_limit"):
            for op in ("historical_fills", "cancel_all"):
                response = self.client.post(f"/market/venue/kalshi/{op}", json={})
                self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.client.post("/trading/venue/kalshi/settlements", json={}).status_code, 401)

    def test_amendment_requires_owned_order_and_existing_guardrails(self):
        record = {"platform": "kalshi", "venue_order_id": "venue-id", "ticker": "KXTEST", "status": "open", "action": "buy", "outcome": "yes", "exchange_index": 0}
        payload = {"audit_order_id": "audit-id", "execute": True, "confirmation": "MANAGE REAL ORDERS", "body": {"ticker": "KXTEST", "side": "bid", "price": "0.5000", "count": "2.00"}}
        with patch.object(self.server, "_check_rate_limit"), patch.object(self.server, "_require_session", return_value={"sub": "user-a"}), patch.object(self.server, "_stored_trading_credentials", return_value={"key": "test"}), patch.object(self.server, "_read_trading_order", return_value=None) as read_order, patch.object(self.server, "_record_trading_risk_event"), patch.object(api, "action") as send:
            response = self.client.post("/trading/venue/kalshi/actions/amend_order", json=payload)
            self.assertEqual(response.status_code, 404)
            read_order.assert_called_once_with("user-a", "audit-id")
            send.assert_not_called()
        with patch.object(self.server, "_check_rate_limit"), patch.object(self.server, "_require_session", return_value={"sub": "user-a"}), patch.object(self.server, "_stored_trading_credentials", return_value={"key": "test"}), patch.object(self.server, "_read_trading_order", return_value=record), patch.object(self.server, "_record_trading_risk_event"), patch.object(trading, "_require_execution_enabled"), patch.object(trading, "preview_order", return_value={"normalized_order": {}}), patch.object(self.server, "_validate_live_trade_guardrails", new=AsyncMock(side_effect=self.server.HTTPException(409, "paused"))) as guard, patch.object(api, "action") as send:
            response = self.client.post("/trading/venue/kalshi/actions/amend_order", json=payload)
            self.assertEqual(response.status_code, 409, response.text)
            guard.assert_awaited_once()
            send.assert_not_called()

    def test_catalog_is_public_and_private_stream_auth_fails_closed(self):
        result = self.client.get("/market/venue/catalog").json()
        self.assertIn("polymarket.holders", result)
        self.assertNotIn("kalshi.settlements", result)
        with patch.object(self.server, "_check_rate_limit"), patch.object(venue_streams, "stream") as stream:
            with self.client.websocket_connect("/ws/venue/polymarket") as ws:
                ws.send_json({"scope": "user"})
                message = ws.receive()
                self.assertEqual(message["code"], 1008)
            stream.assert_not_called()

    def test_successful_decrease_uses_owned_routing_and_updates_audit(self):
        record = {"id": "audit-id", "platform": "kalshi", "venue_order_id": "venue-id", "ticker": "KXTEST", "status": "open", "action": "buy", "outcome": "yes", "exchange_index": 3, "subaccount": 2}
        payload = {"audit_order_id": "audit-id", "execute": True, "confirmation": "MANAGE REAL ORDERS", "body": {"reduce_by": "2.00"}}
        result = {"data": {"order": {"order_id": "venue-id", "status": "resting", "remaining_count_fp": "3.00"}}}
        with patch.object(self.server, "_check_rate_limit"), patch.object(self.server, "_require_session", return_value={"sub": "user-a"}), patch.object(self.server, "_stored_trading_credentials", return_value={"key": "test"}), patch.object(self.server, "_read_trading_order", return_value=record), patch.object(self.server, "_record_trading_risk_event"), patch.object(trading, "_require_execution_enabled"), patch.object(api, "action", return_value=result) as send, patch.object(self.server, "_put_trading_order") as save, patch.object(self.server, "_sync_trade_run_from_order"):
            response = self.client.post("/trading/venue/kalshi/actions/decrease_order", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(send.call_args.args[2], {"order_id": "venue-id", "subaccount": 2})
        self.assertEqual(send.call_args.args[3], {"reduce_by": "2.00", "exchange_index": 3, "market_ticker": "KXTEST"})
        self.assertEqual(save.call_args.args[1]["remaining_quantity"], 3.0)


if __name__ == "__main__":
    unittest.main()
