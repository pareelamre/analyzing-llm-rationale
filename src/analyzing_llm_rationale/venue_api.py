"""Bounded venue operations, using the checked-in September 2026 API contracts.

Only catalogued operations are callable. Account credentials never come from
operation parameters; the HTTP layer supplies the signed-in user's connection.
"""
from __future__ import annotations

import json
import logging
from decimal import Decimal, InvalidOperation
from importlib.resources import files
from typing import Any
from urllib.parse import quote

import requests
from jsonschema import Draft7Validator
from opentelemetry import metrics, trace

from . import market_data, trading

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
operations = metrics.get_meter(__name__).create_counter("venue.operations", unit="1")
CONTRACTS = json.loads(files(__package__).joinpath("venue_contracts.json").read_text(encoding="utf-8"))
BASES = {
    "kalshi": "https://api.elections.kalshi.com/trade-api/v2",
    "clob": "https://clob.polymarket.com",
    "data": "https://data-api.polymarket.com",
}


def contract(platform: str, operation: str, access: str) -> dict:
    spec = CONTRACTS.get(f"{platform}.{operation}")
    if spec is None or spec["access"] != access:
        raise market_data.MarketDataInputError("Unknown operation for this venue and access scope")
    return spec


def catalog(access: str = "public") -> dict:
    return {key: value for key, value in CONTRACTS.items() if value["access"] == access}


def _validate(value: Any, schema: dict, label: str) -> None:
    error = next(Draft7Validator(schema).iter_errors(value), None)
    if error:
        # Do not echo account parameters or credentials in validation errors.
        field = ".".join(str(part) for part in error.absolute_path)
        raise market_data.MarketDataInputError(f"Invalid {label}{'.' + field if field else ''}: {error.validator}")


def prepare(spec: dict, parameters: dict | None, body: Any = None) -> tuple[str, dict, Any]:
    parameters = dict(parameters or {})
    props = {p["name"]: p["schema"] for p in spec["parameters"]}
    required = [p["name"] for p in spec["parameters"] if p["required"]]
    _validate(parameters, {"type": "object", "properties": props, "required": required,
                           "additionalProperties": False}, "parameters")
    body = {} if body is None else body
    _validate(body, spec["body"], "body")
    for low, high in (("start_ts", "end_ts"), ("min_ts", "max_ts"), ("start", "end"), ("from", "to")):
        if low in parameters and high in parameters and parameters[low] >= parameters[high]:
            raise market_data.MarketDataInputError(f"{low} must be before {high}")
    if "candlestick" in spec["path"] and parameters.get("period_interval") not in (1, 60, 1440):
        raise market_data.MarketDataInputError("period_interval must be 1, 60, or 1440")
    if spec["operation"] == "candlesticks" and len(parameters["market_tickers"].split(",")) > 100:
        raise market_data.MarketDataInputError("At most 100 market tickers are allowed")
    if "market" in parameters and "eventId" in parameters:
        raise market_data.MarketDataInputError("market and eventId are mutually exclusive")
    path, query = spec["path"], {}
    for param in spec["parameters"]:
        name = param["name"]
        if name not in parameters:
            continue
        value = parameters[name]
        if param["in"] == "path":
            if not value or value in (".", "..") or any(c in value for c in "/\\?#"):
                raise market_data.MarketDataInputError(f"Invalid path identifier: {name}")
            path = path.replace("{" + name + "}", quote(str(value), safe=""))
        else:
            if isinstance(value, list) and not param["explode"]:
                value = ",".join(str(item) for item in value)
            if isinstance(value, bool):
                value = str(value).lower()
            query[name] = value
    return path, query, body


def _public_request(spec: dict, path: str, query: dict, body: Any) -> Any:
    url = BASES[spec["source"]] + path
    if spec["method"] == "GET":
        return market_data._get_json(url, params=query)
    try:
        response = requests.post(url, json=body, timeout=12, headers={"User-Agent": "foresea-market-bot/1.0"})
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise market_data.MarketDataError("Venue batch request failed") from exc


@tracer.start_as_current_span("venue.read")
def read(platform: str, operation: str, parameters: dict | None = None, body: Any = None,
         *, access: str = "public", creds: dict | None = None) -> dict:
    spec = contract(platform, operation, access)
    if access not in ("public", "account"):
        raise market_data.MarketDataInputError("Read access must be public or account")
    attrs = {"venue": platform, "operation": operation}
    trace.get_current_span().set_attributes(attrs)
    try:
        parameters = dict(parameters or {})
        if access == "account":
            if not creds:
                raise trading.TradingNotConfiguredError("Connect a venue account first")
            if platform == "polymarket":
                client = trading._polymarket_client(creds)
                address_parameter = "user" if spec["source"] == "data" else "maker_address" if operation == "fills" else None
                if address_parameter:
                    if address_parameter in parameters:
                        raise market_data.MarketDataInputError("Account address comes from the connected account")
                    parameters[address_parameter] = trading._polymarket_account_address(client, creds)
        path, query, body = prepare(spec, parameters, body)
        if access == "account" and platform == "kalshi":
            data = trading._kalshi_request("GET", path, creds=creds, params=query)
        elif access == "account" and spec["source"] == "clob":
            # The SDK's convenience order method discards next_cursor. Use its
            # signed transport for one page so callers can explicitly continue.
            data = client._get(f"{client.host}{path}", headers=client._l2_headers("GET", path), params=query)
        else:
            data = _public_request(spec, path, query, body)
        if not isinstance(data, (dict, list)):
            raise market_data.MarketDataError("Venue response must be an object or list")
        result = {"platform": platform, "operation": operation, "data": data}
        # Preserve upstream cursors and expose offset continuation without silently
        # claiming a single page is the complete account history.
        if isinstance(data, dict) and "cursor" in data:
            result["next_cursor"] = data["cursor"] or None
        if isinstance(data, dict) and "next_cursor" in data:
            result["next_cursor"] = None if data["next_cursor"] in ("", "LTE=") else data["next_cursor"]
        if spec["source"] == "data" and "offset" in {p["name"] for p in spec["parameters"]}:
            limit = parameters.get("limit", 10 if operation == "closed_positions" else 100)
            maximum = next(p["schema"].get("maximum") for p in spec["parameters"] if p["name"] == "offset")
            next_offset = parameters.get("offset", 0) + limit
            full = isinstance(data, list) and limit > 0 and len(data) == limit
            result["next_offset"] = next_offset if full and next_offset <= maximum else None
            result["pagination_limit_reached"] = bool(full and next_offset > maximum)
        operations.add(1, {**attrs, "outcome": "success"})
        logger.info("Venue read completed: %s/%s", platform, operation)
        return result
    except Exception as exc:
        operations.add(1, {**attrs, "outcome": "error"})
        logger.warning("Venue read failed: %s/%s", platform, operation)
        if isinstance(exc, (market_data.MarketDataError, market_data.MarketDataInputError, trading.TradingError)):
            raise
        error_type = trading.TradingExecutionError if access == "account" else market_data.MarketDataError
        raise error_type("Venue read failed") from exc


def validate_action(platform: str, operation: str, parameters: dict | None, body: Any,
                    *, execute: bool, confirmation: str, creds: dict | None) -> tuple[dict, str, dict, Any]:
    spec = contract(platform, operation, "write")
    if execute is not True or confirmation != "MANAGE REAL ORDERS":
        raise trading.TradingValidationError("Set execute=true and confirmation='MANAGE REAL ORDERS'")
    if not creds:
        raise trading.TradingNotConfiguredError("Connect a venue account first")
    trading._require_execution_enabled({"order_type": "limit"}, byo=trading._is_byo(creds))
    path, query, body = prepare(spec, parameters, body)
    if operation == "cancel_market_orders" and (not body["market"].strip() or not body["asset_id"].strip()):
        raise trading.TradingValidationError("Scoped cancellation requires nonempty market and asset_id")
    if operation == "decrease_order":
        if ("reduce_by" in body) == ("reduce_to" in body):
            raise trading.TradingValidationError("Provide exactly one of reduce_by or reduce_to")
        try:
            value = Decimal(body.get("reduce_by", body.get("reduce_to")))
            if not value.is_finite() or value < 0 or ("reduce_by" in body and value == 0):
                raise ValueError
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise trading.TradingValidationError("Reduction must be a nonnegative finite count") from exc
    if operation in ("create_order_group", "limit_order_group"):
        try:
            value = Decimal(str(body.get("contracts_limit_fp", body.get("contracts_limit"))))
            if not value.is_finite() or value <= 0:
                raise ValueError
            if "contracts_limit" in body and Decimal(str(body["contracts_limit"])) != value:
                raise ValueError
        except (InvalidOperation, ValueError) as exc:
            raise trading.TradingValidationError("Provide a positive, consistent contracts limit") from exc
    return spec, path, query, body


@tracer.start_as_current_span("venue.action")
def action(platform: str, operation: str, parameters: dict | None = None, body: Any = None,
           *, execute: bool = False, confirmation: str = "", creds: dict | None = None) -> dict:
    spec, path, query, body = validate_action(platform, operation, parameters, body,
                                             execute=execute, confirmation=confirmation, creds=creds)
    attrs = {"venue": platform, "operation": operation}
    trace.get_current_span().set_attributes(attrs)
    try:
        if platform == "kalshi":
            data = trading._kalshi_request(spec["method"], path, creds=creds, params=query,
                                          json_body=body if spec["method"] in ("POST", "PUT") or body else None)
        else:
            client = trading._polymarket_client(creds)
            if operation == "cancel_all":
                data = client.cancel_all()
            elif operation == "cancel_market_orders":
                from py_clob_client_v2.clob_types import OrderMarketCancelParams
                data = client.cancel_market_orders(OrderMarketCancelParams(**body))
            else:
                data = client.post_heartbeat(body["heartbeat_id"])
        operations.add(1, {**attrs, "outcome": "success"})
        logger.info("Venue action completed: %s/%s", platform, operation)
        return {"platform": platform, "operation": operation, "data": data}
    except Exception as exc:
        operations.add(1, {**attrs, "outcome": "error"})
        logger.warning("Venue action failed: %s/%s", platform, operation)
        if isinstance(exc, trading.TradingError):
            raise
        # Never retry writes: a transport failure may follow exchange acceptance.
        raise trading.TradingExecutionError("Venue action failed; reconcile before retrying") from exc
