"""Guarded trading adapters for prediction-market execution.

The public Foresea agent can analyse markets without credentials. This module is
for explicit, signed-in human order submission only: live execution is disabled
by default, credentials come from server-side environment variables, and every
order goes through the same preview/confirmation checks before an exchange call.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse

POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CONFIRMATION_PHRASE = "PLACE REAL ORDER"
DEFAULT_MAX_ORDER_NOTIONAL = Decimal("50")
DEFAULT_TIMEOUT_S = 15
_USER_AGENT = "foresea-trading/0.1"
_TRUTHY = {"1", "true", "yes", "y", "on"}


class TradingError(RuntimeError):
    """Base class for guarded trading errors."""


class TradingValidationError(TradingError):
    """The requested order is malformed or violates local guardrails."""


class TradingDisabledError(TradingError):
    """Live execution is disabled by server configuration."""


class TradingNotConfiguredError(TradingError):
    """The selected venue is missing server-side credentials or SDK support."""


class TradingExecutionError(TradingError):
    """An exchange request failed after passing local validation."""


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def _env_decimal(name: str, default: Decimal) -> Decimal:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = Decimal(str(raw))
    except InvalidOperation as exc:
        raise TradingValidationError(f"{name} must be numeric.") from exc
    if value <= 0:
        raise TradingValidationError(f"{name} must be greater than 0.")
    return value


def _as_decimal(name: str, value: Any, *, required: bool = True) -> Optional[Decimal]:
    if value is None or value == "":
        if required:
            raise TradingValidationError(f"{name} is required.")
        return None
    try:
        dec = Decimal(str(value))
    except InvalidOperation as exc:
        raise TradingValidationError(f"{name} must be numeric.") from exc
    if not dec.is_finite():
        raise TradingValidationError(f"{name} must be finite.")
    return dec


def _normalize_price(value: Any) -> Decimal:
    price = _as_decimal("price", value)
    assert price is not None
    if price <= 0 or price >= 1:
        raise TradingValidationError("price must be greater than 0 and less than 1.")
    return price


def _normalize_quantity(value: Any) -> Decimal:
    quantity = _as_decimal("quantity", value)
    assert quantity is not None
    if quantity <= 0:
        raise TradingValidationError("quantity must be greater than 0.")
    return quantity


def _format_fixed(value: Decimal, places: int) -> str:
    quantum = Decimal("1").scaleb(-places)
    return str(value.quantize(quantum, rounding=ROUND_HALF_UP))


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _clean_platform(value: Any) -> str:
    platform = str(value or "").strip().lower()
    if platform in {"poly", "polymarket"}:
        return "polymarket"
    if platform == "kalshi":
        return "kalshi"
    raise TradingValidationError("platform must be 'polymarket' or 'kalshi'.")


def _clean_action(value: Any) -> str:
    action = str(value or "buy").strip().lower()
    if action not in {"buy", "sell"}:
        raise TradingValidationError("action must be 'buy' or 'sell'.")
    return action


def _clean_outcome(value: Any) -> str:
    outcome = str(value or "yes").strip().lower()
    if outcome not in {"yes", "no"}:
        raise TradingValidationError("outcome must be 'yes' or 'no'.")
    return outcome


def _clean_order_type(value: Any) -> str:
    order_type = str(value or "limit").strip().lower()
    if order_type not in {"limit", "market"}:
        raise TradingValidationError("order_type must be 'limit' or 'market'.")
    return order_type


def _max_order_notional() -> Decimal:
    return _env_decimal("FORESEA_MAX_ORDER_NOTIONAL", DEFAULT_MAX_ORDER_NOTIONAL)


def _polymarket_sdk_available() -> bool:
    try:
        import py_clob_client_v2  # noqa: F401
    except Exception:
        return False
    return True


def _secret_present(name: str) -> bool:
    return bool((os.environ.get(name) or "").strip())


def _kalshi_private_key_present() -> bool:
    return _secret_present("KALSHI_PRIVATE_KEY") or _secret_present("KALSHI_PRIVATE_KEY_FILE")


def _polymarket_configured() -> bool:
    return all(
        _secret_present(name)
        for name in (
            "POLYMARKET_PRIVATE_KEY",
            "POLYMARKET_API_KEY",
            "POLYMARKET_API_SECRET",
            "POLYMARKET_API_PASSPHRASE",
        )
    )


def account_status() -> Dict[str, Any]:
    """Return server-side trading readiness without exposing secret values."""
    kalshi_base = os.environ.get(
        "KALSHI_BASE_URL", "https://external-api.kalshi.com/trade-api/v2"
    ).rstrip("/")
    poly_host = os.environ.get("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com").rstrip("/")
    kalshi_key = _secret_present("KALSHI_API_KEY_ID") or _secret_present("KALSHI_ACCESS_KEY_ID")
    kalshi_key_ready = kalshi_key and _kalshi_private_key_present()
    poly_sdk = _polymarket_sdk_available()
    poly_creds = _polymarket_configured()
    return {
        "trading_enabled": _env_bool("FORESEA_ENABLE_TRADING", False),
        "max_order_notional": _money(_max_order_notional()),
        "allow_market_orders": _env_bool("FORESEA_ALLOW_MARKET_ORDERS", False),
        "confirmation_phrase": CONFIRMATION_PHRASE,
        "credential_source": "server_environment",
        "venues": {
            "kalshi": {
                "configured": kalshi_key_ready,
                "base_url": kalshi_base,
                "api_key_id_present": kalshi_key,
                "private_key_present": _kalshi_private_key_present(),
            },
            "polymarket": {
                "configured": poly_creds and poly_sdk,
                "host": poly_host,
                "sdk_available": poly_sdk,
                "private_key_present": _secret_present("POLYMARKET_PRIVATE_KEY"),
                "api_credentials_present": all(
                    _secret_present(name)
                    for name in (
                        "POLYMARKET_API_KEY",
                        "POLYMARKET_API_SECRET",
                        "POLYMARKET_API_PASSPHRASE",
                    )
                ),
            },
        },
    }


def _json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except ValueError:
            return []
    return []


def _resolve_polymarket_token_id(
    *, slug: Optional[str], market_id: Optional[str], outcome: str
) -> Optional[str]:
    if not slug and not market_id:
        return None
    import requests

    params: Dict[str, str] = {}
    if slug:
        params["slug"] = slug
    if market_id:
        params["id"] = market_id
    try:
        resp = requests.get(
            POLYMARKET_GAMMA_URL,
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=DEFAULT_TIMEOUT_S,
        )
    except Exception as exc:
        raise TradingExecutionError(f"Could not resolve Polymarket token id: {exc}") from exc
    if resp.status_code != 200:
        raise TradingExecutionError(
            f"Could not resolve Polymarket token id: provider returned {resp.status_code}."
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise TradingExecutionError("Could not resolve Polymarket token id: invalid JSON.") from exc
    market = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
    if not market:
        raise TradingValidationError("Polymarket market was not found.")
    labels = [str(x).strip().lower() for x in _json_list(market.get("outcomes"))]
    token_ids = [str(x).strip() for x in _json_list(market.get("clobTokenIds"))]
    if not labels or not token_ids:
        raise TradingValidationError(
            "Polymarket market does not expose CLOB token ids; pass token_id explicitly."
        )
    try:
        idx = labels.index(outcome)
    except ValueError as exc:
        raise TradingValidationError(f"Outcome '{outcome}' not found on Polymarket market.") from exc
    if idx >= len(token_ids) or not token_ids[idx]:
        raise TradingValidationError(f"No token_id found for Polymarket outcome '{outcome}'.")
    return token_ids[idx]


def _guard_notional(action: str, price: Decimal, quantity: Decimal) -> Decimal:
    if action == "buy":
        return price * quantity
    # A sell can be a risk-reducing close or a synthetic opposite exposure. Since
    # this layer does not inspect positions, cap by worst side to avoid
    # underestimating order risk.
    return max(price, Decimal("1") - price) * quantity


def _kalshi_side(action: str, outcome: str) -> tuple[str, str]:
    # Kalshi V2 collapses legacy action/side into directional book_side:
    # buy-yes and sell-no -> bid/yes; buy-no and sell-yes -> ask/no.
    if (action == "buy" and outcome == "yes") or (action == "sell" and outcome == "no"):
        return "bid", "yes"
    return "ask", "no"


def _preview_kalshi(req: Mapping[str, Any], action: str, outcome: str, order_type: str) -> Dict[str, Any]:
    ticker = str(req.get("ticker") or "").strip().upper()
    if not ticker:
        raise TradingValidationError("ticker is required for Kalshi orders.")
    price = _normalize_price(req.get("price"))
    quantity = _normalize_quantity(req.get("quantity"))
    guard_notional = _guard_notional(action, price, quantity)
    max_notional = _max_order_notional()
    if guard_notional > max_notional:
        raise TradingValidationError(
            f"Order notional ${_money(guard_notional):.2f} exceeds FORESEA_MAX_ORDER_NOTIONAL "
            f"${_money(max_notional):.2f}."
        )
    side, outcome_side = _kalshi_side(action, outcome)
    time_in_force = str(
        req.get("time_in_force")
        or ("immediate_or_cancel" if order_type == "market" else "good_till_canceled")
    ).strip().lower()
    allowed_tif = {"good_till_canceled", "immediate_or_cancel", "fill_or_kill"}
    if time_in_force not in allowed_tif:
        raise TradingValidationError(
            "time_in_force must be one of: good_till_canceled, immediate_or_cancel, fill_or_kill."
        )
    client_order_id = str(req.get("client_order_id") or f"foresea-{uuid.uuid4()}").strip()
    if len(client_order_id) > 128:
        raise TradingValidationError("client_order_id must be at most 128 characters.")
    payload: Dict[str, Any] = {
        "ticker": ticker,
        "client_order_id": client_order_id,
        "side": side,
        "count": _format_fixed(quantity, 2),
        "price": _format_fixed(price, 4),
        "time_in_force": time_in_force,
        "post_only": bool(req.get("post_only", False)),
        "cancel_order_on_pause": bool(req.get("cancel_order_on_pause", False)),
        "reduce_only": bool(req.get("reduce_only", False)),
        "exchange_index": int(req.get("exchange_index") or 0),
    }
    if req.get("subaccount") is not None:
        payload["subaccount"] = int(req["subaccount"])
    return {
        "platform": "kalshi",
        "action": action,
        "outcome": outcome,
        "outcome_side": outcome_side,
        "order_type": order_type,
        "price": float(price),
        "quantity": float(quantity),
        "estimated_notional": _money(guard_notional),
        "max_order_notional": _money(max_notional),
        "exchange_order": payload,
        "exchange_path": "/portfolio/events/orders",
    }


def _preview_polymarket(
    req: Mapping[str, Any], action: str, outcome: str, order_type: str
) -> Dict[str, Any]:
    token_id = str(req.get("token_id") or "").strip()
    slug = str(req.get("slug") or "").strip() or None
    market_id = str(req.get("market_id") or "").strip() or None
    if not token_id:
        token_id = _resolve_polymarket_token_id(slug=slug, market_id=market_id, outcome=outcome) or ""
    if not token_id:
        raise TradingValidationError(
            "token_id is required for Polymarket orders, or provide slug/market_id plus outcome."
        )
    price = _normalize_price(req.get("price"))
    quantity = _normalize_quantity(req.get("quantity"))
    guard_notional = _guard_notional(action, price, quantity)
    max_notional = _max_order_notional()
    if guard_notional > max_notional:
        raise TradingValidationError(
            f"Order notional ${_money(guard_notional):.2f} exceeds FORESEA_MAX_ORDER_NOTIONAL "
            f"${_money(max_notional):.2f}."
        )
    order_type_name = str(req.get("time_in_force") or ("FOK" if order_type == "market" else "GTC")).strip().upper()
    allowed = {"GTC", "GTD"} if order_type == "limit" else {"FOK", "FAK"}
    if order_type_name not in allowed:
        raise TradingValidationError(
            f"Polymarket {order_type} orders require time_in_force in {sorted(allowed)}."
        )
    tick_size = str(req.get("tick_size") or "0.01").strip()
    if tick_size not in {"0.1", "0.01", "0.001", "0.0001"}:
        raise TradingValidationError("tick_size must be one of 0.1, 0.01, 0.001, 0.0001.")
    neg_risk = bool(req.get("neg_risk", False))
    side = "BUY" if action == "buy" else "SELL"
    exchange_order: Dict[str, Any] = {
        "token_id": token_id,
        "side": side,
        "price": float(price),
        "order_type": order_type_name,
        "tick_size": tick_size,
        "neg_risk": neg_risk,
        "post_only": bool(req.get("post_only", False)) if order_type == "limit" else False,
    }
    if order_type == "market":
        max_cost = _as_decimal("max_cost", req.get("max_cost"), required=False)
        amount = max_cost if action == "buy" and max_cost is not None else quantity
        exchange_order["amount"] = float(amount)
        exchange_order["amount_type"] = "usd" if action == "buy" else "shares"
    else:
        exchange_order["size"] = float(quantity)
    return {
        "platform": "polymarket",
        "action": action,
        "outcome": outcome,
        "order_type": order_type,
        "price": float(price),
        "quantity": float(quantity),
        "estimated_notional": _money(guard_notional),
        "max_order_notional": _money(max_notional),
        "exchange_order": exchange_order,
    }


def preview_order(req: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize an order without submitting it."""
    platform = _clean_platform(req.get("platform"))
    action = _clean_action(req.get("action"))
    outcome = _clean_outcome(req.get("outcome"))
    order_type = _clean_order_type(req.get("order_type"))
    normalized = (
        _preview_kalshi(req, action, outcome, order_type)
        if platform == "kalshi"
        else _preview_polymarket(req, action, outcome, order_type)
    )
    warnings = [
        "Live trading is disabled unless FORESEA_ENABLE_TRADING=true.",
        f"Live orders require execute=true and confirmation='{CONFIRMATION_PHRASE}'.",
    ]
    if order_type == "market":
        warnings.append("Market/IOC/FOK-style orders require FORESEA_ALLOW_MARKET_ORDERS=true.")
    if platform == "kalshi":
        warnings.append(
            "Kalshi V2 represents direction as bid/ask; action/outcome is normalized to outcome_side."
        )
    return {
        "ok": True,
        "platform": platform,
        "would_execute": False,
        "requires_confirmation": True,
        "confirmation_phrase": CONFIRMATION_PHRASE,
        "trading_enabled": _env_bool("FORESEA_ENABLE_TRADING", False),
        "max_order_notional": normalized["max_order_notional"],
        "estimated_notional": normalized["estimated_notional"],
        "warnings": warnings,
        "normalized_order": normalized,
    }


def _require_execution_enabled(normalized: Mapping[str, Any]) -> None:
    if not _env_bool("FORESEA_ENABLE_TRADING", False):
        raise TradingDisabledError("Live trading is disabled. Set FORESEA_ENABLE_TRADING=true.")
    if normalized.get("order_type") == "market" and not _env_bool("FORESEA_ALLOW_MARKET_ORDERS", False):
        raise TradingDisabledError(
            "Market orders are disabled. Set FORESEA_ALLOW_MARKET_ORDERS=true to allow them."
        )


def _kalshi_private_key_pem() -> str:
    pem = os.environ.get("KALSHI_PRIVATE_KEY")
    if pem:
        return pem.replace("\\n", "\n")
    file_path = os.environ.get("KALSHI_PRIVATE_KEY_FILE")
    if file_path:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read()
    raise TradingNotConfiguredError("Kalshi private key is not configured.")


def _kalshi_auth_headers(method: str, path: str, *, timestamp_ms: Optional[str] = None) -> Dict[str, str]:
    """Create Kalshi REST auth headers for a path including `/trade-api/v2/...`."""
    key_id = os.environ.get("KALSHI_API_KEY_ID") or os.environ.get("KALSHI_ACCESS_KEY_ID")
    if not key_id:
        raise TradingNotConfiguredError("KALSHI_API_KEY_ID is not configured.")
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except Exception as exc:
        raise TradingNotConfiguredError("Install cryptography to enable Kalshi signing.") from exc

    private_key = serialization.load_pem_private_key(
        _kalshi_private_key_pem().encode("utf-8"),
        password=None,
        backend=default_backend(),
    )
    ts = timestamp_ms or str(int(time.time() * 1000))
    path_without_query = path.split("?", 1)[0]
    message = f"{ts}{method.upper()}{path_without_query}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


def _place_kalshi_order(normalized: Mapping[str, Any]) -> Dict[str, Any]:
    if not account_status()["venues"]["kalshi"]["configured"]:
        raise TradingNotConfiguredError("Kalshi trading credentials are not configured.")
    import requests

    base_url = os.environ.get(
        "KALSHI_BASE_URL", "https://external-api.kalshi.com/trade-api/v2"
    ).rstrip("/")
    endpoint_path = str(normalized["exchange_path"])
    parsed = urlparse(base_url)
    signing_path = f"{parsed.path.rstrip('/')}{endpoint_path}"
    headers = {
        **_kalshi_auth_headers("POST", signing_path),
        "Content-Type": "application/json",
        "User-Agent": _USER_AGENT,
    }
    url = f"{base_url}{endpoint_path}"
    payload = dict(normalized["exchange_order"])
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT_S)
    except Exception as exc:
        raise TradingExecutionError(f"Kalshi order request failed: {exc}") from exc
    try:
        body = resp.json()
    except ValueError:
        body = {"text": resp.text[:1000]}
    if resp.status_code not in (200, 201):
        raise TradingExecutionError(f"Kalshi returned status {resp.status_code}: {body}")
    return {"status_code": resp.status_code, "body": body}


def _polymarket_client():
    if not _polymarket_sdk_available():
        raise TradingNotConfiguredError(
            "py-clob-client-v2 is not installed; install the trading extra to enable Polymarket."
        )
    if not _polymarket_configured():
        raise TradingNotConfiguredError("Polymarket CLOB credentials are not configured.")
    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import ApiCreds

    host = os.environ.get("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com")
    chain_id = int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))
    signature_type_raw = os.environ.get("POLYMARKET_SIGNATURE_TYPE")
    signature_type = int(signature_type_raw) if signature_type_raw not in (None, "") else None
    funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS") or None
    creds = ApiCreds(
        api_key=os.environ["POLYMARKET_API_KEY"],
        api_secret=os.environ["POLYMARKET_API_SECRET"],
        api_passphrase=os.environ["POLYMARKET_API_PASSPHRASE"],
    )
    return ClobClient(
        host=host,
        chain_id=chain_id,
        key=os.environ["POLYMARKET_PRIVATE_KEY"],
        creds=creds,
        signature_type=signature_type,
        funder=funder,
    )


def _place_polymarket_order(normalized: Mapping[str, Any]) -> Dict[str, Any]:
    client = _polymarket_client()
    from py_clob_client_v2.clob_types import (
        MarketOrderArgs,
        OrderArgs,
        OrderType,
        PartialCreateOrderOptions,
    )

    order = dict(normalized["exchange_order"])
    options = PartialCreateOrderOptions(
        tick_size=order["tick_size"],
        neg_risk=bool(order["neg_risk"]),
    )
    order_type = getattr(OrderType, order["order_type"])
    try:
        if normalized["order_type"] == "market":
            args = MarketOrderArgs(
                token_id=order["token_id"],
                amount=float(order["amount"]),
                side=order["side"],
                price=float(order["price"]),
                order_type=order_type,
            )
            result = client.create_and_post_market_order(args, options, order_type)
        else:
            args = OrderArgs(
                token_id=order["token_id"],
                price=float(order["price"]),
                size=float(order["size"]),
                side=order["side"],
            )
            result = client.create_and_post_order(
                args,
                options,
                order_type,
                post_only=bool(order.get("post_only", False)),
            )
    except Exception as exc:
        raise TradingExecutionError(f"Polymarket order request failed: {exc}") from exc
    return {"body": result}


def place_order(req: Mapping[str, Any], *, user_id: str) -> Dict[str, Any]:
    """Submit a live order after preview, server enablement, and human confirmation."""
    if not req.get("execute"):
        raise TradingValidationError("Set execute=true to submit a live order.")
    if str(req.get("confirmation") or "") != CONFIRMATION_PHRASE:
        raise TradingValidationError(f"confirmation must be exactly '{CONFIRMATION_PHRASE}'.")
    preview = preview_order(req)
    normalized = preview["normalized_order"]
    _require_execution_enabled(normalized)
    platform = normalized["platform"]
    venue_response = (
        _place_kalshi_order(normalized) if platform == "kalshi" else _place_polymarket_order(normalized)
    )
    return {
        **preview,
        "would_execute": True,
        "submitted": True,
        "user_id": user_id,
        "venue_response": venue_response,
    }
