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
_KALSHI_HOSTS = {
    "external-api.kalshi.com",
    "api.elections.kalshi.com",
    "external-api.demo.kalshi.co",
    "demo-api.kalshi.co",
}
_POLYMARKET_HOSTS = {"clob.polymarket.com"}
_KALSHI_CONNECTION_KEYS = {
    "kalshi_api_key_id",
    "kalshi_private_key",
    "kalshi_base_url",
}
_POLYMARKET_CONNECTION_KEYS = {
    "polymarket_private_key",
    "polymarket_api_key",
    "polymarket_api_secret",
    "polymarket_api_passphrase",
    "polymarket_clob_host",
    "polymarket_chain_id",
    "polymarket_signature_type",
    "polymarket_funder_address",
}


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


# Per-request "bring your own account" credentials. The web/API caller may pass a
# `creds` mapping (keys below) so a signed-in user trades their OWN venue account;
# these are used transiently for one request and never persisted. When a key is
# absent the resolver falls back to the server-side env var, so the shared server
# account keeps working unchanged.
_CRED_ENV = {
    "kalshi_api_key_id": "KALSHI_API_KEY_ID",
    "kalshi_private_key": "KALSHI_PRIVATE_KEY",
    "kalshi_base_url": "KALSHI_BASE_URL",
    "polymarket_private_key": "POLYMARKET_PRIVATE_KEY",
    "polymarket_api_key": "POLYMARKET_API_KEY",
    "polymarket_api_secret": "POLYMARKET_API_SECRET",
    "polymarket_api_passphrase": "POLYMARKET_API_PASSPHRASE",
    "polymarket_clob_host": "POLYMARKET_CLOB_HOST",
    "polymarket_chain_id": "POLYMARKET_CHAIN_ID",
    "polymarket_signature_type": "POLYMARKET_SIGNATURE_TYPE",
    "polymarket_funder_address": "POLYMARKET_FUNDER_ADDRESS",
}

Creds = Optional[Mapping[str, Any]]


def _cv(creds: Creds, key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a credential: per-request `creds` wins, else the server env var."""
    if creds is not None:
        value = creds.get(key)
        if value not in (None, ""):
            return str(value)
    env_name = _CRED_ENV.get(key)
    if env_name is None:
        return default
    return os.environ.get(env_name, default)


def _is_byo(creds: Creds) -> bool:
    """True when the caller supplied at least one own-account credential field."""
    return bool(creds) and any(str(v or "").strip() for v in creds.values())


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


def _connection_url(value: Any, *, platform: str) -> str:
    """Validate a stored user-supplied venue base URL.

    Account connections must never turn the trading service into a generic HTTP
    client. Only the official production/demo Kalshi endpoints and production
    Polymarket CLOB endpoint are accepted.
    """
    url = str(value or "").strip().rstrip("/")
    if not url:
        return ""
    parsed = urlparse(url)
    allowed = _KALSHI_HOSTS if platform == "kalshi" else _POLYMARKET_HOSTS
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.hostname.lower() not in allowed
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise TradingValidationError(
            f"{platform} connection URL must use an official HTTPS API host."
        )
    if platform == "kalshi" and not parsed.path.rstrip("/").endswith("/trade-api/v2"):
        raise TradingValidationError(
            "Kalshi base URL must end with /trade-api/v2."
        )
    if platform == "polymarket" and parsed.path not in ("", "/"):
        raise TradingValidationError("Polymarket CLOB host cannot include a path.")
    return url


def connection_credentials(platform: Any, creds: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a validated, platform-scoped set of credentials for encrypted storage.

    This deliberately reads the mapping directly rather than through ``_cv``:
    an account connection must be complete on its own and must never silently
    borrow a shared deployment credential.
    """
    venue = _clean_platform(platform)
    allowed = _KALSHI_CONNECTION_KEYS if venue == "kalshi" else _POLYMARKET_CONNECTION_KEYS
    cleaned = {
        key: value
        for key, value in dict(creds or {}).items()
        if key in allowed and value not in (None, "")
    }
    if venue == "kalshi":
        missing = [key for key in ("kalshi_api_key_id", "kalshi_private_key") if not str(cleaned.get(key) or "").strip()]
        if missing:
            raise TradingValidationError("Kalshi API key ID and RSA private key are required.")
        cleaned["kalshi_base_url"] = _connection_url(
            cleaned.get("kalshi_base_url") or "https://external-api.kalshi.com/trade-api/v2",
            platform=venue,
        )
        try:
            from cryptography.hazmat.primitives import serialization

            serialization.load_pem_private_key(
                str(cleaned["kalshi_private_key"]).replace("\\n", "\n").encode("utf-8"),
                password=None,
            )
        except TradingError:
            raise
        except Exception as exc:
            raise TradingValidationError("Kalshi private key must be a valid unencrypted PEM key.") from exc
    else:
        required = (
            "polymarket_private_key",
            "polymarket_api_key",
            "polymarket_api_secret",
            "polymarket_api_passphrase",
        )
        missing = [key for key in required if not str(cleaned.get(key) or "").strip()]
        if missing:
            raise TradingValidationError(
                "Polymarket wallet key, API key, API secret, and API passphrase are required."
            )
        cleaned["polymarket_clob_host"] = _connection_url(
            cleaned.get("polymarket_clob_host") or "https://clob.polymarket.com",
            platform=venue,
        )
        if "polymarket_chain_id" in cleaned:
            try:
                cleaned["polymarket_chain_id"] = int(cleaned["polymarket_chain_id"])
            except (TypeError, ValueError) as exc:
                raise TradingValidationError("Polymarket chain ID must be an integer.") from exc
        if "polymarket_signature_type" in cleaned:
            try:
                cleaned["polymarket_signature_type"] = int(cleaned["polymarket_signature_type"])
            except (TypeError, ValueError) as exc:
                raise TradingValidationError("Polymarket signature type must be an integer.") from exc
    return cleaned


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


def _cred_present(creds: Creds, key: str) -> bool:
    return bool((_cv(creds, key) or "").strip())


def _kalshi_private_key_present(creds: Creds = None) -> bool:
    # KALSHI_PRIVATE_KEY_FILE is an env-only deployment option (no BYO equivalent).
    return _cred_present(creds, "kalshi_private_key") or _secret_present("KALSHI_PRIVATE_KEY_FILE")


def _polymarket_configured(creds: Creds = None) -> bool:
    return all(
        _cred_present(creds, key)
        for key in (
            "polymarket_private_key",
            "polymarket_api_key",
            "polymarket_api_secret",
            "polymarket_api_passphrase",
        )
    )


def account_status(creds: Creds = None) -> Dict[str, Any]:
    """Return trading readiness without exposing secret values.

    When `creds` carries per-request own-account credentials, readiness is computed
    against those (with server env as fallback); otherwise it reflects the shared
    server-side account.
    """
    byo = _is_byo(creds)
    kalshi_base = (
        _cv(creds, "kalshi_base_url", "https://external-api.kalshi.com/trade-api/v2")
    ).rstrip("/")
    poly_host = (
        _cv(creds, "polymarket_clob_host", "https://clob.polymarket.com")
    ).rstrip("/")
    kalshi_key = _cred_present(creds, "kalshi_api_key_id") or _secret_present("KALSHI_ACCESS_KEY_ID")
    kalshi_key_ready = kalshi_key and _kalshi_private_key_present(creds)
    poly_sdk = _polymarket_sdk_available()
    poly_creds = _polymarket_configured(creds)
    return {
        "trading_enabled": _env_bool("FORESEA_ENABLE_TRADING", False),
        "byo_trading_enabled": _env_bool("FORESEA_ENABLE_BYO_TRADING", False),
        "max_order_notional": _money(_max_order_notional()),
        "allow_market_orders": _env_bool("FORESEA_ALLOW_MARKET_ORDERS", False),
        "confirmation_phrase": CONFIRMATION_PHRASE,
        "credential_source": "request" if byo else "server_environment",
        "venues": {
            "kalshi": {
                "configured": kalshi_key_ready,
                "base_url": kalshi_base,
                "api_key_id_present": kalshi_key,
                "private_key_present": _kalshi_private_key_present(creds),
            },
            "polymarket": {
                "configured": poly_creds and poly_sdk,
                "host": poly_host,
                "sdk_available": poly_sdk,
                "private_key_present": _cred_present(creds, "polymarket_private_key"),
                "api_credentials_present": all(
                    _cred_present(creds, key)
                    for key in (
                        "polymarket_api_key",
                        "polymarket_api_secret",
                        "polymarket_api_passphrase",
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


def preview_order(req: Mapping[str, Any], creds: Creds = None) -> Dict[str, Any]:
    """Validate and normalize an order without submitting it."""
    byo = _is_byo(creds)
    platform = _clean_platform(req.get("platform"))
    action = _clean_action(req.get("action"))
    outcome = _clean_outcome(req.get("outcome"))
    order_type = _clean_order_type(req.get("order_type"))
    normalized = (
        _preview_kalshi(req, action, outcome, order_type)
        if platform == "kalshi"
        else _preview_polymarket(req, action, outcome, order_type)
    )
    gate_var = "FORESEA_ENABLE_BYO_TRADING" if byo else "FORESEA_ENABLE_TRADING"
    trading_enabled = _env_bool(gate_var, False)
    warnings = [
        f"Live trading is disabled unless {gate_var}=true.",
        f"Live orders require execute=true and confirmation='{CONFIRMATION_PHRASE}'.",
    ]
    if byo:
        warnings.append("Order signs with the credentials you supplied; they are never stored by Foresea.")
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
        "trading_enabled": trading_enabled,
        "max_order_notional": normalized["max_order_notional"],
        "estimated_notional": normalized["estimated_notional"],
        "warnings": warnings,
        "normalized_order": normalized,
    }


def _require_execution_enabled(normalized: Mapping[str, Any], *, byo: bool = False) -> None:
    gate_var = "FORESEA_ENABLE_BYO_TRADING" if byo else "FORESEA_ENABLE_TRADING"
    if not _env_bool(gate_var, False):
        raise TradingDisabledError(f"Live trading is disabled. Set {gate_var}=true.")
    if normalized.get("order_type") == "market" and not _env_bool("FORESEA_ALLOW_MARKET_ORDERS", False):
        raise TradingDisabledError(
            "Market orders are disabled. Set FORESEA_ALLOW_MARKET_ORDERS=true to allow them."
        )


def _kalshi_private_key_pem(creds: Creds = None) -> str:
    pem = _cv(creds, "kalshi_private_key")
    if pem:
        return pem.replace("\\n", "\n")
    file_path = os.environ.get("KALSHI_PRIVATE_KEY_FILE")
    if file_path:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read()
    raise TradingNotConfiguredError("Kalshi private key is not configured.")


def _kalshi_auth_headers(
    method: str, path: str, *, creds: Creds = None, timestamp_ms: Optional[str] = None
) -> Dict[str, str]:
    """Create Kalshi REST auth headers for a path including `/trade-api/v2/...`."""
    key_id = _cv(creds, "kalshi_api_key_id") or os.environ.get("KALSHI_ACCESS_KEY_ID")
    if not key_id:
        raise TradingNotConfiguredError("KALSHI_API_KEY_ID is not configured.")
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except Exception as exc:
        raise TradingNotConfiguredError("Install cryptography to enable Kalshi signing.") from exc

    private_key = serialization.load_pem_private_key(
        _kalshi_private_key_pem(creds).encode("utf-8"),
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


def _place_kalshi_order(normalized: Mapping[str, Any], creds: Creds = None) -> Dict[str, Any]:
    if not account_status(creds)["venues"]["kalshi"]["configured"]:
        raise TradingNotConfiguredError("Kalshi trading credentials are not configured.")
    import requests

    base_url = _cv(
        creds, "kalshi_base_url", "https://external-api.kalshi.com/trade-api/v2"
    ).rstrip("/")
    endpoint_path = str(normalized["exchange_path"])
    parsed = urlparse(base_url)
    signing_path = f"{parsed.path.rstrip('/')}{endpoint_path}"
    headers = {
        **_kalshi_auth_headers("POST", signing_path, creds=creds),
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


def _polymarket_client(creds: Creds = None):
    if not _polymarket_sdk_available():
        raise TradingNotConfiguredError(
            "py-clob-client-v2 is not installed; install the trading extra to enable Polymarket."
        )
    if not _polymarket_configured(creds):
        raise TradingNotConfiguredError("Polymarket CLOB credentials are not configured.")
    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import ApiCreds

    host = _cv(creds, "polymarket_clob_host", "https://clob.polymarket.com")
    chain_id = int(_cv(creds, "polymarket_chain_id", "137"))
    signature_type_raw = _cv(creds, "polymarket_signature_type")
    signature_type = int(signature_type_raw) if signature_type_raw not in (None, "") else None
    funder = _cv(creds, "polymarket_funder_address") or None
    api_creds = ApiCreds(
        api_key=_cv(creds, "polymarket_api_key"),
        api_secret=_cv(creds, "polymarket_api_secret"),
        api_passphrase=_cv(creds, "polymarket_api_passphrase"),
    )
    return ClobClient(
        host=host,
        chain_id=chain_id,
        key=_cv(creds, "polymarket_private_key"),
        creds=api_creds,
        signature_type=signature_type,
        funder=funder,
    )


def _place_polymarket_order(normalized: Mapping[str, Any], creds: Creds = None) -> Dict[str, Any]:
    client = _polymarket_client(creds)
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


def place_order(req: Mapping[str, Any], *, user_id: str, creds: Creds = None) -> Dict[str, Any]:
    """Submit a live order after preview, server enablement, and human confirmation."""
    if not req.get("execute"):
        raise TradingValidationError("Set execute=true to submit a live order.")
    if str(req.get("confirmation") or "") != CONFIRMATION_PHRASE:
        raise TradingValidationError(f"confirmation must be exactly '{CONFIRMATION_PHRASE}'.")
    byo = _is_byo(creds)
    preview = preview_order(req, creds)
    normalized = preview["normalized_order"]
    _require_execution_enabled(normalized, byo=byo)
    platform = normalized["platform"]
    venue_response = (
        _place_kalshi_order(normalized, creds)
        if platform == "kalshi"
        else _place_polymarket_order(normalized, creds)
    )
    return {
        **preview,
        "would_execute": True,
        "submitted": True,
        "user_id": user_id,
        "venue_response": venue_response,
    }


def _response_json(response: Any, *, operation: str) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = {"text": str(getattr(response, "text", ""))[:1000]}
    if not isinstance(body, dict):
        body = {"data": body}
    if not 200 <= int(response.status_code) < 300:
        raise TradingExecutionError(
            f"{operation} returned status {response.status_code}: {body}"
        )
    return body


def _kalshi_request(
    method: str,
    endpoint_path: str,
    *,
    creds: Creds,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Make a signed Kalshi account request without exposing its auth material."""
    import requests

    base_url = _cv(
        creds, "kalshi_base_url", "https://external-api.kalshi.com/trade-api/v2"
    ).rstrip("/")
    parsed = urlparse(base_url)
    signing_path = f"{parsed.path.rstrip('/')}{endpoint_path}"
    headers = {
        **_kalshi_auth_headers(method, signing_path, creds=creds),
        "User-Agent": _USER_AGENT,
    }
    try:
        response = requests.request(
            method.upper(),
            f"{base_url}{endpoint_path}",
            headers=headers,
            params=dict(params or {}),
            timeout=DEFAULT_TIMEOUT_S,
        )
    except Exception as exc:
        raise TradingExecutionError(f"Kalshi {method.upper()} request failed: {exc}") from exc
    return _response_json(response, operation=f"Kalshi {method.upper()} {endpoint_path}")


def _number(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(Decimal(str(value)))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _venue_order_id(source: Mapping[str, Any]) -> Optional[str]:
    for key in ("order_id", "orderID", "id", "orderId"):
        value = source.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _reconciliation_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"filled", "executed", "matched", "complete", "completed"}:
        return "filled"
    if raw in {"canceled", "cancelled", "cancel"}:
        return "canceled"
    if raw in {"rejected", "failed", "error", "unmatched"}:
        return "rejected"
    if raw in {"resting", "live", "open", "active", "pending", "delayed"}:
        return "open"
    return "submitted"


def _normalise_reconciled_order(platform: str, source: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate a venue order response into safe, display-ready audit fields."""
    raw_status = source.get("status") or source.get("order_status")
    if platform == "kalshi":
        filled = _number(source.get("fill_count_fp", source.get("fill_count")))
        remaining = _number(source.get("remaining_count_fp", source.get("remaining_count")))
        quantity = _number(source.get("initial_count_fp", source.get("initial_count")))
        price = _number(source.get("yes_price_dollars", source.get("yes_price")))
        ticker = source.get("ticker")
        token_id = None
    else:
        filled = _number(source.get("size_matched", source.get("matched_size")))
        remaining = _number(source.get("size_remaining", source.get("remaining_size")))
        quantity = _number(source.get("original_size", source.get("size")))
        if quantity is not None and remaining is None and filled is not None:
            remaining = max(0.0, quantity - filled)
        price = _number(source.get("price", source.get("average_price")))
        ticker = None
        token_id = source.get("asset_id", source.get("token_id"))
    return {
        "venue_order_id": _venue_order_id(source),
        "status": _reconciliation_status(raw_status),
        "venue_status": str(raw_status or "submitted"),
        "filled_quantity": filled,
        "remaining_quantity": remaining,
        "quantity": quantity,
        "price": price,
        "ticker": str(ticker) if ticker not in (None, "") else None,
        "token_id": str(token_id) if token_id not in (None, "") else None,
        "updated_at": source.get("last_update_time") or source.get("updated_at") or source.get("created_time"),
    }


def _kalshi_portfolio(creds: Creds, *, limit: int) -> Dict[str, Any]:
    balance = _kalshi_request("GET", "/portfolio/balance", creds=creds)
    positions_payload = _kalshi_request(
        "GET", "/portfolio/positions", creds=creds, params={"limit": limit}
    )
    orders_payload = _kalshi_request(
        "GET", "/portfolio/orders", creds=creds, params={"limit": limit}
    )
    fills_payload = _kalshi_request(
        "GET", "/portfolio/fills", creds=creds, params={"limit": limit}
    )
    positions = []
    for item in positions_payload.get("market_positions") or []:
        if not isinstance(item, Mapping):
            continue
        positions.append(
            {
                "ticker": item.get("ticker"),
                "quantity": _number(item.get("position_fp", item.get("position"))),
                "exposure": _number(item.get("market_exposure_dollars")),
                "realized_pnl": _number(item.get("realized_pnl_dollars")),
                "fees": _number(item.get("fees_paid_dollars")),
                "resting_orders": item.get("resting_orders_count"),
                "updated_at": item.get("last_updated_ts"),
            }
        )
    orders = [
        _normalise_reconciled_order("kalshi", item)
        for item in orders_payload.get("orders") or []
        if isinstance(item, Mapping)
    ]
    fills = []
    for item in fills_payload.get("fills") or []:
        if not isinstance(item, Mapping):
            continue
        fills.append(
            {
                "trade_id": item.get("trade_id"),
                "order_id": item.get("order_id"),
                "ticker": item.get("ticker"),
                "quantity": _number(item.get("count_fp", item.get("count"))),
                "price": _number(item.get("yes_price_dollars", item.get("yes_price"))),
                "fee": _number(item.get("fee_cost_dollars", item.get("fee_cost"))),
                "created_at": item.get("created_time") or item.get("created_ts"),
            }
        )
    return {
        "platform": "kalshi",
        "balance": {
            "available": _number(balance.get("balance")),
            "portfolio_value": _number(balance.get("portfolio_value")),
            "unit": "cents",
            "updated_at": balance.get("updated_ts"),
        },
        "positions": positions,
        "orders": orders,
        "fills": fills,
    }


def _polymarket_account_address(client: Any, creds: Creds) -> str:
    funder = _cv(creds, "polymarket_funder_address")
    if funder:
        return funder
    try:
        return str(client.get_address())
    except Exception as exc:
        raise TradingExecutionError("Could not resolve the Polymarket account address.") from exc


def _polymarket_portfolio(creds: Creds, *, limit: int) -> Dict[str, Any]:
    client = _polymarket_client(creds)
    try:
        allowance = client.get_balance_allowance()
        orders = client.get_open_orders(only_first_page=True)
        trades = client.get_trades(only_first_page=True)
        address = _polymarket_account_address(client, creds)
    except Exception as exc:
        raise TradingExecutionError(f"Polymarket account reconciliation failed: {exc}") from exc

    import requests

    try:
        positions_response = requests.get(
            "https://data-api.polymarket.com/positions",
            params={"user": address, "limit": limit, "sizeThreshold": 0},
            timeout=DEFAULT_TIMEOUT_S,
            headers={"User-Agent": _USER_AGENT},
        )
        positions_payload = _response_json(
            positions_response, operation="Polymarket position lookup"
        )
    except TradingExecutionError:
        raise
    except Exception as exc:
        raise TradingExecutionError(f"Polymarket position lookup failed: {exc}") from exc

    # The Data API returns a list; keep only display-safe position fields.
    positions_source = positions_payload.get("data")
    if not isinstance(positions_source, list):
        positions_source = []
    positions = [
        {
            "token_id": item.get("asset"),
            "outcome": item.get("outcome"),
            "quantity": _number(item.get("size")),
            "average_price": _number(item.get("avgPrice")),
            "current_price": _number(item.get("curPrice")),
            "current_value": _number(item.get("currentValue")),
            "cash_pnl": _number(item.get("cashPnl")),
            "realized_pnl": _number(item.get("realizedPnl")),
            "title": item.get("title"),
            "slug": item.get("slug"),
        }
        for item in positions_source
        if isinstance(item, Mapping)
    ]
    normalised_orders = [
        _normalise_reconciled_order("polymarket", item)
        for item in (orders or [])
        if isinstance(item, Mapping)
    ]
    fills = [
        {
            "trade_id": item.get("id"),
            "order_id": item.get("order_id", item.get("taker_order_id")),
            "token_id": item.get("asset_id"),
            "quantity": _number(item.get("size")),
            "price": _number(item.get("price")),
            "created_at": item.get("match_time") or item.get("created_at"),
        }
        for item in (trades or [])[:limit]
        if isinstance(item, Mapping)
    ]
    return {
        "platform": "polymarket",
        "balance": {
            "available": _number((allowance or {}).get("balance")),
            "allowance": _number((allowance or {}).get("allowance")),
            "unit": "USDC",
        },
        "positions": positions,
        "orders": normalised_orders,
        "fills": fills,
    }


def reconcile_portfolio(platform: Any, creds: Mapping[str, Any], *, limit: int = 100) -> Dict[str, Any]:
    """Fetch a current portfolio, open-order, and fill snapshot from one venue."""
    venue = _clean_platform(platform)
    secure_creds = connection_credentials(venue, creds)
    bounded_limit = max(1, min(int(limit), 100))
    return (
        _kalshi_portfolio(secure_creds, limit=bounded_limit)
        if venue == "kalshi"
        else _polymarket_portfolio(secure_creds, limit=bounded_limit)
    )


def reconcile_order(platform: Any, venue_order_id: str, creds: Mapping[str, Any]) -> Dict[str, Any]:
    """Fetch the live venue state of a specific submitted order."""
    venue = _clean_platform(platform)
    order_id = str(venue_order_id or "").strip()
    if not order_id:
        raise TradingValidationError("venue_order_id is required for reconciliation.")
    secure_creds = connection_credentials(venue, creds)
    if venue == "kalshi":
        source = _kalshi_request(
            "GET", f"/portfolio/orders/{order_id}", creds=secure_creds
        )
        source = source.get("order") if isinstance(source.get("order"), Mapping) else source
    else:
        try:
            source = _polymarket_client(secure_creds).get_order(order_id)
        except Exception as exc:
            raise TradingExecutionError(f"Polymarket order reconciliation failed: {exc}") from exc
    if not isinstance(source, Mapping):
        raise TradingExecutionError("Venue returned an invalid order-reconciliation response.")
    result = _normalise_reconciled_order(venue, source)
    result["venue_order_id"] = result.get("venue_order_id") or order_id
    return result


def cancel_order(
    platform: Any,
    venue_order_id: str,
    creds: Mapping[str, Any],
    *,
    subaccount: Optional[int] = None,
    exchange_index: int = 0,
) -> Dict[str, Any]:
    """Cancel the remaining quantity of a submitted venue order."""
    venue = _clean_platform(platform)
    order_id = str(venue_order_id or "").strip()
    if not order_id:
        raise TradingValidationError("venue_order_id is required to cancel an order.")
    secure_creds = connection_credentials(venue, creds)
    if venue == "kalshi":
        params: Dict[str, Any] = {"exchange_index": int(exchange_index or 0)}
        if subaccount is not None:
            params["subaccount"] = int(subaccount)
        source = _kalshi_request(
            "DELETE",
            f"/portfolio/events/orders/{order_id}",
            creds=secure_creds,
            params=params,
        )
        return {
            "venue_order_id": str(source.get("order_id") or order_id),
            "status": "canceled",
            "venue_status": "canceled",
            "remaining_quantity": 0.0,
            "canceled_quantity": _number(source.get("reduced_by")),
            "updated_at": source.get("ts_ms"),
        }
    try:
        from py_clob_client_v2.clob_types import OrderPayload

        source = _polymarket_client(secure_creds).cancel_order(OrderPayload(orderID=order_id))
    except Exception as exc:
        raise TradingExecutionError(f"Polymarket cancellation failed: {exc}") from exc
    canceled = source.get("canceled") if isinstance(source, Mapping) else []
    if order_id not in (canceled or []):
        detail = source.get("not_canceled") if isinstance(source, Mapping) else source
        raise TradingExecutionError(f"Polymarket did not cancel order {order_id}: {detail}")
    return {
        "venue_order_id": order_id,
        "status": "canceled",
        "venue_status": "canceled",
        "remaining_quantity": 0.0,
    }
