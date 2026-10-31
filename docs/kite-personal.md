# Zerodha Kite Personal

Foresea supports private, read-only account access with Kite Personal in
**Settings → Exchange → Zerodha Kite**. This supports the free Personal tier;
live market quotes, historical candles and streaming require the paid Connect
plan and are outside this integration. See [official pricing](https://zerodha.com/products/api/).

## Connect

1. Create a Personal app in the [Kite developer console](https://developers.kite.trade/).
2. Generate a daily access token using the [official login flow](https://kite.trade/docs/connect/v3/user/).
   Open `https://kite.zerodha.com/connect/login?v=3&api_key=YOUR_API_KEY`, sign in,
   and exchange the resulting `request_token` **locally** for an `access_token`.
   The token exchange sends `api_key`, `request_token`, and
   `SHA256(api_key + request_token + api_secret)` to `POST https://api.kite.trade/session/token`.
   Use the official SDK or a local script; keep the API secret off Foresea and chat.
3. Sign in to Foresea and enter only the API key and access token in Settings.
4. Choose holdings, positions, funds or today's orders, then refresh.

Kite access tokens expire at **6 AM IST the next day** and can be invalidated
earlier through Kite logout. Reconnect with a new token after expiry. “Session
stored” means encrypted credentials exist, not that the daily session is valid.
Disconnect deletes the Foresea credential record; it does not revoke the session
at Zerodha. Logout at Kite if revocation is needed.
Kite tokens can authorize trades; “read-only” describes Foresea's operations,
not a restricted permission scope on the token.

## Private API

All operations require a Foresea session JWT in `Authorization: Bearer …` and are
rate limited. The API key/token are not accepted through query parameters.

| Foresea route | Behavior / upstream GET |
|---|---|
| `GET /integrations/kite` | Safe connection metadata, encryption readiness |
| `PUT /integrations/kite` | JSON `{ "api_key": "…", "access_token": "…" }`; validates `/user/profile`, encrypts and stores |
| `DELETE /integrations/kite` | Deletes only this signed-in user's stored Kite connection |
| `GET /integrations/kite/holdings` | `/portfolio/holdings` |
| `GET /integrations/kite/positions` | `/portfolio/positions` (net and day) |
| `GET /integrations/kite/funds` | `/user/margins` (equity and commodity segments) |
| `GET /integrations/kite/orders` | `/orders` (open and executed orders for today) |

Reads return `platform`, `resource`, `read_only`, `fetched_at` and `data`.
Values retain Kite's field names, quantities and INR amounts. `409` means connect
or renew the Kite session; `429` means wait; `502` means an upstream failure;
`503` means connection storage is unavailable. A Kite error does not invalidate
the user's Foresea session. Successful responses use `Cache-Control: no-store`.

## Deployment prerequisites and boundaries

Disabled by default. Set `FORESEA_ENABLE_KITE_PERSONAL=true` only after the
operator has completed the deployment prerequisites below. No production rollout
or platform approval is implied by this change.

Uses the existing `FORESEA_TRADING_KMS_KEY_NAME`, Cloud KMS IAM and durable
Datastore backend. Saving fails closed without secure, durable storage; there is
no shared server-account credential fallback. KMS envelopes bind credentials to
the signed-in Foresea user and the `kite` namespace. The API secret is never
accepted or stored. Account data is fetched on demand and is not stored in
browser storage, sent into chat, published through MCP or added to shared data.
One private `KiteConnectionIntent/current` entity under each user serializes
credential mutations across replicas. Disconnect replaces that intent and removes
the encrypted credential in one transaction, so an older connect cannot restore it.
The intent record contains a random generation marker, never account data or tokens.

Before offering this integration to third-party customers, the operator must
confirm the applicable approvals and privacy requirements with Zerodha under
[Kite's terms, sections 2, 3 and 4](https://kite.trade/terms/). This implementation
does not establish platform approval. Zerodha's [static-IP guidance](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/static-ip)
distinguishes order placement from read APIs, which remain accessible from any IP.
The API emits account-access log events containing only the internal Foresea
user ID, action and outcome. Under [section 5 of Kite's terms](https://kite.trade/terms/),
the operator must configure the required audit-log retention (at least five years,
or the applicable legal period) before enabling access; this PR does not change
cloud logging retention or provision an audit archive.

There are no order placement, modification, cancellation, position conversion,
GTT or automated trading operations. The integration does not register Kite as a
prediction-market trading venue. No dependency or cloud provisioning is added.
