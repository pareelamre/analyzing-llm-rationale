# marketd — Go market-data ingestion microservice

A small, dependency-free Go microservice that **concurrently retrieves**
prediction-market data from external venues (Polymarket, Kalshi), **normalizes**
it onto a venue-agnostic model, and **serves** it over a JSON HTTP API.

It is the Go counterpart to Foresea's Python `market_data` module, carved out so
the read-heavy market-ingestion path can scale and be deployed independently of
the forecasting app.

## Design

- **Structured API clients** — one per venue (`PolymarketClient`, `KalshiClient`),
  each owning its JSON shape and mapping it onto the shared `Market` type. Adding
  a venue = implement the `Fetcher` interface.
- **Concurrent fan-out** — `Ingestor.Markets` launches one goroutine per source,
  collects results over a channel with a `sync.WaitGroup`, merges, and ranks by
  24h volume. A failing/slow source **degrades gracefully**: its error is
  collected and the healthy sources still return. Per-request `context` deadlines
  propagate to every outbound call.
- **Normalization** — heterogeneous payloads (Polymarket encodes outcomes/prices
  as JSON-stringified arrays and volume as a number; Kalshi quotes prices as
  `*_dollars` **strings**) are unified to `P(Yes) ∈ [0,1]`, Yes/No cents, a web
  URL (Kalshi rooted on the lowercase **series** ticker), and a browse category
  (keyword-inferred, matching the Python categorizer).
- **TTL cache** — generic, concurrency-safe; caches only fully successful
  retrievals so a transient venue outage can't pin a degraded result.
- **Operability** — structured access logs (`log/slog` JSON), `/healthz`, and
  graceful shutdown on SIGINT/SIGTERM.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/markets?q=&category=&limit=` | Concurrently ingested, normalized markets ranked by volume |

```jsonc
// GET /markets?category=Crypto&limit=2
{
  "markets": [
    { "platform": "Polymarket", "ident": "...", "question": "Will Bitcoin be above $66,000 on June 11?",
      "market_url": "https://polymarket.com/market/...", "category": "Crypto",
      "probability": 0.41, "yes_cents": 41, "no_cents": 59, "volume": 12345.6 }
  ],
  "count": 1
}
```

## Run

```bash
# local
go test ./...
go run .                       # listens on :8090 (override with PORT)
curl 'localhost:8090/markets?category=Sports&limit=5'

# container
docker build -t marketd .
docker run -p 8090:8090 marketd
```

### Config (env)

| Var | Default | Description |
|-----|---------|-------------|
| `PORT` | `8090` | Listen port |
| `CACHE_TTL_SECONDS` | `60` | Per-query cache TTL |

## Layout

```
main.go         HTTP server, routes, logging, graceful shutdown
market.go       Market model, Fetcher interface, concurrent Ingestor, categorizer
httpclient.go   shared JSON-over-HTTP client (timeouts, body cap)
polymarket.go   Polymarket Gamma client + normalization
kalshi.go       Kalshi trade-API client + normalization
cache.go        generic TTL cache
*_test.go       normalization + concurrency/merge tests
```
