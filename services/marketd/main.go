// Command marketd is a small Go ingestion microservice for prediction-market
// data. It concurrently retrieves markets from Polymarket and Kalshi via
// structured API clients, normalizes them onto a venue-agnostic model, and serves
// them over a JSON HTTP API — the Go counterpart to Foresea's Python market_data
// module, built to scale the read-heavy market-ingestion path independently.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// _MAX_BATCH_REFS bounds a /quotes request so one caller can't force unbounded
// concurrent upstream fan-out.
const maxBatchRefs = 50

// kalshiCandleLookback is how far back /quote and /quotes request Kalshi
// candlesticks for -- enough for a same-day price trend without an unbounded
// history fetch.
const kalshiCandleLookback = 24 * time.Hour

type server struct {
	ingestor *Ingestor
	kalshi   *KalshiClient
	poly     *PolymarketClient
	cache    *ttlCache[[]Market]
	log      *slog.Logger
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	httpClient := &http.Client{Timeout: 15 * time.Second}
	api := newAPIClient(httpClient)
	polyClient := NewPolymarketClient(api)
	kalshiClient := NewKalshiClient(api)
	ingestor := NewIngestor(polyClient, kalshiClient)

	srv := &server{
		ingestor: ingestor,
		kalshi:   kalshiClient,
		poly:     polyClient,
		cache:    newTTLCache[[]Market](cacheTTL()),
		log:      log,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", srv.handleIndex) // exact "/" only
	mux.HandleFunc("GET /health", srv.handleHealth)
	mux.HandleFunc("GET /healthz", srv.handleHealth) // alias
	mux.HandleFunc("GET /markets", srv.handleMarkets)
	mux.HandleFunc("GET /quote", srv.handleQuote)
	mux.HandleFunc("GET /quotes", srv.handleQuotes)

	addr := ":" + envOr("PORT", "8090")
	httpServer := &http.Server{
		Addr:              addr,
		Handler:           withLogging(log, mux),
		ReadHeaderTimeout: 5 * time.Second,
	}

	// Graceful shutdown on SIGINT/SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Info("marketd listening", "addr", addr)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdownCtx)
}

// handleMarkets serves GET /markets?q=&category=&limit= — concurrently ingested,
// normalized, ranked by volume.
func (s *server) handleMarkets(w http.ResponseWriter, r *http.Request) {
	q := Query{
		Keyword:  r.URL.Query().Get("q"),
		Category: r.URL.Query().Get("category"),
		Limit:    clampLimit(r.URL.Query().Get("limit")),
	}
	cacheKey := q.Keyword + "|" + q.Category + "|" + strconv.Itoa(q.Limit)
	if cached, ok := s.cache.get(cacheKey); ok {
		writeJSON(w, http.StatusOK, marketsResponse{Markets: cached, Count: len(cached), Cached: true})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 12*time.Second)
	defer cancel()

	markets, errs := s.ingestor.Markets(ctx, q)
	srcErrs := make([]string, 0, len(errs))
	for _, e := range errs {
		srcErrs = append(srcErrs, e.Error())
	}
	// Only cache a fully successful retrieval so a transient venue outage doesn't
	// pin a degraded result.
	if len(errs) == 0 {
		s.cache.set(cacheKey, markets)
	}
	writeJSON(w, http.StatusOK, marketsResponse{
		Markets: markets,
		Count:   len(markets),
		Errors:  srcErrs,
	})
}

// handleQuote serves GET /quote?platform=&ident=&extra= -- depth/candles for one
// market. extra is series_ticker for Kalshi (required; candlesticks need both
// the series and market ticker) or the YES-outcome token_id for Polymarket
// (required; order book / price history are keyed on it, not the slug).
func (s *server) handleQuote(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	q := s.fetchQuote(ctx, r.URL.Query().Get("platform"), r.URL.Query().Get("ident"), r.URL.Query().Get("extra"))
	writeJSON(w, http.StatusOK, q)
}

// handleQuotes serves GET /quotes?ref=platform:ident:extra&ref=... -- the batch
// tool: N markets resolve concurrently, in the time of the slowest one rather
// than the sum, with per-ref fault isolation (one bad ref never fails the batch).
func (s *server) handleQuotes(w http.ResponseWriter, r *http.Request) {
	refs := r.URL.Query()["ref"]
	truncated := false
	if len(refs) > maxBatchRefs {
		refs = refs[:maxBatchRefs]
		truncated = true
	}

	ctx, cancel := context.WithTimeout(r.Context(), 12*time.Second)
	defer cancel()

	quotes := make([]Quote, len(refs))
	var wg sync.WaitGroup
	for i, raw := range refs {
		wg.Add(1)
		go func(i int, raw string) {
			defer wg.Done()
			platform, ident, extra := parseRef(raw)
			quotes[i] = s.fetchQuote(ctx, platform, ident, extra)
		}(i, raw)
	}
	wg.Wait()

	writeJSON(w, http.StatusOK, quotesResponse{Quotes: quotes, Count: len(quotes), Truncated: truncated})
}

// fetchQuote dispatches to the venue-specific depth calls. Kalshi order book
// requires signed requests (KALSHI-ACCESS-KEY/SIGNATURE/TIMESTAMP) which this
// credential-free public service intentionally does not implement -- Kalshi
// quotes here carry candles only.
func (s *server) fetchQuote(ctx context.Context, platform, ident, extra string) Quote {
	q := Quote{Platform: platform, Ident: ident, FetchedAt: time.Now().UTC().Format(time.RFC3339)}
	switch strings.ToLower(strings.TrimSpace(platform)) {
	case "kalshi":
		if extra == "" {
			q.Error = "kalshi requires extra=series_ticker"
			return q
		}
		candles, err := s.kalshi.Candlesticks(ctx, extra, ident, kalshiCandleLookback)
		if err != nil {
			q.Error = err.Error()
			return q
		}
		q.Candles = candles
	case "polymarket":
		if extra == "" {
			q.Error = "polymarket requires extra=token_id"
			return q
		}
		book, bookErr := s.poly.OrderBook(ctx, extra)
		candles, historyErr := s.poly.PriceHistory(ctx, extra)
		if bookErr != nil && historyErr != nil {
			q.Error = bookErr.Error()
			return q
		}
		if bookErr == nil {
			q.OrderBook = book
		}
		if historyErr == nil {
			q.Candles = candles
		}
	default:
		q.Error = "platform must be \"kalshi\" or \"polymarket\""
	}
	return q
}

// parseRef splits "platform:ident:extra" (extra optional). ident/extra may not
// themselves contain ":" -- Kalshi tickers and Polymarket slugs/token ids don't.
func parseRef(raw string) (platform, ident, extra string) {
	parts := strings.SplitN(raw, ":", 3)
	if len(parts) > 0 {
		platform = parts[0]
	}
	if len(parts) > 1 {
		ident = parts[1]
	}
	if len(parts) > 2 {
		extra = parts[2]
	}
	return platform, ident, extra
}

type quotesResponse struct {
	Quotes    []Quote `json:"quotes"`
	Count     int     `json:"count"`
	Truncated bool    `json:"truncated,omitempty"`
}

func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// handleIndex serves a small human-friendly landing page at "/", so opening the
// service URL in a browser shows what it is and links to live examples instead
// of a bare 404 / raw JSON.
func (s *server) handleIndex(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(indexHTML))
}

const indexHTML = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>marketd · Foresea</title>
<style>
  body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:640px;margin:8vh auto;padding:0 20px;color:#1a1a1f}
  h1{font-size:22px;margin:0 0 4px} .tag{color:#6b7280;margin:0 0 24px}
  code{background:#f3f4f6;padding:2px 6px;border-radius:6px;font-size:13px}
  a{color:#2563eb;text-decoration:none} a:hover{text-decoration:underline}
  ul{padding-left:18px} li{margin:6px 0} .foot{color:#9ca3af;font-size:12px;margin-top:28px}
</style></head><body>
<h1>marketd</h1>
<p class="tag">Go market-data ingestion microservice — part of Foresea's forecasting stack.</p>
<p>Concurrently retrieves prediction markets from <b>Polymarket</b> and <b>Kalshi</b>,
normalizes them onto one model, and serves them as JSON. This is an API — the
endpoints below return JSON.</p>
<h3>Try it</h3>
<ul>
  <li><a href="/markets?limit=10">/markets?limit=10</a> — trending, ranked by volume</li>
  <li><a href="/markets?category=Crypto&limit=10">/markets?category=Crypto&amp;limit=10</a> — by category</li>
  <li><a href="/markets?q=election&limit=10">/markets?q=election&amp;limit=10</a> — keyword search</li>
  <li><code>/quote?platform=&amp;ident=&amp;extra=</code> — depth/candles for one market (extra = Kalshi series_ticker or Polymarket token_id, both from a Market's SeriesTicker/TokenID field)</li>
  <li><code>/quotes?ref=platform:ident:extra&amp;ref=...</code> — the same, batched (up to 50 refs, concurrent, per-ref fault isolation)</li>
  <li><a href="/health">/health</a> — liveness</li>
</ul>
<p class="foot">Stateless · concurrent (goroutines + context) · per-source fault isolation · TTL-cached.</p>
</body></html>`

type marketsResponse struct {
	Markets []Market `json:"markets"`
	Count   int      `json:"count"`
	Errors  []string `json:"errors,omitempty"`
	Cached  bool     `json:"cached,omitempty"`
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// withLogging emits one structured access log line per request with latency.
func withLogging(log *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"query", r.URL.RawQuery,
			"dur_ms", time.Since(start).Milliseconds(),
		)
	})
}

func clampLimit(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil || n <= 0 {
		return 24
	}
	if n > 100 {
		return 100
	}
	return n
}

func cacheTTL() time.Duration {
	if v := os.Getenv("CACHE_TTL_SECONDS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return time.Duration(n) * time.Second
		}
	}
	return 60 * time.Second
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
