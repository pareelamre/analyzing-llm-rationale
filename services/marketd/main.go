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
	"syscall"
	"time"
)

type server struct {
	ingestor *Ingestor
	cache    *ttlCache[[]Market]
	log      *slog.Logger
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	httpClient := &http.Client{Timeout: 15 * time.Second}
	api := newAPIClient(httpClient)
	ingestor := NewIngestor(
		NewPolymarketClient(api),
		NewKalshiClient(api),
	)

	srv := &server{
		ingestor: ingestor,
		cache:    newTTLCache[[]Market](cacheTTL()),
		log:      log,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", srv.handleHealth)
	mux.HandleFunc("GET /markets", srv.handleMarkets)

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

func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

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
