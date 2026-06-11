package main

import (
	"context"
	"testing"
	"time"
)

func TestNormalizePolymarketBinary(t *testing.T) {
	g := &gammaMarket{
		Slug:          "fed-cut-july",
		Question:      "Will the Fed cut rates in July?",
		Outcomes:      `["Yes", "No"]`,
		OutcomePrices: `["0.63", "0.37"]`,
		Volume24hr:    1234.5,
	}
	m, ok := normalizePolymarket(g)
	if !ok {
		t.Fatal("expected a normalized market")
	}
	if m.Platform != "Polymarket" || m.Ident != "fed-cut-july" {
		t.Fatalf("unexpected identity: %+v", m)
	}
	if m.Probability == nil || *m.Probability != 0.63 {
		t.Fatalf("probability = %v, want 0.63", m.Probability)
	}
	if m.YesCents == nil || *m.YesCents != 63 || *m.NoCents != 37 {
		t.Fatalf("yes/no cents = %v/%v, want 63/37", m.YesCents, m.NoCents)
	}
	if m.MarketURL != "https://polymarket.com/market/fed-cut-july" {
		t.Fatalf("market url = %q", m.MarketURL)
	}
	if m.Category != "Economics" { // "fed" keyword
		t.Fatalf("category = %q, want Economics", m.Category)
	}
}

func TestNormalizePolymarketSkipsNonBinary(t *testing.T) {
	g := &gammaMarket{
		Slug:          "multi",
		Question:      "Who wins?",
		Outcomes:      `["A", "B", "C"]`,
		OutcomePrices: `["0.5", "0.3", "0.2"]`,
	}
	if _, ok := normalizePolymarket(g); ok {
		t.Fatal("expected non-binary market to be skipped")
	}
}

func TestKalshiSeriesURL(t *testing.T) {
	ev := &kalshiEvent{SeriesTicker: ""}
	m := &kalshiMarket{Ticker: "KXFOO-30JAN01-T5", EventTicker: "KXFOO-30JAN01"}
	if got := kalshiMarketURL(ev, m); got != "https://kalshi.com/markets/kxfoo" {
		t.Fatalf("series url = %q, want .../kxfoo", got)
	}
}

func TestCategoryFor(t *testing.T) {
	cases := map[string]string{
		"Will Bitcoin hit $150k?":           "Crypto",
		"Will the Lakers win the NBA title": "Sports",
		"Random unmatched question":         "Other",
	}
	for q, want := range cases {
		if got := categoryFor(q, ""); got != want {
			t.Errorf("categoryFor(%q) = %q, want %q", q, got, want)
		}
	}
	if got := categoryFor("Unmatched", "Companies"); got != "Companies" {
		t.Errorf("raw fallback = %q, want Companies", got)
	}
}

// stubFetcher lets us exercise the concurrent ingestor without network.
type stubFetcher struct {
	name    string
	markets []Market
	delay   time.Duration
	err     error
}

func (s stubFetcher) Name() string { return s.name }
func (s stubFetcher) Fetch(ctx context.Context, _ Query) ([]Market, error) {
	if s.delay > 0 {
		select {
		case <-time.After(s.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return s.markets, s.err
}

func f(v float64) *float64 { return &v }

func TestIngestorMergesConcurrentlyAndRanksByVolume(t *testing.T) {
	a := stubFetcher{name: "a", delay: 30 * time.Millisecond, markets: []Market{{Question: "low", Volume: f(10)}}}
	b := stubFetcher{name: "b", delay: 30 * time.Millisecond, markets: []Market{{Question: "high", Volume: f(99)}}}
	ig := NewIngestor(a, b)

	start := time.Now()
	got, errs := ig.Markets(context.Background(), Query{Limit: 10})
	elapsed := time.Since(start)

	if len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(got) != 2 {
		t.Fatalf("got %d markets, want 2", len(got))
	}
	if got[0].Question != "high" {
		t.Fatalf("expected volume-ranked first = high, got %q", got[0].Question)
	}
	// Concurrency: two 30ms sources should finish well under 60ms serial.
	if elapsed > 55*time.Millisecond {
		t.Fatalf("fetch took %v, expected concurrent (<55ms)", elapsed)
	}
}

func TestIngestorDegradesOnSourceError(t *testing.T) {
	ok := stubFetcher{name: "ok", markets: []Market{{Question: "kept", Volume: f(1)}}}
	bad := stubFetcher{name: "bad", err: context.DeadlineExceeded}
	ig := NewIngestor(ok, bad)

	got, errs := ig.Markets(context.Background(), Query{Limit: 10})
	if len(got) != 1 || got[0].Question != "kept" {
		t.Fatalf("expected the healthy source to still return, got %+v", got)
	}
	if len(errs) != 1 {
		t.Fatalf("expected 1 collected source error, got %d", len(errs))
	}
}
