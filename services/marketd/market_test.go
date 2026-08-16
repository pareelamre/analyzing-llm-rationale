package main

import (
	"context"
	"encoding/json"
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

func TestNormalizePolymarketSetsYesTokenID(t *testing.T) {
	g := &gammaMarket{
		Slug:          "fed-cut-july",
		Question:      "Will the Fed cut rates in July?",
		Outcomes:      `["Yes", "No"]`,
		OutcomePrices: `["0.63", "0.37"]`,
		ClobTokenIds:  `["111111", "222222"]`,
	}
	m, ok := normalizePolymarket(g)
	if !ok {
		t.Fatal("expected a normalized market")
	}
	if m.TokenID != "111111" {
		t.Fatalf("token id = %q, want the YES-outcome id 111111", m.TokenID)
	}
}

func TestNormalizeKalshiSetsSeriesTicker(t *testing.T) {
	ev := &kalshiEvent{Title: "Fed decision", SeriesTicker: "KXFED", Category: "Economics"}
	m := &kalshiMarket{Ticker: "KXFED-25JUN-H", LastPriceDollars: 0.4}
	got, ok := normalizeKalshi(ev, m)
	if !ok {
		t.Fatal("expected a normalized market")
	}
	if got.SeriesTicker != "KXFED" {
		t.Fatalf("series ticker = %q, want KXFED", got.SeriesTicker)
	}
}

// Response shapes below are taken verbatim from docs.kalshi.com and
// docs.polymarket.com example payloads, to catch a wrong JSON tag (which
// encoding/json fails silently on -- a mismatched field just stays zero-valued)
// rather than only finding it against the real API in production.

func TestNormalizeKalshiCandlesParsesDocExample(t *testing.T) {
	raw := `{
		"ticker": "KXFED-25JUN-H",
		"candlesticks": [
			{
				"end_period_ts": 1750000000,
				"yes_bid": {"open_dollars": "0.10", "low_dollars": "0.10", "high_dollars": "0.12", "close_dollars": "0.11"},
				"yes_ask": {"open_dollars": "0.12", "low_dollars": "0.11", "high_dollars": "0.13", "close_dollars": "0.12"},
				"price": {"open_dollars": "0.11", "low_dollars": "0.10", "high_dollars": "0.13", "close_dollars": "0.115", "mean_dollars": "0.115", "previous_dollars": "0.10", "min_dollars": "0.10", "max_dollars": "0.13"},
				"volume_fp": "1500.00",
				"open_interest_fp": "9000.00"
			}
		]
	}`
	var resp kalshiCandlestickResp
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	candles := normalizeKalshiCandles(resp)
	if len(candles) != 1 {
		t.Fatalf("got %d candles, want 1", len(candles))
	}
	c := candles[0]
	if c.EndTime != time.Unix(1750000000, 0).UTC().Format(time.RFC3339) {
		t.Fatalf("end time = %q", c.EndTime)
	}
	if c.Open == nil || *c.Open != 0.11 || c.Close == nil || *c.Close != 0.115 {
		t.Fatalf("open/close = %v/%v, want 0.11/0.115", c.Open, c.Close)
	}
	if c.Volume == nil || *c.Volume != 1500.00 {
		t.Fatalf("volume = %v, want 1500.00", c.Volume)
	}
}

func TestNormalizeKalshiCandlesHandlesNullPriceFields(t *testing.T) {
	raw := `{"candlesticks": [{"end_period_ts": 1, "price": {"open_dollars": null, "close_dollars": "0.5"}, "volume_fp": "0"}]}`
	var resp kalshiCandlestickResp
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	candles := normalizeKalshiCandles(resp)
	if candles[0].Open != nil {
		t.Fatalf("open = %v, want nil for a null field", candles[0].Open)
	}
	if candles[0].Close == nil || *candles[0].Close != 0.5 {
		t.Fatalf("close = %v, want 0.5", candles[0].Close)
	}
}

func TestNormalizePolyBookParsesDocExample(t *testing.T) {
	raw := `{
		"market": "0x1234567890123456789012345678901234567890",
		"asset_id": "0xabc123def456",
		"bids": [{"price": "0.45", "size": "100"}, {"price": "0.44", "size": "200"}],
		"asks": [{"price": "0.46", "size": "150"}, {"price": "0.47", "size": "250"}],
		"min_order_size": "1",
		"tick_size": "0.01"
	}`
	var resp polyBookResp
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	book := normalizePolyBook(resp)
	if len(book.Bids) != 2 || book.Bids[0].Price != 0.45 || book.Bids[0].Size != 100 {
		t.Fatalf("bids = %+v", book.Bids)
	}
	if len(book.Asks) != 2 || book.Asks[0].Price != 0.46 {
		t.Fatalf("asks = %+v", book.Asks)
	}
}

func TestNormalizePolyPriceHistoryParsesDocExample(t *testing.T) {
	raw := `{"history": [{"t": 1750000000, "p": 0.42}, {"t": 1750003600, "p": 0.44}]}`
	var resp polyPriceHistoryResp
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	candles := normalizePolyPriceHistory(resp)
	if len(candles) != 2 {
		t.Fatalf("got %d candles, want 2", len(candles))
	}
	if candles[0].Close == nil || *candles[0].Close != 0.42 {
		t.Fatalf("first close = %v, want 0.42", candles[0].Close)
	}
	if candles[0].EndTime != time.Unix(1750000000, 0).UTC().Format(time.RFC3339) {
		t.Fatalf("end time = %q", candles[0].EndTime)
	}
}

func TestParseRef(t *testing.T) {
	cases := []struct {
		raw                    string
		platform, ident, extra string
	}{
		{"kalshi:KXFED-25JUN-H:KXFED", "kalshi", "KXFED-25JUN-H", "KXFED"},
		{"polymarket:fed-cut-july:111111", "polymarket", "fed-cut-july", "111111"},
		{"polymarket:fed-cut-july", "polymarket", "fed-cut-july", ""},
		{"badref", "badref", "", ""},
	}
	for _, tc := range cases {
		platform, ident, extra := parseRef(tc.raw)
		if platform != tc.platform || ident != tc.ident || extra != tc.extra {
			t.Errorf("parseRef(%q) = (%q, %q, %q), want (%q, %q, %q)",
				tc.raw, platform, ident, extra, tc.platform, tc.ident, tc.extra)
		}
	}
}

func TestFetchQuoteRequiresExtraPerPlatform(t *testing.T) {
	s := &server{kalshi: NewKalshiClient(newAPIClient(nil)), poly: NewPolymarketClient(newAPIClient(nil))}
	if q := s.fetchQuote(context.Background(), "kalshi", "T", ""); q.Error == "" {
		t.Fatal("expected an error when kalshi extra (series_ticker) is missing")
	}
	if q := s.fetchQuote(context.Background(), "polymarket", "s", ""); q.Error == "" {
		t.Fatal("expected an error when polymarket extra (token_id) is missing")
	}
	if q := s.fetchQuote(context.Background(), "dydx", "x", "y"); q.Error == "" {
		t.Fatal("expected an error for an unsupported platform")
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
