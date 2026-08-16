package main

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// Market is the venue-agnostic, normalized shape every source is mapped onto.
// Probability is P(Yes) in [0,1]; YesCents/NoCents are the same expressed the way
// prediction markets quote them (0–100¢).
//
// SeriesTicker (Kalshi only) and TokenID (Polymarket only, the YES-outcome CLOB
// token) are carried through so a caller who already has a Market from /markets
// can fetch candles / order book for it without a second lookup.
type Market struct {
	Platform     string   `json:"platform"`
	Ident        string   `json:"ident"`
	Question     string   `json:"question"`
	MarketURL    string   `json:"market_url"`
	Category     string   `json:"category"`
	Probability  *float64 `json:"probability"`
	YesCents     *int     `json:"yes_cents,omitempty"`
	NoCents      *int     `json:"no_cents,omitempty"`
	CloseTime    string   `json:"close_time,omitempty"`
	Volume       *float64 `json:"volume,omitempty"`
	SeriesTicker string   `json:"series_ticker,omitempty"`
	TokenID      string   `json:"token_id,omitempty"`
}

// flexFloat decodes a numeric value whether the venue encodes it as a JSON
// number (Polymarket volume) or a JSON string (Kalshi *_dollars fields).
type flexFloat float64

func (f *flexFloat) UnmarshalJSON(b []byte) error {
	s := strings.Trim(string(b), `"`)
	if s == "" || s == "null" {
		return nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return nil // tolerate unexpected junk rather than failing the whole decode
	}
	*f = flexFloat(v)
	return nil
}

func (f flexFloat) ptr() *float64 {
	if f == 0 {
		return nil
	}
	v := float64(f)
	return &v
}

// Query is the normalized filter applied across every venue.
type Query struct {
	Keyword  string
	Category string
	Limit    int
}

// OrderBookLevel is one price/size pair on one side of a book.
type OrderBookLevel struct {
	Price float64 `json:"price"`
	Size  float64 `json:"size"`
}

// OrderBook is bids/asks for the YES side of a binary market, best price first.
type OrderBook struct {
	Bids []OrderBookLevel `json:"bids"`
	Asks []OrderBookLevel `json:"asks"`
}

// Candle is one OHLC period. Any of Open/High/Low/Close/Volume may be nil if the
// venue didn't report it for that period.
type Candle struct {
	EndTime string   `json:"end_time"` // RFC3339
	Open    *float64 `json:"open,omitempty"`
	High    *float64 `json:"high,omitempty"`
	Low     *float64 `json:"low,omitempty"`
	Close   *float64 `json:"close,omitempty"`
	Volume  *float64 `json:"volume,omitempty"`
}

// Quote is depth/candle data for one market -- a focused enrichment endpoint,
// not a re-derivation of the base listing fields /markets already serves.
// FetchedAt is the staleness signal every response carries. Error is set (with
// the other fields left empty) when this one ref failed, so a batch of refs
// degrades per-ref rather than failing as a whole.
type Quote struct {
	Platform  string     `json:"platform"`
	Ident     string     `json:"ident"`
	OrderBook *OrderBook `json:"order_book,omitempty"`
	Candles   []Candle   `json:"candles,omitempty"`
	FetchedAt string     `json:"fetched_at"`
	Error     string     `json:"error,omitempty"`
}

// Fetcher is one external market source (e.g. Polymarket, Kalshi). Implementations
// own their HTTP client and JSON shape; the ingestor only sees normalized Markets.
type Fetcher interface {
	Name() string
	Fetch(ctx context.Context, q Query) ([]Market, error)
}

// Ingestor fans out a Query to every source concurrently and merges the results.
type Ingestor struct {
	sources []Fetcher
}

func NewIngestor(sources ...Fetcher) *Ingestor { return &Ingestor{sources: sources} }

// Markets retrieves from all sources concurrently (one goroutine per source),
// merges, ranks by volume, and applies the limit. A failing source degrades
// gracefully: its error is collected and the others still return.
func (ig *Ingestor) Markets(ctx context.Context, q Query) ([]Market, []error) {
	type result struct {
		markets []Market
		err     error
	}
	ch := make(chan result, len(ig.sources))
	var wg sync.WaitGroup
	for _, src := range ig.sources {
		wg.Add(1)
		go func(src Fetcher) {
			defer wg.Done()
			m, err := src.Fetch(ctx, q)
			if err != nil {
				err = fmt.Errorf("%s: %w", src.Name(), err)
			}
			ch <- result{markets: m, err: err}
		}(src)
	}
	go func() { wg.Wait(); close(ch) }()

	var all []Market
	var errs []error
	for r := range ch {
		if r.err != nil {
			errs = append(errs, r.err)
		}
		all = append(all, r.markets...)
	}

	sort.SliceStable(all, func(i, j int) bool {
		return volOf(all[i]) > volOf(all[j])
	})
	if q.Limit > 0 && len(all) > q.Limit {
		all = all[:q.Limit]
	}
	return all, errs
}

func volOf(m Market) float64 {
	if m.Volume == nil {
		return 0
	}
	return *m.Volume
}

// probability helpers shared by the venue clients.
func yesNoCents(p float64) (yes, no int) {
	yes = int(p*100 + 0.5)
	if yes < 0 {
		yes = 0
	} else if yes > 100 {
		yes = 100
	}
	return yes, 100 - yes
}

// categoryKeywords mirrors the Python categorizer so the Go service produces the
// same browse categories. Polymarket's listing has no category field, so we infer
// from the question; Kalshi's own category is used as a fallback.
var categoryKeywords = []struct {
	name     string
	keywords []string
}{
	{"Politics", []string{"election", "president", "senate", "congress", "trump", "biden", "vote", "governor", "parliament", "government", "shutdown", "poll", "democrat", "republican"}},
	{"Crypto", []string{"bitcoin", "btc", "ethereum", " eth ", "crypto", "solana", "dogecoin", "token", "blockchain", "coinbase"}},
	{"Sports", []string{"nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "world cup", "super bowl", "premier league", "champion", "playoff", "tournament", " vs ", "ufc", "golf", "tennis", "olympic"}},
	{"Economics", []string{"fed", "interest rate", "inflation", "gdp", "jobs report", "recession", "cpi", "unemployment", "economy", "rate cut"}},
	{"Entertainment", []string{"movie", "film", "oscar", "album", "song", "box office", "grammy", "emmy", "netflix", "celebrity", "season", "spotify"}},
	{"Tech", []string{"openai", "tesla", "apple", "google", "nvidia", "chip", "gpt", "spacex", "artificial intelligence", "iphone", "software"}},
	{"World", []string{"ukraine", "russia", "china", "israel", "gaza", "war", "nuclear", "climate", "nato", "ceasefire"}},
}

func categoryFor(question, raw string) string {
	q := strings.ToLower(question)
	for _, c := range categoryKeywords {
		for _, kw := range c.keywords {
			if strings.Contains(q, kw) {
				return c.name
			}
		}
	}
	if raw != "" {
		return raw
	}
	return "Other"
}

func matchesQuery(m Market, q Query) bool {
	if q.Keyword != "" && !strings.Contains(strings.ToLower(m.Question), strings.ToLower(q.Keyword)) {
		return false
	}
	if q.Category != "" && !strings.EqualFold(m.Category, q.Category) {
		return false
	}
	return true
}
