package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const polymarketGammaURL = "https://gamma-api.polymarket.com/markets"

// PolymarketClient is a structured API client for Polymarket's Gamma API.
type PolymarketClient struct{ api *apiClient }

func NewPolymarketClient(api *apiClient) *PolymarketClient { return &PolymarketClient{api: api} }

func (c *PolymarketClient) Name() string { return "polymarket" }

// gammaMarket is the subset of the Gamma market object we consume. Gamma encodes
// outcomes, prices, and CLOB token ids as JSON-stringified arrays, so they arrive
// as strings.
type gammaMarket struct {
	Slug          string    `json:"slug"`
	Question      string    `json:"question"`
	Title         string    `json:"title"`
	Outcomes      string    `json:"outcomes"`
	OutcomePrices string    `json:"outcomePrices"`
	ClobTokenIds  string    `json:"clobTokenIds"`
	EndDate       string    `json:"endDate"`
	Volume24hr    flexFloat `json:"volume24hr"`
	Volume        flexFloat `json:"volume"`
}

func (c *PolymarketClient) Fetch(ctx context.Context, q Query) ([]Market, error) {
	limit := q.Limit
	if limit <= 0 {
		limit = 20
	}
	// Fetch a deeper candidate pool so post-filtering by keyword/category still
	// yields enough results; Gamma orders by 24h volume.
	mult := 12
	if q.Keyword != "" || q.Category != "" {
		mult = 60 // matches may be sparse outside the very top of the volume tail
	}
	candidates := limit * mult
	if candidates > 500 {
		candidates = 500
	}
	params := url.Values{}
	params.Set("active", "true")
	params.Set("closed", "false")
	params.Set("order", "volume24hr")
	params.Set("ascending", "false")
	params.Set("limit", strconv.Itoa(candidates))

	var raw []gammaMarket
	if err := c.api.getJSON(ctx, polymarketGammaURL+"?"+params.Encode(), &raw); err != nil {
		return nil, err
	}

	out := make([]Market, 0, limit)
	for i := range raw {
		m, ok := normalizePolymarket(&raw[i])
		if !ok {
			continue
		}
		if !matchesQuery(m, q) {
			continue
		}
		out = append(out, m)
		if len(out) >= limit {
			break
		}
	}
	return out, nil
}

func normalizePolymarket(g *gammaMarket) (Market, bool) {
	labels := parseJSONStringArray(g.Outcomes)
	priceStrs := parseJSONStringArray(g.OutcomePrices)
	if len(labels) != 2 || len(priceStrs) != 2 {
		return Market{}, false
	}
	// Keep only binary Yes/No markets.
	li, ni := indexOfFold(labels, "yes"), indexOfFold(labels, "no")
	if li < 0 || ni < 0 {
		return Market{}, false
	}
	yes, err := strconv.ParseFloat(strings.TrimSpace(priceStrs[li]), 64)
	if err != nil {
		return Market{}, false
	}
	question := g.Question
	if question == "" {
		question = g.Title
	}
	yc, nc := yesNoCents(yes)
	market := Market{
		Platform:    "Polymarket",
		Ident:       g.Slug,
		Question:    question,
		MarketURL:   fmt.Sprintf("https://polymarket.com/market/%s", g.Slug),
		Category:    categoryFor(question, ""),
		Probability: &yes,
		YesCents:    &yc,
		NoCents:     &nc,
		CloseTime:   g.EndDate,
	}
	if v := g.Volume24hr.ptr(); v != nil {
		market.Volume = v
	} else if v := g.Volume.ptr(); v != nil {
		market.Volume = v
	}
	if tokenIDs := parseJSONStringArray(g.ClobTokenIds); len(tokenIDs) == 2 {
		market.TokenID = strings.TrimSpace(tokenIDs[li])
	}
	return market, true
}

const (
	polymarketBookURL          = "https://clob.polymarket.com/book"
	polymarketPricesHistoryURL = "https://clob.polymarket.com/prices-history"
)

type polyBookLevel struct {
	Price string `json:"price"`
	Size  string `json:"size"`
}

type polyBookResp struct {
	Bids []polyBookLevel `json:"bids"`
	Asks []polyBookLevel `json:"asks"`
}

// OrderBook fetches the live order book for a Polymarket CLOB token (the
// YES-outcome TokenID on a normalized Market). Public endpoint, no auth.
func (c *PolymarketClient) OrderBook(ctx context.Context, tokenID string) (*OrderBook, error) {
	params := url.Values{"token_id": {tokenID}}
	var resp polyBookResp
	if err := c.api.getJSON(ctx, polymarketBookURL+"?"+params.Encode(), &resp); err != nil {
		return nil, err
	}
	return normalizePolyBook(resp), nil
}

func normalizePolyBook(resp polyBookResp) *OrderBook {
	return &OrderBook{
		Bids: polyLevels(resp.Bids),
		Asks: polyLevels(resp.Asks),
	}
}

func polyLevels(raw []polyBookLevel) []OrderBookLevel {
	out := make([]OrderBookLevel, 0, len(raw))
	for _, lvl := range raw {
		price, err := strconv.ParseFloat(lvl.Price, 64)
		if err != nil {
			continue
		}
		size, err := strconv.ParseFloat(lvl.Size, 64)
		if err != nil {
			continue
		}
		out = append(out, OrderBookLevel{Price: price, Size: size})
	}
	return out
}

type polyPriceHistoryResp struct {
	History []polyPricePoint `json:"history"`
}

type polyPricePoint struct {
	T int64   `json:"t"`
	P float64 `json:"p"`
}

// PriceHistory fetches recent price points for a Polymarket CLOB token and
// reshapes them onto the shared Candle type (Close only -- this venue reports a
// price series, not per-period OHLC). Public endpoint, no auth.
//
// "interval" is a lookback window (last 1 day), not a bucket size; "fidelity" is
// the actual point spacing in minutes. Without fidelity, interval=1h alone
// returns ~60 one-minute-apart points -- far more than a batch response should
// carry per market. 60-minute fidelity over a 1-day window gives ~24 points,
// matching Kalshi's hourly candles for a consistent cross-venue shape.
func (c *PolymarketClient) PriceHistory(ctx context.Context, tokenID string) ([]Candle, error) {
	params := url.Values{"market": {tokenID}, "interval": {"1d"}, "fidelity": {"60"}}
	var resp polyPriceHistoryResp
	if err := c.api.getJSON(ctx, polymarketPricesHistoryURL+"?"+params.Encode(), &resp); err != nil {
		return nil, err
	}
	return normalizePolyPriceHistory(resp), nil
}

func normalizePolyPriceHistory(resp polyPriceHistoryResp) []Candle {
	out := make([]Candle, 0, len(resp.History))
	for _, pt := range resp.History {
		price := pt.P
		out = append(out, Candle{
			EndTime: time.Unix(pt.T, 0).UTC().Format(time.RFC3339),
			Close:   &price,
		})
	}
	return out
}

// parseJSONStringArray decodes a JSON-stringified array (Gamma's encoding) into
// a Go slice, e.g. `"[\"Yes\", \"No\"]"` -> ["Yes","No"].
func parseJSONStringArray(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	var arr []string
	if err := json.Unmarshal([]byte(s), &arr); err != nil {
		return nil
	}
	return arr
}

func indexOfFold(xs []string, want string) int {
	for i, x := range xs {
		if strings.EqualFold(strings.TrimSpace(x), want) {
			return i
		}
	}
	return -1
}
