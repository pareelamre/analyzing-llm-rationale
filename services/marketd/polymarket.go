package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"strings"
)

const polymarketGammaURL = "https://gamma-api.polymarket.com/markets"

// PolymarketClient is a structured API client for Polymarket's Gamma API.
type PolymarketClient struct{ api *apiClient }

func NewPolymarketClient(api *apiClient) *PolymarketClient { return &PolymarketClient{api: api} }

func (c *PolymarketClient) Name() string { return "polymarket" }

// gammaMarket is the subset of the Gamma market object we consume. Gamma encodes
// outcomes and prices as JSON-stringified arrays, so they arrive as strings.
type gammaMarket struct {
	Slug          string    `json:"slug"`
	Question      string    `json:"question"`
	Title         string    `json:"title"`
	Outcomes      string    `json:"outcomes"`
	OutcomePrices string    `json:"outcomePrices"`
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
	return market, true
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
