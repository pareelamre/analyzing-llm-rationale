package main

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"
)

const kalshiEventsURL = "https://api.elections.kalshi.com/trade-api/v2/events"

// KalshiClient is a structured API client for Kalshi's public trade API. It pulls
// markets grouped under events (the flat /markets listing is saturated by
// auto-generated multi-leg "MVE" parlay markets).
type KalshiClient struct{ api *apiClient }

func NewKalshiClient(api *apiClient) *KalshiClient { return &KalshiClient{api: api} }

func (c *KalshiClient) Name() string { return "kalshi" }

type kalshiEventsResp struct {
	Events []kalshiEvent `json:"events"`
}

type kalshiEvent struct {
	Title        string         `json:"title"`
	Category     string         `json:"category"`
	SeriesTicker string         `json:"series_ticker"`
	Markets      []kalshiMarket `json:"markets"`
}

type kalshiMarket struct {
	Ticker           string    `json:"ticker"`
	EventTicker      string    `json:"event_ticker"`
	YesSubTitle      string    `json:"yes_sub_title"`
	MVECollection    string    `json:"mve_collection_ticker"`
	LastPriceDollars flexFloat `json:"last_price_dollars"`
	YesBidDollars    flexFloat `json:"yes_bid_dollars"`
	YesAskDollars    flexFloat `json:"yes_ask_dollars"`
	CloseTime        string    `json:"close_time"`
	Volume24hFP      flexFloat `json:"volume_24h_fp"`
}

func (c *KalshiClient) Fetch(ctx context.Context, q Query) ([]Market, error) {
	limit := q.Limit
	if limit <= 0 {
		limit = 20
	}
	params := url.Values{}
	params.Set("status", "open")
	params.Set("with_nested_markets", "true")
	params.Set("limit", "200")

	var resp kalshiEventsResp
	if err := c.api.getJSON(ctx, kalshiEventsURL+"?"+params.Encode(), &resp); err != nil {
		return nil, err
	}

	out := make([]Market, 0, limit)
	for ei := range resp.Events {
		ev := &resp.Events[ei]
		for mi := range ev.Markets {
			m, ok := normalizeKalshi(ev, &ev.Markets[mi])
			if !ok {
				continue
			}
			if !matchesQuery(m, q) {
				continue
			}
			out = append(out, m)
			if len(out) >= limit {
				return out, nil
			}
		}
	}
	return out, nil
}

func normalizeKalshi(ev *kalshiEvent, m *kalshiMarket) (Market, bool) {
	if m.MVECollection != "" {
		return Market{}, false // skip multi-leg parlay legs
	}
	prob, ok := kalshiProbability(m)
	if !ok {
		return Market{}, false
	}
	question := ev.Title
	if sub := strings.TrimSpace(m.YesSubTitle); sub != "" && !strings.Contains(strings.ToLower(ev.Title), strings.ToLower(sub)) {
		question = fmt.Sprintf("%s — %s", ev.Title, sub)
	}
	yc, nc := yesNoCents(prob)
	out := Market{
		Platform:     "Kalshi",
		Ident:        m.Ticker,
		Question:     question,
		MarketURL:    kalshiMarketURL(ev, m),
		Category:     categoryFor(question, ev.Category),
		Probability:  &prob,
		YesCents:     &yc,
		NoCents:      &nc,
		CloseTime:    m.CloseTime,
		SeriesTicker: ev.SeriesTicker,
	}
	if v := m.Volume24hFP.ptr(); v != nil {
		out.Volume = v
	}
	return out, true
}

// kalshiProbability prefers the last trade, else the bid/ask midpoint.
func kalshiProbability(m *kalshiMarket) (float64, bool) {
	last := float64(m.LastPriceDollars)
	if last > 0 && last < 1 {
		return last, true
	}
	bid, ask := float64(m.YesBidDollars), float64(m.YesAskDollars)
	if bid > 0 || ask < 1 {
		mid := (bid + ask) / 2
		if mid > 0 && mid < 1 {
			return mid, true
		}
	}
	return 0, false
}

// kalshiMarketURL builds the web URL, which is rooted on the (lowercase) series
// ticker — NOT the event ticker (which carries a date suffix and 404s).
func kalshiMarketURL(ev *kalshiEvent, m *kalshiMarket) string {
	series := strings.TrimSpace(ev.SeriesTicker)
	if series == "" {
		base := m.EventTicker
		if base == "" {
			base = m.Ticker
		}
		if i := strings.LastIndex(base, "-"); i > 0 {
			base = base[:i]
		}
		series = base
	}
	series = strings.ToLower(series)
	if series == "" {
		return ""
	}
	return "https://kalshi.com/markets/" + series
}

const kalshiCandlesticksURL = "https://api.elections.kalshi.com/trade-api/v2/series/%s/markets/%s/candlesticks"

type kalshiCandlestickResp struct {
	Candlesticks []kalshiCandlestick `json:"candlesticks"`
}

type kalshiCandlestick struct {
	EndPeriodTs int64            `json:"end_period_ts"`
	Price       kalshiPriceRange `json:"price"`
	VolumeFP    flexFloat        `json:"volume_fp"`
}

type kalshiPriceRange struct {
	OpenDollars  *flexFloat `json:"open_dollars"`
	LowDollars   *flexFloat `json:"low_dollars"`
	HighDollars  *flexFloat `json:"high_dollars"`
	CloseDollars *flexFloat `json:"close_dollars"`
}

// Candlesticks fetches recent hourly OHLC candles for one market. seriesTicker
// comes from the Market's SeriesTicker field (set in normalizeKalshi) -- Kalshi
// requires both the series and market ticker, there's no by-ticker-alone lookup.
// Public endpoint (no auth headers), unlike the order book endpoint.
func (c *KalshiClient) Candlesticks(ctx context.Context, seriesTicker, ticker string, lookback time.Duration) ([]Candle, error) {
	now := time.Now().UTC()
	params := url.Values{}
	params.Set("start_ts", fmt.Sprintf("%d", now.Add(-lookback).Unix()))
	params.Set("end_ts", fmt.Sprintf("%d", now.Unix()))
	params.Set("period_interval", "60") // hourly

	reqURL := fmt.Sprintf(kalshiCandlesticksURL, url.PathEscape(seriesTicker), url.PathEscape(ticker)) + "?" + params.Encode()
	var resp kalshiCandlestickResp
	if err := c.api.getJSON(ctx, reqURL, &resp); err != nil {
		return nil, err
	}
	return normalizeKalshiCandles(resp), nil
}

func normalizeKalshiCandles(resp kalshiCandlestickResp) []Candle {
	out := make([]Candle, 0, len(resp.Candlesticks))
	for _, cs := range resp.Candlesticks {
		out = append(out, Candle{
			EndTime: time.Unix(cs.EndPeriodTs, 0).UTC().Format(time.RFC3339),
			Open:    flexPtr(cs.Price.OpenDollars),
			High:    flexPtr(cs.Price.HighDollars),
			Low:     flexPtr(cs.Price.LowDollars),
			Close:   flexPtr(cs.Price.CloseDollars),
			Volume:  cs.VolumeFP.ptr(),
		})
	}
	return out
}

func flexPtr(f *flexFloat) *float64 {
	if f == nil {
		return nil
	}
	return f.ptr()
}
