package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// apiClient is a thin, reusable JSON-over-HTTP client shared by the venue
// fetchers. It pins a timeout via the caller's context and caps the response
// body so a misbehaving upstream can't exhaust memory.
type apiClient struct {
	http      *http.Client
	userAgent string
	maxBody   int64
}

func newAPIClient(hc *http.Client) *apiClient {
	return &apiClient{http: hc, userAgent: "foresea-marketd/0.1", maxBody: 16 << 20}
}

// getJSON performs GET url and decodes the response into out.
func (c *apiClient) getJSON(ctx context.Context, url string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", c.userAgent)
	req.Header.Set("Accept", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body := io.LimitReader(resp.Body, c.maxBody)
	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(body, 512))
		return fmt.Errorf("status %d: %s", resp.StatusCode, string(snippet))
	}
	if err := json.NewDecoder(body).Decode(out); err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	return nil
}
