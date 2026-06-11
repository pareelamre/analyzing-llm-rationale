package main

import (
	"sync"
	"time"
)

// ttlCache is a tiny concurrency-safe cache with per-entry expiry, used to avoid
// hammering the venues on repeated identical queries.
type ttlCache[V any] struct {
	mu  sync.RWMutex
	ttl time.Duration
	m   map[string]cacheEntry[V]
}

type cacheEntry[V any] struct {
	val V
	exp time.Time
}

func newTTLCache[V any](ttl time.Duration) *ttlCache[V] {
	return &ttlCache[V]{ttl: ttl, m: make(map[string]cacheEntry[V])}
}

func (c *ttlCache[V]) get(key string) (V, bool) {
	c.mu.RLock()
	e, ok := c.m[key]
	c.mu.RUnlock()
	if !ok || time.Now().After(e.exp) {
		var zero V
		return zero, false
	}
	return e.val, true
}

func (c *ttlCache[V]) set(key string, val V) {
	c.mu.Lock()
	c.m[key] = cacheEntry[V]{val: val, exp: time.Now().Add(c.ttl)}
	c.mu.Unlock()
}
