"""Kite Connect market-data contracts using synthetic broker responses."""
from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from analyzing_llm_rationale import kite_account, kite_market_data, server

CREDS = {"api_key": "fixturekey", "access_token": "fixturetoken"}
QUOTE = {"instrument_token": 12345, "last_price": 123.5, "ohlc": {"close": 120}, "timestamp": "2026-11-01 15:30:00"}
CANDLE = ["2026-11-01T09:15:00+0530", 120, 125, 119, 124, 100, 4]


class KiteMarketClientTests(unittest.TestCase):
    def test_quote_snapshots_and_missing_symbols(self):
        with mock.patch.object(kite_account.requests, "get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = {"status": "success", "data": {"NSE:INFY": QUOTE}}
            result = kite_market_data.quotes(CREDS, ["NSE:INFY", "BSE:ABSENT"])
            self.assertEqual(result["quotes"]["NSE:INFY"]["last_price"], 123.5)
            self.assertEqual(result["missing_instruments"], ["BSE:ABSENT"])
            self.assertEqual(get.call_args.args, ("https://api.kite.trade/quote",))
            self.assertEqual(get.call_args.kwargs["params"], [("i", "NSE:INFY"), ("i", "BSE:ABSENT")])
            self.assertFalse(get.call_args.kwargs["allow_redirects"])

    def test_history_resolves_symbol_token_and_preserves_ohlcv_oi(self):
        with mock.patch.object(kite_account.requests, "get") as get:
            get.return_value.status_code = 200
            get.return_value.json.side_effect = [
                {"status": "success", "data": {"NSE:INFY": QUOTE}},
                {"status": "success", "data": {"candles": [CANDLE]}},
            ]
            result = kite_market_data.history(CREDS, "NSE:INFY", "day", "2026-11-01", "2026-11-01")
            self.assertEqual(result["instrument"], "NSE:INFY")
            self.assertEqual(result["candles"], [["2026-11-01T09:15:00+05:30", *CANDLE[1:]]])
            self.assertEqual(get.call_args.args, ("https://api.kite.trade/instruments/historical/12345/day",))
            self.assertEqual(get.call_args.kwargs["params"]["from"], "2026-11-01 00:00:00")
            self.assertEqual(get.call_args.kwargs["params"]["to"], "2026-11-01 23:59:59")
            self.assertEqual(get.call_args.kwargs["params"]["oi"], 1)

    def test_invalid_symbols_dates_intervals_and_ranges_do_not_fetch(self):
        with mock.patch.object(kite_account.requests, "get") as get:
            for instruments in ([], ["https://evil.invalid"], ["NSE:INFY\r\nInjected"], ["NSE:INFY"] * 501):
                with self.assertRaises(kite_account.KiteAccountError):
                    kite_market_data.quotes(CREDS, instruments)
            for interval, start, end in (("bad", "2026-01-01", "2026-01-02"), ("day", "bad", "2026-01-02"), ("day", "2026-02-01", "2026-01-01"), ("minute", "2025-01-01", "2026-01-01")):
                with self.assertRaises(kite_account.KiteAccountError):
                    kite_market_data.history(CREDS, "NSE:INFY", interval, start, end)
            get.assert_not_called()

    def test_market_permission_and_bad_numbers_fail_safely(self):
        with mock.patch.object(kite_account.requests, "get") as get:
            get.return_value.status_code = 403
            get.return_value.json.return_value = {"message": "fixturetoken"}
            with self.assertRaises(kite_account.KiteAccountError) as raised:
                kite_market_data.quotes(CREDS, ["NSE:INFY"])
            self.assertEqual(raised.exception.status_code, 403)
            self.assertIn("Connect", str(raised.exception))
            self.assertNotIn("fixturetoken", str(raised.exception))
            get.return_value.status_code = 200
            get.return_value.json.return_value = {"status": "success", "data": {"NSE:INFY": {**QUOTE, "last_price": float("nan")}}}
            with self.assertRaises(kite_account.KiteAccountError):
                kite_market_data.quotes(CREDS, ["NSE:INFY"])

    def test_history_rejects_malformed_out_of_range_and_unordered_candles(self):
        for candles in ([CANDLE[:-2]], [[CANDLE[0], 120, 119, 125, 124, 100]],
                        [["2025-01-01T09:15:00+0530", *CANDLE[1:]]], [CANDLE, CANDLE],
                        [[CANDLE[0], 120, 125, 119, float("inf"), 100]], [["bad", *CANDLE[1:]]]):
            with self.subTest(candles=candles), mock.patch.object(kite_market_data, "request_data", side_effect=[{"NSE:INFY": QUOTE}, {"candles": candles}]):
                with self.assertRaises(kite_account.KiteAccountError) as raised:
                    kite_market_data.history(CREDS, "NSE:INFY", "day", "2026-11-01", "2026-11-01")
                self.assertEqual(raised.exception.status_code, 502)

    def test_utc_ranges_convert_to_ist_and_empty_history_is_valid(self):
        with mock.patch.object(kite_market_data, "request_data", side_effect=[{"NSE:INFY": QUOTE}, {"candles": []}]) as request:
            result = kite_market_data.history(CREDS, "nse:infy", "minute", "2026-11-01T03:45:00Z", "2026-11-01T04:45:00Z")
            self.assertEqual(result["candles"], [])
            self.assertEqual(request.call_args.kwargs["params"]["from"], "2026-11-01 09:15:00")
            self.assertEqual(request.call_args.kwargs["params"]["to"], "2026-11-01 10:15:00")

    def test_missing_history_symbol_never_fetches_candles(self):
        with mock.patch.object(kite_market_data, "request_data", return_value={}) as request:
            with self.assertRaises(kite_account.KiteAccountError) as raised:
                kite_market_data.history(CREDS, "NSE:ABSENT", "day", "2026-11-01", "2026-11-01")
            self.assertEqual(raised.exception.status_code, 404)
            self.assertEqual(request.call_count, 1)


class KiteMarketRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        self.headers = {"Authorization": "Bearer " + server._issue_session("market-owner", "fixture@example.invalid", "Fixture", "")}

    def test_routes_require_session_and_use_owner_connection(self):
        for path in ("/integrations/kite/quotes?i=NSE:INFY", "/integrations/kite/history?instrument=NSE:INFY&from=2026-11-01&to=2026-11-01"):
            self.assertEqual(self.client.get(path).status_code, 401)
        with mock.patch.dict(os.environ, {"FORESEA_ENABLE_KITE_PERSONAL": "true"}), mock.patch.object(server, "_check_rate_limit"), mock.patch.object(server, "_stored_trading_credentials", return_value=CREDS) as stored, mock.patch.object(kite_market_data, "quotes", return_value={"quotes": {"NSE:INFY": QUOTE}, "missing_instruments": []}) as quotes:
            response = self.client.get("/integrations/kite/quotes?i=NSE:INFY", headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["data"]["quotes"]["NSE:INFY"]["last_price"], 123.5)
            self.assertIn("no-store", response.headers["cache-control"])
            stored.assert_called_once_with("market-owner", "kite")
            quotes.assert_called_once_with(CREDS, ["NSE:INFY"])
            self.assertNotIn("fixturetoken", response.text)

    def test_unconnected_and_disabled_do_not_request_market_data(self):
        with mock.patch.dict(os.environ, {"FORESEA_ENABLE_KITE_PERSONAL": "true"}), mock.patch.object(server, "_check_rate_limit"), mock.patch.object(server, "_stored_trading_credentials", return_value=None), mock.patch.object(kite_market_data, "quotes") as quotes:
            self.assertEqual(self.client.get("/integrations/kite/quotes?i=NSE:INFY", headers=self.headers).status_code, 409)
            quotes.assert_not_called()
        with mock.patch.dict(os.environ, {"FORESEA_ENABLE_KITE_PERSONAL": "false"}), mock.patch.object(server, "_check_rate_limit"), mock.patch.object(kite_market_data, "quotes") as quotes:
            self.assertEqual(self.client.get("/integrations/kite/quotes?i=NSE:INFY", headers=self.headers).status_code, 503)
            quotes.assert_not_called()

    def test_history_route_and_cross_user_isolation(self):
        other_headers = {"Authorization": "Bearer " + server._issue_session("other-owner", "other@example.invalid", "Other", "")}
        with mock.patch.dict(os.environ, {"FORESEA_ENABLE_KITE_PERSONAL": "true"}), mock.patch.object(server, "_check_rate_limit"), mock.patch.object(server, "_stored_trading_credentials", side_effect=lambda owner, platform: CREDS if owner == "market-owner" else None) as stored, mock.patch.object(kite_market_data, "history", return_value={"candles": []}) as history:
            path = "/integrations/kite/history?instrument=NSE:INFY&interval=day&from=2026-11-01&to=2026-11-01"
            response = self.client.get(path, headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("no-store", response.headers["cache-control"])
            self.assertEqual(response.json()["data"], {"candles": []})
            history.assert_called_once_with(CREDS, instrument="NSE:INFY", interval="day", start="2026-11-01", end="2026-11-01")
            self.assertEqual(self.client.get(path, headers=other_headers).status_code, 409)
            self.assertEqual(history.call_count, 1)
            self.assertEqual(stored.call_args.args, ("other-owner", "kite"))


class KiteMarketUiTests(unittest.TestCase):
    def test_native_text_requests_and_stale_response_isolation(self):
        script = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync('frontend/index.html', 'utf8');
const code = html.slice(html.indexOf('// Private Kite data stays'), html.indexOf('// ── BYO Model Providers Manager'));
class Element {
  constructor() { this.children=[]; this.value=''; this.style={}; this.isConnected=true; this._text=''; this.attrs={}; }
  setAttribute(key,value) { this.attrs[key]=value; }
  set textContent(value) { this._text=value; this.children=[]; }
  get textContent() { return this._text; }
  set innerHTML(value) { throw Error('provider text became HTML'); }
  replaceChildren() { this.children=[]; this._text=''; }
  append(child) { this.children.push(child); }
  createTHead() { const child=new Element(); this.append(child); return child; }
  createTBody() { return this.createTHead(); }
  insertRow() { return this.createTHead(); }
  insertCell() { return this.createTHead(); }
}
const ids=['kiteMarketData','kiteMarketMessage','kiteMarketSymbol','kiteMarketInterval','kiteMarketFrom','kiteMarketTo'];
const elements=new Map(ids.map(id => [id,new Element()]));
elements.get('kiteMarketSymbol').value='NSE:INFY';
elements.get('kiteMarketInterval').value='day';
elements.get('kiteMarketFrom').value=elements.get('kiteMarketTo').value='2026-11-01';
let session='Bearer owner', resolve, request;
const context=vm.createContext({
  document: {getElementById: id => elements.get(id), createElement: () => new Element(), createElementNS: () => new Element()},
  authHeaders: () => ({Authorization:session}), URLSearchParams,
  _ebVenueFilter:'kite', _ebView:'markets',
  fetch: (url, options) => { request={url,options}; return new Promise(done => {resolve=done;}); },
});
vm.runInContext(code,context);
const target=elements.get('kiteMarketData');
context.renderKiteMarket('quotes',{quotes:{'<img onerror=attack()>':{last_price:0,ohlc:{}}},missing_instruments:[]},target);
assert.equal(target.children[0].children[1].children[0].children[0].textContent,'<img onerror=attack()>');
assert.equal(target.children[0].children[1].children[0].children[1].textContent,'0');
target.replaceChildren();
const candles=Array.from({length:1600},(_,i)=>[new Date(Date.UTC(2026,0,1)+i*60000).toISOString(),100,i===1?1000:100,100,i===1?1000:100,1]);
context.renderKiteMarket('history',{instrument:'NSE:INFY',interval:'minute',candles},target);
assert.ok(target.children[2].textContent.startsWith('Close: 100–1000'),'reported range lost an unsampled extreme');
assert.ok(target.children[1].attrs['aria-label'].includes('Range 100 to 1000'));
assert.equal(target.children[3].children[1].children.length,200);
(async () => {
  let pending=context.kiteMarketAction('quotes');
  assert.equal(request.url,'/integrations/kite/quotes?i=NSE%3AINFY');
  assert.equal(request.options.headers.Authorization,'Bearer owner');
  assert.equal(request.options.cache,'no-store');
  session='Bearer another-owner';
  resolve({ok:true,json:async()=>({data:{quotes:{PRIVATE:{last_price:123}},missing_instruments:[]}})});
  await pending;
  assert.equal(target.children.length,0);
  pending=context.kiteMarketAction('history');
  assert.ok(request.url.includes('from=2026-11-01'));
  context.clearKiteMarketData(); context._ebVenueFilter='kalshi';
  resolve({ok:true,json:async()=>({data:{candles:[['PRIVATE',1,2,0,1,3]]}})});
  await pending;
  assert.equal(target.children.length,0);
  assert.equal(elements.get('kiteMarketMessage').textContent,'');
  context._ebVenueFilter='kite';
  pending=context.kiteMarketAction('quotes');
  resolve({ok:false,status:403,json:async()=>({detail:'Check paid Kite Connect subscription.'})});
  await pending;
  assert.equal(elements.get('kiteMarketMessage').textContent,'Check paid Kite Connect subscription.');
  assert.equal(target.children.length,0);
  for (const id of ['kiteApiKey','kiteAccessToken','kiteAccountData','kiteAccountMessage','kiteChip','kiteResource']) elements.set(id,new Element());
  elements.get('kiteResource').value='holdings';
  const quoteReply={ok:true,json:async()=>({data:{quotes:{PRIVATE:{last_price:123}},missing_instruments:[]},fetched_at:'2026-11-01T00:00:00Z'})};
  for (const action of ['connect','disconnect']) {
    for (const quoteFirst of [true,false]) {
      // A concurrent status read must not disable connection-change cleanup.
      const calls=[];
      context.fetch=(url,options)=>new Promise(done=>calls.push({url,options,reply:done}));
      const mutation=context.kiteAccountAction(action);
      const quote=context.kiteMarketAction('quotes');
      const status=context.kiteAccountAction('status');
      calls[2].reply({ok:true,json:async()=>({connected:true,enabled:true})}); await status;
      if (quoteFirst) { calls[1].reply(quoteReply); await quote; assert.ok(target.children.length>0); }
      calls[0].reply({ok:true,json:async()=>({connected:action==='connect',enabled:true})}); await mutation;
      if (!quoteFirst) { calls[1].reply(quoteReply); await quote; }
      assert.equal(target.children.length,0,'market data survived '+action+' completion');
      assert.equal(elements.get('kiteMarketMessage').textContent,'');
    }
  }
})().catch(error => {console.error(error);process.exitCode=1;});
"""
        result = subprocess.run(["node", "-e", script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
