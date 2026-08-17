from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402


class _FakeDsKey:
    """Mimics google.cloud.datastore.Key enough for benchmark_tools' Datastore
    account store: multi-segment ancestor paths, equality/hashing by path."""

    def __init__(self, path_parts):
        self.path_parts = tuple(path_parts)
        self.kind = path_parts[-2]
        self.name = path_parts[-1]

    def __eq__(self, other):
        return isinstance(other, _FakeDsKey) and self.path_parts == other.path_parts

    def __hash__(self):
        return hash(self.path_parts)


class _FakeDsEntity(dict):
    def __init__(self, key=None, exclude_from_indexes=()):
        super().__init__()
        self.key = key


class _FakeDsQuery:
    def __init__(self, store, kind, ancestor):
        self._store = store
        self._kind = kind
        self._ancestor_prefix = ancestor.path_parts if ancestor is not None else None

    def fetch(self):
        out = []
        for key, entity in self._store.items():
            if key.kind != self._kind:
                continue
            if self._ancestor_prefix is not None and key.path_parts[: len(self._ancestor_prefix)] != self._ancestor_prefix:
                continue
            out.append(entity)
        return out


class _FakeDsTransaction:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDsClient:
    """In-memory stand-in for google.cloud.datastore.Client, covering just
    the operations benchmark_tools' _ds_* account store uses: multi-segment
    ancestor keys, get/put/delete, kind+ancestor queries, and a transaction
    context manager (no real conflict simulation -- this tests the happy
    path's read/write shape, not Datastore's optimistic-concurrency retry)."""

    def __init__(self):
        self.store = {}

    def key(self, *path_parts):
        return _FakeDsKey(path_parts)

    def get(self, key):
        return self.store.get(key)

    def put(self, entity):
        self.store[entity.key] = entity

    def delete(self, key):
        self.store.pop(key, None)

    def query(self, kind=None, ancestor=None):
        return _FakeDsQuery(self.store, kind, ancestor)

    def transaction(self):
        return _FakeDsTransaction()


def _install_fake_datastore(test_case: unittest.TestCase) -> _FakeDsClient:
    """Patch benchmark_tools onto a fresh in-memory fake Datastore client and
    stub the google.cloud.datastore module so `_ds.Entity(...)` works
    offline -- same approach tests/test_auth_dedup.py uses for server.py."""
    client = _FakeDsClient()
    patcher = mock.patch.object(benchmark_tools, "_get_account_datastore", lambda: client)
    patcher.start()
    test_case.addCleanup(patcher.stop)
    original_module = sys.modules.get("google.cloud.datastore")
    sys.modules["google.cloud.datastore"] = types.SimpleNamespace(Entity=_FakeDsEntity)

    def _restore():
        if original_module is not None:
            sys.modules["google.cloud.datastore"] = original_module
        else:
            sys.modules.pop("google.cloud.datastore", None)

    test_case.addCleanup(_restore)
    return client


def _fetch_kalshi_quotes(quotes):
    """Build a fetch_kalshi side_effect from a per-ticker spec, for tests
    that need place_trade's shadow order to be marketable (see
    _resolve_shadow_marketability -- an order with no live quote no longer
    fills at all, so most place_trade tests need one).

    ``quotes[ticker]`` may be:
      - a float: used as both yes_ask and no_ask (fine when a test only
        trades one side, or deliberately wants an implausible quote where
        both sides are quoted cheap, e.g. the netting-arb tests).
      - a dict: passed through as the raw quote payload (for yes_ask/no_ask
        that must differ, e.g. opening then closing a position).
      - a list: consumed one entry per call for that ticker (each entry a
        float or dict as above), for a test whose ask must change between
        successive calls on the same ticker (e.g. averaging in at two
        different prices).
    A ticker with no entry raises MarketDataError, matching a real lookup
    failure for an unmocked ticker rather than silently defaulting.
    """
    call_counts: dict = {}

    def _fetch(ticker):
        spec = quotes.get(ticker)
        if spec is None:
            raise market_data.MarketDataError(f"no mock quote configured for {ticker}")
        if isinstance(spec, list):
            idx = call_counts.get(ticker, 0)
            call_counts[ticker] = idx + 1
            spec = spec[min(idx, len(spec) - 1)]
        if isinstance(spec, (int, float)):
            return {"yes_ask": spec, "no_ask": spec}
        return dict(spec)

    return _fetch


class BenchmarkToolTests(unittest.TestCase):
    def setUp(self):
        # place_trade's shadow path checks live Kalshi quotes (see
        # _resolve_shadow_marketability) and no longer fills an order with no
        # live quote at all -- default to "no quote available" so a test that
        # forgets to mock a quote fails loudly (MarketDataError propagating
        # into an unfilled/rejected result) instead of silently depending on
        # real network access. Tests that need a fill mock fetch_kalshi
        # locally via _fetch_kalshi_quotes(...).
        patcher = mock.patch(
            "analyzing_llm_rationale.market_data.fetch_kalshi",
            side_effect=market_data.MarketDataError("market data disabled in tests"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_manage_notes_add_search_edit_delete_with_limits(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "notes.json"
            ctx = benchmark_tools.ToolContext(agent_id="model-a")

            added = benchmark_tools.manage_notes(
                {"action": "add", "text": "Watch the FOMC date.", "tags": ["rates"]},
                ctx,
                path=path,
            )
            self.assertTrue(added["ok"])
            note_id = added["note"]["id"]

            found = benchmark_tools.manage_notes(
                {"action": "search", "query": "fomc"},
                ctx,
                path=path,
            )
            self.assertEqual([n["id"] for n in found["notes"]], [note_id])

            edited = benchmark_tools.manage_notes(
                {"action": "edit", "id": note_id, "text": "Watch CPI before FOMC."},
                ctx,
                path=path,
            )
            self.assertEqual(edited["note"]["text"], "Watch CPI before FOMC.")

            deleted = benchmark_tools.manage_notes(
                {"action": "delete", "id": note_id},
                ctx,
                path=path,
            )
            self.assertEqual(deleted["deleted"], 1)

            too_long = benchmark_tools.manage_notes(
                {"action": "add", "text": "x" * (benchmark_tools.MAX_NOTE_CHARS + 1)},
                ctx,
                path=path,
            )
            self.assertFalse(too_long["ok"])

    def test_web_search_filters_blacklisted_citations(self):
        from analyzing_llm_rationale import news_pipeline

        articles = [
            {"title": "Bad", "url": "https://coinmarketcap.com/currencies/x", "summary": "Bad summary."},
            {"title": "Good", "url": "https://example.com/story", "summary": "Good summary."},
        ]
        mock_pipeline = mock.Mock()
        mock_pipeline.fetch_summarize_rank.return_value = articles

        with mock.patch.object(news_pipeline, "NewsPipeline", return_value=mock_pipeline) as pipeline_cls:
            result = benchmark_tools.web_search({"query": "test market"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["sources"], [{"title": "Good", "url": "https://example.com/story"}])
        self.assertEqual(result["blocked_results"], 1)
        self.assertIn("Good summary.", result["summary"])
        self.assertNotIn("Bad summary.", result["summary"])
        pipeline_cls.assert_called_once_with(fetch_sources=benchmark_tools.WEB_SEARCH_SOURCES)
        mock_pipeline.fetch_summarize_rank.assert_called_once_with(
            "test market", top_k=benchmark_tools.WEB_SEARCH_TOP_K
        )

    def test_web_search_does_not_require_an_openai_key(self):
        # Regression: web_search used to hit api.openai.com and required
        # OPENAI_API_KEY, which was never configured as a repo secret, so
        # every single call failed. It now runs through the keyless
        # multi-source news pipeline instead (SCADS_AI_API_KEY, already
        # required elsewhere, covers query planning/summarization).
        from analyzing_llm_rationale import news_pipeline

        mock_pipeline = mock.Mock()
        mock_pipeline.fetch_summarize_rank.return_value = []

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(news_pipeline, "NewsPipeline", return_value=mock_pipeline),
        ):
            result = benchmark_tools.web_search({"query": "test market"})

        self.assertTrue(result["ok"])

    def test_web_search_requires_a_query(self):
        result = benchmark_tools.web_search({"query": "  "})
        self.assertFalse(result["ok"])
        self.assertIn("query is required", result["error"])

    def test_place_trade_defaults_to_shadow_kalshi_buy(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXTEST": 0.42}),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXTEST", "side": "yes", "price": 0.42, "quantity": 2},
                    ctx,
                )

        self.assertTrue(result["ok"])
        self.assertFalse(result["submitted"])
        self.assertEqual(result["mode"], "shadow")
        self.assertEqual(result["normalized_order"]["platform"], "kalshi")
        self.assertEqual(result["normalized_order"]["action"], "buy")
        self.assertEqual(result["normalized_order"]["outcome"], "yes")
        self.assertEqual(
            result["normalized_order"]["exchange_order"]["time_in_force"],
            "immediate_or_cancel",
        )
        self.assertFalse(result["normalized_order"]["exchange_order"]["post_only"])
        self.assertEqual(result["execution"]["filled_quantity"], 2.0)
        self.assertTrue(result["risk_guard"]["allowed"])

    def test_place_trade_forces_ioc_and_rejects_resting_order_options(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXIOC": 0.42}),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {
                        "ticker": "KXIOC",
                        "side": "yes",
                        "price": 0.42,
                        "quantity": 2,
                        "time_in_force": "good_till_canceled",
                        "post_only": True,
                        "order_type": "market",
                    },
                    ctx,
                )

        self.assertTrue(result["ok"])
        order = result["normalized_order"]["exchange_order"]
        self.assertEqual(order["time_in_force"], "immediate_or_cancel")
        self.assertFalse(order["post_only"])
        self.assertTrue(any("time_in_force" in w for w in result["warnings"]))
        self.assertTrue(any("post_only" in w for w in result["warnings"]))
        self.assertTrue(any("order_type" in w for w in result["warnings"]))

    def test_place_trade_rejects_live_mode_and_never_calls_trading_place_order(self):
        # place_trade is called from an autonomous LLM tool loop, which cannot
        # supply real human confirmation and does not route through
        # create_trading_run/execute_trading_run/the guardrail chain/the kill
        # switch. FORESEA_AGENT_PLACE_TRADE_MODE=live used to let it call
        # trading.place_order directly with a self-supplied confirmation
        # phrase and shared server credentials -- it must now be rejected
        # outright, before trading.place_order is ever reached.
        from analyzing_llm_rationale import trading

        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_PLACE_TRADE_MODE": "live",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(trading, "place_order") as fake_place_order,
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXPARTIAL", "side": "yes", "price": 0.40, "quantity": 10},
                    ctx,
                )
                fake_place_order.assert_not_called()

        self.assertFalse(result["ok"])
        self.assertIn("must be 'shadow'", result["error"])

    def test_place_trade_rejects_single_market_concentration_over_15_percent(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "0.15",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXCONC": 0.10}),
                ),
            ):
                first = benchmark_tools.place_trade(
                    {"ticker": "KXCONC", "side": "yes", "price": 0.10, "quantity": 100},
                    ctx,
                )
                second = benchmark_tools.place_trade(
                    {"ticker": "KXCONC", "side": "yes", "price": 0.10, "quantity": 60},
                    ctx,
                )

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertTrue(second["rejected"])
        self.assertEqual(second["reason"], "concentration_limit")
        self.assertEqual(second["risk_guard"]["concentration_cap"], 15.0)
        self.assertEqual(second["risk_guard"]["market_cost_basis_after"], 16.0)

    def test_rejected_trade_stores_zero_cash_delta_not_the_hypothetical_amount(self):
        # A rejected order never reaches the exchange -- no cash moves, so
        # its agent_actions row must record cash_delta=0, not the delta the
        # risk-guard preview computed before rejecting it. Storing that
        # hypothetical value here previously corrupted every downstream
        # cash-delta-based reconstruction (agent_trading_stats.agent_equity_curve).
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "0.15",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXCONC": 0.10}),
                ),
            ):
                first = benchmark_tools.place_trade(
                    {"ticker": "KXCONC", "side": "yes", "price": 0.10, "quantity": 100},
                    ctx,
                )
                second = benchmark_tools.place_trade(
                    {"ticker": "KXCONC", "side": "yes", "price": 0.10, "quantity": 60},
                    ctx,
                )
            conn = sqlite3.connect(db_path)
            try:
                cash_after = conn.execute(
                    "SELECT cash FROM agent_accounts WHERE agent_id = 'model-a'"
                ).fetchone()[0]
                rejected_row = conn.execute(
                    "SELECT cash_delta, cash_required FROM agent_actions "
                    "WHERE agent_id = 'model-a' AND action_type = 'rejected_trade'"
                ).fetchone()
            finally:
                conn.close()

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        # The would-be cost stays visible in risk_guard/cash_required (the
        # tool's own response and the audit row) -- only cash_delta, which
        # readers trust as "cash that actually moved", must be zero.
        self.assertGreater(second["risk_guard"]["cash_required"], 0.0)
        self.assertIsNotNone(rejected_row)
        self.assertEqual(rejected_row[0], 0.0)
        self.assertGreater(rejected_row[1], 0.0)
        # Confirms the rejection genuinely never touched cash.
        self.assertAlmostEqual(cash_after, 100.0 - first["risk_guard"]["cash_required"])

    def test_place_trade_rejects_orders_that_are_insolvent_after_fees(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "10",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXSOLV": 0.95}),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXSOLV", "side": "yes", "price": 0.95, "quantity": 10.5},
                    ctx,
                )

        self.assertFalse(result["ok"])
        self.assertTrue(result["rejected"])
        self.assertEqual(result["reason"], "solvency")
        self.assertGreater(result["risk_guard"]["cash_required"], result["risk_guard"]["cash_before"])

    def test_place_trade_netting_payout_counts_for_solvency(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "10",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes(
                        {"KXNET": {"yes_ask": 0.40, "no_ask": 0.59}}
                    ),
                ),
            ):
                opened = benchmark_tools.place_trade(
                    {"ticker": "KXNET", "side": "yes", "price": 0.40, "quantity": 10},
                    ctx,
                )
                closed = benchmark_tools.place_trade(
                    {"ticker": "KXNET", "side": "no", "price": 0.59, "quantity": 10},
                    ctx,
                )

        self.assertTrue(opened["ok"])
        self.assertTrue(closed["ok"])
        self.assertEqual(closed["risk_guard"]["netting_payout"], 10.0)
        self.assertEqual(closed["risk_guard"]["cash_required"], 0.0)
        self.assertEqual(closed["risk_guard"]["market_cost_basis_after"], 0.0)
        self.assertEqual(closed["account"]["n_open_positions"], 0)

    def test_place_trade_allows_a_large_quote_verified_netting_profit(self):
        # This codebase's netting-arb guard used to reject any netting close
        # realizing more than a flat $0.15/pair, regardless of why -- but
        # PredictionArena's own methodology (whose design this benchmark
        # follows) applies no such cap: realized PnL on a netting close is
        # simply payout minus both legs' cost (see its "Selling: Buy NO to
        # Sell YES" worked example). A YES+NO close at the SAME price on the
        # same ticker (as if a real market briefly summed to ~$0.24, not
        # ~$1.00) is a degenerate case that can't happen on Kalshi in
        # practice -- yes_ask/no_ask are derived from the same unified order
        # book -- but it's the simplest way to force a large arb_per_pair and
        # confirm the guard no longer rejects it, since the trade's price was
        # still quote-verified by _resolve_shadow_marketability (not
        # agent-guessed), so the resulting profit is trusted like any other.
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "1000",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXARB": 0.12}),
                ),
            ):
                opened = benchmark_tools.place_trade(
                    {"ticker": "KXARB", "side": "yes", "price": 0.12, "quantity": 10},
                    ctx,
                )
                closed = benchmark_tools.place_trade(
                    {"ticker": "KXARB", "side": "no", "price": 0.12, "quantity": 10},
                    ctx,
                )

        self.assertTrue(opened["ok"])
        self.assertTrue(closed["ok"])
        self.assertEqual(closed["risk_guard"]["netting_payout"], 10.0)
        # payout(10) - old_basis(1.2) - new_basis(1.2) - fee_alloc(0.07392),
        # realized in full, not capped at $0.15/pair.
        self.assertAlmostEqual(closed["account"]["realized_pnl"], 7.52608, places=4)

    def test_place_trade_shadow_fill_clamps_to_live_ask(self):
        # A marketable shadow order (price crosses the real ask) should fill
        # at the real ask, not at whatever more-aggressive price was asked
        # for -- mirrors how a real IOC order can't pay worse than the book.
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value={"yes_bid": 0.85, "yes_ask": 0.88},
                ),
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXQUOTED", "side": "yes", "price": 0.95, "quantity": 2},
                    ctx,
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["execution"]["filled_quantity"], 2.0)
        self.assertEqual(result["normalized_order"]["price"], 0.88)

    def test_place_trade_shadow_order_below_market_does_not_fill(self):
        # A shadow order priced below the real ask would never cross a real
        # book -- it should record zero fill, not "assume full" at a price
        # no real counterparty would take.
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value={"yes_bid": 0.85, "yes_ask": 0.88},
                ),
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXQUOTED", "side": "yes", "price": 0.10, "quantity": 2},
                    ctx,
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["execution"]["filled_quantity"], 0.0)
        self.assertEqual(result["execution"]["fill_status"], "shadow_unfilled_below_market")
        self.assertEqual(result["account"]["n_open_positions"], 0)

    def test_place_trade_does_not_fill_when_no_live_quote_is_available(self):
        # Regression: _resolve_shadow_marketability used to trust the
        # caller's price outright when a live quote couldn't be fetched,
        # reasoning that the netting-arb guard would catch abuse -- it
        # doesn't fully, since that guard only fires on the *closing* leg of
        # a netted pair. A single directional entry booked at a fabricated
        # price during a quote outage could sit open indefinitely, marked to
        # market later against a real quote and silently inflating the shown
        # unrealized P&L. No live quote now means no fill, same as a real IOC
        # order would see with no book to route against.
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
            }
            # setUp's default fetch_kalshi mock raises MarketDataError for
            # every ticker -- exercise that default directly rather than
            # overriding it, since it's exactly the "no quote available" case.
            with mock.patch.dict(os.environ, env, clear=False):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXNOQUOTE", "side": "yes", "price": 0.05, "quantity": 100},
                    ctx,
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["execution"]["filled_quantity"], 0.0)
        self.assertEqual(result["execution"]["fill_status"], "shadow_quote_unavailable")
        self.assertEqual(result["account"]["n_open_positions"], 0)

    def test_place_trade_rejects_per_cycle_spend_over_limit(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "0.009",  # $0.90 at $100 account value
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXCYCLE1": 0.40, "KXCYCLE2": 0.10}),
                ),
            ):
                first = benchmark_tools.place_trade(
                    {"ticker": "KXCYCLE1", "side": "yes", "price": 0.40, "quantity": 2},
                    ctx,
                )
                second = benchmark_tools.place_trade(
                    {"ticker": "KXCYCLE2", "side": "yes", "price": 0.10, "quantity": 1},
                    ctx,
                )

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertTrue(second["rejected"])
        self.assertEqual(second["reason"], "per_cycle_spend")
        self.assertGreater(
            second["risk_guard"]["cycle_spend_after"],
            second["risk_guard"]["per_cycle_spend_limit"],
        )

    def test_per_cycle_spend_limit_scales_with_account_value(self):
        # Regression: this used to be a flat dollar amount (DEFAULT was
        # $500/cycle regardless of account size) -- now it's a percentage, so
        # the same FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT must yield a
        # bigger dollar limit for a bigger account, not a fixed number.
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        def _cap_at(account_value: str) -> float:
            with tempfile.TemporaryDirectory() as td:
                env = {
                    "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                    "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                    "FORESEA_AGENT_ACCOUNT_VALUE": account_value,
                    "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                    "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "0.2",
                    "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                }
                with mock.patch.dict(os.environ, env, clear=False):
                    result = benchmark_tools.place_trade(
                        {"ticker": "KXSCALE", "side": "yes", "price": 0.10, "quantity": 1},
                        ctx,
                    )
            return result["risk_guard"]["per_cycle_spend_limit"]

        self.assertAlmostEqual(_cap_at("100"), 20.0)
        self.assertAlmostEqual(_cap_at("10000"), 2000.0)

    def test_place_trade_updates_weighted_average_entry_in_positions_table(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXAVG": [0.60, 0.70]}),
                ),
            ):
                first = benchmark_tools.place_trade(
                    {"ticker": "KXAVG", "side": "yes", "price": 0.60, "quantity": 10},
                    ctx,
                )
                second = benchmark_tools.place_trade(
                    {"ticker": "KXAVG", "side": "yes", "price": 0.70, "quantity": 5},
                    ctx,
                )
            conn = sqlite3.connect(db_path)
            try:
                pos = conn.execute(
                    """
                    SELECT quantity, cost_basis, avg_entry_price
                    FROM agent_positions
                    WHERE agent_id = 'model-a' AND ticker = 'KXAVG' AND side = 'yes'
                    """
                ).fetchone()
                actions = conn.execute("SELECT COUNT(*) FROM agent_actions").fetchone()[0]
            finally:
                conn.close()

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos[0], 15.0)
        self.assertAlmostEqual(pos[1], 9.5)
        self.assertAlmostEqual(pos[2], 9.5 / 15.0)
        self.assertEqual(actions, 2)

    def test_place_trade_settles_open_positions_before_new_cycle(self):
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            base_env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_SETTLEMENT_FEE_RATE": "0.014",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }

            def resolve(ticker):
                return 1 if ticker == "KXSETTLE" else None

            with (
                mock.patch.dict(os.environ, {**base_env, "FORESEA_AGENT_CYCLE_ID": "cycle-1"}, clear=False),
                mock.patch("analyzing_llm_rationale.market_data.resolve_kalshi", side_effect=resolve),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXSETTLE": 0.40}),
                ),
            ):
                opened = benchmark_tools.place_trade(
                    {"ticker": "KXSETTLE", "side": "yes", "price": 0.40, "quantity": 10},
                    ctx,
                )

            with (
                mock.patch.dict(os.environ, {**base_env, "FORESEA_AGENT_CYCLE_ID": "cycle-2"}, clear=False),
                mock.patch("analyzing_llm_rationale.market_data.resolve_kalshi", side_effect=resolve),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXOTHER": 0.10}),
                ),
            ):
                after_settlement = benchmark_tools.place_trade(
                    {"ticker": "KXOTHER", "side": "yes", "price": 0.10, "quantity": 1},
                    ctx,
                )

            conn = sqlite3.connect(db_path)
            try:
                settlement = conn.execute(
                    """
                    SELECT payout, settlement_fee, realized_pnl
                    FROM agent_actions
                    WHERE action_type = 'settlement' AND ticker = 'KXSETTLE'
                    """
                ).fetchone()
                remaining = conn.execute(
                    """
                    SELECT COUNT(*) FROM agent_positions
                    WHERE ticker = 'KXSETTLE'
                    """
                ).fetchone()[0]
            finally:
                conn.close()

        self.assertTrue(opened["ok"])
        self.assertTrue(after_settlement["ok"])
        self.assertEqual(after_settlement["risk_guard"]["settlements_before_trade"][0]["ticker"], "KXSETTLE")
        self.assertIsNotNone(settlement)
        self.assertAlmostEqual(settlement[0], 10.0)
        self.assertAlmostEqual(settlement[1], 0.14)
        self.assertAlmostEqual(settlement[2], 5.86)
        self.assertEqual(remaining, 0)

    def test_agent_cycles_table_round_trips(self):
        # Stores per-cycle thesis/transcript/candidates -- the source for the
        # agentic-trading transparency feed. Not written by any existing
        # function yet, so this just locks in the schema itself.
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path)}, clear=False
            ):
                conn = benchmark_tools._account_conn()
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO agent_cycles
                        (agent_id, cycle_id, ts, thesis, transcript_json, steps, truncated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "gpt-oss-120b", "2026-08-11T00:00", "2026-08-11T00:00:05+00:00",
                            "Held flat this cycle.", '{"candidates_offered": ["KXTEST"]}',
                            2, 0,
                        ),
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT agent_id, cycle_id, thesis, steps, truncated FROM agent_cycles"
                        " WHERE agent_id = ? AND cycle_id = ?",
                        ("gpt-oss-120b", "2026-08-11T00:00"),
                    ).fetchone()
                finally:
                    conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row["thesis"], "Held flat this cycle.")
        self.assertEqual(row["steps"], 2)
        self.assertEqual(row["truncated"], 0)

    # -- Datastore-backed account store (used when FORESEA_AGENT_ACCOUNT_DB_PATH
    # is unset -- the live Cloud Run case, which has no persistent volume for
    # the SQLite default). These mirror the equivalent SQLite-path tests
    # above to confirm the two backends behave identically from place_trade's
    # perspective.

    def test_place_trade_datastore_backend_nets_positions(self):
        client = _install_fake_datastore(self)
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "10",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes(
                        {"KXDSNET": {"yes_ask": 0.40, "no_ask": 0.59}}
                    ),
                ),
            ):
                self.assertNotIn("FORESEA_AGENT_ACCOUNT_DB_PATH", os.environ)
                opened = benchmark_tools.place_trade(
                    {"ticker": "KXDSNET", "side": "yes", "price": 0.40, "quantity": 10},
                    ctx,
                )
                closed = benchmark_tools.place_trade(
                    {"ticker": "KXDSNET", "side": "no", "price": 0.59, "quantity": 10},
                    ctx,
                )

        self.assertTrue(opened["ok"])
        self.assertTrue(closed["ok"])
        self.assertEqual(closed["risk_guard"]["netting_payout"], 10.0)
        self.assertEqual(closed["risk_guard"]["cash_required"], 0.0)
        self.assertEqual(closed["account"]["n_open_positions"], 0)
        # Confirms the fake client actually persisted to its store (not just
        # that place_trade's in-memory response looked right), and that the
        # two agree with each other.
        account_entity = client.get(client.key("AgentTradingAccount", "model-a"))
        self.assertAlmostEqual(float(account_entity["cash"]), closed["account"]["cash"])

    def test_place_trade_datastore_backend_allows_a_large_quote_verified_netting_profit(self):
        # Datastore-backend twin of the SQLite test above -- same degenerate
        # same-price quote, confirming the guard's removal applies to both
        # account-store backends, not just the default SQLite one.
        _install_fake_datastore(self)
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "1000",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_CYCLE_ID": "cycle-1",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXDSARB": 0.12}),
                ),
            ):
                opened = benchmark_tools.place_trade(
                    {"ticker": "KXDSARB", "side": "yes", "price": 0.12, "quantity": 10},
                    ctx,
                )
                closed = benchmark_tools.place_trade(
                    {"ticker": "KXDSARB", "side": "no", "price": 0.12, "quantity": 10},
                    ctx,
                )

        self.assertTrue(opened["ok"])
        self.assertTrue(closed["ok"])
        self.assertEqual(closed["risk_guard"]["netting_payout"], 10.0)
        self.assertAlmostEqual(closed["account"]["realized_pnl"], 7.52608, places=4)

    def test_place_trade_datastore_backend_settles_open_positions(self):
        _install_fake_datastore(self)
        ctx = benchmark_tools.ToolContext(agent_id="model-a")

        with tempfile.TemporaryDirectory() as td:
            base_env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_VALUE": "100",
                "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
                "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
                "FORESEA_AGENT_SETTLEMENT_FEE_RATE": "0.014",
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }

            def resolve(ticker):
                return 1 if ticker == "KXDSSETTLE" else None

            with (
                mock.patch.dict(os.environ, {**base_env, "FORESEA_AGENT_CYCLE_ID": "cycle-1"}, clear=False),
                mock.patch("analyzing_llm_rationale.market_data.resolve_kalshi", side_effect=resolve),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXDSSETTLE": 0.40}),
                ),
            ):
                opened = benchmark_tools.place_trade(
                    {"ticker": "KXDSSETTLE", "side": "yes", "price": 0.40, "quantity": 10},
                    ctx,
                )

            with (
                mock.patch.dict(os.environ, {**base_env, "FORESEA_AGENT_CYCLE_ID": "cycle-2"}, clear=False),
                mock.patch("analyzing_llm_rationale.market_data.resolve_kalshi", side_effect=resolve),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({"KXDSOTHER": 0.10}),
                ),
            ):
                after_settlement = benchmark_tools.place_trade(
                    {"ticker": "KXDSOTHER", "side": "yes", "price": 0.10, "quantity": 1},
                    ctx,
                )

        self.assertTrue(opened["ok"])
        self.assertTrue(after_settlement["ok"])
        self.assertEqual(
            after_settlement["risk_guard"]["settlements_before_trade"][0]["ticker"], "KXDSSETTLE"
        )
        settlement = after_settlement["risk_guard"]["settlements_before_trade"][0]
        self.assertAlmostEqual(settlement["payout"], 10.0)
        self.assertAlmostEqual(settlement["settlement_fee"], 0.14)
        self.assertAlmostEqual(settlement["realized_pnl"], 5.86)


class KalshiTakerFeeRateTests(unittest.TestCase):
    """place_trade only ever takes liquidity (immediate-or-cancel, no
    resting orders), so _kalshi_fee should prefer Kalshi's own live taker
    rate over the flat KALSHI_FEE_COEFFICIENT estimate when it's available,
    and fall back safely (never raise) when it isn't."""

    def _fresh_cache(self):
        return mock.patch.dict(benchmark_tools._KALSHI_TAKER_FEE_RATE_CACHE, {}, clear=True)

    def test_parses_a_flat_taker_fee_rate(self):
        self.assertEqual(benchmark_tools._parse_taker_fee_rate({"taker_fee_rates": 0.05}), 0.05)

    def test_parses_the_first_tier_from_a_tiered_list(self):
        tiers = {"taker_fee_rates": [{"rate": 0.06}, {"rate": 0.02}]}
        self.assertEqual(benchmark_tools._parse_taker_fee_rate(tiers), 0.06)

    def test_returns_none_for_an_unparseable_or_missing_shape(self):
        self.assertIsNone(benchmark_tools._parse_taker_fee_rate({"taker_fee_rates": "garbage"}))
        self.assertIsNone(benchmark_tools._parse_taker_fee_rate({}))
        self.assertIsNone(benchmark_tools._parse_taker_fee_rate({"taker_fee_rates": [{"rate": "n/a"}]}))

    def test_kalshi_fee_applies_the_live_rate_directly_to_notional(self):
        with (
            self._fresh_cache(),
            mock.patch(
                "analyzing_llm_rationale.trading.get_kalshi_fee_tiers",
                return_value={"taker_fee_rates": 0.05},
            ),
        ):
            fee = benchmark_tools._kalshi_fee(0.40, 10)
        # rate * price * quantity, NOT the parabolic price*(1-price) estimate.
        self.assertAlmostEqual(fee, 0.05 * 0.40 * 10)

    def test_kalshi_fee_falls_back_to_the_estimate_when_the_lookup_fails(self):
        with (
            self._fresh_cache(),
            mock.patch(
                "analyzing_llm_rationale.trading.get_kalshi_fee_tiers",
                side_effect=RuntimeError("KALSHI_API_KEY_ID is not configured."),
            ),
        ):
            fee = benchmark_tools._kalshi_fee(0.40, 10)
        self.assertAlmostEqual(
            fee, benchmark_tools.KALSHI_FEE_COEFFICIENT * 10 * 0.40 * (1.0 - 0.40)
        )

    def test_kalshi_fee_falls_back_when_the_response_has_no_usable_rate(self):
        with (
            self._fresh_cache(),
            mock.patch(
                "analyzing_llm_rationale.trading.get_kalshi_fee_tiers",
                return_value={"unrelated_field": 1},
            ),
        ):
            fee = benchmark_tools._kalshi_fee(0.40, 10)
        self.assertAlmostEqual(
            fee, benchmark_tools.KALSHI_FEE_COEFFICIENT * 10 * 0.40 * (1.0 - 0.40)
        )

    def test_live_rate_lookup_is_memoized_within_a_process(self):
        with (
            self._fresh_cache(),
            mock.patch(
                "analyzing_llm_rationale.trading.get_kalshi_fee_tiers",
                return_value={"taker_fee_rates": 0.05},
            ) as get_tiers,
        ):
            benchmark_tools._kalshi_taker_fee_rate()
            benchmark_tools._kalshi_taker_fee_rate()
        self.assertEqual(get_tiers.call_count, 1)


if __name__ == "__main__":
    unittest.main()
