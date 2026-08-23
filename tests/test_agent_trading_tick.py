from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "agent_trading_tick.py"
_SPEC = importlib.util.spec_from_file_location("agent_trading_tick_test_module", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
agent_trading_tick = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(agent_trading_tick)

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402


def _quote(ident, question="Q?", bid=0.4, ask=0.45, close="2026-09-01T00:00:00Z",
           opens="2026-05-01T00:00:00Z", resolution_criteria=None):
    quote = {
        "platform": "Kalshi", "ident": ident, "question": question,
        "probability": (bid + ask) / 2, "yes_bid": bid, "yes_ask": ask,
        "close_time": close, "created_time": opens,
    }
    if resolution_criteria is not None:
        quote["resolution_criteria"] = resolution_criteria
    return quote


class ShadowModeAssertionTests(unittest.TestCase):
    def test_default_is_shadow_and_passes(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORESEA_AGENT_PLACE_TRADE_MODE", None)
            agent_trading_tick._assert_shadow_mode()  # must not raise

    def test_explicit_shadow_passes(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_PLACE_TRADE_MODE": "shadow"}, clear=False):
            agent_trading_tick._assert_shadow_mode()  # must not raise

    def test_live_mode_raises(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_PLACE_TRADE_MODE": "live"}, clear=False):
            with self.assertRaises(RuntimeError):
                agent_trading_tick._assert_shadow_mode()


class MaxOrderNotionalTests(unittest.TestCase):
    def test_defaults_to_8pct_of_the_10k_default_account_value(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in ("FORESEA_AGENT_ACCOUNT_VALUE", "FORESEA_MAX_ORDER_NOTIONAL"):
                os.environ.pop(key, None)
            result = agent_trading_tick._configure_max_order_notional()
            self.assertAlmostEqual(result, 800.0)
            self.assertEqual(os.environ["FORESEA_MAX_ORDER_NOTIONAL"], "800.0")

    def test_scales_with_a_custom_account_value(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_ACCOUNT_VALUE": "20000"}, clear=False):
            result = agent_trading_tick._configure_max_order_notional()
        self.assertAlmostEqual(result, 1600.0)

    def test_pct_itself_is_env_overridable(self):
        # MAX_ORDER_NOTIONAL_PCT is a module-level constant read once at
        # import time (like every other env-derived constant in this file:
        # CANDIDATE_COUNT, MAX_TOOL_STEPS, ...), so overriding it per-test
        # means patching the loaded attribute directly, not os.environ.
        with (
            mock.patch.dict(os.environ, {"FORESEA_AGENT_ACCOUNT_VALUE": "10000"}, clear=False),
            mock.patch.object(agent_trading_tick, "MAX_ORDER_NOTIONAL_PCT", 0.05),
        ):
            result = agent_trading_tick._configure_max_order_notional()
        self.assertAlmostEqual(result, 500.0)

    def test_scoped_to_this_process_not_a_global_trading_default(self):
        # This must override trading.py's own module-level default only via
        # the env var it reads live at guard-check time -- never mutate the
        # shared DEFAULT_MAX_ORDER_NOTIONAL constant itself, or every other
        # trading path (human BYO trading included) would inherit it too.
        from analyzing_llm_rationale import trading

        with mock.patch.dict(os.environ, {"FORESEA_AGENT_ACCOUNT_VALUE": "10000"}, clear=False):
            agent_trading_tick._configure_max_order_notional()
            self.assertEqual(trading._max_order_notional(), Decimal("800.0"))
        self.assertEqual(trading.DEFAULT_MAX_ORDER_NOTIONAL, Decimal("50"))


class CurrentAccountValueTests(unittest.TestCase):
    def test_no_open_positions_is_just_cash(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}, clear=False
            ):
                with benchmark_tools._account_transaction() as conn:
                    value = agent_trading_tick._current_account_value(conn, "model-a", [])
        self.assertAlmostEqual(value, benchmark_tools.DEFAULT_AGENT_ACCOUNT_VALUE)

    def test_marks_an_open_position_to_the_live_bid_not_cost_basis(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}, clear=False
            ):
                ctx = benchmark_tools.ToolContext(agent_id="model-a")
                # Opens 10 contracts @ 0.40 (cost basis $4); the market has
                # since moved to a 0.60 bid -- the guards must see the
                # current $6 mark, not the stale $4 cost basis.
                with mock.patch.object(
                    market_data, "fetch_kalshi", return_value=_quote("KXTEST", bid=0.38, ask=0.40),
                ):
                    benchmark_tools.place_trade(
                        {"ticker": "KXTEST", "side": "yes", "price": 0.40, "quantity": 10}, ctx,
                    )
                with benchmark_tools._account_transaction() as conn:
                    value = agent_trading_tick._current_account_value(
                        conn, "model-a", [_quote("KXTEST", bid=0.60, ask=0.62)]
                    )
        # starting cash ($10000) - cost basis ($4) - fee ($0.168), plus the
        # position marked to the live $6 bid (10 contracts @ 0.60).
        self.assertGreater(value, benchmark_tools.DEFAULT_AGENT_ACCOUNT_VALUE - 4.0)
        self.assertAlmostEqual(value, benchmark_tools.DEFAULT_AGENT_ACCOUNT_VALUE - 4.0 - 0.168 + 6.0, places=2)

    def test_falls_back_to_cost_basis_when_no_live_quote_is_available(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}, clear=False
            ):
                ctx = benchmark_tools.ToolContext(agent_id="model-a")
                with mock.patch.object(
                    market_data, "fetch_kalshi", return_value=_quote("KXTEST", bid=0.38, ask=0.40),
                ):
                    benchmark_tools.place_trade(
                        {"ticker": "KXTEST", "side": "yes", "price": 0.40, "quantity": 10}, ctx,
                    )
                with benchmark_tools._account_transaction() as conn:
                    # held_quotes is empty -- as if the re-quote fetch failed.
                    value = agent_trading_tick._current_account_value(conn, "model-a", [])
        self.assertAlmostEqual(value, benchmark_tools.DEFAULT_AGENT_ACCOUNT_VALUE - 0.168)


def _poly_quote(ident, question="Q?", bid=0.4, ask=0.45, close="2026-09-01T00:00:00Z",
                 opens="2026-05-01T00:00:00Z"):
    return {
        "platform": "Polymarket", "ident": ident, "question": question,
        "probability": (bid + ask) / 2, "yes_bid": bid, "yes_ask": ask,
        "close_time": close, "created_time": opens,
    }


class CandidateSelectionTests(unittest.TestCase):
    def test_discover_candidates_excludes_known_tickers_and_caps_count(self):
        listed = [_quote("KXA"), _quote("KXB"), _quote("KXC"), _quote("KXD")]
        with (
            mock.patch.object(market_data, "list_kalshi", return_value=listed),
            mock.patch.object(market_data, "list_polymarket", return_value=[]),
            mock.patch.object(agent_trading_tick, "CANDIDATE_COUNT", 2),
        ):
            found = agent_trading_tick._discover_candidates({"KXA"})
        self.assertEqual([q["ident"] for q in found], ["KXB", "KXC"])

    def test_discover_candidates_skips_unpriced_markets(self):
        unpriced = dict(_quote("KXE"))
        unpriced["probability"] = None
        with (
            mock.patch.object(market_data, "list_kalshi", return_value=[unpriced, _quote("KXF")]),
            mock.patch.object(market_data, "list_polymarket", return_value=[]),
        ):
            found = agent_trading_tick._discover_candidates(set())
        self.assertEqual([q["ident"] for q in found], ["KXF"])

    def test_discover_candidates_survives_market_data_error(self):
        with (
            mock.patch.object(
                market_data, "list_kalshi", side_effect=market_data.MarketDataError("boom")
            ),
            mock.patch.object(market_data, "list_polymarket", return_value=[]),
        ):
            found = agent_trading_tick._discover_candidates(set())
        self.assertEqual(found, [])

    def test_discover_candidates_paginates_kalshi_listing(self):
        # Regression test: Kalshi's /events page isn't sorted by close_time, so
        # without paginate=True the unpaginated first page can (and, observed
        # live, does) contain zero markets in the close-day window even though
        # thousands of qualifying markets exist on later pages -- silently
        # starving every cycle of candidates.
        with (
            mock.patch.object(market_data, "list_kalshi", return_value=[_quote("KXA")]) as mocked,
            mock.patch.object(market_data, "list_polymarket", return_value=[]),
        ):
            agent_trading_tick._discover_candidates(set())
        self.assertTrue(mocked.call_args.kwargs.get("paginate") is True)

    def test_discover_candidates_round_robins_across_both_venues(self):
        # A shortfall in one venue's listing must not starve the other's --
        # both venues get a look-in every round, in venue order.
        with (
            mock.patch.object(market_data, "list_kalshi", return_value=[_quote("KXA"), _quote("KXB")]),
            mock.patch.object(market_data, "list_polymarket", return_value=[_poly_quote("poly-a"), _poly_quote("poly-b")]),
            mock.patch.object(agent_trading_tick, "CANDIDATE_COUNT", 3),
        ):
            found = agent_trading_tick._discover_candidates(set())
        self.assertEqual([q["ident"] for q in found], ["KXA", "poly-a", "KXB"])

    def test_discover_candidates_fills_from_the_other_venue_when_one_is_short(self):
        with (
            mock.patch.object(market_data, "list_kalshi", return_value=[_quote("KXA")]),
            mock.patch.object(market_data, "list_polymarket", return_value=[_poly_quote("poly-a"), _poly_quote("poly-b")]),
            mock.patch.object(agent_trading_tick, "CANDIDATE_COUNT", 3),
        ):
            found = agent_trading_tick._discover_candidates(set())
        self.assertEqual([q["ident"] for q in found], ["KXA", "poly-a", "poly-b"])

    def test_discover_candidates_survives_polymarket_market_data_error(self):
        with (
            mock.patch.object(market_data, "list_kalshi", return_value=[_quote("KXA")]),
            mock.patch.object(
                market_data, "list_polymarket", side_effect=market_data.MarketDataError("boom")
            ),
        ):
            found = agent_trading_tick._discover_candidates(set())
        self.assertEqual([q["ident"] for q in found], ["KXA"])

    def test_requote_held_skips_failed_lookups(self):
        def fake_fetch(ticker):
            if ticker == "KXBAD":
                raise market_data.MarketDataError("gone")
            return _quote(ticker)

        with mock.patch.object(market_data, "fetch_kalshi", side_effect=fake_fetch):
            quotes = agent_trading_tick._requote_held([("kalshi", "KXGOOD"), ("kalshi", "KXBAD")])
        self.assertEqual([q["ident"] for q in quotes], ["KXGOOD"])

    def test_requote_held_routes_polymarket_positions_to_fetch_polymarket(self):
        with (
            mock.patch.object(market_data, "fetch_kalshi", return_value=_quote("KXGOOD")),
            mock.patch.object(market_data, "fetch_polymarket", return_value=_poly_quote("poly-a")) as poly_mock,
        ):
            quotes = agent_trading_tick._requote_held(
                [("kalshi", "KXGOOD"), ("polymarket", "poly-a")]
            )
        self.assertEqual([q["ident"] for q in quotes], ["KXGOOD", "poly-a"])
        poly_mock.assert_called_once_with(slug="poly-a")


class CandidateLineFormattingTests(unittest.TestCase):
    def test_candidate_line_shows_both_ends_of_the_resolution_window(self):
        # Regression: a real live trade found the ticker's close date but not
        # its open date, so agents couldn't tell that real news they found
        # (e.g. a cabinet departure) happened *before* this specific market's
        # window started -- and bought a losing position on real-but-stale
        # evidence that actually belonged to an earlier, already-resolved
        # ticker in the same recurring series (KXFOO-26APR vs
        # KXFOO-26MAY22-26SEP).
        line = agent_trading_tick._fmt_candidate_line(
            _quote("KXFOO", opens="2026-05-22T18:00:00Z", close="2026-09-02T03:59:00Z")
        )
        self.assertIn("2026-05-22T18:00:00Z", line)
        self.assertIn("2026-09-02T03:59:00Z", line)
        self.assertIn("resolution window", line)

    def test_candidate_line_handles_a_missing_open_date(self):
        quote = _quote("KXBAR")
        del quote["created_time"]
        line = agent_trading_tick._fmt_candidate_line(quote)
        self.assertIn("unknown", line)

    def test_candidate_line_shows_the_venue_tag(self):
        kalshi_line = agent_trading_tick._fmt_candidate_line(_quote("KXFOO"))
        poly_line = agent_trading_tick._fmt_candidate_line(_poly_quote("some-poly-slug"))
        self.assertIn("[kalshi]", kalshi_line)
        self.assertIn("[polymarket]", poly_line)

    def test_candidate_line_shows_the_derived_no_price_not_just_yes(self):
        # Regression: a real live position was rejected trying to close a
        # YES holding by buying NO at ~its YES entry price (implausible
        # netting arb) -- the candidate line only ever showed yes bid/ask,
        # so the agent had no live NO price to read and had to guess one
        # instead of pricing off the real market. NO isn't a field Kalshi
        # returns; it's derived from the YES book the same way
        # place_trade's own guards derive it (accounting.MarketQuote), so
        # what's shown here is guaranteed to match what execution checks.
        line = agent_trading_tick._fmt_candidate_line(_quote("KXFOO", bid=0.40, ask=0.45))
        self.assertIn("yes bid/ask 0.40/0.45", line)
        self.assertIn("no bid/ask 0.55/0.60", line)

    def test_candidate_line_surfaces_the_venue_s_resolution_rules(self):
        # Regression: resolution_criteria was already fetched fresh every
        # cycle by market_data.py's fetch_kalshi/fetch_polymarket (it's
        # right there on every quote), but never actually shown to the
        # agent -- which had to infer what qualifies from the question text
        # alone. Observed live: an agent reasoned a cabinet-departure market
        # should resolve YES for a specific person's resignation, when that
        # market's own rules explicitly excluded that person's case.
        line = agent_trading_tick._fmt_candidate_line(
            _quote("KXCABLEAVE", resolution_criteria="Excludes departures by Person X specifically.")
        )
        self.assertIn("Resolution rules:", line)
        self.assertIn("Excludes departures by Person X specifically.", line)

    def test_candidate_line_omits_the_rules_line_when_none_is_available(self):
        # Not every quote has rules text (a venue outage, a market missing
        # it) -- must degrade to the old line, not print an empty/None rules
        # line that looks like a real "no restrictions" answer.
        line = agent_trading_tick._fmt_candidate_line(_quote("KXFOO"))
        self.assertNotIn("Resolution rules:", line)

    def test_candidate_line_supplies_full_resolution_rules_text(self):
        full_rules = "All legal definitions, exclusions, and criteria: " + ("rule_detail " * 200).strip()
        line = agent_trading_tick._fmt_candidate_line(
            _quote("KXFOO", resolution_criteria=full_rules)
        )
        self.assertIn("Resolution rules: " + full_rules, line)

    def test_candidate_line_shows_kalshi_expected_resolution_when_available(self):
        quote = _quote("KXFOO")
        quote["expected_expiration_time"] = "2026-08-21T00:00:00Z"

        line = agent_trading_tick._fmt_candidate_line(quote)

        self.assertIn("Expected underlying resolution: 2026-08-21T00:00:00Z", line)


class PaperCalibrationContextTests(unittest.TestCase):
    def test_uses_kalshi_calibration_evidence_as_a_net_of_cost_baseline(self):
        quote = _quote("KXMARKET", question="Will the next CPI print exceed expectations?")
        quote.update({"category": "Economics", "platform": "Kalshi", "close_time": "2026-08-21T00:00:00Z", "volume": 1000})

        context = agent_trading_tick._paper_calibration_context(
            quote, now=agent_trading_tick.datetime(2026, 8, 20, tzinfo=agent_trading_tick.timezone.utc)
        )

        self.assertIn("Kalshi Research, Calibration in Prediction Markets", context)
        self.assertIn("true resolution time", context)
        self.assertIn("derive P(YES) independently", context)
        self.assertIn("spread, fees", context)

    def test_uses_expected_expiration_not_administrative_close_for_kalshi_timing(self):
        quote = _quote("KXMARKET", question="Will Team A win this game?")
        quote.update({
            "category": "Sports",
            "platform": "Kalshi",
            "close_time": "2026-08-20T01:00:00Z",
            "expected_expiration_time": "2026-08-21T00:00:00Z",
        })

        self.assertEqual(
            agent_trading_tick._calibration_resolution_time(quote),
            "2026-08-21T00:00:00Z",
        )
        context = agent_trading_tick._paper_calibration_context(
            quote, now=agent_trading_tick.datetime(2026, 8, 20, tzinfo=agent_trading_tick.timezone.utc)
        )
        self.assertIn("expected expiration", context)

    def test_flags_kalshi_quotes_without_reported_volume(self):
        quote = _quote("KXMARKET", question="Will Team A win this game?")
        quote.update({"category": "Sports", "platform": "Kalshi", "close_time": "2026-08-21T00:00:00Z"})

        context = agent_trading_tick._paper_calibration_context(
            quote, now=agent_trading_tick.datetime(2026, 8, 20, tzinfo=agent_trading_tick.timezone.utc)
        )

        self.assertIn("No positive reported volume", context)

    def test_adds_political_compression_prior_without_mechanically_repricing(self):
        quote = _quote("KXPRESIDENT", question="Will the next president win reelection?")
        quote.update({"category": "Politics", "platform": "Kalshi", "close_time": "2026-10-01T00:00:00Z"})

        context = agent_trading_tick._paper_calibration_context(
            quote, now=agent_trading_tick.datetime(2026, 8, 20, tzinfo=agent_trading_tick.timezone.utc)
        )

        self.assertIn("Political prices were persistently compressed", context)
        self.assertIn("do not mechanically extremise", context)
        self.assertIn("large political prints", context)

    def test_adds_short_horizon_weather_caution(self):
        quote = _quote("KXWEATHER", question="Will New York temperature exceed 90F?")
        quote.update({"category": "Weather", "close_time": "2026-08-21T00:00:00Z"})

        context = agent_trading_tick._paper_calibration_context(
            quote, now=agent_trading_tick.datetime(2026, 8, 20, tzinfo=agent_trading_tick.timezone.utc)
        )

        self.assertIn("Short-horizon weather prices", context)
        self.assertIn("independent evidence", context)

    def test_omits_an_unvalidated_non_kalshi_domain_horizon_combination(self):
        quote = _quote("KXOTHER", question="Will an unrelated custom event happen?")
        quote.update({"category": "Other", "platform": "Polymarket", "close_time": "2026-08-23T00:00:00Z"})

        self.assertEqual(
            agent_trading_tick._paper_calibration_context(
                quote, now=agent_trading_tick.datetime(2026, 8, 20, tzinfo=agent_trading_tick.timezone.utc)
            ),
            "",
        )


class ModelBackstopCharsTests(unittest.TestCase):
    def test_uses_the_model_s_verified_context_window_when_known(self):
        # glm-5.2-fp8 is the only one of the ten agent-trading models whose
        # deployed context length SCADS AI's own /v1/models listing actually
        # publishes (524288 tokens, checked 2026-08-18). Half reserved for
        # the rest of the real prompt (system prompt, ~17 tool specs, ReAct
        # loop history), 4 chars/token.
        self.assertEqual(
            agent_trading_tick._model_backstop_chars("glm-5.2-fp8"),
            int(524288 * 4 * 0.5),
        )

    def test_falls_back_to_a_conservative_default_for_unverified_models(self):
        # Every other model, including all three scads-alias-* aliases
        # (SCADS doesn't publish what they route to), gets one shared
        # conservative default rather than a guessed per-model figure that
        # could overestimate real capacity.
        for model in ("llama-3.3-70b-instruct", "scads-alias-reasoning", "scads-alias-code", "scads-alias-ha"):
            with self.subTest(model=model):
                self.assertEqual(agent_trading_tick._model_backstop_chars(model), int(128000 * 4 * 0.5))

    def test_falls_back_for_an_unrecognized_model_name(self):
        # Must never raise -- an unknown model (e.g. a typo'd env var) still
        # needs a usable backstop, not a crash before the cycle even starts.
        self.assertEqual(agent_trading_tick._model_backstop_chars("not-a-real-model"), int(128000 * 4 * 0.5))


class BuildQuestionTests(unittest.TestCase):
    def test_instructs_checking_the_resolution_window_before_trading_on_news(self):
        question = agent_trading_tick._build_question("PORTFOLIO", "CANDIDATES")
        self.assertIn("resolution window", question)
        self.assertIn("different date ranges", question)

    def test_fits_under_the_hard_server_limit_in_the_normal_case(self):
        candidates = agent_trading_tick._build_candidates_block(
            [], [_quote(f"KX{i}") for i in range(3)],
        )
        question = agent_trading_tick._build_question("PORTFOLIO", candidates)
        # server.py no longer caps AgentAnalyzeRequest.question at all;
        # MAX_QUESTION_CHARS is a pure sanity backstop, not a real limit.
        self.assertLessEqual(len(question), agent_trading_tick.MAX_QUESTION_CHARS)

    def test_does_not_trim_a_realistic_candidates_block_at_all(self):
        # Regression: the old 1900-char budget (server-side hard limit was
        # 2000) was so tight relative to the fixed instruction text that a
        # single held position plus a handful of offered candidates
        # routinely triggered trimming -- and since trimming cut raw
        # characters rather than whole lines, it silently destroyed almost
        # the entire candidates block, including every Polymarket candidate,
        # down to a mid-word fragment of the first held position. Agents
        # were never actually shown the candidates they were supposedly
        # being offered. Confirmed live 2026-08-18. With real headroom, a
        # normal cycle (1 held position, 3 new candidates) must survive
        # completely intact.
        held = [_quote("KXHELD", question="An existing held position")]
        new = [_quote(f"KX{i}", question=f"A brand new candidate market number {i}") for i in range(3)]
        candidates = agent_trading_tick._build_candidates_block(held, new)
        portfolio = (
            "=== Your portfolio (shadow account -- paper trading, no real money) ===\n"
            "Cash: $9500.00 | Starting cash: $10000.00\n"
            "Realized P&L: $0.00 | Fees paid so far: $2.00\n"
            "Open positions:\n"
            "  - KXHELD yes: 10.0 contracts, avg entry 0.42, cost basis $4.20\n"
            "Your own reasoning from the previous cycle: No compelling edge, holding steady."
        )
        question = agent_trading_tick._build_question(portfolio, candidates)
        for q in held + new:
            self.assertIn(q["ident"], question)
        self.assertNotIn("more omitted for space", question)

    def test_trims_the_candidates_block_by_whole_lines_not_mid_line(self):
        # A genuinely pathological number of candidates still has to fit
        # somewhere -- verify the overflow is handled by dropping whole
        # trailing lines (leaving every surviving candidate fully readable),
        # never by slicing raw characters mid-line.
        new = [_quote(f"KX{i}", question=f"Candidate market number {i} with a fairly long question text") for i in range(1500)]
        candidates = agent_trading_tick._build_candidates_block([], new)
        question = agent_trading_tick._build_question("PORTFOLIO", candidates)
        self.assertLessEqual(len(question), agent_trading_tick.MAX_QUESTION_CHARS)
        self.assertIn(agent_trading_tick._TRADING_INSTRUCTION, question)
        self.assertIn("(more omitted for space)", question)
        # Every candidate line that DID survive must be a complete, intact
        # line, not a fragment cut off mid-word.
        for line in question.splitlines():
            if line.strip().startswith("- [kalshi] KX"):
                self.assertIn("resolution window", line, f"truncated mid-line: {line!r}")

    def test_never_exceeds_the_limit_when_portfolio_block_alone_is_huge(self):
        # Regression: a real portfolio_block (many open positions + a
        # near-max-length previous-cycle reasoning excerpt) can exceed the
        # budget on its own, even with an empty candidates block -- must not
        # be allowed to silently return an overlong question, breaking every
        # subsequent cycle with AgentAnalyzeRequest's server-side validation
        # error (observed live for 4 of 10 models on 2026-08-18).
        huge_portfolio = (
            "=== Your portfolio ===\n"
            + "\n".join(f"  - KXTEST{i} yes: 10.0 contracts, avg entry 0.42" for i in range(6000))
            + "\nYour own reasoning from the previous cycle: " + ("z" * 2000)
        )
        question = agent_trading_tick._build_question(huge_portfolio, "")
        self.assertLessEqual(len(question), agent_trading_tick.MAX_QUESTION_CHARS)
        self.assertIn(agent_trading_tick._TRADING_INSTRUCTION, question)

    def test_never_exceeds_the_limit_with_huge_portfolio_and_huge_candidates(self):
        # Forces the absolute-last-resort clamp: single unsplittable "lines"
        # (no newlines) too large to keep even one of, so both blocks trim
        # away to nothing and the final hard clamp on the portfolio text
        # itself must still guarantee the limit is never exceeded.
        huge_portfolio = "y" * 150000
        huge_candidates = "x" * 150000
        question = agent_trading_tick._build_question(huge_portfolio, huge_candidates)
        self.assertLessEqual(len(question), agent_trading_tick.MAX_QUESTION_CHARS)
        self.assertIn(agent_trading_tick._TRADING_INSTRUCTION, question)


class PortfolioBlockTests(unittest.TestCase):
    def test_portfolio_block_reports_cash_and_positions(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}, clear=False
            ):
                ctx = benchmark_tools.ToolContext(agent_id="model-a")
                with mock.patch.object(
                    market_data, "fetch_kalshi", return_value=_quote("KXTEST", bid=0.40, ask=0.42),
                ):
                    benchmark_tools.place_trade(
                        {"ticker": "KXTEST", "side": "yes", "price": 0.42, "quantity": 2}, ctx,
                    )
                with benchmark_tools._account_transaction() as conn:
                    block = agent_trading_tick._build_portfolio_block(conn, "model-a", "Held flat last time.")

        self.assertIn("KXTEST", block)
        self.assertIn("Held flat last time.", block)
        self.assertIn("Open positions:", block)

    def test_portfolio_block_reports_no_positions(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}, clear=False
            ):
                with benchmark_tools._account_transaction() as conn:
                    block = agent_trading_tick._build_portfolio_block(conn, "model-b", None)

        self.assertIn("Open positions: none.", block)
        self.assertNotIn("previous cycle", block)

    def test_portfolio_block_supplies_full_last_thesis(self):
        full_thesis = "x" * 2500
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}, clear=False
            ):
                with benchmark_tools._account_transaction() as conn:
                    block = agent_trading_tick._build_portfolio_block(conn, "model-c", full_thesis)

        self.assertIn(full_thesis, block)
        self.assertIn("Your own reasoning from the previous cycle:", block)


class LearningContextTests(unittest.TestCase):
    def test_refreshes_each_realized_action_once_and_builds_calibration_context(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path)}, clear=False
            ):
                with benchmark_tools._account_transaction() as conn:
                    benchmark_tools._record_account_action(
                        conn,
                        agent_id="model-learning",
                        action_type="settlement",
                        cycle_id="settled-cycle",
                        platform="kalshi",
                        ticker="KXLEARN",
                        side="yes",
                        realized_pnl=-12.5,
                        outcome="no",
                    )
                    self.assertEqual(agent_trading_tick._refresh_learning(conn, "model-learning"), 1)
                    # A retry must not add a duplicate lesson for the same immutable action.
                    self.assertEqual(agent_trading_tick._refresh_learning(conn, "model-learning"), 0)
                    block = agent_trading_tick._build_learning_block(conn, "model-learning")
                    learning_count = conn.execute(
                        "SELECT COUNT(*) FROM agent_learning WHERE agent_id = ?", ("model-learning",)
                    ).fetchone()[0]

        self.assertEqual(learning_count, 1)
        self.assertIn("1 loss-making", block)
        self.assertIn("KXLEARN", block)
        self.assertIn("resolution rules", block)
        self.assertIn("Risk caps and eligibility rules are unchanged", block)

    def test_learning_context_is_visible_to_the_next_agent_turn(self):
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_NOTES_PATH": str(Path(td) / "notes.json"),
                "FORESEA_AGENT_CYCLE_ID": "learning-cycle",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with benchmark_tools._account_transaction() as conn:
                    benchmark_tools._record_account_action(
                        conn,
                        agent_id="model-learning-turn",
                        action_type="trade",
                        cycle_id="prior-cycle",
                        platform="polymarket",
                        ticker="learn-market",
                        side="yes",
                        realized_pairs=5,
                        realized_pnl=4.25,
                        outcome="realized",
                    )
                with (
                    mock.patch.object(agent_trading_tick, "_init_local_agent"),
                    mock.patch.object(benchmark_tools, "_settle_agent_open_positions", return_value=[]),
                    mock.patch.object(market_data, "list_kalshi", return_value=[]),
                    mock.patch.object(market_data, "list_polymarket", return_value=[]),
                    mock.patch.object(
                        agent_trading_tick, "_call_agent_analyze",
                        return_value=SimpleNamespace(thesis="Passed.", tool_transcript=[]),
                    ) as call_mock,
                ):
                    agent_trading_tick.run_cycle("model-learning-turn")

        question = call_mock.call_args[0][0]
        self.assertIn("Learning from your resolved shadow trades", question)
        self.assertIn("learn-market", question)
        self.assertIn("1 profitable", question)


class RunCycleTests(unittest.TestCase):
    def _fake_report(self, thesis="Passed this cycle.", transcript=None):
        return SimpleNamespace(thesis=thesis, tool_transcript=transcript or [])

    def test_run_cycle_persists_thesis_and_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_NOTES_PATH": str(Path(td) / "notes.json"),
                "FORESEA_AGENT_CYCLE_ID": "test-cycle-1",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(agent_trading_tick, "_init_local_agent"),
                mock.patch.object(market_data, "list_kalshi", return_value=[_quote("KXNEW")]),
                mock.patch.object(market_data, "list_polymarket", return_value=[]),
                mock.patch.object(
                    agent_trading_tick, "_call_agent_analyze",
                    return_value=self._fake_report(thesis="Bought KXNEW on a genuine edge.",
                                                     transcript=[{"action": "place_trade"}]),
                ) as call_mock,
            ):
                agent_trading_tick.run_cycle("model-c")

                with benchmark_tools._account_transaction() as conn:
                    row = conn.execute(
                        "SELECT * FROM agent_cycles WHERE agent_id = ? AND cycle_id = ?",
                        ("model-c", "test-cycle-1"),
                    ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row["thesis"], "Bought KXNEW on a genuine edge.")
        self.assertEqual(row["steps"], 1)
        self.assertIn("KXNEW", row["transcript_json"])
        # The question passed to the tool loop must actually mention the candidate.
        question_arg = call_mock.call_args[0][0]
        self.assertIn("KXNEW", question_arg)
        self.assertIn("Your portfolio", question_arg)

    def test_run_cycle_configures_order_notional_from_current_account_value_before_the_tool_loop(self):
        # Regression: _configure_max_order_notional() used to run before
        # held_quotes existed, reading the static starting-cash baseline --
        # it must now run after held_quotes is available and reflect the
        # account's real current value (so a profitable agent's order cap
        # actually grows, and a losing agent's actually shrinks).
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_NOTES_PATH": str(Path(td) / "notes.json"),
                "FORESEA_AGENT_CYCLE_ID": "test-cycle-2",
            }
            seen = {}

            async def _capture_env(question):
                seen["FORESEA_MAX_ORDER_NOTIONAL"] = os.environ.get("FORESEA_MAX_ORDER_NOTIONAL")
                seen["FORESEA_AGENT_ACCOUNT_VALUE"] = os.environ.get("FORESEA_AGENT_ACCOUNT_VALUE")
                return self._fake_report()

            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(agent_trading_tick, "_init_local_agent"),
                mock.patch.object(market_data, "list_kalshi", return_value=[]),
                mock.patch.object(market_data, "list_polymarket", return_value=[]),
                mock.patch.object(agent_trading_tick, "_call_agent_analyze", side_effect=_capture_env),
            ):
                agent_trading_tick.run_cycle("model-d")

        self.assertEqual(seen["FORESEA_AGENT_ACCOUNT_VALUE"], str(benchmark_tools.DEFAULT_AGENT_ACCOUNT_VALUE))
        self.assertEqual(seen["FORESEA_MAX_ORDER_NOTIONAL"], "800.0")

    def test_run_cycle_offers_held_positions_for_exit(self):
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_NOTES_PATH": str(Path(td) / "notes.json"),
                "FORESEA_AGENT_CYCLE_ID": "test-cycle-2",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = benchmark_tools.ToolContext(agent_id="model-d")
                with mock.patch.object(
                    market_data, "fetch_kalshi", return_value=_quote("KXHELD", bid=0.48, ask=0.50),
                ):
                    benchmark_tools.place_trade(
                        {"ticker": "KXHELD", "side": "yes", "price": 0.5, "quantity": 3}, ctx,
                    )

                with (
                    mock.patch.object(agent_trading_tick, "_init_local_agent"),
                    mock.patch.object(market_data, "list_kalshi", return_value=[]),
                    mock.patch.object(market_data, "list_polymarket", return_value=[]),
                    mock.patch.object(
                        market_data, "fetch_kalshi", return_value=_quote("KXHELD", question="Held market"),
                    ),
                    mock.patch.object(
                        agent_trading_tick, "_call_agent_analyze",
                        return_value=self._fake_report(),
                    ) as call_mock,
                ):
                    agent_trading_tick.run_cycle("model-d")

        question_arg = call_mock.call_args[0][0]
        self.assertIn("KXHELD", question_arg)
        self.assertIn("currently hold", question_arg)

    def test_run_cycle_refuses_when_not_shadow(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_PLACE_TRADE_MODE": "live"}, clear=False):
            with self.assertRaises(RuntimeError):
                agent_trading_tick.run_cycle("model-e")


class AgentAnalyzeRetryTests(unittest.TestCase):
    def test_retries_then_raises_after_exhausting_attempts(self):
        import asyncio

        calls = []

        async def _always_fails(req, request=None):
            calls.append(1)
            raise RuntimeError("upstream down")

        with (
            mock.patch.object(agent_trading_tick, "AGENT_ANALYZE_RETRIES", 2),
            mock.patch.object(agent_trading_tick, "AGENT_ANALYZE_RETRY_BACKOFF_S", 0.0),
        ):
            with mock.patch(
                "analyzing_llm_rationale.server.agent_analyze", side_effect=_always_fails,
            ):
                with self.assertRaises(RuntimeError):
                    asyncio.run(agent_trading_tick._call_agent_analyze("question text"))
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
