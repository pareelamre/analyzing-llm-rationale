from __future__ import annotations

import importlib.util
import json
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

    def test_discover_candidates_reserves_one_source_verified_weather_market(self):
        weather = _quote(
            "KXWEATHER",
            question="What will the highest temperature in Chicago be today?",
            resolution_criteria="NWS Daily Climate Report for station KORD.",
        )
        weather["category"] = "Weather"
        general = _quote("KXGENERAL")

        def list_kalshi(**kwargs):
            return [weather] if kwargs.get("category") == "Weather" else [general]

        with (
            mock.patch.object(market_data, "list_kalshi", side_effect=list_kalshi),
            mock.patch.object(market_data, "list_polymarket", return_value=[]),
            mock.patch.object(agent_trading_tick, "CANDIDATE_COUNT", 2),
            mock.patch.object(agent_trading_tick, "WEATHER_CANDIDATE_QUOTA", 1),
        ):
            found = agent_trading_tick._discover_candidates(set())

        self.assertEqual([quote["ident"] for quote in found], ["KXWEATHER", "KXGENERAL"])

    def test_weather_research_count_reads_only_declared_tool_calls(self):
        transcript = [
            {"tool": "weather_market_research"},
            {"tool": "get_market"},
            {"action": "weather_market_research"},
        ]
        self.assertEqual(agent_trading_tick._weather_research_call_count(transcript), 2)

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

    def test_candidate_line_shows_participation_depth(self):
        # Kalshi Research finds Brier falls monotonically with volume within
        # every horizon, so depth is the agent's best read on whether a price
        # is worth disputing. It previously saw only whether volume existed,
        # which cannot separate a market with 40 contracts traded from one
        # with 40,000 -- the difference between a disputable price and one
        # that is effectively unbeatable.
        quote = _quote("KXFOO")
        quote["volume"] = 41234.0
        quote["liquidity"] = 8800.0
        line = agent_trading_tick._fmt_candidate_line(quote)
        self.assertIn("volume 41,234", line)
        self.assertIn("depth/open interest 8,800", line)

    def _board(self, payload=None):
        resp = mock.Mock(status_code=200)
        resp.json.return_value = json.loads(payload or '{"leaderboard": [{"agent_id": "winner", "return_pct": 5.05, "trade_count": 14, "realized_count": 6, "win_rate": 0.33},{"agent_id": "middle", "return_pct": 0.45, "trade_count": 17, "realized_count": 12, "win_rate": 0.25},{"agent_id": "loser", "return_pct": -15.21, "trade_count": 25, "realized_count": 15, "win_rate": 0.27}]}')
        return mock.patch.object(agent_trading_tick.requests, "get", return_value=resp)

    def test_standings_rank_every_agent_and_mark_the_reader(self):
        # Each model runs its cycle with only its own shadow account, so it
        # has never had any way to know where it stands. The published board
        # is the one place the whole field is visible.
        with self._board():
            block = agent_trading_tick._leaderboard_block("loser")
        self.assertIn("1. winner", block)
        self.assertIn("3. loser", block)
        self.assertIn("<-- YOU", block)
        self.assertIn("You are 3 of 3", block)
        self.assertIn("20.26pp off the lead", block)

    def test_the_agent_directly_above_is_named_as_the_target(self):
        # "Eighth of eight, 20pp off the lead" is abstract and unreachable.
        # "0.69pp behind minimax, pass them first" is a target. Rank is taken
        # one place at a time.
        with self._board():
            block = agent_trading_tick._leaderboard_block("loser")
        self.assertIn("Directly above you is middle", block)
        self.assertIn("15.66pp", block)   # loser -15.21 vs middle +0.45
        self.assertIn("pass them first", block)

    def test_the_leader_is_told_who_is_chasing(self):
        with self._board():
            block = agent_trading_tick._leaderboard_block("winner")
        self.assertIn("You are first", block)
        self.assertIn("middle", block)
        self.assertIn("reading this same board", block)

    def test_the_leader_is_told_it_is_defending_not_chasing(self):
        with self._board():
            block = agent_trading_tick._leaderboard_block("winner")
        self.assertIn("You are first", block)
        self.assertIn("hands it over", block)
        self.assertNotIn("pass them first", block)

    def test_standings_say_plainly_that_activity_is_not_rank(self):
        # "Trade more to climb" is the obvious inference from a leaderboard
        # and it is the wrong one here: the most active agent is last, and
        # the leader got there on fewer trades.
        with self._board():
            block = agent_trading_tick._leaderboard_block("middle")
        self.assertIn("traded the most (25)", block)
        self.assertIn("being right, not from being busy", block)

    def test_each_model_is_told_the_hand_it_actually_holds(self):
        # Context windows run 65k to 1,048,576 across the eight agents -- a
        # sixteenfold spread. A model with a million tokens should read more
        # than the field can; one with 65k is wasting a cycle imitating it.
        # Telling each what it has is how a particular model's best shows up
        # instead of eight models converging on the same shallow pass.
        biggest = agent_trading_tick._own_capability_line("glm-5-3-flash")
        smallest = agent_trading_tick._own_capability_line("llama-3.3-70b-instruct")
        middle = agent_trading_tick._own_capability_line("gemma-4-26b-a4b-it")

        self.assertIn("largest of any agent", biggest)
        self.assertIn("1,048,576", biggest)
        self.assertIn("smallest here", smallest)
        self.assertIn("cannot out-read this field", smallest)
        self.assertIn("65,536", smallest)
        # The middle of the field gets neither claim.
        self.assertNotIn("largest of any agent", middle)
        self.assertNotIn("smallest here", middle)

    def test_an_unknown_model_gets_no_capability_claim(self):
        self.assertEqual(agent_trading_tick._own_capability_line("not-a-model"), "")

    def test_an_unreachable_board_does_not_cost_the_cycle(self):
        # A scoreboard outage must never take a trading cycle with it.
        with mock.patch.object(agent_trading_tick.requests, "get",
                               side_effect=RuntimeError("board down")):
            self.assertEqual(agent_trading_tick._leaderboard_block("loser"), "")
        resp = mock.Mock(status_code=503)
        with mock.patch.object(agent_trading_tick.requests, "get", return_value=resp):
            self.assertEqual(agent_trading_tick._leaderboard_block("loser"), "")

    def test_the_instruction_states_the_competition_and_its_trap(self):
        text = agent_trading_tick._TRADING_INSTRUCTION
        self.assertIn("YOU ARE COMPETING", text)
        self.assertIn("finish top by return", text)
        # ...and immediately guards the inference it invites.
        self.assertIn("picking better, not by picking more", text)

    def test_candidates_are_ranked_by_the_edge_they_require(self):
        # The menu, not the gate, was the binding constraint on trade rate.
        # A trade needs its edge to beat half-spread + fee + floor; measured
        # live that runs 3pp to 49pp, and only ~24% of sides sit at or below
        # 4pp. Three markets taken in listing order therefore usually held
        # nothing an agent could act on however good its read was. Ranking by
        # hurdle puts the reachable markets in front of it every cycle.
        cheap = _quote("KXCHEAP", bid=0.07, ask=0.08)
        mid = _quote("KXMID", bid=0.40, ask=0.45)
        wide = _quote("KXWIDE", bid=0.00, ask=0.93)

        ranked = sorted([wide, mid, cheap], key=agent_trading_tick._edge_hurdle_pp)

        self.assertEqual([q["ident"] for q in ranked],
                         ["KXCHEAP", "KXMID", "KXWIDE"])
        self.assertLess(agent_trading_tick._edge_hurdle_pp(cheap), 0.04)
        self.assertGreater(agent_trading_tick._edge_hurdle_pp(wide), 0.20)

    def test_an_untradeable_market_sorts_last_rather_than_first(self):
        # A side quoted at or above 1.00 has no reachable bar at all. It must
        # not sort to the front by looking like a zero hurdle.
        dead = _quote("KXDEAD")
        for key in ("yes_bid", "yes_ask", "no_bid", "no_ask"):
            dead.pop(key, None)
        ok = _quote("KXOK", bid=0.40, ask=0.45)
        self.assertEqual(agent_trading_tick._edge_hurdle_pp(dead), float("inf"))
        self.assertEqual(
            [q["ident"] for q in sorted([dead, ok],
                                        key=agent_trading_tick._edge_hurdle_pp)],
            ["KXOK", "KXDEAD"])

    def test_the_menu_is_wide_enough_to_contain_a_reachable_market(self):
        # At 3, a random draw usually contained none. Ranking only helps if
        # there is a pool to rank.
        self.assertGreaterEqual(agent_trading_tick.CANDIDATE_COUNT, 8)

    def test_display_and_ranking_read_the_same_hurdle(self):
        # If these ever diverge, an agent is ranked toward one market and
        # told the number for another.
        quote = _quote("KXQ", bid=0.07, ask=0.08)
        cheapest = min(agent_trading_tick._edge_hurdle_by_side(quote).values())
        self.assertEqual(agent_trading_tick._edge_hurdle_pp(quote), cheapest)
        self.assertIn("%.1fpp" % (100 * cheapest),
                      agent_trading_tick._fmt_edge_hurdle(quote))

    def test_candidate_line_states_the_edge_needed_to_clear_the_gate(self):
        # A position only clears when its net edge beats fees plus the
        # min-net-edge floor, measured against the executable ask rather than
        # the mid -- so the true hurdle is half-spread + fee + floor. Across
        # live candidates that runs from 3pp to 49pp: the same 2% floor
        # demanding 16x more conviction depending on the book, with none of it
        # visible. An agent saw two quotes and could not tell that one needed
        # a near-certainty to be tradeable at all.
        tight = _quote("KXTIGHT")
        tight.update({"yes_bid": 0.07, "yes_ask": 0.08, "no_bid": 0.92, "no_ask": 0.93})
        wide = _quote("KXWIDE")
        wide.update({"yes_bid": 0.0, "yes_ask": 0.93, "no_bid": 0.07, "no_ask": 1.00})

        tight_line = agent_trading_tick._fmt_candidate_line(tight)
        wide_line = agent_trading_tick._fmt_candidate_line(wide)

        self.assertIn("edge needed vs mid", tight_line)
        self.assertIn("yes +3.0pp", tight_line)
        # A one-sided book demands a near-certainty on the side that can pay.
        self.assertIn("yes +49.0pp", wide_line)
        # ...and the side quoted at 1.00 is not offered as tradeable at all,
        # matching the no_executable_price rejection in place_trade.
        self.assertNotIn("no +", wide_line)

    def test_edge_hurdle_is_omitted_without_a_two_sided_quote(self):
        quote = _quote("KXNOBOOK")
        for key in ("yes_bid", "yes_ask", "no_bid", "no_ask"):
            quote.pop(key, None)
        self.assertIn("edge hurdle n/a", agent_trading_tick._fmt_candidate_line(quote))

    def test_candidate_line_says_so_when_participation_is_unreported(self):
        # Silence must not read as "thin, therefore beatable".
        quote = _quote("KXFOO")
        quote.pop("volume", None)
        quote["liquidity"] = 0
        line = agent_trading_tick._fmt_candidate_line(quote)
        self.assertIn("participation unreported", line)
        self.assertNotIn("volume 0", line)

    def test_candidate_horizon_reaches_the_region_where_edge_survives(self):
        # The study's sharpest result: long-dated markets never reach a 0.05
        # Brier at any level of participation, while near-close prices sit at
        # ~0.02. A 30-day ceiling excluded that region entirely, leaving
        # agents hunting edge only where prices are closest to efficient.
        self.assertGreaterEqual(agent_trading_tick.MAX_CLOSE_DAYS, 90)

    def test_discrepancy_discipline_cites_measured_figures_not_an_unsourced_rate(self):
        # "perceived edges >20pp fail 73.4% of the time" was presented to every
        # model as measured fact with no traceable source. Claims that steer
        # every forecast have to be attributable.
        from analyzing_llm_rationale import agent_capabilities as _ac

        instruction = agent_trading_tick._TRADING_INSTRUCTION
        note = _ac.build_grounding_note({"n_snapshots_resolved": 5, "overall": {}})
        for text in (instruction, note):
            self.assertNotIn("73.4", text)
            self.assertIn("2,243,741", text)

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


class FailureClassificationTests(unittest.TestCase):
    def test_a_wrapped_read_timeout_is_a_timeout_not_an_outage(self):
        # The agent path re-raises a read timeout as "503 ... temporarily
        # unavailable". Classifying on that text filed every timeout as a
        # provider outage, which is how glm-5-3 was reported down for days
        # while SCADS listed it up with tools enabled -- it was just slower
        # than the 120s read timeout. Retrying an "outage" that is really a
        # timeout also burns a second full timeout window for nothing.
        import requests

        root = requests.exceptions.ReadTimeout("The read operation timed out")
        wrapped = RuntimeError(
            "503: The 'glm-5-3' forecasting model is temporarily unavailable. "
            "Please retry in a moment, or pass a different `model` in the request."
        )
        wrapped.__cause__ = root

        self.assertEqual(agent_trading_tick._failure_kind(wrapped), "provider_timeout")

    def test_a_genuine_outage_is_still_an_outage(self):
        # The timeout check must not swallow real 503s.
        exc = RuntimeError("503: upstream service unavailable")
        self.assertEqual(agent_trading_tick._failure_kind(exc), "provider_unavailable")

    def test_a_wrapped_rate_limit_is_still_a_rate_limit(self):
        # The safety property that keeps 429s out of the retry path.
        exc = RuntimeError("503: temporarily unavailable (upstream returned HTTP 429)")
        self.assertEqual(agent_trading_tick._failure_kind(exc), "provider_rate_limited")

    def test_a_timeout_is_not_retried_as_a_transient_503(self):
        # provider_unavailable earns a second attempt; a timeout must not,
        # or a slow model costs two full timeout windows per cycle.
        import requests

        root = requests.exceptions.ReadTimeout("The read operation timed out")
        wrapped = RuntimeError("503: The 'glm-5-3' forecasting model is temporarily unavailable.")
        wrapped.__cause__ = root
        self.assertNotEqual(agent_trading_tick._failure_kind(wrapped), "provider_unavailable")

    def test_agent_path_gives_a_slow_model_room_to_answer(self):
        # The asyncio ceiling was already disabled here, but the HTTP client's
        # own socket read timeout was never set on this path, so "no ceiling"
        # still meant a hard 120s from the provider dataclass default.
        from analyzing_llm_rationale import server

        self.assertGreater(server._AGENT_TOOL_PROVIDER_READ_TIMEOUT_S, 120.0)

    def test_glm_5_3_defaults_to_long_read_timeout_in_all_provider_selectors(self):
        # PR #413 passed _AGENT_TOOL_PROVIDER_READ_TIMEOUT_S to _select_agent_provider,
        # but when an agent calls its `forecast` tool, _tool_forecast delegates to
        # predict(), which calls _select_predict_provider and _scads_alt_provider.
        # If neither defaults glm-5-3 to the 600s timeout, the forecast tool call
        # still dies at 120s.
        import os
        from unittest import mock

        from analyzing_llm_rationale import server

        with mock.patch.dict(os.environ, {"SCADS_AI_API_KEY": "test-key"}):
            # 1. Direct _scads_alt_provider without request_timeout_s
            provider = server._scads_alt_provider("glm-5-3")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.request_timeout_s, server._AGENT_TOOL_PROVIDER_READ_TIMEOUT_S)

            # Other models retain standard default
            fast_provider = server._scads_alt_provider("glm-5-3-flash")
            self.assertIsNotNone(fast_provider)
            self.assertEqual(fast_provider.request_timeout_s, 120.0)

            # 2. _select_predict_provider for glm-5-3 (used inside _tool_forecast)
            req = server.PredictRequest(question="Will X happen?", model="glm-5-3")
            pred_provider, _, _ = server._select_predict_provider(req)
            self.assertEqual(pred_provider.request_timeout_s, server._AGENT_TOOL_PROVIDER_READ_TIMEOUT_S)


class DecisionQualityInstructionTests(unittest.TestCase):
    def test_requires_fresh_evidence_for_new_risk_and_rule_window_checks(self):
        instruction = agent_trading_tick._TRADING_INSTRUCTION

        self.assertIn("RESEARCH QUALITY GATE", instruction)
        self.assertIn("dated, material evidence update", instruction)
        self.assertIn("A new position without visible resolution rules", instruction)
        self.assertIn("PASS -- not a guess", instruction)

    def test_offers_a_bounded_strategy_menu_with_safe_orderbook_arb_handling(self):
        instruction = agent_trading_tick._TRADING_INSTRUCTION

        self.assertIn("EVIDENCE_EDGE", instruction)
        self.assertIn("CATALYST_EDGE", instruction)
        self.assertIn("ORDERBOOK_ARBITRAGE_RESEARCH", instruction)
        self.assertIn("POSITION_RISK_REDUCTION", instruction)
        self.assertIn("never submit a single leg as 'arbitrage'", instruction)


class StrategySelectionTests(unittest.TestCase):
    def test_extracts_only_known_strategy_labels(self):
        self.assertEqual(
            agent_trading_tick._selected_strategy("- **Strategy**: [CATALYST_EDGE]"),
            "catalyst_edge",
        )
        self.assertEqual(
            agent_trading_tick._selected_strategy("- **Strategy**: definitely profitable"),
            "unreported",
        )
        self.assertEqual(agent_trading_tick._selected_strategy("No strategy field"), "unreported")


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

    def test_candidate_includes_settlement_aware_weather_context(self):
        quote = _quote("KXWEATHER", question="What will the highest temperature in Chicago be today?")
        quote.update({
            "category": "Weather",
            "resolution_criteria": "Resolves by the NWS Daily Climate Report for Chicago O'Hare.",
        })

        line = agent_trading_tick._fmt_candidate_line(quote)

        self.assertIn("Weather contract type: daily temperature", line)
        self.assertIn("Official settlement source: NWS Daily Climate Report", line)

    def test_weather_thesis_forecast_persists_type_and_settlement_source(self):
        thesis = (
            "- **Action**: PASS\n"
            "- **Market & Venue**: [KXWEATHER] on [Kalshi]\n"
            "- **Model Probability**: [70%] vs **Market Price**: [60%]"
        )
        quote = _quote("KXWEATHER", question="What will the highest temperature in Chicago be today?")
        quote.update({
            "category": "Weather",
            "resolution_criteria": "NWS Daily Climate Report for station KORD.",
        })
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}, clear=False
            ):
                with benchmark_tools._account_transaction() as conn:
                    agent_trading_tick._persist_thesis_forecasts(
                        conn, "weather-agent", "weather-cycle", thesis, [], [quote], "evidence_edge",
                        forecast_ts="2026-08-20T00:00:00+00:00",
                    )
                    row = conn.execute(
                        "SELECT weather_market_type, weather_settlement_source FROM agent_thesis_forecasts"
                    ).fetchone()
                    conn.execute(
                        """
                        UPDATE agent_thesis_forecasts
                        SET resolved_outcome = 1, brier_score = 0.09, market_brier_score = 0.16
                        """
                    )
                    learning = agent_trading_tick._build_learning_block(conn, "weather-agent")

        self.assertEqual(row["weather_market_type"], "daily_temperature")
        self.assertEqual(row["weather_settlement_source"], "nws_daily_climate_report")
        self.assertIn("Weather calibration by contract type/source", learning)
        self.assertIn("daily_temperature / nws_daily_climate_report", learning)

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
        # glm-5-3 is the only one of the ten agent-trading models whose
        # deployed context length SCADS AI's own /v1/models listing actually
        # publishes (524288 tokens, checked 2026-08-18). Half reserved for
        # the rest of the real prompt (system prompt, ~17 tool specs, ReAct
        # loop history), 4 chars/token.
        self.assertEqual(
            agent_trading_tick._model_backstop_chars("glm-5-3"),
            int(524288 * 4 * 0.5),
        )

    def test_falls_back_to_a_conservative_default_for_unverified_models(self):
        # Every other model, including all three scads-alias-* aliases
        # (SCADS doesn't publish what they route to), gets one shared
        # conservative default rather than a guessed per-model figure that
        # could overestimate real capacity.
        for model in ("scads-alias-reasoning", "scads-alias-code", "scads-alias-ha"):
            with self.subTest(model=model):
                self.assertEqual(agent_trading_tick._model_backstop_chars(model), int(128000 * 4 * 0.5))

    def test_a_published_window_smaller_than_the_default_is_respected(self):
        # llama-3.3-70b-instruct's real window is 65,536 -- half the 128k
        # default it used to fall back to, so its prompt budget was being
        # derived from twice its actual capacity.
        self.assertEqual(
            agent_trading_tick._model_backstop_chars("llama-3.3-70b-instruct"),
            int(65536 * 4 * 0.5),
        )
        self.assertLess(
            agent_trading_tick._model_backstop_chars("llama-3.3-70b-instruct"),
            int(128000 * 4 * 0.5),
        )

    def test_falls_back_for_an_unrecognized_model_name(self):
        # Must never raise -- an unknown model (e.g. a typo'd env var) still
        # needs a usable backstop, not a crash before the cycle even starts.
        self.assertEqual(agent_trading_tick._model_backstop_chars("not-a-real-model"), int(128000 * 4 * 0.5))


class DeclaredThesisProbabilityTests(unittest.TestCase):
    def test_parses_a_probability_line_with_a_qualifier_before_vs(self):
        # Verbatim from deepseek-v4-flash: a complete, correctly stated figure
        # that the old pattern rejected because "(no-change)" sat between the
        # percentage and "vs". Reconciliation then refused the trade for having
        # no calibrated P(YES), so a researched, decided position was never
        # opened and the agent showed zero positions.
        thesis = chr(10).join([
            "### 1. Decision & Execution",
            "- **Action**: BUY NO",
            "- **Market & Venue**: will-there-be-no-change-615 on Polymarket",
            "- **Order Sizing**: Quarter Kelly 5% cap, quantity derived by tool.",
            "- **Model Probability**: 40% (no-change) vs **Market Price**: 48.5%"
            " mid / 52% NO ask (Edge: ~+8.5pp vs mid).",
        ])
        declared = agent_trading_tick._declared_thesis_execution(thesis)
        self.assertIsNotNone(declared)
        self.assertEqual(declared["action"], "BUY NO")
        self.assertEqual(declared["model_probability"], 0.4)
        self.assertEqual(declared["sizing_mode"], "quarter_kelly")

    def test_parses_the_plain_form_without_a_qualifier(self):
        thesis = chr(10).join([
            "- **Action**: BUY YES",
            "- **Market & Venue**: KXTEST on Kalshi",
            "- **Order Sizing**: Edge Kelly",
            "- **Model Probability**: 62% vs **Market Price**: 50%",
        ])
        declared = agent_trading_tick._declared_thesis_execution(thesis)
        self.assertEqual(declared["model_probability"], 0.62)
        self.assertEqual(declared["sizing_mode"], "edge_kelly")

    def test_loose_fallback_tolerates_markdown_and_approximation(self):
        for text in ("**P(YES)**: ~37%", "calibrated P(YES) of 37%", "model probability: 37%"):
            with self.subTest(text=text):
                match = agent_trading_tick._LOOSE_MODEL_PROBABILITY_RE.search(text)
                self.assertIsNotNone(match, text)
                self.assertEqual(match.group(1), "37")


class EventMeritGateTests(unittest.TestCase):
    def test_prompt_requires_event_merit_not_just_a_price_gap(self):
        # llama-3.3-70b-instruct traded 90 times on 19,802 of notional for a
        # 23% win rate and -793 gross: symptoms of trading numeric gaps rather
        # than events it understood. The gate asks for the mechanism, the
        # informational advantage, and the falsifier before opening.
        question = agent_trading_tick._build_question("PORTFOLIO", "CANDIDATES")
        for requirement in (
            "EVENT MERIT GATE",
            "Trade the event, not the number",
            "mechanism",
            "better informed",
            "prove you wrong",
        ):
            self.assertIn(requirement, question)

    def test_prompt_states_a_bare_disagreement_is_not_an_edge(self):
        question = agent_trading_tick._build_question("PORTFOLIO", "CANDIDATES")
        self.assertIn("you do not have an edge, you have a disagreement", question)


class RecalledNotesTests(unittest.TestCase):
    def _write_notes(self, directory, agent_id, notes):
        path = Path(directory) / "notes.json"
        path.write_text(json.dumps({agent_id: notes}), encoding="utf-8")
        return path

    def test_prior_notes_are_injected_into_the_cycle_prompt(self):
        # Recall used to require the model to spend one of its 4 tool steps
        # calling manage_notes. gemma, gpt-oss and qwen each wrote notes over
        # dozens of cycles and read them back zero times, so memory was
        # written and never used. Surface it directly instead.
        with tempfile.TemporaryDirectory() as td:
            path = self._write_notes(td, "model-a", [
                {"text": "Fed Sep 2026: model 45% vs market 42%, edge below threshold.",
                 "tags": ["fed"], "created_at": "2026-09-01T00:00:00+00:00"},
            ])
            with mock.patch.dict(os.environ, {"FORESEA_AGENT_NOTES_PATH": str(path)}):
                question = agent_trading_tick._build_question(
                    "PORTFOLIO", "CANDIDATES", "model-a",
                )
        self.assertIn("Your notes from previous cycles", question)
        self.assertIn("edge below threshold", question)
        self.assertIn("[fed]", question)

    def test_only_the_agents_own_notes_are_recalled(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "notes.json"
            path.write_text(json.dumps({
                "model-a": [{"text": "mine", "created_at": "2026-09-01T00:00:00+00:00"}],
                "model-b": [{"text": "someone else's", "created_at": "2026-09-01T00:00:00+00:00"}],
            }), encoding="utf-8")
            with mock.patch.dict(os.environ, {"FORESEA_AGENT_NOTES_PATH": str(path)}):
                question = agent_trading_tick._build_question("P", "C", "model-a")
        self.assertIn("mine", question)
        self.assertNotIn("someone else's", question)

    def test_no_notes_adds_no_section(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_notes(td, "model-a", [])
            with mock.patch.dict(os.environ, {"FORESEA_AGENT_NOTES_PATH": str(path)}):
                question = agent_trading_tick._build_question("P", "C", "model-a")
        self.assertNotIn("Your notes from previous cycles", question)

    def test_recall_survives_an_unreadable_notes_file(self):
        # A missing or corrupt notes file must never break a trading cycle.
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_NOTES_PATH": "/nonexistent/notes.json"}):
            question = agent_trading_tick._build_question("P", "C", "model-a")
        self.assertIn("PORTFOLIO" if "PORTFOLIO" in question else "P", question)
        self.assertNotIn("Your notes from previous cycles", question)


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

    def test_a_block_trimmed_to_the_limit_never_costs_every_candidate(self):
        # The omission marker used to be appended after the budget was already
        # spent, so a trimmed block came back LONGER than requested.
        # _build_question reads that as "still too long" and moves to its next
        # step, which drops the candidates block outright -- so a block landing
        # within ~30 chars of the limit cost the agent every market it was
        # being offered, silently, instead of costing it one line. Lengthening
        # a candidate line by 29 characters was enough to trigger it.
        block = chr(10).join(
            "  - [kalshi] KX%d: filler line of a realistic width" % i
            for i in range(400))
        for budget in range(200, 400):
            trimmed = agent_trading_tick._trim_block_to_lines(block, budget)
            self.assertLessEqual(
                len(trimmed), budget,
                "trimming to %d returned %d chars" % (budget, len(trimmed)))

    def test_candidates_survive_a_block_that_lands_on_the_limit(self):
        new = [_quote(f"KX{i}", question=f"Candidate market number {i} with a fairly long question")
               for i in range(1500)]
        candidates = agent_trading_tick._build_candidates_block([], new)
        with mock.patch.object(agent_trading_tick, "MAX_QUESTION_CHARS", 200_000):
            question = agent_trading_tick._build_question("PORTFOLIO", candidates)
        self.assertLessEqual(len(question), 200_000)
        # The whole point of the cycle is that markets are visible.
        self.assertIn("Markets you can act on", question)
        self.assertIn("(more omitted for space)", question)

    def test_trims_the_candidates_block_by_whole_lines_not_mid_line(self):
        # A genuinely pathological number of candidates still has to fit
        # somewhere -- verify the overflow is handled by dropping whole
        # trailing lines (leaving every surviving candidate fully readable),
        # never by slicing raw characters mid-line.
        #
        # The cap is opt-in now (AGENT_TRADING_MAX_QUESTION_CHARS, unlimited
        # by default), so set one explicitly to exercise the retained trimming
        # machinery rather than asserting against an unlimited budget.
        cap = 200_000
        new = [_quote(f"KX{i}", question=f"Candidate market number {i} with a fairly long question text") for i in range(1500)]
        candidates = agent_trading_tick._build_candidates_block([], new)
        with mock.patch.object(agent_trading_tick, "MAX_QUESTION_CHARS", cap):
            question = agent_trading_tick._build_question("PORTFOLIO", candidates)
        self.assertLessEqual(len(question), cap)
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
    def test_trading_instruction_explains_close_netting_pnl(self):
        self.assertIn("CLOSE ACCOUNTING", agent_trading_tick._TRADING_INSTRUCTION)
        self.assertIn("$1.00 per matched pair", agent_trading_tick._TRADING_INSTRUCTION)

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

    def test_portfolio_block_carries_compact_prior_state_and_research_sources(self):
        full_thesis = (
            "### 1. Decision & Execution\n- **Action**: HOLD\n"
            "### 3. Model Edge & Valuation\n- **Model Probability**: 55%\n" + "x" * 2500
        )
        transcript = json.dumps({"tool_transcript": [{
            "action": "web_search",
            "observation": json.dumps({
                "summary": "Official update confirms the event date has not changed.",
                "sources": [{"title": "Official source", "url": "https://example.test/research"}],
            }),
        }]})
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}, clear=False
            ):
                with benchmark_tools._account_transaction() as conn:
                    block = agent_trading_tick._build_portfolio_block(
                        conn, "model-c", full_thesis, last_transcript=transcript
                    )

        self.assertNotIn(full_thesis, block)
        self.assertIn("Prior thesis state", block)
        self.assertIn("Research carried forward", block)
        self.assertIn("https://example.test/research", block)
        self.assertLess(len(block), 2200)


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


class ThesisForecastLearningTests(unittest.TestCase):
    def test_persists_explicit_thesis_probability_and_scores_only_final_settlement(self):
        thesis = (
            "### 0. Research Delta\n"
            "- **Strategy**: [EVIDENCE_EDGE]\n"
            "- **New evidence**: Official update https://example.test confirms the qualifying event.\n"
            "### 1. Decision & Execution\n"
            "- **Action**: BUY YES\n"
            "- **Market & Venue**: [KXLEARN-26] on [Kalshi]\n"
            "### 3. Model Edge & Valuation\n"
            "- **Model Probability**: [70%] vs **Market Price**: [60%] (Edge: [+10%])"
        )
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path)}, clear=False
            ):
                with benchmark_tools._account_transaction() as conn:
                    recorded = agent_trading_tick._persist_thesis_forecasts(
                        conn, "model-learning", "cycle-forecast", thesis, [],
                        [_quote("KXLEARN-26", bid=0.59, ask=0.61)], "evidence_edge",
                        forecast_ts="2026-08-20T00:00:00+00:00",
                    )
                    self.assertEqual(recorded, 1)
                    # A voluntary close is deliberately not a truth label.
                    benchmark_tools._record_account_action(
                        conn, agent_id="model-learning", action_type="trade",
                        cycle_id="cycle-close", platform="kalshi", ticker="KXLEARN-26",
                        outcome="realized", realized_pairs=1, realized_pnl=1.0,
                    )
                    self.assertEqual(agent_trading_tick._refresh_thesis_forecast_outcomes(conn, "model-learning"), 0)
                    benchmark_tools._record_account_action(
                        conn, agent_id="model-learning", action_type="settlement",
                        cycle_id="cycle-settlement", platform="kalshi", ticker="KXLEARN-26",
                        outcome="yes",
                    )
                    self.assertEqual(agent_trading_tick._refresh_thesis_forecast_outcomes(conn, "model-learning"), 1)
                    row = conn.execute(
                        "SELECT model_probability, market_probability, resolved_outcome, brier_score, market_brier_score "
                        "FROM agent_thesis_forecasts"
                    ).fetchone()

        self.assertAlmostEqual(row["model_probability"], 0.70)
        self.assertAlmostEqual(row["market_probability"], 0.60)
        self.assertEqual(row["resolved_outcome"], 1)
        self.assertAlmostEqual(row["brier_score"], 0.09)
        self.assertAlmostEqual(row["market_brier_score"], 0.16)

    def test_trade_tool_probability_is_a_fallback_when_the_final_thesis_omits_the_field(self):
        records = agent_trading_tick._thesis_forecast_records(
            "### 1. Decision & Execution\n- **Action**: BUY YES",
            [{"action": "place_trade", "args": {
                "platform": "polymarket", "ticker": "learn-market", "model_probability": 0.63,
                "side": "yes",
            }}],
            [{"platform": "polymarket", "ident": "learn-market", "probability": 0.55}],
            "evidence_edge",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["ticker"], "learn-market")
        self.assertAlmostEqual(records[0]["model_probability"], 0.63)
        self.assertAlmostEqual(records[0]["market_probability"], 0.55)

    def test_a_no_side_probability_is_stored_as_p_yes_not_inverted(self):
        """A "95% NO" is a 5% P(YES); storing 0.95 would invert calibration."""
        records = agent_trading_tick._thesis_forecast_records(
            "- **Market & Venue**: [KXNO-26] on [Kalshi]\n"
            "- **Model Probability**: 95% NO vs **Market Price**: 15.5% NO",
            [],
            [],
            "news_research",
        )
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["model_probability"], 0.05)
        self.assertAlmostEqual(records[0]["market_probability"], 0.845)

    def test_a_bare_side_qualifier_no_longer_discards_the_forecast(self):
        """gpt-oss-120b and glm-5-3-flash both write the side without brackets."""
        records = agent_trading_tick._thesis_forecast_records(
            "- **Market & Venue**: [KXSIDE-26] on [Kalshi]\n"
            "- **Model Probability**: 5% YES (95% NO) vs **Market Price**: 84.5% YES",
            [],
            [],
            "news_research",
        )
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["model_probability"], 0.05)
        self.assertAlmostEqual(records[0]["market_probability"], 0.845)

    def test_a_pass_that_names_no_market_is_recovered_from_the_quoted_price(self):
        """A PASS is a free calibration point, so do not discard it."""
        records = agent_trading_tick._thesis_forecast_records(
            "- **Action**: PASS\n"
            "- **Market & Venue**: No new position\n"
            "- **Model Probability**: ~83% YES vs **Market Price**: 84.5% (Edge: -1.5pp)",
            [],
            [
                {"platform": "polymarket", "ident": "ceasefire-26", "probability": 0.845},
                {"platform": "kalshi", "ident": "KXOTHER-26", "probability": 0.155},
            ],
            "news_research",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["ticker"], "ceasefire-26")
        self.assertEqual(records[0]["platform"], "polymarket")
        self.assertEqual(records[0]["action"], "PASS")
        self.assertAlmostEqual(records[0]["model_probability"], 0.83)

    def test_an_ambiguous_quoted_price_records_nothing_rather_than_guessing(self):
        """Two candidates at one price make the reference unresolvable."""
        records = agent_trading_tick._thesis_forecast_records(
            "- **Action**: PASS\n"
            "- **Market & Venue**: No new position\n"
            "- **Model Probability**: ~83% vs **Market Price**: 84.5%",
            [],
            [
                {"platform": "polymarket", "ident": "ceasefire-26", "probability": 0.845},
                {"platform": "kalshi", "ident": "KXTWIN-26", "probability": 0.845},
            ],
            "news_research",
        )
        self.assertEqual(records, [])

    def test_the_prompt_no_longer_tells_agents_to_erase_the_market_on_a_pass(self):
        """The old wording cost five of every eight scoreable forecasts."""
        template = agent_trading_tick._TRADING_INSTRUCTION
        self.assertNotIn("write 'No new position' for HOLD/PASS", template)
        self.assertIn("Never write 'No new position' or 'N/A' here", template)
        self.assertIn("even when you PASS or HOLD", template)

    def test_scores_a_pass_forecast_from_a_bounded_venue_resolution_lookup(self):
        thesis = (
            "- **Market & Venue**: [KXPASS-26] on [Kalshi]\n"
            "- **Model Probability**: [25%] vs **Market Price**: [40%] (Edge: [-15%])"
        )
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "accounts.sqlite"
            with mock.patch.dict(
                os.environ, {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(db_path)}, clear=False
            ):
                with benchmark_tools._account_transaction() as conn:
                    agent_trading_tick._persist_thesis_forecasts(
                        conn, "model-pass", "cycle-pass", thesis, [], [], "evidence_edge",
                        forecast_ts="2026-08-20T00:00:00+00:00",
                    )
                    self.assertEqual(
                        agent_trading_tick._refresh_thesis_forecast_outcomes(
                            conn, "model-pass", {("kalshi", "KXPASS-26"): (0, "2026-08-21T00:00:00+00:00")}
                        ),
                        1,
                    )
                    row = conn.execute(
                        "SELECT resolved_outcome, brier_score FROM agent_thesis_forecasts"
                    ).fetchone()

        self.assertEqual(row["resolved_outcome"], 0)
        self.assertAlmostEqual(row["brier_score"], 0.0625)


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

    def test_run_cycle_executes_a_declared_buy_thesis_when_the_tool_was_not_called(self):
        """A final BUY is a paper-trade commitment, not merely display copy."""
        thesis = """### 0. Research Delta
- **Strategy**: EVIDENCE_EDGE
- **New evidence**: A dated, material source changed the forecast.
- **Belief update**: 55% -> 75%

### 1. Decision & Execution
- **Action**: BUY YES
- **Market & Venue**: KXDECLARED on Kalshi
- **Order Sizing**: Quarter Kelly 5% cap

### 2. Resolution Rules & Compliance Audit
- **Rules Verification**: Verified
- **Observation Window**: Verified

### 3. Model Edge & Valuation
- **Model Probability**: 75% vs **Market Price**: 45% (Edge: +30%)

### 4. Catalysts & Invalidation
- **Key Catalysts / Dates**: Tomorrow
- **Invalidation Trigger**: Contradictory primary source"""
        quote = _quote("KXDECLARED", bid=0.44, ask=0.45)
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_NOTES_PATH": str(Path(td) / "notes.json"),
                "FORESEA_AGENT_CYCLE_ID": "declared-buy-cycle",
                "FORESEA_AGENT_PLACE_TRADE_MODE": "shadow",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(agent_trading_tick, "_init_local_agent"),
                mock.patch.object(market_data, "list_kalshi", return_value=[quote]),
                mock.patch.object(market_data, "list_polymarket", return_value=[]),
                mock.patch.object(market_data, "fetch_kalshi", return_value=quote),
                mock.patch.object(
                    agent_trading_tick, "_call_agent_analyze",
                    return_value=self._fake_report(thesis=thesis),
                ),
            ):
                summary = agent_trading_tick.run_cycle("model-declared-buy")
                with benchmark_tools._account_transaction() as conn:
                    position = conn.execute(
                        "SELECT side, quantity FROM agent_positions WHERE agent_id = ? AND ticker = ?",
                        ("model-declared-buy", "KXDECLARED"),
                    ).fetchone()
                    cycle = conn.execute(
                        "SELECT thesis, transcript_json FROM agent_cycles WHERE agent_id = ? AND cycle_id = ?",
                        ("model-declared-buy", "declared-buy-cycle"),
                    ).fetchone()

        self.assertEqual(summary["paper_execution_outcome"], "filled")
        self.assertIsNotNone(position)
        self.assertEqual(position["side"], "yes")
        self.assertGreater(float(position["quantity"]), 0)
        self.assertIn("PAPER ORDER FILLED", cycle["thesis"])
        transcript = json.loads(cycle["transcript_json"])["tool_transcript"]
        self.assertEqual(transcript[-1]["source"], "thesis_execution_reconciliation")
        self.assertEqual(transcript[-1]["args"]["sizing_mode"], "quarter_kelly")

    def test_declared_buy_without_an_exact_current_candidate_is_not_guessed(self):
        thesis = """### 1. Decision & Execution
- **Action**: BUY YES
- **Market & Venue**: KXNOTOFFERED on Kalshi
- **Order Sizing**: Quarter Kelly 5% cap

### 3. Model Edge & Valuation
- **Model Probability**: 70% vs **Market Price**: 40% (Edge: +30%)"""
        with mock.patch.object(benchmark_tools, "place_trade") as place_trade:
            rendered, result = agent_trading_tick._reconcile_thesis_execution(
                agent_id="model-unmatched",
                thesis=thesis,
                transcript=[],
                candidates=[_quote("KXOTHER")],
            )

        place_trade.assert_not_called()
        self.assertEqual(result["outcome"], "not_offered")
        self.assertIn("**Paper execution**: NOT EXECUTED", rendered)

    def test_declared_thesis_reports_the_recorded_direct_paper_fill(self):
        thesis = """### 1. Decision & Execution
- **Action**: BUY YES
- **Market & Venue**: KXDIRECT on Kalshi
- **Order Sizing**: Quarter Kelly 5% cap

### 3. Model Edge & Valuation
- **Model Probability**: 70% vs **Market Price**: 40% (Edge: +30%)"""
        transcript = [{
            "action": "place_trade",
            "args": {"ticker": "KXDIRECT"},
            "observation": json.dumps({"ok": True, "execution": {"filled_quantity": 5}}),
        }]
        with mock.patch.object(benchmark_tools, "place_trade") as place_trade:
            rendered, result = agent_trading_tick._reconcile_thesis_execution(
                agent_id="model-direct-fill", thesis=thesis, transcript=transcript, candidates=[]
            )

        place_trade.assert_not_called()
        self.assertEqual(result["outcome"], "filled")
        self.assertIn("**Paper execution**: PAPER ORDER FILLED", rendered)

    def test_recorded_paper_attempt_replaces_a_false_live_trading_disabled_claim(self):
        thesis = """### 1. Decision & Execution
- **Action**: **HOLD (trade to close was attempted but live trading is disabled; position remains open)**
- **Paper execution**: not available
"""
        transcript = [{
            "action": "place_trade",
            "args": {"ticker": "KXCLOSE"},
            "observation": json.dumps({"ok": True, "execution": {"filled_quantity": 2.5}}),
        }]

        rendered, result = agent_trading_tick._reconcile_thesis_execution(
            agent_id="model-close-simulation", thesis=thesis, transcript=transcript, candidates=[]
        )

        self.assertEqual(result["outcome"], "filled")
        self.assertIn("CLOSE ATTEMPT RECORDED", rendered)
        self.assertIn("**Paper execution**: PAPER ORDER FILLED", rendered)
        self.assertNotIn("live trading is disabled", rendered.lower())

    def test_recorded_tool_error_is_not_presented_as_a_guardrail_rejection(self):
        thesis = """### 1. Decision & Execution
- **Action**: BUY YES
- **Market & Venue**: KXERROR on Kalshi
- **Order Sizing**: Quarter Kelly 5% cap

### 3. Model Edge & Valuation
- **Model Probability**: 70% vs **Market Price**: 40% (Edge: +30%)"""
        transcript = [{
            "action": "place_trade",
            "args": {"ticker": "KXERROR"},
            "observation": json.dumps({
                "ok": False,
                "tool": "place_trade",
                "error": "float() argument must be a string or a real number, not 'NoneType'",
            }),
        }]

        rendered, result = agent_trading_tick._reconcile_thesis_execution(
            agent_id="model-tool-error", thesis=thesis, transcript=transcript, candidates=[]
        )

        self.assertEqual(result["outcome"], "error")
        self.assertIn("**Paper execution**: PAPER ORDER ERROR", rendered)
        self.assertIn("NoneType", rendered)
        self.assertNotIn("rejected by a guardrail", rendered)

    def test_legacy_truncated_successful_trade_is_recovered_as_a_fill(self):
        thesis = "BUY YES on KXLEGACY on Kalshi"
        transcript = [{
            "action": "place_trade",
            "observation": (
                '{"ok": true, "execution": {"filled_quantity": 4.25}, '
                '"account": {"open_positions": ['
            ),
        }]

        rendered, result = agent_trading_tick._reconcile_thesis_execution(
            agent_id="model-legacy-fill", thesis=thesis, transcript=transcript, candidates=[]
        )

        self.assertEqual(result["outcome"], "filled")
        self.assertIn("PAPER ORDER FILLED", rendered)
        self.assertIn("retained execution summary", rendered)

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

    def test_a_transient_503_earns_a_second_attempt_but_a_429_does_not(self):
        # A bare 503 is transient -- the same model answers minutes later, and
        # losing the cycle costs a whole model on the board. A 429 is the
        # opposite: retrying multiplies requests during the very incident that
        # caused it. SCADS reports a quota rejection as a 503 whose text names
        # the 429, so the wrapped form must be treated as a 429, not a 503.
        import asyncio

        cases = [
            ("503: The model is temporarily unavailable. Please retry.", 2),
            ("503: temporarily unavailable (upstream returned HTTP 429)", 1),
            ("429: rate limit exceeded", 1),
            ("Request timed out after 120s", 2),
        ]
        for message, expected_calls in cases:
            with self.subTest(message=message):
                calls = []

                async def _fails(req, request=None, _m=message, _calls=calls):
                    _calls.append(1)
                    raise RuntimeError(_m)

                with (
                    mock.patch.object(agent_trading_tick, "AGENT_ANALYZE_RETRIES", 1),
                    mock.patch.object(
                        agent_trading_tick, "AGENT_ANALYZE_UNAVAILABLE_RETRIES", 2),
                    mock.patch.object(
                        agent_trading_tick, "AGENT_ANALYZE_RETRY_BACKOFF_S", 0.0),
                    mock.patch(
                        "analyzing_llm_rationale.server.agent_analyze", side_effect=_fails),
                ):
                    with self.assertRaises(RuntimeError):
                        asyncio.run(
                            agent_trading_tick._call_agent_analyze("question text"))
                self.assertEqual(len(calls), expected_calls)

    def test_retries_use_exponential_async_backoff_before_a_recovery(self):
        import asyncio

        report = SimpleNamespace(thesis="Recovered.", tool_transcript=[])
        with (
            mock.patch.object(agent_trading_tick, "AGENT_ANALYZE_RETRIES", 4),
            mock.patch.object(agent_trading_tick, "AGENT_ANALYZE_RETRY_BACKOFF_S", 20.0),
            mock.patch("analyzing_llm_rationale.server.agent_analyze", side_effect=[
                RuntimeError("temporary 503"), RuntimeError("temporary 503"), report,
            ]),
            mock.patch.object(agent_trading_tick.asyncio, "sleep", new_callable=mock.AsyncMock) as sleep,
        ):
            actual = asyncio.run(agent_trading_tick._call_agent_analyze("question text"))

        self.assertIs(actual, report)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [20.0, 40.0])

    def test_agent_analyze_request_requires_kelly_sizing_for_benchmark_trades(self):
        import asyncio

        captured = {}

        async def _capture(req, request=None):
            captured["require_kelly_sizing"] = req.benchmark_tools
            return SimpleNamespace(thesis="Passed.", tool_transcript=[])

        with mock.patch("analyzing_llm_rationale.server.agent_analyze", side_effect=_capture):
            asyncio.run(agent_trading_tick._call_agent_analyze("question text"))

        self.assertTrue(captured["require_kelly_sizing"])

    def test_agent_analyze_request_binds_the_worker_model(self):
        import asyncio

        captured = {}

        async def _capture(req, request=None):
            captured["model"] = req.model
            return SimpleNamespace(thesis="Passed.", tool_transcript=[])

        with (
            mock.patch.object(agent_trading_tick, "MODEL", "glm-5-3-flash"),
            mock.patch("analyzing_llm_rationale.server.agent_analyze", side_effect=_capture),
        ):
            asyncio.run(agent_trading_tick._call_agent_analyze("question text"))

        self.assertEqual(captured["model"], "glm-5-3-flash")


class CycleTelemetryTests(unittest.TestCase):
    def test_scads_status_precheck_matches_router_model_and_fails_open(self):
        response = mock.Mock()
        response.json.return_value = {
            "models": {
                "default": [{
                    "name": "zai-org/GLM-5.3-Flash",
                    "real_name": "zai-org/GLM-5.3-Flash",
                    "state": "down",
                }]
            }
        }
        response.raise_for_status.return_value = None
        with (
            mock.patch.object(agent_trading_tick, "SCADS_STATUS_PRECHECK", True),
            mock.patch.object(agent_trading_tick.requests, "get", return_value=response),
        ):
            state, detail = agent_trading_tick._scads_model_readiness("glm-5-3-flash")
        self.assertEqual(state, "down")
        self.assertIn("GLM-5.3-Flash", detail)

        with (
            mock.patch.object(agent_trading_tick, "SCADS_STATUS_PRECHECK", True),
            mock.patch.object(agent_trading_tick.requests, "get", side_effect=RuntimeError("status host down")),
        ):
            state, detail = agent_trading_tick._scads_model_readiness("glm-5-3-flash")
        self.assertIsNone(state)
        self.assertIsNone(detail)

    def test_failure_detail_uses_chained_provider_status_without_leaking_response(self):
        upstream = RuntimeError("status=503 body=authorization=super-secret-response")
        wrapper = RuntimeError("The forecasting model is temporarily unavailable.")
        wrapper.__cause__ = upstream

        self.assertEqual(
            agent_trading_tick._failure_detail(wrapper),
            "Upstream provider returned HTTP 503.",
        )

    def test_main_persists_a_successful_structured_cycle_telemetry_record(self):
        with tempfile.TemporaryDirectory() as td:
            env = {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}
            summary = {
                "candidate_count": 3,
                "tool_steps": 2,
                "settled_count": 1,
                "thesis_published": True,
                "forecast_records": 1,
                "paper_execution_outcome": "filled",
                "provider_model": "zai-org/GLM-5.3-Flash",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(agent_trading_tick, "MODEL", "model-telemetry"),
                mock.patch.object(benchmark_tools, "_current_cycle_id", return_value="cycle-telemetry"),
                mock.patch.object(agent_trading_tick, "init_observability"),
                mock.patch.object(agent_trading_tick, "_scads_model_readiness", return_value=(None, None)),
                mock.patch.object(agent_trading_tick, "run_cycle", return_value=summary) as run,
            ):
                self.assertEqual(agent_trading_tick.main(), 0)
                run.assert_called_once_with("model-telemetry", cycle_id="cycle-telemetry")
                with benchmark_tools._account_transaction() as conn:
                    row = conn.execute("SELECT * FROM agent_cycle_telemetry").fetchone()

        self.assertEqual(row["outcome"], "success")
        self.assertEqual(row["candidate_count"], 3)
        self.assertEqual(row["tool_steps"], 2)
        self.assertEqual(row["settled_count"], 1)
        self.assertEqual(row["thesis_published"], 1)
        self.assertEqual(row["forecast_records"], 1)
        self.assertEqual(row["paper_execution_outcome"], "filled")
        self.assertEqual(row["provider_model"], "zai-org/GLM-5.3-Flash")
        self.assertIsNotNone(row["finished_at"])
        self.assertIsNotNone(row["duration_ms"])

    def test_main_persists_a_provider_failure_with_a_safe_category(self):
        with tempfile.TemporaryDirectory() as td:
            env = {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(agent_trading_tick, "MODEL", "model-telemetry"),
                mock.patch.object(benchmark_tools, "_current_cycle_id", return_value="cycle-provider-503"),
                mock.patch.object(agent_trading_tick, "init_observability"),
                mock.patch.object(agent_trading_tick, "_scads_model_readiness", return_value=(None, None)),
                mock.patch.object(
                    agent_trading_tick,
                    "run_cycle",
                    side_effect=RuntimeError("503: The forecasting model is temporarily unavailable."),
                ),
            ):
                self.assertEqual(agent_trading_tick.main(), agent_trading_tick.PROVIDER_DEGRADATION_EXIT_CODE)
                with benchmark_tools._account_transaction() as conn:
                    row = conn.execute("SELECT * FROM agent_cycle_telemetry").fetchone()

        self.assertEqual(row["outcome"], "failure")
        self.assertEqual(row["failure_kind"], "provider_unavailable")
        self.assertEqual(row["failure_detail"], "Upstream provider returned HTTP 503.")
        self.assertIsNotNone(row["finished_at"])

    def test_main_defers_a_model_explicitly_paused_by_scads_without_running_tools(self):
        with tempfile.TemporaryDirectory() as td:
            env = {"FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite")}
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(agent_trading_tick, "MODEL", "model-paused"),
                mock.patch.object(benchmark_tools, "_current_cycle_id", return_value="cycle-paused"),
                mock.patch.object(agent_trading_tick, "init_observability"),
                mock.patch.object(
                    agent_trading_tick,
                    "_scads_model_readiness",
                    return_value=("down", "SCADS status check reports model-paused as down."),
                ),
                mock.patch.object(agent_trading_tick, "run_cycle") as run,
            ):
                self.assertEqual(agent_trading_tick.main(), agent_trading_tick.PROVIDER_DEGRADATION_EXIT_CODE)
                with benchmark_tools._account_transaction() as conn:
                    row = conn.execute("SELECT * FROM agent_cycle_telemetry").fetchone()

        run.assert_not_called()
        self.assertEqual(row["outcome"], "deferred")
        self.assertEqual(row["failure_kind"], "provider_paused")
        self.assertIn("status check", row["failure_detail"])


if __name__ == "__main__":
    unittest.main()
