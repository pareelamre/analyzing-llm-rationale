"""Agent capability upgrades: built-in skills, track-record grounding, ReAct loop."""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import agent_capabilities as ac  # noqa: E402


class BuiltinSkillsTests(unittest.TestCase):
    def test_four_skills_with_name_and_instruction(self):
        skills = ac.builtin_skills()
        self.assertEqual({s["name"] for s in skills},
                         {"Base rate", "Scenario decomposition", "Red team", "Key drivers"})
        self.assertTrue(all(s["instruction"] for s in skills))


class GroundingNoteTests(unittest.TestCase):
    def test_none_aggregate(self):
        self.assertEqual(ac.build_grounding_note(None), "")

    def test_no_resolved_yet(self):
        note = ac.build_grounding_note({"n_snapshots_resolved": 0})
        self.assertIn("no resolved forecasts", note.lower())

    def test_with_skill_and_calibration(self):
        agg = {"n_snapshots_resolved": 40,
               "overall": {"skill_vs_market": 0.03},
               "calibration_model": {"applied": True, "raw_ece": 0.12}}
        note = ac.build_grounding_note(agg)
        self.assertIn("40 resolved", note)
        self.assertIn("skill vs market", note.lower())
        # With no resolved bins there is no evidence of a tail bias, so the
        # note must not assert one. It previously claimed a longshot bias
        # unconditionally, in a fixed direction, whatever the record said.
        self.assertNotIn("longshot", note.lower())

    def test_tail_advice_follows_the_record_instead_of_a_fixed_direction(self):
        # The hardcoded line told every model it had "overpriced" longshots and
        # to shade tail YES estimates DOWN. The live record said the opposite:
        # forecasts averaging 6.4% resolved YES 11.9% of the time across 1,486
        # resolutions, so that advice pushed every tail estimate the wrong way.
        under = {"n_snapshots_resolved": 2710, "calibration": [
            {"avg_predicted": 0.035, "observed_yes_rate": 0.082, "n": 1107},
            {"avg_predicted": 0.149, "observed_yes_rate": 0.227, "n": 379},
        ]}
        over = {"n_snapshots_resolved": 2710, "calibration": [
            {"avg_predicted": 0.035, "observed_yes_rate": 0.010, "n": 1107},
            {"avg_predicted": 0.149, "observed_yes_rate": 0.090, "n": 379},
        ]}
        under_note = ac.build_grounding_note(under).lower()
        over_note = ac.build_grounding_note(over).lower()

        self.assertIn("under-priced", under_note)
        self.assertIn("do not reflexively shade tail yes estimates down", under_note)
        self.assertNotIn("over-priced", under_note)
        # Same code, opposite record, opposite advice.
        self.assertIn("over-priced", over_note)
        self.assertIn("shade tail yes estimates down", over_note)

    def test_high_confidence_bias_is_measured_not_asserted(self):
        agg = {"n_snapshots_resolved": 2710, "calibration": [
            {"avg_predicted": 0.846, "observed_yes_rate": 0.681, "n": 94},
        ]}
        note = ac.build_grounding_note(agg)
        # The figure that used to be written in by hand as "~68%".
        self.assertIn("85%", note)
        self.assertIn("68%", note)
        self.assertIn("94 resolved", note)

    def test_a_thin_bin_does_not_become_a_stated_bias(self):
        agg = {"n_snapshots_resolved": 12, "calibration": [
            {"avg_predicted": 0.04, "observed_yes_rate": 0.90, "n": 3},
        ]}
        note = ac.build_grounding_note(agg).lower()
        self.assertNotIn("tail bias", note)
        self.assertNotIn("under-priced", note)


class ParseActionTests(unittest.TestCase):
    def test_parses_action(self):
        a = ac.parse_action('{"thought":"x","action":"get_market","args":{"platform":"poly"}}')
        self.assertEqual(a["action"], "get_market")
        self.assertEqual(a["args"]["platform"], "poly")

    def test_parses_final_in_code_fence(self):
        a = ac.parse_action('```json\n{"final": "the answer"}\n```')
        self.assertEqual(a["final"], "the answer")

    def test_no_json_returns_none(self):
        self.assertIsNone(ac.parse_action("just prose, no json here"))

    def test_salvages_the_live_glm_malformed_envelope(self):
        # Verbatim from glm-5.2-fp8's store: a stray `,"` after the thought
        # string makes this invalid JSON. It repeated identically in 24 of its
        # last 25 cycles, each ending at step 0 having called no tool at all,
        # with this envelope stored as the model's public thesis.
        raw = (
            '{"thought": "I need to research all three candidate markets to form '
            'independent probability assessments. Let me start with parallel web '
            'searches on each topic."," "action": "web_search", "args": {"query": '
            '"Matthew Stafford retirement 2026 NFL Rams news 2025"}}'
        )
        a = ac.parse_action(raw)
        self.assertIsNotNone(a)
        self.assertEqual(a["action"], "web_search")
        self.assertEqual(a["args"]["query"], "Matthew Stafford retirement 2026 NFL Rams news 2025")

    def test_salvage_keeps_nested_arg_objects_intact(self):
        a = ac.parse_action('{"thought": "x",, "action": "place_trade", '
                            '"args": {"order": {"qty": 3}, "ticker": "T"}}')
        self.assertEqual(a["action"], "place_trade")
        self.assertEqual(a["args"], {"order": {"qty": 3}, "ticker": "T"})

    def test_salvage_applies_tool_aliases(self):
        a = ac.parse_action('{"thought": "x",, "action": "google_search", "args": {"query": "q"}}')
        self.assertEqual(a["action"], "web_search")

    def test_salvage_does_not_invent_a_call_from_prose(self):
        # Prose that merely mentions an action word must stay a final answer.
        self.assertIsNone(ac.parse_action("I will use web_search to check the odds."))

    def test_salvage_requires_an_explicit_tool_name(self):
        self.assertIsNone(ac.parse_action('{"thought": "just thinking",, "args": {"query": "q"}}'))

    def test_normalizes_openai_style_native_function_call(self):
        # Regression: minimax-m3 (live, observed) defaults to its own native
        # function-calling shape instead of the prompted {"action", "args"}
        # schema. That JSON parses fine but has no "action" key, so it used
        # to be silently treated as a final answer -- the tool never ran even
        # though the model clearly meant to call it.
        a = ac.parse_action('{"name": "web_search", "parameters": {"query": "cabinet news"}}')
        self.assertEqual(a["action"], "web_search")
        self.assertEqual(a["args"]["query"], "cabinet news")

    def test_parses_action_with_nested_braces_in_string(self):
        text = '{"thought": "Checking rate {cut > 25bps} conditions", "action": "web_search", "args": {"query": "rates"}}'
        a = ac.parse_action(text)
        self.assertIsNotNone(a)
        self.assertEqual(a["action"], "web_search")
        self.assertEqual(a["args"]["query"], "rates")

    def test_normalizes_action_with_parameters_key(self):
        text = '{"action": "web_search", "parameters": {"query": "Fed cut"}}'
        a = ac.parse_action(text)
        self.assertIsNotNone(a)
        self.assertEqual(a["action"], "web_search")
        self.assertEqual(a["args"], {"query": "Fed cut"})
        self.assertNotIn("final", a)

    def test_normalizes_arguments_key_as_well_as_parameters(self):
        a = ac.parse_action('{"name": "web_search", "arguments": {"query": "x"}}')
        self.assertEqual(a["action"], "web_search")
        self.assertEqual(a["args"], {"query": "x"})

    def test_normalizes_stringified_arguments(self):
        # Some providers hand back arguments as a JSON-encoded string rather
        # than a nested object (the raw OpenAI tool_calls convention).
        a = ac.parse_action('{"name": "web_search", "arguments": "{\\"query\\": \\"x\\"}"}')
        self.assertEqual(a["args"], {"query": "x"})

    def test_malformed_stringified_arguments_falls_back_to_empty_dict(self):
        a = ac.parse_action('{"name": "web_search", "arguments": "not json"}')
        self.assertEqual(a["action"], "web_search")
        self.assertEqual(a["args"], {})

    def test_does_not_touch_an_already_well_formed_final_answer(self):
        a = ac.parse_action('{"name": "irrelevant", "final": "the answer"}')
        self.assertEqual(a["final"], "the answer")
        self.assertNotIn("action", a)


class SystemPromptTests(unittest.TestCase):
    def test_extra_rules_included(self):
        rule = "You MUST call the forecast tool before answering."
        prompt = ac.build_system_prompt([{"name": "forecast", "description": "d"}], 4, extra_rules=rule)
        self.assertIn(rule, prompt)
        self.assertIn("forecast", prompt)


class ToolLoopTests(unittest.TestCase):
    def test_calls_tool_then_finalizes(self):
        seen = []

        async def get_market(args):
            seen.append(args)
            return "price 42%"

        turns = iter([
            '{"thought":"check price","action":"get_market","args":{"platform":"polymarket"}}',
            '{"thought":"done","final":"It trades around 42%."}',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "where does it trade?", {"get_market": get_market},
            [{"name": "get_market", "description": "fetch price"}], chat_fn, max_steps=5))
        self.assertEqual(res["answer"], "It trades around 42%.")
        self.assertEqual(len(res["transcript"]), 1)
        self.assertEqual(res["transcript"][0]["action"], "get_market")
        self.assertFalse(res["truncated"])
        self.assertEqual(seen, [{"platform": "polymarket"}])

    def test_calls_tool_when_model_uses_its_native_function_call_shape(self):
        # End-to-end regression for the minimax-m3 bug: the model ignores the
        # prompted {"action", "args"} schema and emits OpenAI-style native
        # function-calling JSON instead. The tool must still actually run.
        seen = []

        async def web_search(args):
            seen.append(args)
            return "5 sources found"

        turns = iter([
            '{"name": "web_search", "parameters": {"query": "cabinet news"}}',
            '{"final": "No new departures found."}',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "any cabinet news?", {"web_search": web_search},
            [{"name": "web_search", "description": "search the web"}], chat_fn, max_steps=5))
        self.assertEqual(len(res["transcript"]), 1)
        self.assertEqual(res["transcript"][0]["action"], "web_search")
        self.assertEqual(seen, [{"query": "cabinet news"}])
        self.assertEqual(res["answer"], "No new departures found.")

    def test_unknown_tool_is_handled(self):
        turns = iter([
            '{"action":"nonexistent","args":{}}',
            '{"final":"done anyway"}',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=3))
        self.assertIn("unknown tool", res["transcript"][0]["observation"])
        self.assertEqual(res["answer"], "done anyway")

    def test_large_paper_trade_observation_stays_valid_json(self):
        async def place_trade(_args):
            return json.dumps({
                "ok": True,
                "tool": "place_trade",
                "action_id": "audit-123",
                "execution": {"fill_status": "shadow_assumed_full", "filled_quantity": 7.5},
                "account": {"open_positions": [{"ticker": "X", "padding": "x" * 10_000}]},
            })

        turns = iter([
            '{"action":"place_trade","args":{"ticker":"X"}}',
            '{"final":"Recorded."}',
        ])

        seen_messages = []

        async def chat_fn(messages):
            seen_messages.append(messages)
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "q", {"place_trade": place_trade}, [], chat_fn, max_steps=2, obs_limit=4_000
        ))
        observation = json.loads(res["transcript"][0]["observation"])
        self.assertTrue(observation["ok"])
        self.assertEqual(observation["action_id"], "audit-123")
        self.assertEqual(observation["execution"]["filled_quantity"], 7.5)
        self.assertEqual(observation["account"]["open_positions"][0]["padding"], "x" * 10_000)
        context = seen_messages[1][-1]["content"].removeprefix("Observation: ")
        context_observation = json.loads(context)
        self.assertTrue(context_observation["observation_truncated"])
        self.assertNotIn("account", context_observation)

    def test_long_tool_loop_compacts_old_prompt_observations_but_keeps_audit_full(self):
        async def research(_args):
            return "evidence-" * 1_000

        turns = iter([
            '{"action":"research","args":{"round":1}}',
            '{"action":"research","args":{"round":2}}',
            '{"action":"research","args":{"round":3}}',
            '{"action":"research","args":{"round":4}}',
            '{"final":"The completed research supports a cautious trade."}',
        ])
        seen_messages = []

        async def chat_fn(messages):
            seen_messages.append([dict(message) for message in messages])
            return next(turns)

        result = asyncio.run(ac.run_tool_loop(
            "Research this market.",
            {"research": research},
            [{"name": "research", "description": "collect evidence"}],
            chat_fn,
            max_steps=4,
            obs_limit=4_000,
            observation_context_limit=9_000,
        ))

        # The model's final turn stays under the prompt budget, while the
        # durable transcript remains complete for audit/replay.
        final_observations = [
            message["content"]
            for message in seen_messages[-1]
            if message["content"].startswith("Observation: ")
        ]
        self.assertLessEqual(sum(map(len, final_observations)), 9_000)
        self.assertTrue(any("Earlier observation compacted" in item for item in final_observations))
        self.assertEqual(len(result["transcript"]), 4)
        self.assertTrue(all(len(step["observation"]) == 9_000 for step in result["transcript"]))

    def test_truncates_at_max_steps(self):
        async def tool(args):
            return "obs"

        async def chat_fn(messages):
            # Always calls a tool, never finalizes -> must hit max_steps.
            return '{"action":"t","args":{}}'

        res = asyncio.run(ac.run_tool_loop(
            "q", {"t": tool}, [{"name": "t", "description": "d"}], chat_fn, max_steps=3))
        self.assertTrue(res["truncated"])
        self.assertEqual(res["steps"], 3)
        self.assertEqual(len(res["transcript"]), 3)

    def test_does_not_publish_a_tool_call_when_forced_to_finalize(self):
        async def tool(_args):
            return "observation"

        turns = iter([
            '{"action":"t","args":{}}',
            '{"action":"t","args":{}}',
            # A provider that ignores the finalization instruction must not
            # leak its raw action object into a public thesis.
            '{"thought":"I should search again","action":"t","args":{}}',
        ])

        async def chat_fn(_messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "q", {"t": tool}, [{"name": "t", "description": "d"}], chat_fn, max_steps=2))
        self.assertEqual(res["answer"], "")
        self.assertTrue(res["finalization_failed"])
        self.assertTrue(res["truncated"])

    def test_on_step_called_once_per_tool_call_with_index_thought_action_error(self):
        seen = []

        async def get_market(args):
            return "price 42%"

        async def on_step(step):
            seen.append(step)

        turns = iter([
            '{"thought":"check price","action":"get_market","args":{"platform":"polymarket"}}',
            '{"final":"It trades around 42%."}',
        ])

        async def chat_fn(messages):
            return next(turns)

        asyncio.run(ac.run_tool_loop(
            "where does it trade?", {"get_market": get_market},
            [{"name": "get_market", "description": "fetch price"}], chat_fn, max_steps=5,
            on_step=on_step))

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0], {
            "index": 0, "thought": "check price", "action": "get_market",
            "args": {"platform": "polymarket"}, "observation": "price 42%", "error": False,
        })

    def test_on_step_marks_error_true_for_unknown_tool(self):
        seen = []

        async def on_step(step):
            seen.append(step)

        turns = iter([
            '{"action":"nonexistent","args":{}}',
            '{"final":"done anyway"}',
        ])

        async def chat_fn(messages):
            return next(turns)

        asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=3, on_step=on_step))
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0]["error"])

    def test_on_step_marks_error_true_for_tool_that_raises(self):
        seen = []

        async def boom(args):
            raise RuntimeError("tool exploded")

        async def on_step(step):
            seen.append(step)

        turns = iter([
            '{"action":"boom","args":{}}',
            '{"final":"done anyway"}',
        ])

        async def chat_fn(messages):
            return next(turns)

        asyncio.run(ac.run_tool_loop(
            "q", {"boom": boom}, [{"name": "boom", "description": "d"}], chat_fn, max_steps=3,
            on_step=on_step))
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0]["error"])

    def test_on_step_exception_does_not_break_the_loop(self):
        async def get_market(args):
            return "price 42%"

        async def on_step(step):
            raise RuntimeError("persistence hiccup")

        turns = iter([
            '{"action":"get_market","args":{}}',
            '{"final":"It trades around 42%."}',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "q", {"get_market": get_market}, [{"name": "get_market", "description": "d"}],
            chat_fn, max_steps=5, on_step=on_step))
        self.assertEqual(res["answer"], "It trades around 42%.")
        self.assertEqual(len(res["transcript"]), 1)

    def test_on_step_not_called_on_final_answer_turn(self):
        seen = []

        async def on_step(step):
            seen.append(step)

        async def chat_fn(messages):
            return '{"final": "no tools needed"}'

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=3, on_step=on_step))
        self.assertEqual(res["answer"], "no tools needed")
        self.assertEqual(seen, [])

    def test_on_step_start_fires_before_the_tool_runs_with_no_observation_yet(self):
        order = []

        async def get_market(args):
            order.append("tool_ran")
            return "price 42%"

        async def on_step_start(step):
            order.append(("start", step))

        turns = iter([
            '{"thought":"check price","action":"get_market","args":{"platform":"polymarket"}}',
            '{"final":"It trades around 42%."}',
        ])

        async def chat_fn(messages):
            return next(turns)

        asyncio.run(ac.run_tool_loop(
            "where does it trade?", {"get_market": get_market},
            [{"name": "get_market", "description": "fetch price"}], chat_fn, max_steps=5,
            on_step_start=on_step_start))

        self.assertEqual(len(order), 2)
        self.assertEqual(order[0], ("start", {
            "index": 0, "thought": "check price", "action": "get_market",
            "args": {"platform": "polymarket"},
        }))
        self.assertEqual(order[1], "tool_ran")

    def test_on_step_start_still_fires_when_the_tool_subsequently_raises(self):
        seen = []

        async def boom(args):
            raise RuntimeError("tool exploded")

        async def on_step_start(step):
            seen.append(step)

        turns = iter([
            '{"action":"boom","args":{}}',
            '{"final":"done anyway"}',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "q", {"boom": boom}, [{"name": "boom", "description": "d"}], chat_fn, max_steps=3,
            on_step_start=on_step_start))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["action"], "boom")
        self.assertEqual(res["answer"], "done anyway")

    def test_on_step_start_exception_does_not_break_the_loop(self):
        async def get_market(args):
            return "price 42%"

        async def on_step_start(step):
            raise RuntimeError("persistence hiccup")

        turns = iter([
            '{"action":"get_market","args":{}}',
            '{"final":"It trades around 42%."}',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "q", {"get_market": get_market}, [{"name": "get_market", "description": "d"}],
            chat_fn, max_steps=5, on_step_start=on_step_start))
        self.assertEqual(res["answer"], "It trades around 42%.")
        self.assertEqual(len(res["transcript"]), 1)

    def test_on_step_start_not_called_on_final_answer_turn(self):
        seen = []

        async def on_step_start(step):
            seen.append(step)

        async def chat_fn(messages):
            return '{"final": "no tools needed"}'

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=3, on_step_start=on_step_start))
        self.assertEqual(res["answer"], "no tools needed")
        self.assertEqual(seen, [])

    def test_token_budget_stops_a_runaway_cycle(self):
        # Measured live: one cycle runs 10-21k tokens against a shared
        # 10,000-token-per-minute provider quota, so a deep cycle spends the
        # whole minute and the next model in the serial lane collects the 429.
        calls = []

        async def big_tool(args):
            calls.append(1)
            return "x" * 40000   # ~10k tokens of observation per step

        async def chat_fn(messages):
            return '{"action":"big","args":{}}'

        res = asyncio.run(ac.run_tool_loop(
            "q", {"big": big_tool}, [{"name": "big", "description": "d"}],
            chat_fn, max_steps=16, token_budget=20000))

        # Far fewer than the 16 steps allowed -- stopped on cost, not count.
        self.assertLess(len(calls), 16)
        self.assertGreaterEqual(len(calls), 1)
        self.assertTrue(res["truncated"])

    def test_default_budget_is_a_runaway_guard_not_a_per_cycle_limiter(self):
        # At 20,000 this stopped every healthy model after 2-3 tool calls of
        # the 16 allowed, because the loop resends the whole conversation each
        # turn. The default has to clear the deepest cycle actually observed
        # (7 tool calls) and still catch a true runaway.
        from analyzing_llm_rationale.server import _AGENT_TOOL_TOKEN_BUDGET as default

        async def realistic_tool(_args):
            return "x" * 6000        # ~1.5k tokens of observation, as measured

        async def chat_fn(_messages):
            return '{"action":"t","args":{}}'

        spec = [{"name": "t", "description": "d"}]
        capped = asyncio.run(ac.run_tool_loop(
            "q" * 4000, {"t": realistic_tool}, spec, chat_fn,
            max_steps=16, token_budget=20000))
        current = asyncio.run(ac.run_tool_loop(
            "q" * 4000, {"t": realistic_tool}, spec, chat_fn,
            max_steps=16, token_budget=default))
        uncapped = asyncio.run(ac.run_tool_loop(
            "q" * 4000, {"t": realistic_tool}, spec, chat_fn,
            max_steps=16, token_budget=None))

        # The regression this guards: the old ceiling truncated an ordinary
        # deep cycle less than half way through.
        self.assertEqual(capped["stop_reason"], "token_budget")
        self.assertLess(len(capped["transcript"]), 16)
        # At the current default that same cycle is indistinguishable from
        # having no budget at all -- depth is decided by max_steps, not cost.
        self.assertEqual(current["stop_reason"], "max_steps")
        self.assertEqual(len(current["transcript"]), len(uncapped["transcript"]))
        self.assertEqual(current["tokens_used"], uncapped["tokens_used"])
        # ...and the guard still sits meaningfully above that cycle's cost,
        # so a genuine runaway is still caught.
        self.assertGreater(default, uncapped["tokens_used"])

    def test_budget_stop_is_distinguishable_from_running_out_of_steps(self):
        # Both stops report truncated=True and steps=max_steps, so without an
        # explicit reason a budget-capped cycle is indistinguishable from one
        # that simply used every step -- which is how a capped tick went
        # undiagnosed. The reason has to survive to the caller.
        async def big_tool(_args):
            return "x" * 40000

        async def small_tool(_args):
            return "obs"

        async def chat_fn_big(_messages):
            return '{"action":"big","args":{}}'

        async def chat_fn_small(_messages):
            return '{"action":"t","args":{}}'

        capped = asyncio.run(ac.run_tool_loop(
            "q", {"big": big_tool}, [{"name": "big", "description": "d"}],
            chat_fn_big, max_steps=16, token_budget=20000))
        exhausted = asyncio.run(ac.run_tool_loop(
            "q", {"t": small_tool}, [{"name": "t", "description": "d"}],
            chat_fn_small, max_steps=3))

        # Same truncated flag, different cause.
        self.assertTrue(capped["truncated"])
        self.assertTrue(exhausted["truncated"])
        self.assertEqual(capped["stop_reason"], "token_budget")
        self.assertEqual(exhausted["stop_reason"], "max_steps")
        # And the budget stop reports what it actually managed, not max_steps.
        self.assertLess(capped["steps_completed"], 16)
        self.assertGreaterEqual(capped["steps_completed"], 1)
        self.assertGreater(capped["tokens_used"], 0)

    def test_no_budget_means_no_token_cap(self):
        calls = []

        async def tool(args):
            calls.append(1)
            return "x" * 40000

        async def chat_fn(messages):
            return '{"action":"t","args":{}}'

        res = asyncio.run(ac.run_tool_loop(
            "q", {"t": tool}, [{"name": "t", "description": "d"}],
            chat_fn, max_steps=4))
        self.assertEqual(len(calls), 4)
        self.assertTrue(res["truncated"])

    def test_a_cheap_cycle_keeps_every_step_it_wants(self):
        # The budget must not penalise a small cycle.
        calls = []

        async def tool(args):
            calls.append(1)
            return "short"

        turns = iter(['{"action":"t","args":{}}'] * 3 + ['{"final":"done, no edge found"}'])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "q", {"t": tool}, [{"name": "t", "description": "d"}],
            chat_fn, max_steps=16, token_budget=20000))
        self.assertEqual(len(calls), 3)
        self.assertIn("no edge", res["answer"])

    def test_keyless_json_is_retried_when_the_caller_has_no_backstop(self):
        # Live: gemma-4-26b-a4b-it and gpt-oss-120b both ended at step 0 with
        # valid JSON that was neither an action nor a final, and the board
        # rendered "completed a research pass but did not return a publishable
        # final thesis". The agent-trading path has no forecast backstop to
        # salvage that shape, so ask once for a usable turn.
        turns = iter([
            '{"query": "fed rates", "topn": 5}',
            '{"final": "PASS - market fairly priced, my 47% vs 48%."}',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "q", {}, [], chat_fn, max_steps=5, retry_unusable_final=True))
        self.assertIn("fairly priced", res["answer"])

    def test_keyless_json_is_still_accepted_when_a_backstop_exists(self):
        # Default behaviour is unchanged: /agent/analyze without benchmark
        # tools relies on the deterministic forecast backstop to extract
        # structure from exactly this shape.
        calls = []

        async def chat_fn(messages):
            calls.append(1)
            return '{"probability": 0.7, "rationale": "because"}'

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=5))
        self.assertEqual(len(calls), 1)
        self.assertEqual(res["steps"], 0)
        self.assertIn("probability", res["answer"])

    def test_a_structured_final_does_not_crash_the_loop(self):
        # Live: gemma-4-26b-a4b-it returned an object in `final`. Calling
        # .strip() on it raised AttributeError, which reached the board as a
        # bare 502 and discarded the whole cycle.
        async def chat_fn(messages):
            return json.dumps({"final": {"action": "PASS", "reason": "no edge"}})

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=3))
        self.assertIn("PASS", res["answer"])
        self.assertIn("no edge", res["answer"])

    def test_answer_coercion_handles_every_shape(self):
        self.assertEqual(ac._coerce_answer_text("  hi  "), "hi")
        self.assertEqual(ac._coerce_answer_text(None), "")
        self.assertIn("PASS", ac._coerce_answer_text({"action": "PASS"}))
        self.assertIn("a", ac._coerce_answer_text(["a", "b"]))
        self.assertEqual(ac._coerce_answer_text(42), "42")

    def test_bare_verdict_with_no_research_gets_one_chance_to_do_the_work(self):
        # Live: llama-3.3-70b-instruct returned "PASS" on turn 0 with zero tool
        # calls, in under a second, five times in two days -- and the board
        # rendered a completely blank card.
        async def web_search(args):
            return "5 sources"

        turns = iter([
            '{"final": "PASS"}',
            '{"action":"web_search","args":{"query":"fed"}}',
            '{"final":"PASS - market is fairly priced at 48% vs my 47%."}',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "assess", {"web_search": web_search},
            [{"name": "web_search", "description": "d"}], chat_fn, max_steps=5))

        self.assertIn("fairly priced", res["answer"])
        self.assertEqual(len(res["transcript"]), 1)

    def test_a_reasoned_pass_is_accepted_immediately(self):
        # The nudge must not punish an agent that explains itself.
        async def chat_fn(messages):
            return '{"final": "PASS - no edge: my 47% vs market 48%, inside fees."}'

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=5))
        self.assertEqual(res["steps"], 0)
        self.assertIn("no edge", res["answer"])

    def test_bare_verdict_is_accepted_once_research_has_been_done(self):
        # Having called a tool, a terse PASS is a real decision.
        async def web_search(args):
            return "5 sources"

        turns = iter([
            '{"action":"web_search","args":{"query":"fed"}}',
            '{"final": "PASS"}',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "q", {"web_search": web_search},
            [{"name": "web_search", "description": "d"}], chat_fn, max_steps=5))
        self.assertEqual(res["answer"], "PASS")

    def test_retries_once_after_unparseable_turn_then_succeeds(self):
        # Regression for minimax-m3: a turn with no parseable JSON at all used
        # to be accepted as the final answer immediately, wasting the whole
        # cycle. It should now get one corrective nudge before that happens.
        seen_messages = []

        async def get_market(args):
            return "price 42%"

        turns = iter([
            "I'll research the current state of this market.",
            '{"action":"get_market","args":{}}',
            '{"final":"It trades around 42%."}',
        ])

        async def chat_fn(messages):
            seen_messages.append(list(messages))
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "where does it trade?", {"get_market": get_market},
            [{"name": "get_market", "description": "fetch price"}], chat_fn, max_steps=5))

        self.assertEqual(res["answer"], "It trades around 42%.")
        self.assertFalse(res["truncated"])
        self.assertEqual(len(res["transcript"]), 1)
        self.assertIn("could not be parsed", seen_messages[1][-1]["content"])

    def test_retries_once_when_turn_is_a_foreign_tool_call_dialect(self):
        # Regression for minimax-m3: it sometimes emits Claude-style
        # <function_calls><invoke> XML instead of the prompted JSON schema.
        # parse_action can't find any JSON in that at all.
        async def get_market(args):
            return "price 42%"

        turns = iter([
            '<function_calls><invoke name="get_market">',
            '{"action":"get_market","args":{}}',
            '{"final":"done"}',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "q", {"get_market": get_market},
            [{"name": "get_market", "description": "d"}], chat_fn, max_steps=5))
        self.assertEqual(res["answer"], "done")
        self.assertEqual(len(res["transcript"]), 1)

    def test_gives_up_after_one_failed_retry_and_returns_raw_text(self):
        turns = iter([
            "I'll research the current state of this market.",
            'I am still just thinking out loud, not calling anything.',
        ])

        async def chat_fn(messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=5))
        self.assertEqual(res["answer"], "I am still just thinking out loud, not calling anything.")
        self.assertEqual(res["steps"], 1)
        self.assertEqual(res["transcript"], [])
        self.assertFalse(res["truncated"])

    def test_no_retry_when_no_step_budget_remains(self):
        calls = []

        async def chat_fn(messages):
            calls.append(1)
            return "I'll research the current state of this market."

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=1))
        self.assertEqual(len(calls), 1)
        self.assertEqual(res["answer"], "I'll research the current state of this market.")
        self.assertEqual(res["steps"], 0)

    def test_no_retry_for_valid_json_missing_action_key(self):
        # A model that answers directly with structured data (e.g. raw
        # forecast fields) instead of a tool-call envelope is valid JSON,
        # just not one of the two recognized shapes -- this must still be
        # treated as an immediate final answer (a deterministic backstop may
        # extract structure from it), not retried like genuinely unparseable
        # output.
        calls = []

        async def chat_fn(messages):
            calls.append(1)
            return '{"probability": 0.7, "rationale": "because"}'

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=5))
        self.assertEqual(len(calls), 1)
        self.assertEqual(res["steps"], 0)
        self.assertIn("probability", res["answer"])

    def test_a_fluent_thesis_with_no_tool_calls_is_sent_back_for_research(self):
        # minimax-m3, live: published a full house-format thesis on a
        # zero-tool cycle -- "### 0. Research Delta", a strategy line and
        # specific probabilities -- having fetched no market and searched no
        # evidence. The old guard only caught a bare one-liner, so fluency
        # stood in for work and the board showed priors as research.
        searched = []
        polished = (
            '{"final": "### 0. Research Delta\\n- **Strategy**: PASS\\n'
            '- **New evidence**: No material new evidence this cycle. Prior research '
            'already established the incumbent is returning.\\n'
            '- **Belief update**: No material change. P(YES) ~2%."}'
        )

        async def web_search(_args):
            searched.append(1)
            return "3 sources"

        turns = iter([
            polished,
            '{"thought":"check it","action":"web_search","args":{"q":"x"}}',
            '{"final": "PASS - checked the market, no edge after fees."}',
        ])

        async def chat_fn(_messages):
            return next(turns)

        res = asyncio.run(ac.run_tool_loop(
            "q", {"web_search": web_search},
            [{"name": "web_search", "description": "d"}], chat_fn, max_steps=5))

        # Sent back once, and the work actually happened before the verdict.
        self.assertEqual(len(searched), 1)
        self.assertEqual(len(res["transcript"]), 1)
        self.assertIn("no edge", res["answer"])

    def test_a_tool_less_turn_still_accepts_a_reasoned_answer(self):
        # The counterpart: a stage that deliberately offers no tools must not
        # be nagged to call one that does not exist.
        calls = []

        async def chat_fn(_messages):
            calls.append(1)
            return '{"final": "PASS - no edge: my 47% vs market 48%, inside fees."}'

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=5))
        self.assertEqual(len(calls), 1)
        self.assertEqual(res["steps"], 0)

    def test_retry_does_not_fire_for_a_well_formed_final_answer(self):
        calls = []

        async def chat_fn(messages):
            calls.append(1)
            return '{"final": "no tools needed"}'

        res = asyncio.run(ac.run_tool_loop("q", {}, [], chat_fn, max_steps=5))
        self.assertEqual(len(calls), 1)
        self.assertEqual(res["answer"], "no tools needed")
        self.assertEqual(res["steps"], 0)


if __name__ == "__main__":
    unittest.main()
