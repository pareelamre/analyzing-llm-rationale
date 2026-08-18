"""Agent capability upgrades: built-in skills, track-record grounding, ReAct loop."""
from __future__ import annotations

import asyncio
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
        self.assertIn("longshot", note.lower())


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


if __name__ == "__main__":
    unittest.main()
