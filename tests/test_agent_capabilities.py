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


if __name__ == "__main__":
    unittest.main()
