"""SSE keep-alive: slow steps and a slow first token never leave the agent stream silent."""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import server  # noqa: E402


class KeepaliveTests(unittest.TestCase):
    def test_a_slow_step_gets_keepalive_comments_and_then_its_result(self):
        async def slow():
            await asyncio.sleep(0.07)
            return "ready"

        async def run():
            task = asyncio.ensure_future(slow())
            pings = [p async for p in server._keepalive_until(task, interval=0.02)]
            return pings, task.result()

        pings, result = asyncio.run(run())
        self.assertGreaterEqual(len(pings), 2)
        self.assertTrue(all(p == ": keep-alive\n\n" for p in pings))
        self.assertEqual(result, "ready")

    def test_a_fast_step_gets_none(self):
        async def run():
            task = asyncio.ensure_future(asyncio.sleep(0, result=1))
            return [p async for p in server._keepalive_until(task, interval=1.0)], task.result()

        self.assertEqual(asyncio.run(run()), ([], 1))

    def test_a_slow_first_token_is_padded_and_every_token_still_arrives_in_order(self):
        async def tokens():
            await asyncio.sleep(0.07)
            for t in ("a", "b", "c"):
                yield t

        async def run():
            return [x async for x in server._with_keepalive(tokens(), interval=0.02)]

        out = asyncio.run(run())
        self.assertIsNone(out[0])
        self.assertEqual([x for x in out if x is not None], ["a", "b", "c"])

    def test_an_error_in_the_stream_still_raises(self):
        async def broken():
            yield "a"
            raise RuntimeError("provider failed")

        async def run():
            return [x async for x in server._with_keepalive(broken(), interval=1.0)]

        with self.assertRaises(RuntimeError):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
