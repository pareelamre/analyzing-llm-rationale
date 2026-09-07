"""The radar streams repeated themselves every three seconds.

Measured against the live SSE endpoint over a fifteen-second window:

    5 radar_tick events, 26,798 bytes each
    distinct payloads once the timestamp is removed: 1 of 5

So roughly 32 MB per hour per connected client, to say nothing had
happened. The edge-board record behind it is republished about every
fifteen minutes, and RADAR_CACHE_TTL is unset in production, so each of
those ticks also recomputed the radar from scratch.

Both streams now send a tick on connect and thereafter only when the
markets change. The quiet ticks are heartbeats rather than silence,
because an idle SSE connection through a proxy is closed and a client
cannot otherwise tell a quiet market from a dead stream.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import server as server_module  # noqa: E402


def _radar(*questions):
    return server_module.RadarResponse(
        updated_at="2026-09-07T22:00:00Z",
        markets=[
            server_module.RadarMarket(
                id=f"id{i}", ident=f"K{i}", platform="Kalshi", question=q,
                market_probability=0.5, model_probability=0.6,
                edge=0.1, abs_edge=0.1, side="YES",
            )
            for i, q in enumerate(questions)
        ],
    )


class FingerprintTests(unittest.TestCase):
    def test_the_same_markets_fingerprint_the_same(self):
        a = server_module._radar_markets_fingerprint(_radar("Will X?", "Will Y?"))
        b = server_module._radar_markets_fingerprint(_radar("Will X?", "Will Y?"))
        self.assertEqual(a, b)

    def test_different_markets_fingerprint_differently(self):
        a = server_module._radar_markets_fingerprint(_radar("Will X?"))
        b = server_module._radar_markets_fingerprint(_radar("Will Z?"))
        self.assertNotEqual(a, b)

    def test_a_changed_probability_is_a_change(self):
        one = _radar("Will X?")
        two = _radar("Will X?")
        two.markets[0].market_probability = 0.9
        self.assertNotEqual(
            server_module._radar_markets_fingerprint(one),
            server_module._radar_markets_fingerprint(two),
        )

    def test_the_fingerprint_ignores_nothing_that_a_client_would_see(self):
        """It is built from the same model_dump the tick sends."""
        radar = _radar("Will X?")
        sent = json.dumps(
            [m.model_dump(mode="json") for m in radar.markets],
            sort_keys=True, separators=(",", ":"), default=str,
        )
        self.assertEqual(server_module._radar_markets_fingerprint(radar), sent)


class SseStreamTests(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, radars, ticks):
        """Drive the generator over a fixed sequence of radar snapshots."""
        calls = iter(radars)
        request = mock.Mock()

        async def not_disconnected():
            return False

        request.is_disconnected = not_disconnected
        out = []
        with (
            mock.patch.object(server_module, "_radar_from_track_record",
                              side_effect=lambda limit: next(calls)),
            mock.patch.object(server_module.asyncio, "sleep",
                              new=mock.AsyncMock(return_value=None)),
        ):
            response = await server_module.stream_radar(request)
            agen = response.body_iterator
            for _ in range(ticks):
                out.append(await agen.__anext__())
        return out

    async def test_an_unchanged_radar_sends_one_tick_then_heartbeats(self):
        same = [_radar("Will X?") for _ in range(5)]
        events = await self._collect(same, 5)

        self.assertEqual(events[0].count("event: radar_tick"), 1)
        for event in events[1:]:
            self.assertIn("event: heartbeat", event)
            self.assertNotIn("event: radar_tick", event)

    async def test_a_heartbeat_is_far_smaller_than_a_tick(self):
        events = await self._collect([_radar("Will X?")] * 2, 2)
        tick, heartbeat = events[0], events[1]
        self.assertLess(len(heartbeat), len(tick) // 2)

    async def test_a_change_sends_a_new_tick(self):
        radars = [_radar("Will X?"), _radar("Will X?"), _radar("Will Y?")]
        events = await self._collect(radars, 3)

        self.assertIn("event: radar_tick", events[0])
        self.assertIn("event: heartbeat", events[1])
        self.assertIn("event: radar_tick", events[2])
        self.assertIn("Will Y?", events[2])

    async def test_the_first_tick_always_carries_the_markets(self):
        """A client connecting mid-quiet-period still gets current state."""
        events = await self._collect([_radar("Will X?")], 1)
        payload = json.loads(events[0].split("data: ", 1)[1].strip())
        self.assertEqual(len(payload["markets"]), 1)
        self.assertEqual(payload["markets"][0]["question"], "Will X?")


if __name__ == "__main__":
    unittest.main()
