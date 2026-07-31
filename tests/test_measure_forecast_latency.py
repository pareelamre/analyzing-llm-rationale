from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock


def _load_measure_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "measure_forecast_latency.py"
    spec = importlib.util.spec_from_file_location("measure_forecast_latency", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ForecastLatencyMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.measure = _load_measure_module()

    def test_stream_parser_records_timing_breakdown_and_served_model(self) -> None:
        def event(name: str, payload: dict[str, object]) -> list[bytes]:
            return [
                f"event: {name}\n".encode(),
                f"data: {json.dumps(payload)}\n".encode(),
                b"\n",
            ]

        lines = [
            *event("meta", {"status": "streaming", "prepare_ms": 42, "model_key": "minimax-m3"}),
            *event("delta", {"text": "hello", "first_delta_ms": 75, "provider_first_delta_ms": 33}),
            *event(
                "done",
                {
                    "response": {
                        "model_key": "minimax-m3",
                        "served_model_name": "MiniMaxAI/MiniMax-M3-MXFP8",
                    }
                },
            ),
        ]

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def readline(self):
                if lines:
                    return lines.pop(0)
                return b""

        with mock.patch.object(self.measure.urllib.request, "urlopen", return_value=FakeResponse()):
            result = self.measure._post_stream(
                "https://foresea.test/predict/stream",
                {"question": "Will it happen?"},
                {"content-type": "application/json"},
                5,
            )

        self.assertEqual(result["status"], 200)
        self.assertEqual(result["server_prepare_ms"], 42)
        self.assertEqual(result["server_first_delta_ms"], 75)
        self.assertEqual(result["provider_first_delta_ms"], 33)
        self.assertEqual(result["delta_chars"], 5)
        self.assertEqual(result["response_model_key"], "minimax-m3")
        self.assertEqual(result["served_model_name"], "MiniMaxAI/MiniMax-M3-MXFP8")

    def test_stream_parser_treats_sse_error_as_failed_measurement(self) -> None:
        def event(name: str, payload: dict[str, object]) -> list[bytes]:
            return [
                f"event: {name}\n".encode(),
                f"data: {json.dumps(payload)}\n".encode(),
                b"\n",
            ]

        lines = [
            *event("meta", {"status": "streaming", "prepare_ms": 1, "model_key": "minimax-m3"}),
            *event("error", {"status_code": 503, "detail": "temporarily unavailable"}),
        ]

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def readline(self):
                if lines:
                    return lines.pop(0)
                return b""

        with mock.patch.object(self.measure.urllib.request, "urlopen", return_value=FakeResponse()):
            result = self.measure._post_stream(
                "https://foresea.test/predict/stream",
                {"question": "Will it happen?"},
                {"content-type": "application/json"},
                5,
            )

        self.assertEqual(result["status"], 503)
        self.assertEqual(result["server_prepare_ms"], 1)
        self.assertEqual(result["error_event"]["detail"], "temporarily unavailable")


if __name__ == "__main__":
    unittest.main()
