from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_agent_trading_board.py"
_SPEC = importlib.util.spec_from_file_location("build_agent_trading_board_test_module", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
board_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(board_script)

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402


def _seed_store(path: Path, agent_id: str, *, with_position: bool = False):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    benchmark_tools._ensure_account_schema(conn)
    conn.execute(
        "INSERT INTO agent_accounts (agent_id, starting_cash, cash, realized_pnl, "
        "fees_paid, settlement_fees_paid, updated_at) VALUES (?, 10000, 9950, 0, 0, 0, ?)",
        (agent_id, "2026-08-11T00:00:00+00:00"),
    )
    if with_position:
        conn.execute(
            "INSERT INTO agent_positions (agent_id, platform, ticker, side, quantity, "
            "cost_basis, avg_entry_price, updated_at) VALUES (?, 'kalshi', 'KXFOO-26', "
            "'yes', 50, 20, 0.4, ?)",
            (agent_id, "2026-08-11T00:00:00+00:00"),
        )
    conn.commit()
    conn.close()


class OpenStoreTests(unittest.TestCase):
    def test_missing_store_falls_back_to_an_empty_in_memory_db(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(board_script, "STORE_DIR", Path(td)):
                conn = board_script._open_store("nobody-traded-yet")
                rows = conn.execute("SELECT * FROM agent_accounts").fetchall()
        self.assertEqual(rows, [])

    def test_existing_store_is_read(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model-a"
            model_dir.mkdir()
            _seed_store(model_dir / "store.sqlite", "model-a")
            with mock.patch.object(board_script, "STORE_DIR", Path(td)):
                conn = board_script._open_store("model-a")
                row = conn.execute("SELECT agent_id FROM agent_accounts").fetchone()
                conn.close()
        self.assertEqual(row["agent_id"], "model-a")


class HeldTickersAndQuotesTests(unittest.TestCase):
    def test_held_tickers_only_counts_open_quantity(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td) / "model-a"
            model_dir.mkdir()
            _seed_store(model_dir / "store.sqlite", "model-a", with_position=True)
            with mock.patch.object(board_script, "STORE_DIR", Path(td)):
                conn = board_script._open_store("model-a")
                held = board_script._held_tickers(conn)
                conn.close()
        self.assertEqual(held, {"KXFOO-26"})

    def test_fetch_quotes_skips_failed_lookups_and_keys_lowercase_platform(self):
        def fake_fetch(ticker):
            if ticker == "KXBAD":
                raise market_data.MarketDataError("gone")
            return {"platform": "Kalshi", "ident": ticker, "yes_bid": 0.5}

        with mock.patch.object(market_data, "fetch_kalshi", side_effect=fake_fetch):
            quotes = board_script._fetch_quotes({"KXGOOD", "KXBAD"})
        self.assertEqual(set(quotes.keys()), {("kalshi", "KXGOOD")})


class BuildBoardTests(unittest.TestCase):
    def test_build_board_aggregates_across_models_and_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            store_dir = Path(td) / "stores"
            store_dir.mkdir()
            (store_dir / "model-a").mkdir()
            (store_dir / "model-b").mkdir()
            _seed_store(store_dir / "model-a" / "store.sqlite", "model-a", with_position=True)
            _seed_store(store_dir / "model-b" / "store.sqlite", "model-b")

            with (
                mock.patch.object(board_script, "STORE_DIR", store_dir),
                mock.patch.object(board_script, "_chat_capable_models", return_value=["model-a", "model-b"]),
                mock.patch.object(
                    market_data, "fetch_kalshi",
                    return_value={"platform": "Kalshi", "ident": "KXFOO-26", "yes_bid": 0.6},
                ),
            ):
                board = board_script.build_board()

        self.assertEqual(board["mode"], "shadow")
        self.assertEqual(sorted(board["models"]), ["model-a", "model-b"])
        self.assertEqual(len(board["leaderboard"]), 2)
        agent_ids = {row["agent_id"] for row in board["leaderboard"]}
        self.assertEqual(agent_ids, {"model-a", "model-b"})
        model_a_row = next(r for r in board["leaderboard"] if r["agent_id"] == "model-a")
        # 50 contracts @ 0.6 bid = 30 liquidation value on top of 9950 cash
        self.assertAlmostEqual(model_a_row["account_value"], 9950.0 + 30.0)
        self.assertIn("model-a", board["equity_curves"])
        self.assertIn("model-b", board["equity_curves"])
        self.assertIn("model-a", board["eligibility"])
        self.assertIn("model-b", board["eligibility"])
        self.assertIn("eligible", board["eligibility"]["model-a"])
        # Must be JSON-serializable end to end (no stray sqlite3.Row/etc leaking through).
        json.dumps(board)

    def test_main_writes_output_file(self):
        with tempfile.TemporaryDirectory() as td:
            store_dir = Path(td) / "stores"
            store_dir.mkdir()
            out_path = Path(td) / "out" / "agent_trading_live.json"
            with (
                mock.patch.object(board_script, "STORE_DIR", store_dir),
                mock.patch.object(board_script, "OUTPUT_PATH", out_path),
                mock.patch.object(board_script, "_chat_capable_models", return_value=["model-a"]),
            ):
                rc = board_script.main()
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["models"], ["model-a"])


if __name__ == "__main__":
    unittest.main()
