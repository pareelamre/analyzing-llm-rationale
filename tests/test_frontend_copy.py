import unittest
from pathlib import Path


class FrontendCopyTests(unittest.TestCase):
    def test_edge_board_omits_paper_return_language(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        renderer = index.split("function renderEdgeBoard(d) {", 1)[1].split(
            "function _fmtDate", 1
        )[0]

        self.assertNotIn("paper pnl", renderer.lower())
        self.assertNotIn("paper trading", renderer.lower())
        self.assertNotIn("paper roi", renderer.lower())
        self.assertIn("resolved strategy performance", renderer.lower())
        self.assertIn("resolved markets (p&amp;l)", renderer.lower())
        self.assertIn("saved forecasts (research)", renderer.lower())
        self.assertIn("one settlement", renderer.lower())

    def test_edge_board_model_comparison_chart_tracks_scads_models(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("const _MTM_MODEL_COLORS", index)
        self.assertIn("'scads-alias-code':", index)
        self.assertIn("'scads-alias-reasoning':", index)
        self.assertIn("'kimi-k2.7-code':", index)
        self.assertIn("raw.includes('kimi-k2.7')", index)
        # padR is derived from the head-bubble radius so the newest value is
        # never clipped at the right edge.
        self.assertIn("const padL = 60, padR = HEAD_R + 24, padT = 26", index)
        self.assertIn("legendRowH = 17", index)
        self.assertIn('id="eb-mtm-svg"', index)
        self.assertIn('id="eb-mtm-table-body"', index)
        self.assertIn("data-eq-id", index)
        self.assertIn("const _equityChartStates = new Map()", index)
        self.assertIn("_attachEdgeBoardChartHovers(host)", index)

    def test_mtm_curves_carry_head_labels(self):

        """_equitySvg skips the end-of-line bubble unless the curve sets `head`;
        _renderMtmPage used to omit it, so the MTM chart rendered as bare lines."""
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        renderer = index.split("function _renderMtmPage(d) {", 1)[1].split(
            "function renderEdgeBoard", 1
        )[0]

        self.assertIn("head: _ebModelHead(m.model)", renderer)
        self.assertIn("headTitle:", renderer)

    def test_equity_chart_scales_and_bounds_head_bubbles(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        chart = index.split("function _equitySvg(", 1)[1].split(
            "function _attachEquityHover", 1
        )[0]

        # Responsive height, not a fixed px box that letterboxes when narrow.
        self.assertIn("height:auto", chart)
        # The chart grows to guarantee room for every head bubble rather than
        # dropping the ones that don't fit — this is the "I want all 17
        # bubbles" behaviour, not a fixed-height cap.
        self.assertIn("requiredHeadsHeight", chart)
        self.assertIn("H = padT + cH + padB", chart)
        self.assertNotIn("headCapacity", chart)
        self.assertNotIn("shownHeads", chart)
        # Headroom keeps extremes off the frame.
        self.assertIn("const headPad", chart)
        # One curve missing timestamps must not drop the whole chart to index mode.
        self.assertIn("const hasTimes = timedCurves.length > 0", chart)

    def test_edge_board_polling_survives_history_navigation(self):
        """applyHistoryState closes every overlay before re-opening the requested
        one; the re-open must restart polling rather than bail on the still-set
        `open` class, or the charts freeze."""
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertIn("function _startEdgeTimers()", index)
        self.assertIn("function _stopEdgeTimers()", index)
        opener = index.split("async function openEdgeBoard(", 1)[1].split(
            "async function openArbitrage", 1
        )[0]
        self.assertIn("_startEdgeTimers()", opener)
        # The early-return path must still revive the timers.
        self.assertNotIn("if (ov.classList.contains('open')) return;", opener)
        closer = index.split("function closeEdgeBoard(", 1)[1].split(
            "function _ebAskQuestion", 1
        )[0]
        self.assertIn("seq !== _edgeOpenSeq", closer)

    def test_equity_tooltip_is_bounded(self):
        """A row-per-curve tooltip with ~17 models grows taller than the chart;
        the old code clamped only the top edge, so it ran off the bottom."""
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        hover = index.split("function _attachEquityHover(container) {", 1)[1].split(
            "function _attachEdgeBoardChartHovers", 1
        )[0]

        self.assertIn("TIP_MAX_ROWS", hover)
        self.assertIn("more</div>", hover)
        # Clamp against the measured box on both axes, not a hardcoded guess.
        self.assertIn("tip.offsetWidth", hover)
        self.assertIn("tip.offsetHeight", hover)
        self.assertIn("if (top + th > cr.height)", hover)
        self.assertNotIn("lft + 150 > container.offsetWidth", hover)

    def test_edge_board_refresh_rerenders_full_surface(self):

        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        refresher = index.split("async function refreshEdgeForecasts() {", 1)[1].split(
            "function _fmtEquityDate", 1
        )[0]

        self.assertIn("host.innerHTML = renderEdgeBoard(d)", refresher)
        self.assertIn("const previousPage = _ebPage", refresher)
        self.assertIn("_attachEdgeBoardChartHovers(host)", refresher)
        self.assertIn("refreshEdgePrices()", refresher)

    def test_mtm_sizing_controls_use_delegated_button_handlers(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        renderer = index.split("function _renderMtmPage(d) {", 1)[1].split(
            "function renderEdgeBoard", 1
        )[0]
        edge_renderer = index.split("function renderEdgeBoard(d) {", 1)[1].split(
            "function _fmtDate", 1
        )[0]

        self.assertIn("document.addEventListener('click'", index)
        self.assertIn('id="eb-mtm-container"', renderer)
        self.assertIn('id="eb-mtm-header-title"', renderer)
        self.assertIn("eb-mtm-heading-row", renderer)
        self.assertIn("eb-sizing-tabs", renderer)
        self.assertIn("data-mtm-sizing-mode=\"quarter_kelly\"", renderer)
        self.assertIn("quarter_kelly_by_model", index)
        self.assertIn("d[`${sizingMode}_by_model`]", index)
        self.assertIn("event_type === 'settlement'", index)
        self.assertIn("m.model !== 'crowd-follow'", index)
        self.assertIn("m.model !== 'market-follow'", index)
        self.assertIn("data-mtm-sizing-mode=\"growth_1pct\"", renderer)
        self.assertIn("data-mtm-sizing-mode=\"growth_2pct\"", renderer)
        self.assertNotIn("data-mtm-strategy=", renderer)
        self.assertIn("data-eb-view=\"mtm\"", edge_renderer)
        self.assertNotIn("onclick=\"_setMtmSizingMode", renderer)
        self.assertNotIn("onclick=\"_ebSetView", edge_renderer)

    def test_edge_board_growth_curve_normalizes_multiplier_to_bankroll(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        normalizer = index.split("function _ebNormalizeBankrollCurve(s) {", 1)[1].split(
            "function _ebPnlCurve", 1
        )[0]

        self.assertIn("maxAbs <= 10", normalizer)
        self.assertIn("v * EB_STARTING_BANKROLL", normalizer)
        self.assertIn("v * (EB_STARTING_BANKROLL / 100)", normalizer)

    def test_edge_board_strategy_chart_uses_compound_growth_curve(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        self.assertIn("function _ebPnlCurve(s)", index)

    def test_edge_board_equity_chart_insets_points_from_edges(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        renderer = index.split("function _equitySvg(", 1)[1].split(
            "function _attachEquityHover", 1
        )[0]

        self.assertIn("const pointInsetX", renderer)
        self.assertIn("const innerW", renderer)
        self.assertIn("padL + pointInsetX + ((t - tMin) / tSpan) * innerW", renderer)
        self.assertIn("pointInsetX, innerW", renderer)

    def test_edge_board_omits_mark_to_market_summary_cards(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

        self.assertNotIn('id="eb-mtm-primary-v"', index)
        self.assertNotIn('id="eb-mtm-last-marked-v"', index)
        self.assertNotIn("Council account value", index)
        self.assertNotIn("cash + bid liquidation", index)
        self.assertNotIn("chart heartbeat", index)
        self.assertNotIn("SCADS + baseline", index)

    def test_server_compact_pnl_strategy_preserves_growth_curve_ts(self):
        from analyzing_llm_rationale.server import _compact_pnl_strategy
        strategy = {
            "n_bets": 5,
            "roi": 0.1,
            "growth_curve": [10000, 10500],
            "growth_curve_ts": ["2026-06-01T00:00:00+00:00", "2026-06-02T00:00:00+00:00"],
            "equity_curve": [0, 500],
            "equity_curve_ts": ["2026-06-01T00:00:00+00:00", "2026-06-02T00:00:00+00:00"],
        }
        compact = _compact_pnl_strategy(strategy)
        self.assertIn("growth_curve_ts", compact)
        self.assertEqual(len(compact["growth_curve"]), len(compact["growth_curve_ts"]))

    def test_equity_hover_uses_per_curve_points_length_in_index_mode(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        hover = index.split("function _attachEquityHover(container) {", 1)[1].split(
            "function _attachEdgeBoardChartHovers", 1
        )[0]

        self.assertIn("const cLen = c.points ? c.points.length : 0;", hover)
        self.assertIn("idx = Math.max(0, Math.min(cLen - 1, Math.round(frac * (cLen - 1))));", hover)


class AgentTradingBoardFrontendTests(unittest.TestCase):
    """The agentic shadow-trading page (Milestone 2) -- checked against both
    frontend/index.html (source) and static/index.html (served copy), since
    they're kept in sync by hand and have previously drifted."""

    def _both(self):
        root = Path(__file__).resolve().parents[1]
        return {
            "frontend": (root / "frontend" / "index.html").read_text(encoding="utf-8"),
            "static": (root / "static" / "index.html").read_text(encoding="utf-8"),
        }

    def test_side_nav_link_opens_edge_board_agentic_tab_not_its_own_overlay(self):
        # The agentic board lives inside the Edge board's "Agentic" tab, not a
        # standalone overlay -- confirm the old overlay is gone and the
        # side-nav button now routes through openEdgeBoard()/_ebSetView().
        for name, index in self._both().items():
            with self.subTest(file=name):
                self.assertIn('onclick="openAgentTradingBoard()"', index)
                self.assertNotIn('id="agentTradingOverlay"', index)
                self.assertNotIn('id="agentTradingBody"', index)
                self.assertNotIn("closeAgentTradingBoard", index)
                opener = index.split("async function openAgentTradingBoard()", 1)[1].split(
                    "async function loadAgentTradingSection", 1
                )[0]
                self.assertIn("await openEdgeBoard()", opener)
                self.assertIn("_ebSetView('agentic')", opener)

    def test_shadow_only_banner_is_unconditional_not_derived_from_fetched_data(self):
        # The safety banner must be a literal string baked into every render
        # path (including the loading/error states), never something read off
        # the fetched payload -- a compromised or stale artifact must not be
        # able to make this text disappear.
        for name, index in self._both().items():
            with self.subTest(file=name):
                renderer = index.split("function renderAgentTradingBoard(d) {", 1)[1].split(
                    "async function removePersonalLedgerEntry", 1
                )[0]
                self.assertIn("Shadow / paper trading only", renderer)
                self.assertIn("const banner = ", renderer)
                # Present in the loading and error early-return branches too.
                self.assertIn("host.innerHTML = `${banner}<p", renderer)

    def test_fetches_the_dedicated_board_endpoint(self):
        for name, index in self._both().items():
            with self.subTest(file=name):
                loader = index.split("async function loadAgentTradingSection() {", 1)[1].split(
                    "function _agentTradingActivityLine", 1
                )[0]
                self.assertIn("_liveJsonFetch('/agent-trading/board')", loader)

    def test_agentic_is_a_real_edge_board_tab(self):
        for name, index in self._both().items():
            with self.subTest(file=name):
                self.assertIn('data-eb-view="agentic">Agentic</button>', index)
                self.assertIn("['mtm', 'agentic'].includes(view)", index)
                self.assertIn('id="eb-agentic-section"', index)
                # _ebSetView triggers the fetch when switching onto the tab.
                setview = index.split("function _ebSetView(view) {", 1)[1].split(
                    "function _equitySvg", 1
                )[0]
                self.assertIn("loadAgentTradingSection()", setview)

    def test_periodic_refresh_and_initial_load_both_repopulate_the_agentic_tab(self):
        # Regression test: renderEdgeBoard() only ever emits a "Loading…"
        # placeholder for the Agentic tab (its data isn't part of the
        # /edge-board payload), so both loadEdgeBoard()'s first-open fetch
        # and refreshEdgeForecasts()'s 5-minute poll fully replace #edgeBody
        # via renderEdgeBoard() -- each must independently re-trigger
        # loadAgentTradingSection() or the tab gets stuck loading forever
        # (hit exactly this in manual browser verification: openEdgeBoard()
        # doesn't await its own data load, so _ebSetView('agentic') can run
        # before _ebLastData exists and silently no-ops).
        for name, index in self._both().items():
            with self.subTest(file=name):
                load_edge_board = index.split("async function loadEdgeBoard() {", 1)[1].split(
                    "// ── Watchlist full page", 1
                )[0]
                self.assertIn("if (_ebView === 'agentic') loadAgentTradingSection();", load_edge_board)
                refresher = index.split("async function refreshEdgeForecasts() {", 1)[1].split(
                    "function _fmtEquityDate", 1
                )[0]
                self.assertIn("if (_ebView === 'agentic') loadAgentTradingSection();", refresher)

    def test_leaderboard_and_activity_render_real_fields(self):
        for name, index in self._both().items():
            with self.subTest(file=name):
                renderer = index.split("function renderAgentTradingBoard(d) {", 1)[1].split(
                    "async function removePersonalLedgerEntry", 1
                )[0]
                self.assertIn("d.leaderboard", renderer)
                self.assertIn("d.equity_curves", renderer)
                self.assertIn("d.recent_activity", renderer)
                self.assertIn("row.win_rate", renderer)
                self.assertIn("row.trade_count", renderer)
                self.assertIn("_equitySvg(curves", renderer)


if __name__ == "__main__":
    unittest.main()
