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
        self.assertIn("live foresea forecasts", renderer.lower())
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
        self.assertIn("'kimi-k3':", index)
        self.assertIn("'gemma-4-26b-a4b-it':", index)
        self.assertIn("raw.includes('kimi-k3')", index)
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
            "async function openForecastDrift", 1
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

    def test_mtm_methodology_uses_quiet_agentic_style_not_a_callout_banner(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        renderer = index.split("function _renderMtmPage(d) {", 1)[1].split(
            "function renderEdgeBoard", 1
        )[0]

        self.assertIn('class="mtm-methodology"', renderer)
        self.assertIn("mtm-methodology-label", renderer)
        self.assertNotIn("background:rgba(6,182,212,0.05)", renderer)

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

    def test_server_compact_mtm_rows_preserve_risk_stats(self):
        from analyzing_llm_rationale.server import _compact_mark_to_market_by_model

        compact = _compact_mark_to_market_by_model([{
            "model": "test-model",
            "sharpe": 1.25,
            "max_drawdown": 0.12,
            "account": {"account_value": 10_000.0, "value_curve": []},
        }])

        self.assertEqual(compact[0]["sharpe"], 1.25)
        self.assertEqual(compact[0]["max_drawdown"], 0.12)

    def test_mtm_renderer_keeps_risk_cells_aligned_after_sizing_switch(self):
        index = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")
        updater = index.split("function _updateMtmViewInPlace() {", 1)[1].split(
            "function _refreshActiveEdgeBoardHost", 1
        )[0]

        self.assertIn("const risk = _mtmRiskStats(m);", updater)
        self.assertIn("<td><b>${sharpeStr}</b></td>", updater)
        self.assertIn("<td>${account.n_trades ?? m.n_trades ?? 0}</td>", updater)

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

    def test_no_dedicated_agentic_nav_entry_point_left(self):
        # The agentic board lives inside the Edge board's "Agentic" tab, not a
        # standalone overlay or its own side-nav link -- Edge board's existing
        # nav button is the only way in now (user-requested: a separate
        # "Agentic trading" side-nav entry was redundant with it).
        for name, index in self._both().items():
            with self.subTest(file=name):
                self.assertNotIn("openAgentTradingBoard", index)
                self.assertNotIn("Agentic trading", index)
                self.assertNotIn('id="agentTradingOverlay"', index)
                self.assertNotIn('id="agentTradingBody"', index)
                self.assertNotIn("closeAgentTradingBoard", index)

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

    def test_agentic_feed_has_a_per_model_latest_thesis_fallback(self):
        for name, index in self._both().items():
            with self.subTest(file=name):
                renderer = index.split("function renderAgentTradingBoard(d) {", 1)[1].split(
                    "async function removePersonalLedgerEntry", 1
                )[0]
                self.assertIn("const latestTheses = d.latest_theses || {};", renderer)
                self.assertIn("latestTheses[activeFilter]", renderer)
                self.assertIn("No published thesis for", renderer)
                self.assertIn("const banner = ", renderer)
                # Present in the loading and error early-return branches too.
                self.assertIn("host.innerHTML = `${banner}<p", renderer)

    def test_equity_curve_head_bubbles_show_the_model_abbreviation_not_a_boolean(self):
        # Each curve's end-of-line bubble renders String(curve.head) as its
        # label (see _equitySvg) -- a literal `head: true` on the curve object
        # renders as the text "true" on every bubble instead of the model's
        # short abbreviation (e.g. "COD", "L33").
        for name, index in self._both().items():
            with self.subTest(file=name):
                renderer = index.split("function renderAgentTradingBoard(d) {", 1)[1].split(
                    "async function removePersonalLedgerEntry", 1
                )[0]
                self.assertIn("head: _ebModelHead(row.agent_id)", renderer)
                self.assertNotIn("head: true", renderer)

    def test_fetches_the_dedicated_board_endpoint(self):
        for name, index in self._both().items():
            with self.subTest(file=name):
                loader = index.split("async function loadAgentTradingSection() {", 1)[1].split(
                    "function _agentTradingActivityLine", 1
                )[0]
                self.assertIn("_liveJsonFetch('/agent-trading/board')", loader)
        loader = self._both()["frontend"].split("async function loadAgentTradingSection() {", 1)[1].split(
            "function _agentTradingActivityLine", 1
        )[0]
        self.assertIn("_agentTradingRequest", loader)
        self.assertIn("_agentTradingCachedBoard", loader)
        self.assertIn("_agentTradingBoardSignature", loader)

    def test_agentic_thesis_cards_hide_legacy_na_placeholders(self):
        index = self._both()["frontend"]
        formatter = index.split("function _thesisBulletForDisplay(rawBullet) {", 1)[1].split(
            "function _formatThesisHtml", 1
        )[0]
        self.assertIn("unavailableComparison", formatter)
        self.assertIn("not assessed", formatter)
        self.assertIn("visibleBlocks", index)

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

    def test_periodic_refresh_repopulates_the_agentic_tab(self):
        # Regression test: renderEdgeBoard() only ever emits a "Loading…"
        # placeholder for the Agentic tab (its data isn't part of the
        # /edge-board payload), and refreshEdgeForecasts()'s 5-minute poll
        # fully replaces #edgeBody via renderEdgeBoard() -- without this it
        # would wipe the tab back to that placeholder forever once it's the
        # active view.
        #
        # loadEdgeBoard() (the first-ever Edge board open) does NOT need the
        # same trigger: the Agentic tab button can't exist -- let alone be
        # clicked -- until renderEdgeBoard() has produced real markup at
        # least once, and that only happens after loadEdgeBoard()'s own
        # fetch resolves. So _ebView can't already be 'agentic' the first
        # time loadEdgeBoard() runs; adding the same trigger there would be
        # dead code (this was true before, and became load-bearing-false
        # once the side-nav shortcut that could open Edge board and
        # immediately switch tabs -- racing this same fetch -- was removed).
        for name, index in self._both().items():
            with self.subTest(file=name):
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

    def test_dialog_brand_links_close_the_dialog_not_force_landing(self):
        # Regression: the "Foresea" brand link inside each overlay's header
        # called showLanding() unconditionally, so clicking it while signed
        # into the app (e.g. from the Edge board) yanked the user all the way
        # out to the marketing landing page instead of just closing the
        # dialog like the adjacent Close button does. Confirmed live before
        # the fix: currentView flipped from 'app' to 'landing' on click.
        overlays = [
            ("trackOverlay", "closeTrackRecord()"),
            ("edgeOverlay", "closeEdgeBoard()"),
            ("watchOverlay", "closeWatchlistPage()"),
            ("ledgerOverlay", "closePersonalLedger()"),
        ]
        for name, index in self._both().items():
            with self.subTest(file=name):
                for overlay_id, close_call in overlays:
                    marker = f'id="{overlay_id}"'
                    self.assertIn(marker, index)
                    section = index.split(marker, 1)[1][:600]
                    brand_button = section.split("brand-home", 1)[1][:120]
                    self.assertIn(f'onclick="{close_call}"', brand_button)
                    self.assertNotIn("showLanding()", brand_button)

    def test_edge_board_load_failure_resets_the_loaded_flag_and_offers_retry(self):
        # Regression: edgeLoaded was set true before the fetch even started
        # and never reset on failure, so one transient /edge-board error
        # permanently stuck the whole edge board (all three tabs, including
        # Agentic) on "not available yet" for the rest of the page session --
        # neither closing/reopening the dialog nor switching tabs retried it,
        # only a hard page refresh recovered.
        for name, index in self._both().items():
            with self.subTest(file=name):
                loader = index.split("async function loadEdgeBoard() {", 1)[1].split(
                    "async function openWatchlistPage", 1
                )[0]
                catch_block = loader.split("} catch (e) {", 1)[1]
                self.assertIn("edgeLoaded = false", catch_block)
                self.assertIn("loadEdgeBoard();", catch_block)
                self.assertIn(">Retry<", catch_block)

    def test_track_record_load_failure_resets_the_loaded_flag_and_offers_retry(self):
        for name, index in self._both().items():
            with self.subTest(file=name):
                loader = index.split("async function loadTrackRecord() {", 1)[1].split(
                    "let edgeLoaded", 1
                )[0]
                catch_block = loader.split("} catch (e) {", 1)[1]
                self.assertIn("trackLoaded = false", catch_block)
                self.assertIn("loadTrackRecord();", catch_block)
                self.assertIn(">Retry<", catch_block)

    def test_agentic_section_error_state_offers_retry(self):
        for name, index in self._both().items():
            with self.subTest(file=name):
                renderer = index.split("function renderAgentTradingBoard(d) {", 1)[1].split(
                    "async function removePersonalLedgerEntry", 1
                )[0]
                error_branch = renderer.split("if (d && d.error) {", 1)[1][:400]
                self.assertIn("loadAgentTradingSection();", error_branch)
                self.assertIn(">Retry<", error_branch)


if __name__ == "__main__":
    unittest.main()
