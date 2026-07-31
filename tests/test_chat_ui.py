from __future__ import annotations

import unittest
from pathlib import Path


class ChatUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        static_dir = Path(__file__).resolve().parents[1] / "static"
        cls.index_html = (static_dir / "index.html").read_text(encoding="utf-8")
        cls.foresea_html = "\n".join(
            (static_dir / name).read_text(encoding="utf-8")
            for name in ("index.html", "agents.html", "trade.html")
        )

    def test_composer_uses_the_shell_as_its_focus_indicator(self) -> None:
        self.assertIn(
            ".input-box #questionInput:focus-visible {\n      outline: none;",
            self.index_html,
        )

    def test_hidden_live_status_does_not_leave_an_empty_pill(self) -> None:
        self.assertIn(
            ".chat-live-status {\n      display: none !important;",
            self.index_html,
        )
        self.assertIn(
            'class="chat-live-status" aria-hidden="true"',
            self.index_html,
        )

    def test_chat_content_stays_in_the_shared_reading_column(self) -> None:
        self.assertIn(
            "div.className = 'max-w-2xl mx-auto w-full flex justify-end';",
            self.index_html,
        )
        self.assertIn("lg:max-w-[640px]", self.index_html)

    def test_stop_icon_is_geometric_and_centered(self) -> None:
        self.assertIn(
            ".send-btn-stop-icon {\n"
            "      background: currentColor;\n"
            "      border-radius: 1px;\n"
            "      display: block;\n"
            "      height: 8px;\n"
            "      width: 8px;",
            self.index_html,
        )
        self.assertIn("place-items: center;", self.index_html)
        self.assertIn(
            "'<span class=\"send-btn-stop-icon\" aria-hidden=\"true\"></span>'",
            self.index_html,
        )
        self.assertNotIn("btn.textContent = isSending ? '■' : '↑';", self.index_html)

    def test_ui_icons_do_not_depend_on_text_glyphs(self) -> None:
        for glyph in (
            "✕",
            "✎",
            "⧉",
            "■",
            "↻",
            "↑",
            "⌄",
            "☆",
            "★",
            "✓",
            "⚠",
            "←",
            "→",
            "↗",
            "＋",
            "▲",
            "▼",
            "📈",
            "⚗",
        ):
            with self.subTest(glyph=glyph):
                self.assertNotIn(glyph, self.foresea_html)
        self.assertIn('<symbol id="ui-x"', self.index_html)
        self.assertIn("function uiIcon(name, className = '')", self.index_html)

    def test_forecast_wait_state_is_accessible_and_server_driven(self) -> None:
        self.assertIn('class="msg-ai forecast-wait"', self.index_html)
        self.assertIn('role="status" aria-live="polite" aria-atomic="true"', self.index_html)
        self.assertIn(
            "typingEl._setForecastProgress?.(data.status, data);",
            self.index_html,
        )
        self.assertIn("Researching the market", self.index_html)
        self.assertIn("Writing the forecast", self.index_html)
        self.assertNotIn(
            "const phases = ['Gathering evidence",
            self.index_html,
        )

    def test_forecast_wait_state_respects_reduced_motion(self) -> None:
        self.assertIn("@media (prefers-reduced-motion: reduce)", self.index_html)
        self.assertIn(".forecast-wait-orbit", self.index_html)
        self.assertIn(".forecast-wait-scan::after", self.index_html)

    def test_wait_experience_emits_bounded_product_events(self) -> None:
        self.assertIn("trackEvent('forecast_wait_started', { surface });", self.index_html)
        self.assertIn("trackEvent('forecast_wait_phase'", self.index_html)
        self.assertIn("trackEvent('forecast_wait_finished'", self.index_html)
        self.assertIn("duration_bucket: _waitDurationBucket", self.index_html)

    def test_streamed_turn_settles_without_redrawing_the_thread(self) -> None:
        self.assertIn("settlePromptActions(request.userMessageId);", self.index_html)
        finally_block = self.index_html.split(
            "} finally {\n    settlePromptActions(request.userMessageId);",
            1,
        )[1].split("\n  }\n}", 1)[0]
        self.assertNotIn("renderMessages();", finally_block)

    def test_stopped_stream_preserves_partial_answer(self) -> None:
        self.assertIn("request.cancelStream?.();", self.index_html)
        self.assertIn("reader.cancel().catch(() => {});", self.index_html)
        self.assertIn("if (streamCancelled || signal.aborted)", self.index_html)
        self.assertIn("if (text) finish(null, 'cancelled');", self.index_html)
        self.assertIn("data.generation_stopped = true;", self.index_html)
        self.assertIn("forecast_stopped", self.index_html)

    def test_response_is_saved_to_its_originating_conversation(self) -> None:
        self.assertIn(
            "appendMessage(request.convId, { role: 'assistant'",
            self.index_html,
        )
        self.assertIn("activeId !== convId", self.index_html)
        self.assertIn(
            "else if (activeRequests.has(id)) stopGeneration(id);",
            self.index_html,
        )
        self.assertIn("const activeRequests = new Map();", self.index_html)
        self.assertIn("activeRequests.set(request.convId, request);", self.index_html)
        self.assertIn("if (!question || activeRequests.has(activeId)) return;", self.index_html)

    def test_evidence_sources_render_as_clickable_chips(self) -> None:
        self.assertIn("function sourceFeedHtml(sources) {", self.index_html)
        self.assertIn('class="source-chip hover:opacity-80"', self.index_html)
        self.assertIn('href="${escAttr(s.url)}"', self.index_html)
        self.assertIn("const sources   = data.evidence_sources || [];", self.index_html)
        self.assertIn("${sourceFeedHtml(sources)}", self.index_html)

    def test_agent_answers_preserve_evidence_sources_for_chips(self) -> None:
        self.assertIn(
            "const evidenceSources = Array.isArray(report?.evidence_sources) ? report.evidence_sources : [];",
            self.index_html,
        )
        self.assertIn("evidence_sources: evidenceSources,", self.index_html)
        self.assertIn("_agentReportToMarkdown(report, { includeSources: false })", self.index_html)
        self.assertIn("const includeSources = options.includeSources !== false;", self.index_html)

    def test_track_chat_uses_smart_roi_against_crowd_baseline(self) -> None:
        self.assertIn("_trackCurve(smart, '#10b981', 'Smart ROI')", self.index_html)
        self.assertIn("_trackCurve(crowd, '#94a3b8', 'crowd-follow baseline', true)", self.index_html)
        self.assertIn(">Smart ROI</div>", self.index_html)
        self.assertIn(">vs crowd-follow</div>", self.index_html)
        self.assertIn("crowd-follow is the zero-edge baseline", self.index_html)
        self.assertNotIn("Best paper ROI", self.index_html)

    def test_edge_board_uses_smart_roi_with_crowd_baseline(self) -> None:
        self.assertIn("const strategy = isCrowd ? (m.paper_pnl || {}).flat : ((m.paper_pnl || {}).smart", self.index_html)
        self.assertIn("function _roiDeltaPct(x)", self.index_html)
        self.assertIn("Smart ROI minus the crowd-follow baseline", self.index_html)
        self.assertIn("_roiDeltaPct(smartVsCrowd)", self.index_html)
        self.assertIn('id="eb-pnl-edge-v"', self.index_html)
        self.assertIn(">vs crowd-follow</div>", self.index_html)
        self.assertIn(">crowd-follow baseline</div>", self.index_html)
        self.assertIn("<th>Smart ROI</th><th>vs crowd</th>", self.index_html)
        self.assertNotIn('id="eb-pnl-flat-v"', self.index_html)
        self.assertNotIn("<th>Flat ROI</th>", self.index_html)

    def test_edge_board_renders_mark_to_market_account_chart(self) -> None:
        self.assertIn("const mtm = d.mark_to_market_account || null;", self.index_html)
        self.assertIn("const mtmModels = (d.mark_to_market_by_model || []).filter", self.index_html)
        self.assertIn("const mtmCycleMinutes = d.mark_to_market_cycle_minutes || 15;", self.index_html)
        self.assertIn("Shadow benchmark account refreshed", self.index_html)
        self.assertIn("const curves = mtmModels.map", self.index_html)
        self.assertIn("value_curve || []", self.index_html)
        self.assertIn("Shadow mark-to-market account", self.index_html)
        self.assertIn("bid-side liquidation value", self.index_html)
        self.assertIn("Resolved strategy performance", self.index_html)
        rendered_headings = self.index_html.split('return `<div class="tr-inner">', 1)[1].split(
            "function _fmtDate",
            1,
        )[0]
        self.assertLess(
            rendered_headings.index("Shadow mark-to-market account"),
            rendered_headings.index("Model comparison"),
        )

    def test_auth_modal_exposes_google_and_github_providers(self) -> None:
        self.assertIn('id="googleFallbackBtn"', self.index_html)
        self.assertIn('id="githubBtn"', self.index_html)
        self.assertIn("function refreshAuthSocial()", self.index_html)
        self.assertIn("if (social) social.hidden = false;", self.index_html)
        self.assertIn("githubBtn.disabled = !_ghClientId;", self.index_html)
        self.assertIn("_gClientId = cfg.google_client_id || null;", self.index_html)
        self.assertNotIn("if (!cfg.google_client_id) return;", self.index_html)

    def test_signed_out_users_can_try_the_forecast_desk(self) -> None:
        launch_body = self.index_html.split("function launchApp", 1)[1].split(
            "function fillExample", 1
        )[0]
        send_preamble = self.index_html.split("async function sendQuestion", 1)[1].split(
            "_hideSlashPalette();", 1
        )[0]
        landing_stream_body = self.index_html.split("async function _streamLandingQuestion", 1)[
            1
        ].split("function _landingOpenInApp", 1)[0]
        start_example_body = self.index_html.split("function startExample", 1)[1].split(
            "let _landingLastQuestion",
            1,
        )[0]
        submit_landing_body = self.index_html.split("function submitLandingQuestion", 1)[1].split(
            "async function _streamLandingQuestion",
            1,
        )[0]

        self.assertNotIn("openAuthModal('login')", launch_body)
        self.assertNotIn("_queueForecastAfterSignIn", launch_body)
        self.assertNotIn("_queueForecastAfterSignIn", send_preamble)
        self.assertNotIn("_queueForecastAfterSignIn", landing_stream_body)
        self.assertIn("startForecast(question)", start_example_body)
        self.assertIn("startForecast(question)", submit_landing_body)
        self.assertNotIn("_streamLandingQuestion(question)", start_example_body)
        self.assertNotIn("_streamLandingQuestion(question)", submit_landing_body)

    def test_edge_board_uses_static_hash_route(self) -> None:
        self.assertIn("'edge-landing':  '#edge'", self.index_html)
        edge_opener = self.index_html.split("async function openEdgeBoard", 1)[1].split(
            "async function openArbitrage",
            1,
        )[0]

        self.assertIn("setHistoryState('edge-landing')", edge_opener)
        self.assertNotIn("setHistoryState(currentView === 'app' ? 'edge-app'", edge_opener)

if __name__ == "__main__":
    unittest.main()
