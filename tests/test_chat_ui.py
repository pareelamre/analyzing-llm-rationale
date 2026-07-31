from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ChatUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_html = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

    def test_composer_uses_the_shell_as_its_focus_indicator(self) -> None:
        self.assertRegex(
            self.index_html,
            r"\.input-box #questionInput:focus-visible\s*\{\s*outline: none;",
        )

    def test_hidden_live_status_does_not_leave_an_empty_pill(self) -> None:
        self.assertRegex(
            self.index_html,
            r"\.chat-live-status\s*\{\s*display: none !important;",
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

    def test_landing_forecast_renders_evidence_source_bubbles(self) -> None:
        self.assertIn(
            "makeAIBubble(response, response.variant, { includeActions: false })",
            self.index_html,
        )
        self.assertIn(
            "contentEl.innerHTML = _landingForecastHtml(data.response, text);",
            self.index_html,
        )
        self.assertIn(
            "contentEl.innerHTML = _landingForecastHtml(d);",
            self.index_html,
        )

    def test_generated_forecast_thesis_uses_markdown_formatting(self) -> None:
        self.assertIn(
            '<div class="chat-md text-[15px] text-gray-700 leading-relaxed">'
            "${renderMarkdown(_chatText(rationale))}</div>",
            self.index_html,
        )

    def test_auth_providers_refresh_after_async_config_load(self) -> None:
        self.assertIn(
            "const authProvidersReady = _ensureAuthProviders();",
            self.index_html,
        )
        self.assertRegex(
            self.index_html,
            r"_syncAuthProviderVisibility\(\);\s+_ensureAuthProviders\(\);",
        )
        self.assertIn('id="googleFallbackBtn"', self.index_html)
        self.assertIn("githubBtn.disabled = !_ghClientId;", self.index_html)
        self.assertIn("Google sign-in is not configured on this server.", self.index_html)

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

    def test_edge_board_renders_mtm_chart_without_summary_cards(self) -> None:
        self.assertIn("_ebSetView('mtm')", self.index_html)
        self.assertIn("Shadow mark-to-market account", self.index_html)
        self.assertIn('id="eb-mtm-svg"', self.index_html)
        self.assertIn('id="eb-mtm-table-body"', self.index_html)
        self.assertNotIn("Council account value", self.index_html)
        self.assertNotIn("cash + bid liquidation", self.index_html)
        self.assertNotIn("chart heartbeat", self.index_html)
        self.assertNotIn("SCADS + baseline", self.index_html)

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
        history_body = self.index_html.split("function applyHistoryState", 1)[1].split(
            "window.addEventListener('popstate'",
            1,
        )[0]
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
        self.assertNotIn("_queueForecastAfterSignIn", submit_landing_body)
        self.assertNotIn("function _queueForecastAfterSignIn", self.index_html)
        self.assertNotIn("openAuthModal('login')", history_body)
        self.assertIn("startForecast(question)", start_example_body)
        self.assertIn("_streamLandingQuestion(question)", submit_landing_body)
        self.assertNotIn("_streamLandingQuestion(question)", start_example_body)
        self.assertIn("attach_evidence: false", landing_stream_body)
        self.assertIn("evidence_top_k: 3", landing_stream_body)
        self.assertIn("max_tokens: 384", landing_stream_body)

    def test_streaming_forecast_emits_timing_events(self) -> None:
        self.assertIn("function _trackForecastStreamTiming", self.index_html)
        self.assertIn("forecast_stream_timing", self.index_html)
        self.assertIn("'first_delta'", self.index_html)
        self.assertIn("server_first_delta_ms: data.first_delta_ms", self.index_html)
        self.assertIn("server_prepare_ms: data.prepare_ms", self.index_html)

    def test_streaming_forecasts_batch_markdown_renders(self) -> None:
        predict_body = self.index_html.split("async function streamPredict", 1)[1].split(
            "async function streamAgentAnalyze",
            1,
        )[0]
        agent_body = self.index_html.split("async function streamAgentAnalyze", 1)[1].split(
            "async function sendQuestion",
            1,
        )[0]
        landing_body = self.index_html.split("async function _streamLandingQuestion", 1)[1].split(
            "function _landingOpenInApp",
            1,
        )[0]

        for stream_body in (predict_body, agent_body, landing_body):
            delta_body = stream_body.split("event === 'delta'", 1)[1].split(
                "event === 'done'",
                1,
            )[0]
            self.assertIn("const scheduleRender = () =>", stream_body)
            self.assertIn("requestAnimationFrame(() =>", stream_body)
            self.assertIn("scheduleRender();", delta_body)
            self.assertNotIn("contentEl.innerHTML = renderMarkdown(_chatText(text))", delta_body)

    def test_interactive_forecasts_use_lighter_evidence_defaults(self) -> None:
        send_body = self.index_html.split("async function sendQuestion", 1)[1].split(
            "if (activeModel)",
            1,
        )[0]
        self.assertIn("body.evidence_top_k = 3;", send_body)
        self.assertIn("if (chat_mode) body.max_tokens = 384;", send_body)
        self.assertIn(
            "body.attach_evidence = !(shortFollowup || attachedEvidence.length || hasCompleteMarketContext);",
            send_body,
        )
        self.assertIn("const hasCompleteMarketContext", send_body)
        self.assertIn("let extractedMarketProbability = null;", send_body)

    def test_prompt_window_exposes_builtin_chat_model_selector(self) -> None:
        self.assertIn('id="promptModelSelect"', self.index_html)
        self.assertIn("const BUILTIN_CHAT_MODELS_FALLBACK", self.index_html)
        self.assertIn("function setBuiltinChatModel", self.index_html)
        self.assertIn("body.model = builtinChatModel;", self.index_html)
        self.assertIn("served_model_name", self.index_html)
        self.assertNotIn("alias-image-generation", self.index_html)
        self.assertNotIn("alias-vision", self.index_html)

    def test_new_conversation_opens_plain_empty_chat(self) -> None:
        empty_branch = self.index_html.split("function renderMessages", 1)[1].split(
            "let i = 0;",
            1,
        )[0]

        self.assertIn(">New conversation</button>", self.index_html)
        self.assertIn("title: 'New conversation'", self.index_html)
        self.assertNotIn("New market brief", self.index_html)
        self.assertIn("if (!conv || conv.messages.length === 0)", empty_branch)
        self.assertNotIn("Foresea market desk", empty_branch)
        self.assertNotIn("Intelligence stack", empty_branch)
        self.assertNotIn("Live brief format", empty_branch)

    def test_conversations_get_stable_hash_urls(self) -> None:
        self.assertIn("const CHAT_HASH_PREFIX = '#chat/';", self.index_html)
        self.assertIn("function conversationHash(id)", self.index_html)
        self.assertIn("function conversationIdFromHash", self.index_html)
        self.assertIn("setHistoryState('chat', historyMode, { convId: id });", self.index_html)
        self.assertIn("conversationIdFromHash()", self.index_html)
        self.assertIn("activateConv(urlConversationId, { updateHistory: false });", self.index_html)

if __name__ == "__main__":
    unittest.main()
