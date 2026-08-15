from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


class ChatUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_html = (
            Path(__file__).resolve().parents[1] / "static" / "index.html"
        ).read_text(encoding="utf-8")

    def test_composer_uses_the_shell_as_its_focus_indicator(self) -> None:
        self.assertIn(
            ".input-box #questionInput:focus-visible {\n      outline: none;",
            self.index_html,
        )

    def test_composer_keeps_prompt_and_controls_in_distinct_areas(self) -> None:
        self.assertIn('class="input-box composer-shell"', self.index_html)
        self.assertIn('class="composer-actions"', self.index_html)
        self.assertIn('class="composer-model-picker"', self.index_html)
        self.assertIn('class="composer-model-label">Model</span>', self.index_html)
        self.assertIn('placeholder="What would you like to forecast?"', self.index_html)
        self.assertIn('aria-label="Choose chat model"', self.index_html)
        self.assertIn('@media (max-width: 640px) {', self.index_html)
        self.assertIn('.composer-shell { gap: 7px; grid-template-columns: 1fr;', self.index_html)

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

    def test_chat_forecast_renders_probability_score(self) -> None:
        self.assertIn("model_probability: d.model_probability ?? null", self.index_html)
        self.assertIn("raw.lastIndexOf('[p')", self.index_html)

    def test_plain_chat_replies_do_not_render_forecast_card(self) -> None:
        chat_reply = self.index_html.split("if (qtype === 'chat')", 1)[1].split(
            "if (qtype === 'multiple_choice'", 1
        )[0]
        self.assertIn(
            "const chatForecastProbability = normalizedProbability(data.model_probability);",
            chat_reply,
        )
        self.assertIn("const isChatForecast = chatForecastProbability != null;", chat_reply)
        self.assertNotIn("ledgerProbability(data)", chat_reply)
        self.assertIn("isChatForecast ? `<div class=\"market-lens", chat_reply)
        self.assertIn("isChatForecast && options.includeActions !== false", chat_reply)
        self.assertIn("function normalizedProbability", self.index_html)
        self.assertIn("function statedForecastProbability", self.index_html)
        self.assertIn("data?.model_probability", self.index_html)

    def test_chat_probability_fallback_rejects_market_quotes_and_invalid_values(self) -> None:
        script = r'''
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('frontend/index.html', 'utf8');
const helpers = source.match(/function clamp01[\s\S]*?function extractMarketProbability/)[0]
  .replace(/function extractMarketProbability$/, '');
const ledger = source.match(/function ledgerProbability[\s\S]*?\n}\r?\n\r?\nfunction forecastActionsHtml/)[0]
  .replace(/\r?\n\r?\nfunction forecastActionsHtml$/, '');
const sandbox = {};
vm.runInNewContext(`${helpers}\n${ledger}`, sandbox);
const cases = [
  ["I estimate there's a strong probability—around **75%**—that this happens.", null, 0.75],
  ['The market probability is 20%, but I cannot make a forecast.', null, null],
  ['The market is at 20%. I estimate the chance is around 75%.', null, 0.75],
  ['I estimate a 125% chance.', null, null],
  ['I estimate a 75% chance.', 0, 0],
];
for (const [rationale, explicit, expected] of cases) {
  const actual = sandbox.ledgerProbability({ model_probability: explicit, confidence: null, rationale });
  if (actual !== expected) throw new Error(`expected ${expected}, got ${actual}`);
}
'''
        result = subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_chat_forecast_can_be_saved_to_a_personal_ledger(self) -> None:
        self.assertIn(">Personal ledger", self.index_html)
        self.assertIn("Add to personal ledger", self.index_html)
        self.assertIn("/personal-ledger/", self.index_html)
        self.assertIn("function openPersonalLedger", self.index_html)

    def test_exchange_connection_never_persists_credentials_in_browser_storage(self) -> None:
        self.assertIn("/trading/connections/", self.index_html)
        self.assertIn("FORESEA_TRADING_KMS_KEY_NAME", (
            Path(__file__).resolve().parents[1] / "src" / "analyzing_llm_rationale" / "server.py"
        ).read_text(encoding="utf-8"))
        self.assertNotIn("_saveVenueCredsStore", self.index_html)
        self.assertNotIn("_allVenueCreds", self.index_html)
        self.assertIn("localStorage.removeItem('foresea_venue_creds')", self.index_html)

    def test_signed_in_analytics_uses_the_session_header_not_send_beacon(self) -> None:
        self.assertIn("function _sendAnalytics(url, payload)", self.index_html)
        self.assertIn("const { token } = _loadStoredSession();", self.index_html)
        self.assertIn("if (token) {", self.index_html)
        self.assertIn("headers: authHeaders({ 'Content-Type': 'application/json' })", self.index_html)
        self.assertIn("// sendBeacon cannot attach the session header.", self.index_html)
        self.assertIn("_sendAnalytics('/analytics/visit', payload);", self.index_html)
        self.assertIn("_sendAnalytics('/analytics/event', payload);", self.index_html)
        self.assertIn("trackEvent('forecast_started'", self.index_html)
        self.assertIn("trackEvent('track_record_opened'", self.index_html)
        self.assertIn("trackEvent('edge_board_opened'", self.index_html)
        self.assertIn("trackEvent('watchlist_opened'", self.index_html)
        self.assertIn("trackEvent('personal_ledger_opened'", self.index_html)

    def test_app_uses_refresh_safe_paths_and_migrates_legacy_hash_links(self) -> None:
        self.assertIn("const CHAT_PATH_PREFIX = '/chat/';", self.index_html)
        self.assertIn("'app':           '/ask'", self.index_html)
        self.assertIn("'edge-landing':  '/edge'", self.index_html)
        self.assertIn("'watch-landing': '/watchlist'", self.index_html)
        self.assertIn("'ledger-landing': '/ledger'", self.index_html)
        self.assertIn("function legacyRouteFromHash", self.index_html)
        self.assertIn("history.replaceState(", self.index_html)
        self.assertIn("const _initConversationId = conversationIdFromPath();", self.index_html)

    def test_workspace_panel_uses_plain_language_controls(self) -> None:
        self.assertIn(">Your workspace<", self.index_html)
        self.assertIn("Your saved context", self.index_html)
        self.assertIn("Add a note or link", self.index_html)
        self.assertIn("Your record", self.index_html)
        self.assertIn("function openWorkspaceContext", self.index_html)
        self.assertIn("Research is Foresea's always-on answering workflow.", self.index_html)
        self.assertNotIn("workspaceStandardMode", self.index_html)
        self.assertNotIn("setWorkspaceAnswerMode", self.index_html)
        self.assertNotIn("How Foresea answers", self.index_html)

    def test_workspace_panel_summarizes_private_context_and_ledger(self) -> None:
        self.assertIn("No notes added", self.index_html)
        self.assertIn("note${_workspaceKbDocs.length === 1 ? '' : 's'} saved", self.index_html)
        self.assertIn("marked correct or wrong", self.index_html)
        self.assertIn("renderWorkspacePanel();", self.index_html)

    def test_browser_history_closes_personal_ledger_overlay(self) -> None:
        history_body = self.index_html.split("function applyHistoryState", 1)[1].split(
            "window.addEventListener('popstate'", 1
        )[0]
        self.assertIn("closePersonalLedger({ updateHistory: false });", history_body)
        self.assertIn("function personalLedgerRationalePreview", self.index_html)
        self.assertIn("View full reasoning", self.index_html)
        self.assertIn("function beginPersonalLedgerSignIn", self.index_html)
        self.assertIn("foresea_pending_ledger_after_signin", self.index_html)
        self.assertIn("await openPersonalLedger();", self.index_html)
        self.assertIn("function markPersonalLedgerVerdict", self.index_html)
        self.assertIn("/verdict", self.index_html)
        self.assertIn("Correct</button>", self.index_html)
        self.assertIn("Wrong</button>", self.index_html)

    def test_auth_providers_refresh_after_async_config_load(self) -> None:
        self.assertIn(
            "const authProvidersReady = _ensureAuthProviders();",
            self.index_html,
        )
        self.assertIn(
            "_syncAuthProviderVisibility();\n  _ensureAuthProviders();",
            self.index_html,
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

    def test_edge_board_renders_mtm_chart_without_summary_cards(self) -> None:
        self.assertIn('data-eb-view="mtm"', self.index_html)
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

if __name__ == "__main__":
    unittest.main()
