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

    def test_settings_modal_hosts_connections_and_copied_agents_with_no_duplicate_ids(self) -> None:
        self.assertIn("function openSettingsModal", self.index_html)
        self.assertIn("function closeSettingsModal", self.index_html)
        self.assertIn("function _showSettingsSection", self.index_html)
        self.assertIn('id="settingsOverlay"', self.index_html)
        self.assertNotIn('id="tradingPanel"', self.index_html)
        self.assertNotIn('id="copiedAgentList"', self.index_html)
        for hosting_id in ("kalshiChip", "polyChip", "vcKalshiKeyId", "vcPolyPriv"):
            self.assertEqual(
                self.index_html.count(f'id="{hosting_id}"'), 1,
                f'expected exactly one id="{hosting_id}" (relocated, not duplicated)',
            )
        self.assertIn("onclick=\"openSettingsModal()\"", self.index_html)
        self.assertIn("saveVenueCreds('kalshi')", self.index_html)

    def test_your_workspace_panel_is_removed_from_the_sidebar(self) -> None:
        # The old sidebar panel (saved context, record, agent runs, and a
        # nested "Preferences" with model/skills/knowledge base) was removed
        # outright, not folded into Settings -- Settings only hosts
        # Connections and Copied agents.
        self.assertNotIn(">Your workspace<", self.index_html)
        self.assertNotIn('id="configDetails"', self.index_html)
        self.assertNotIn('id="workspacePreferences"', self.index_html)
        self.assertNotIn('id="modelList"', self.index_html)
        self.assertNotIn('id="skillList"', self.index_html)
        self.assertNotIn('id="kbList"', self.index_html)
        self.assertNotIn('id="workspaceAgentRunsTimeline"', self.index_html)

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

    def test_agent_research_can_open_a_review_only_trade_run(self) -> None:
        self.assertIn("Review Trade Run", self.index_html)
        self.assertIn("function openTradeRunFromBubble", self.index_html)
        self.assertIn("function tradeRunHandoff", self.index_html)
        self.assertIn(
            "I've prepared this as a reviewable order — nothing is placed. "
            "Choose your own price and size in the terminal.",
            self.index_html,
        )
        self.assertIn("no exchange credentials, price, or size", self.index_html)

    def test_trade_run_handoff_html_explains_a_trading_question_with_no_resolved_market(self) -> None:
        # tradeRunHandoffHtml must never stay silent on a trading-flavored
        # question just because no market was resolved this turn — but it
        # must also never say anything implying Foresea itself can execute.
        script = r'''
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('frontend/index.html', 'utf8');
const helpers = source.match(/const _TRADING_INTENT_RE[\s\S]*?\n}\r?\n\r?\nasync function addForecastToPersonalLedger/)[0]
  .replace(/\r?\n\r?\nasync function addForecastToPersonalLedger$/, '');
const sandbox = { uiIcon: () => '' };
vm.runInNewContext(`${helpers}\nfunction tradeRunHandoff(){ return null; }`, sandbox);

const withIntent = sandbox.tradeRunHandoffHtml({ question: 'should I buy this or sell it?' });
if (!withIntent.includes("I can't place trades directly")) throw new Error('missing honest no-market note: ' + withIntent);
if (withIntent.includes('Review Trade Run')) throw new Error('must not render the button without a resolved handoff');

const withoutIntent = sandbox.tradeRunHandoffHtml({ question: 'What will the weather be like tomorrow?' });
if (withoutIntent !== '') throw new Error('ordinary forecast question must render nothing: ' + withoutIntent);
'''
        result = subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_market_context_from_a_card_persists_across_conversation_turns(self) -> None:
        # A market card only attaches its context to the very next message
        # (_pendingMarketCtx is one-shot); a plain follow-up like "can you
        # place a trade on that?" must still see the market via the
        # conversation's remembered activeMarketCtx, not lose it silently.
        script = r'''
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('frontend/index.html', 'utf8');
const sandbox = {
  localStorage: (() => {
    const data = {};
    return {
      getItem: k => (k in data ? data[k] : null),
      setItem: (k, v) => { data[k] = String(v); },
      removeItem: k => { delete data[k]; },
    };
  })(),
};
vm.createContext(sandbox);
vm.runInContext(source.match(/let storageKey = STORAGE_KEY;[\s\S]*?function upsertConv[\s\S]*?\n}\r?\n/)[0]
  .replace('let storageKey = STORAGE_KEY;', "let storageKey = 'test'; const STORAGE_KEY = 'test'; function queueConversationSync(){}"),
  sandbox);

const convId = 'conv_test';
sandbox.upsertConv(convId, { id: convId, messages: [] });

// Simulate the market-card-attached turn's bookkeeping.
const mCtx = { market_probability: 0.42, market_url: 'https://polymarket.com/event/x', market_platform: 'Polymarket', market_ident: 'x', model: 'some-model' };
const marketOnly = Object.assign({}, mCtx);
delete marketOnly.model;
sandbox.upsertConv(convId, { activeMarketCtx: marketOnly });

// A later, plain-text turn with no fresh card falls back to the remembered context.
const remembered = sandbox.getConv(convId).activeMarketCtx;
if (!remembered || remembered.market_ident !== 'x' || remembered.market_platform !== 'Polymarket') {
  throw new Error('remembered market context missing or wrong: ' + JSON.stringify(remembered));
}
if ('model' in remembered) throw new Error('model preference must not be carried forward with the market context');

// A different market card click overwrites the remembered value.
sandbox.upsertConv(convId, { activeMarketCtx: { market_platform: 'Kalshi', market_ident: 'y' } });
const overwritten = sandbox.getConv(convId).activeMarketCtx;
if (overwritten.market_ident !== 'y') throw new Error('a new market card must overwrite the remembered one');
'''
        result = subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_effort_tier_caption_only_appears_for_the_simple_tier(self) -> None:
        # A thinner analysis pass must never be silent -- but standard/deep
        # responses (the vast majority) must stay exactly as before. A plain
        # chat reply with no forecast component (a greeting) has no "analysis
        # pass" to caption, so it must stay silent even at the simple tier.
        self.assertIn("function effortTierCaptionHtml", self.index_html)
        self.assertIn("${effortTierCaptionHtml(data)}", self.index_html)
        script = r'''
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('frontend/index.html', 'utf8');
const clamp01Fn = source.match(/function clamp01[\s\S]*?\n}\r?\n/)[0];
const normalizedProbabilityFn = source.match(/function normalizedProbability[\s\S]*?\n}\r?\n/)[0];
const fn = source.match(/function effortTierCaptionHtml[\s\S]*?\n}\r?\n/)[0];
const sandbox = {};
vm.runInNewContext(clamp01Fn + normalizedProbabilityFn + fn, sandbox);

const simple = sandbox.effortTierCaptionHtml({ effort_tier: 'simple' });
if (!simple.includes('Quick take')) throw new Error('missing quick-take caption: ' + simple);

const simpleChatForecast = sandbox.effortTierCaptionHtml({ effort_tier: 'simple', question_type: 'chat', model_probability: 0.6 });
if (!simpleChatForecast.includes('Quick take')) throw new Error('missing quick-take caption for a simple-tier chat forecast: ' + simpleChatForecast);

const simpleGreeting = sandbox.effortTierCaptionHtml({ effort_tier: 'simple', question_type: 'chat', model_probability: null });
if (simpleGreeting !== '') throw new Error(`expected empty string for a plain chat greeting, got: ${simpleGreeting}`);

for (const tier of ['standard', 'deep', null, undefined]) {
  const out = sandbox.effortTierCaptionHtml({ effort_tier: tier });
  if (out !== '') throw new Error(`expected empty string for tier=${tier}, got: ${out}`);
}
'''
        result = subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_agent_progress_is_a_single_evolving_line_not_a_stage_checklist(self) -> None:
        # A plain "hello" (or any message) must not render a wall of
        # pre-declared research-pipeline stage labels -- only whatever
        # stage is actually happening right now, as one line of text.
        self.assertNotIn('class="agent-steps"', self.index_html)
        self.assertNotIn('class="agent-step"', self.index_html)
        script = r'''
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync('frontend/index.html', 'utf8');
const fn = source.match(/function addAgentProgress[\s\S]*?\n}\r?\n/)[0];
const document = {
  createElement: () => ({ set innerHTML(html) { this._html = html; }, get innerHTML() { return this._html; },
    querySelector(sel) {
      if (sel === '.agent-progress-title') {
        const m = /class="agent-progress-title">([^<]*)</.exec(this._html);
        const node = this;
        return {
          get textContent() { return m ? m[1] : null; },
          set textContent(v) { node._html = node._html.replace(/(class="agent-progress-title">)[^<]*(<)/, `$1${v}$2`); },
        };
      }
      return null;
    },
    querySelectorAll: () => [],
  }),
  getElementById: () => ({ appendChild: () => {} }),
};
const sandbox = { document, scrollBottom: () => {}, escHtml: s => s };
vm.createContext(sandbox);
vm.runInContext(fn, sandbox);

const bubble = sandbox.addAgentProgress(['Resolving the market', 'Gathering evidence', 'Forecasting', 'Pricing the edge', 'Recommending']);
if (bubble.innerHTML.includes('agent-step')) throw new Error('checklist markup must not be rendered: ' + bubble.innerHTML);
if (!bubble.innerHTML.includes('Agent working...')) throw new Error('missing initial status line: ' + bubble.innerHTML);

bubble._setAgentProgress('Gathering evidence');
if (!bubble.innerHTML.includes('>Gathering evidence<')) throw new Error('status line did not update in place: ' + bubble.innerHTML);
'''
        result = subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_agent_run_steps_are_visible_via_an_expandable_panel(self) -> None:
        self.assertIn("function toggleAgentRunSteps", self.index_html)
        self.assertIn("function _agentRunStepRowHtml", self.index_html)
        self.assertIn("run.steps || []", self.index_html)
        self.assertIn("No tool-loop steps recorded for this run.", self.index_html)

    def test_agent_run_step_started_but_not_completed_renders_as_unknown_not_success(self) -> None:
        # A step with started_at but no completed_at means the process
        # crashed or was recycled mid-step -- must not fall through to the
        # success checkmark, which would misrepresent an unknown outcome as
        # a completed one.
        self.assertIn("!!step.started_at && !step.completed_at", self.index_html)
        self.assertIn("Started but never completed", self.index_html)

    def test_agentic_trading_board_omits_ambiguous_promotion_eligibility(self) -> None:
        self.assertNotIn("d.eligibility || {}", self.index_html)
        self.assertNotIn("function _eligibilityReasonText", self.index_html)
        self.assertNotIn(">Eligible</th>", self.index_html)

    def test_users_can_copy_a_public_agent_without_private_trading_state(self) -> None:
        self.assertNotIn("Copy agent", self.index_html)
        self.assertNotIn(">Copied agents<", self.index_html)
        self.assertNotIn('id="copiedAgentList"', self.index_html)
        self.assertIn("function syncCopiedAgentsFromServer", self.index_html)
        self.assertIn("function _agentSkillsForRun", self.index_html)
        self.assertIn("agentBody.agent_profile_id = copiedAgent.profileId", self.index_html)
        self.assertIn("agentBody.model = copiedAgent.sourceAgentId", self.index_html)
        self.assertIn("private research recipe", self.index_html)
        self.assertIn("no private history, context, trading permissions, or exchange connection", self.index_html)

    def test_app_uses_refresh_safe_paths_and_migrates_legacy_hash_links(self) -> None:
        self.assertIn("const CHAT_PATH_PREFIX = '/chat/';", self.index_html)
        self.assertIn("'app':           '/ask'", self.index_html)
        self.assertIn("'edge-landing':  '/edge'", self.index_html)
        self.assertIn("'watch-landing': '/watchlist'", self.index_html)
        self.assertIn("'ledger-landing': '/ledger'", self.index_html)
        self.assertIn("function legacyRouteFromHash", self.index_html)
        self.assertIn("history.replaceState(", self.index_html)
        self.assertIn("const _initConversationId = conversationIdFromPath();", self.index_html)

    def test_direct_route_load_skips_the_landing_page_flash(self) -> None:
        # Regression: the whole SPA shell renders the landing page visible by
        # default in the raw HTML -- JS only hides it and opens the right
        # overlay/app-shell once the big inline <script> near the end of
        # <body> runs. On a direct load or refresh of e.g. /edge, that gap
        # showed the landing page (mid entrance animation) before flipping to
        # the edge board a moment later. A tiny inline script at the end of
        # <head> now computes the route before first paint and stamps it as
        # data-boot-view on <html>; CSS keyed off that attribute shows the
        # right shell/overlay immediately, and _bootRouting suppresses the
        # overlay's own fade-in Motion.animate call for that one initial
        # open only (it's already visible, so fading it from 0 would itself
        # flash) without touching later user-driven opens.
        head = self.index_html.split("<head>", 1)[1].split("</head>", 1)[0]
        self.assertIn("data-boot-view", head)
        self.assertIn("history.state && history.state.view", head)
        self.assertIn("/^\\/edge(?:\\/(mtm|agentic))?$/", head)
        self.assertIn("/^\\/chat\\/[^/]+$/", head)

        self.assertIn(
            'html[data-boot-view]:not([data-boot-view="landing"]) #landingPage { display: none !important; }',
            self.index_html,
        )
        self.assertIn('html[data-boot-view="edge-app"] #appShell', self.index_html)
        self.assertIn('html[data-boot-view^="edge-"] #edgeOverlay', self.index_html)
        self.assertIn('html[data-boot-view^="track-"] #trackOverlay', self.index_html)
        self.assertIn('html[data-boot-view^="watch-"] #watchOverlay', self.index_html)
        self.assertIn('html[data-boot-view^="ledger-"] #ledgerOverlay', self.index_html)

        self.assertIn("let _bootRouting = false;", self.index_html)
        self.assertIn(
            "_bootRouting = true;\nif (!_maybeStandaloneWatchlist()) "
            "applyHistoryState({ view: _initView, convId: _initConversationId, panel: _initPanel });\n"
            "_bootRouting = false;",
            self.index_html,
        )
        # Every overlay/app-shell fade-in is guarded, so the first (boot)
        # open never re-fades content the boot CSS already made visible.
        for guarded in (
            "if (!_bootRouting && window.Motion && !prefersReduced()) {\n"
            "    Motion.animate(app, { opacity: [0, 1]",
            "if (!_bootRouting && window.Motion && !prefersReduced()) {\n"
            "    Motion.animate(ov, { opacity: [0, 1] }, { duration: 0.3, easing: EASE });\n"
            "  }\n"
            "  if (!trackLoaded)",
            "if (!_bootRouting && window.Motion && !prefersReduced()) {\n"
            "    Motion.animate(ov, { opacity: [0, 1] }, { duration: 0.3, easing: EASE });\n"
            "  }\n"
            "  if (!edgeLoaded)",
            "if (!_bootRouting && window.Motion && !prefersReduced()) "
            "Motion.animate(ov, { opacity: [0, 1] }, { duration: 0.3, easing: EASE });\n"
            "  renderWatchlistPage();",
            "if (!_bootRouting && window.Motion && !prefersReduced()) "
            "Motion.animate(ov, { opacity: [0, 1] }, { duration: 0.3, easing: EASE });\n"
            "  await refreshPersonalLedger();",
        ):
            self.assertIn(guarded, self.index_html)

    def test_silent_session_restore_does_not_hijack_the_current_route(self) -> None:
        # Regression: initAuth() restores a saved session on every page load,
        # racing the synchronous initial-route logic below it. If that restore
        # won the race after currentView was already 'app' (e.g. the edge
        # board or track record overlay open on top of the chat shell),
        # activateConv's default history push silently rewrote the URL to
        # /chat/{id} -- so a *later* refresh landed on plain chat instead of
        # whatever overlay/route the user actually had open.
        init_auth_body = self.index_html.split("async function initAuth", 1)[1].split(
            "// Close the auth modal on Escape.", 1
        )[0]
        self.assertIn(
            "await syncConversationsAfterSignIn(user, { updateHistory: false });",
            init_auth_body,
        )
        self.assertIn(
            "async function syncConversationsAfterSignIn(user, { updateHistory = true } = {}) {",
            self.index_html,
        )
        self.assertIn(
            "activateConv(activeId && store.conversations[activeId] ? activeId : ids[0], { updateHistory });",
            self.index_html,
        )
        self.assertIn("else newConversation({ updateHistory });", self.index_html)
        # The explicit sign-in flow (_afterSignIn) is a real user action, not
        # a silent restore, and should keep pushing history as before.
        after_sign_in_body = self.index_html.split("async function _afterSignIn", 1)[1].split(
            "function ", 1
        )[0]
        self.assertIn("await syncConversationsAfterSignIn(user);", after_sign_in_body)

    def test_settings_modal_scrolls_past_the_global_lenis_smooth_scroll(self) -> None:
        # Regression: the app runs Lenis globally for buttery document-level
        # smooth scroll (see initMotion). Every other overlay's scrollable
        # body (trackBody/edgeBody/watchBody/ledgerBody, the chat messages
        # pane, the sidebar) opts out via data-lenis-prevent so Lenis lets
        # native overflow scrolling happen inside them instead of hijacking
        # the wheel event for the document. The Settings modal never got
        # that attribute, so scrolling inside it silently did nothing.
        settings_modal = self.index_html.split('id="settingsOverlay"', 1)[1].split(
            "<!-- ── Track record overlay", 1
        )[0]
        self.assertIn('class="settings-nav" data-lenis-prevent', settings_modal)
        self.assertIn('class="settings-body" data-lenis-prevent', settings_modal)
        # And a visible, app-consistent thin scrollbar instead of the
        # default browser one.
        self.assertIn(".settings-nav::-webkit-scrollbar", self.index_html)
        self.assertIn(".settings-body::-webkit-scrollbar", self.index_html)

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

    def test_chat_markdown_renders_fenced_code_blocks_and_preserves_currency(self) -> None:
        script = r'''
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('static/index.html', 'utf8');

const sandbox = { window: {}, document: { getElementById: () => null, addEventListener: () => {} } };
const start = html.indexOf('function escHtml(s) {');
const end = html.indexOf('// ── Motion helpers', start);
vm.runInNewContext(html.slice(start, end), sandbox);

// 1. Code blocks
const codeOutput = sandbox.renderMarkdown('```python\ndef test():\n    return 42\n```');
if (!codeOutput.includes('<pre class="md-pre"><code class="md-codeblock language-python">def test():\n    return 42</code></pre>')) {
  throw new Error(`Code block rendering failed: ${codeOutput}`);
}

// 2. Currency preservation
const currencyOutput = sandbox.renderMarkdown('The price moved from $50 to $60, earning $10 per share.');
if (!currencyOutput.includes('$50 to $60') || !currencyOutput.includes('$10')) {
  throw new Error(`Currency formatting failed: ${currencyOutput}`);
}

// 3. URL date preservation
const urlOutput = sandbox.renderMarkdown('Link: [data](https://example.com/2026-08-23/results)');
if (!urlOutput.includes('https://example.com/2026-08-23/results')) {
  throw new Error(`URL date preservation failed: ${urlOutput}`);
}

// 4. JSON prose extraction
const jsonText = sandbox._chatText(JSON.stringify({
  reasoning: 'Evidence indicates high probability.',
  predicted_answer: 'YES',
  confidence: 0.85
}));
if (!jsonText.includes('Evidence indicates high probability.') || !jsonText.includes('**YES**')) {
  throw new Error(`JSON reasoning extraction failed: ${jsonText}`);
}
'''
        result = subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_trade_run_opens_the_connections_ui_when_no_venue_linked(self) -> None:
        # Functional test, not a text search: it runs the real page JS and
        # calls openTradeModal the way the "Review Trade Run" button does,
        # with a signed-in user and no connected exchange -- the state in
        # which the button previously did nothing at all because it reached
        # for a #tradingPanel that no longer exists.
        script = r'''
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('static/index.html', 'utf8');

// Stateful DOM: same id returns the same element, and classList really tracks.
const els = new Map();
const makeEl = (id = '') => {
  const classes = new Set();
  const el = {
    id, style: {}, hidden: false, textContent: '', innerHTML: '', open: false,
    classList: {
      contains: (c) => classes.has(c),
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      toggle: (c) => (classes.has(c) ? classes.delete(c) : classes.add(c)),
    },
    _classes: classes,
    setAttribute: () => {}, getAttribute: () => null, focus: () => {},
    scrollIntoView: () => {}, querySelector: () => null, querySelectorAll: () => [],
    appendChild: () => {}, insertAdjacentHTML: () => {},
  };
  return el;
};
const getEl = (id) => { if (!els.has(id)) els.set(id, makeEl(id)); return els.get(id); };

const storageMock = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
const sandbox = {
  window: { location: { pathname: '/', hash: '', search: '' }, addEventListener: () => {}, matchMedia: () => ({ matches: false }), scrollTo: () => {}, localStorage: storageMock, sessionStorage: storageMock, open: () => ({}) },
  document: { body: makeEl('body'), getElementById: getEl, querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {}, createElement: () => makeEl() },
  navigator: { sendBeacon: () => true },
  history: { pushState: () => {}, replaceState: () => {}, state: null },
  localStorage: storageMock, sessionStorage: storageMock,
  URL: globalThis.URL, URLSearchParams: globalThis.URLSearchParams,
  Blob: globalThis.Blob, crypto: globalThis.crypto,
  setTimeout: (f) => { if (typeof f === 'function') f(); return 1; },
  clearTimeout: () => {}, setInterval: () => 1, clearInterval: () => {},
  console: { log: () => {}, warn: () => {}, error: () => {} },
  fetch: () => Promise.resolve({ ok: false, json: async () => ({}) }),
  trackEvent: () => {}, Motion: null,
};
const context = vm.createContext(sandbox);
const scriptMatches = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)];
let fullJs = '';
for (const match of scriptMatches) {
  const attrs = match[1];
  if (!attrs.includes('src=') && !attrs.includes('ld+json')) fullJs += match[2] + '\n';
}
vm.runInContext(fullJs, context);

// Signed in, but no exchange connected -- the reported failing state.
// currentUser is a top-level `let`, which in a VM is a lexical binding and
// NOT a property of the context -- it has to be assigned inside the VM.
vm.runInContext(`
  currentUser = { id: 'u1' };
  _loadVenueConnections = async () => {};
  _venueConnected = () => false;
  renderVenueStatus = () => {};
  _loadModelProviders = () => {};
  _closeAcctMenu = () => {};
  openSidebar = () => {};
  openAuthModal = () => { throw new Error('should not ask for login when signed in'); };
`, context);

(async () => {
  await context.openTradeModal({ platform: 'kalshi', ident: 'KXTEST', question: 'Will X happen?' });

  const overlay = els.get('settingsOverlay');
  if (!overlay || !overlay.classList.contains('show')) {
    throw new Error('settings modal did not open: the button is still a no-op');
  }
  const err = els.get('venueErr');
  if (!err || !String(err.textContent).toLowerCase().includes('connect')) {
    throw new Error(`no guidance shown to the user, got: ${err && err.textContent}`);
  }
  if (err.hidden) throw new Error('guidance element left hidden');
})().catch((e) => { console.error(e.message); process.exit(1); });
'''
        result = subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_clicking_an_edge_board_question_closes_the_board(self) -> None:
        # Functional test for the reported bug: clicking a question routed to
        # the chat but left the Edge Board covering it. closeEdgeBoard defers
        # removing .open behind a fade, and startForecast's routing bumps
        # _edgeOpenSeq in the same tick, which the deferred finish() reads as
        # "re-opened while fading" and so leaves the overlay open.
        script = r'''
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('static/index.html', 'utf8');

const els = new Map();
const makeEl = (id = '') => {
  const classes = new Set();
  return {
    id, style: {}, hidden: false, textContent: '', innerHTML: '', dataset: {},
    classList: {
      contains: (c) => classes.has(c), add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      toggle: (c) => (classes.has(c) ? classes.delete(c) : classes.add(c)),
    },
    _classes: classes,
    setAttribute: () => {}, getAttribute: () => null, focus: () => {},
    scrollIntoView: () => {}, querySelector: () => null, querySelectorAll: () => [],
    appendChild: () => {}, insertAdjacentHTML: () => {},
    addEventListener: () => {}, removeEventListener: () => {},
    getBoundingClientRect: () => ({ top: 0, left: 0, width: 0, height: 0 }),
  };
};
const getEl = (id) => { if (!els.has(id)) els.set(id, makeEl(id)); return els.get(id); };
const storageMock = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
const sandbox = {
  window: { location: { pathname: '/edge', hash: '', search: '' }, addEventListener: () => {}, matchMedia: () => ({ matches: false }), scrollTo: () => {}, localStorage: storageMock, sessionStorage: storageMock },
  document: { body: makeEl('body'), getElementById: getEl, querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {}, createElement: () => makeEl() },
  navigator: { sendBeacon: () => true },
  history: { pushState: () => {}, replaceState: () => {}, state: null },
  localStorage: storageMock, sessionStorage: storageMock,
  URL: globalThis.URL, URLSearchParams: globalThis.URLSearchParams,
  Blob: globalThis.Blob, crypto: globalThis.crypto,
  setTimeout: () => 1, requestAnimationFrame: () => 1, cancelAnimationFrame: () => {},
  clearTimeout: () => {}, setInterval: () => 1, clearInterval: () => {},
  console: { log: () => {}, warn: () => {}, error: () => {} },
  fetch: () => Promise.resolve({ ok: false, json: async () => ({}) }),
  trackEvent: () => {}, Motion: null,
};
const context = vm.createContext(sandbox);
const scriptMatches = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)];
let fullJs = '';
for (const match of scriptMatches) {
  const attrs = match[1];
  if (!attrs.includes('src=') && !attrs.includes('ld+json')) fullJs += match[2] + '\n';
}
try { vm.runInContext(fullJs, context); } catch (_) { /* load-time DOM stubs are incomplete by design */ }

// Real Motion is present in production, so exercise the animated path: that
// is the path that regressed. Its promise resolves on a later microtask,
// exactly like the browser.
// closeEdgeBoard reads window.Motion; in a VM that is a different binding
// from the bare Motion global, so set the one the code actually checks.
sandbox.window.Motion = { animate: () => ({ finished: Promise.resolve() }), stagger: () => 0 };
vm.runInContext(`
  Motion = { animate: () => ({ finished: Promise.resolve() }), stagger: () => 0 };
  prefersReduced = () => false;
  _ebBoardData = [{ ident: 'mkt-1', question: 'Will X happen?', platform: 'Kalshi', market_probability: 0.4 }];
  startForecast = () => { _edgeOpenSeq++; };   // routing bumps the token
  _captureEdgeBoardScroll = () => {};
  _saveEdgeBoardScrollToHistory = () => {};
  _stopEdgeTimers = () => {};
`, context);

const ov = getEl('edgeOverlay');

// Reproduce the race directly on closeEdgeBoard, which is where it lives.
// Deferred close + a routing bump in the same tick = finish() sees a changed
// token, reads it as "re-opened while fading", and leaves the overlay open.
const run = async (opts) => {
  ov.classList.add('open');
  vm.runInContext('_edgeOpenSeq = 0;', context);
  context.closeEdgeBoard(opts);
  vm.runInContext('_edgeOpenSeq++;', context);   // startForecast's routing
  await new Promise((r) => setImmediate(r));
  await new Promise((r) => setImmediate(r));
  return ov.classList.contains('open');
};

(async () => {
  const stillOpenWhenDeferred = await run({ updateHistory: false });
  if (!stillOpenWhenDeferred) {
    throw new Error('harness did not reproduce the deferred-close race; this test would not catch a regression');
  }
  const stillOpenWhenImmediate = await run({ updateHistory: false, immediate: true });
  if (stillOpenWhenImmediate) {
    throw new Error('edge board still open after an immediate close: the board would cover the chat');
  }
})().catch((e) => { console.error(e.message); process.exit(1); });
'''
        result = subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_edge_board_navigation_and_history_routing(self) -> None:
        script = r'''
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('static/index.html', 'utf8');

const storageMock = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
const makeEl = (id = '') => ({ id, classList: { contains: () => false, add: () => {}, remove: () => {}, toggle: () => {} }, style: {}, attributes: {}, setAttribute: () => {}, getAttribute: () => null, focus: () => {}, querySelector: () => null, querySelectorAll: () => [] });
const sandbox = {
  window: { location: { pathname: '/', hash: '', search: '' }, addEventListener: () => {}, matchMedia: () => ({ matches: false }), scrollTo: () => {}, localStorage: storageMock, sessionStorage: storageMock },
  document: { body: makeEl('body'), getElementById: (id) => makeEl(id), querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {} },
  navigator: { sendBeacon: () => true },
  history: { pushState: () => {}, replaceState: () => {}, state: null },
  localStorage: storageMock,
  sessionStorage: storageMock,
  Blob: globalThis.Blob,
  URL: globalThis.URL,
  URLSearchParams: globalThis.URLSearchParams,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  console: { log: () => {}, warn: () => {}, error: () => {} },
  fetch: globalThis.fetch || (() => Promise.resolve({ ok: false, json: async () => ({}) })),
  trackEvent: () => {},
  Motion: null,
};

const context = vm.createContext(sandbox);
const scriptMatches = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)];
let fullJs = '';
for (const match of scriptMatches) {
  const attrs = match[1];
  if (!attrs.includes('src=') && !attrs.includes('ld+json')) fullJs += match[2] + '\n';
}
vm.runInContext(fullJs, context);

// 1. History URLs for Edge Board panels
if (sandbox.historyUrlFor('edge-landing', null, 'markets') !== '/edge') {
  throw new Error(`historyUrlFor markets failed: ${sandbox.historyUrlFor('edge-landing', null, 'markets')}`);
}
if (sandbox.historyUrlFor('edge-landing', null, 'mtm') !== '/edge/mtm') {
  throw new Error(`historyUrlFor mtm failed: ${sandbox.historyUrlFor('edge-landing', null, 'mtm')}`);
}
if (sandbox.historyUrlFor('edge-landing', null, 'agentic') !== '/edge/agentic') {
  throw new Error(`historyUrlFor agentic failed: ${sandbox.historyUrlFor('edge-landing', null, 'agentic')}`);
}
if (sandbox.historyUrlFor('edge-app', null, 'mtm') !== '/edge/mtm') {
  throw new Error(`historyUrlFor edge-app mtm failed: ${sandbox.historyUrlFor('edge-app', null, 'mtm')}`);
}

// 2. Parse edge paths
const pMarkets = sandbox.parseEdgePath('/edge');
if (!pMarkets || pMarkets.panel !== 'markets') throw new Error('parseEdgePath /edge failed');
const pMtm = sandbox.parseEdgePath('/edge/mtm');
if (!pMtm || pMtm.panel !== 'mtm') throw new Error('parseEdgePath /edge/mtm failed');
const pAgentic = sandbox.parseEdgePath('/edge/agentic');
if (!pAgentic || pAgentic.panel !== 'agentic') throw new Error('parseEdgePath /edge/agentic failed');

// 3. Legacy hash routes
const hMtm = sandbox.legacyRouteFromHash('#edge/mtm');
if (!hMtm || hMtm.panel !== 'mtm' || hMtm.view !== 'edge-landing') throw new Error('legacyRouteFromHash #edge/mtm failed');
const hAgentic = sandbox.legacyRouteFromHash('#edge/agentic');
if (!hAgentic || hAgentic.panel !== 'agentic' || hAgentic.view !== 'edge-landing') throw new Error('legacyRouteFromHash #edge/agentic failed');
const hAppMtm = sandbox.legacyRouteFromHash('#edge-app/mtm');
if (!hAppMtm || hAppMtm.panel !== 'mtm' || hAppMtm.view !== 'edge-app') throw new Error('legacyRouteFromHash #edge-app/mtm failed');

// 4. View switching
sandbox._ebSetView('mtm');
if (vm.runInContext('_ebView', context) !== 'mtm') throw new Error('_ebSetView mtm failed');
sandbox._ebSetView('agentic');
if (vm.runInContext('_ebView', context) !== 'agentic') throw new Error('_ebSetView agentic failed');
sandbox._ebSetView('markets');
if (vm.runInContext('_ebView', context) !== 'markets') throw new Error('_ebSetView markets failed');
'''
        result = subprocess.run(
            ["node", "-e", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()


