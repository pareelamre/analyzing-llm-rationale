type Json = Record<string, any>;

document.documentElement.dataset.twinDeskModule = "loaded";

const byId = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const money = (value: unknown): string => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `$${parsed.toFixed(2)}` : "—";
};
const timestamp = (value: unknown): string => {
  if (!value) return "unknown time";
  const parsed = new Date(String(value));
  return Number.isNaN(parsed.getTime()) ? "unknown time" : parsed.toLocaleString();
};
const requestId = (prefix: string): string => `${prefix}-${crypto.randomUUID()}`;

let twinStatus: Json | null = null;
let twinPortfolio: Json | null = null;
let twinReadiness: Json | null = null;
let activeMandate: Json | null = null;
let decisionCursor: string | null = null;
let commandCursor: string | null = null;

function authHeaders(json = false): HeadersInit {
  const token = localStorage.getItem("foresea_session");
  return {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(json ? { "Content-Type": "application/json" } : {}),
  };
}

async function api(path: string, init: RequestInit = {}): Promise<Json> {
  const response = await fetch(path, { ...init, headers: { ...authHeaders(Boolean(init.body)), ...(init.headers || {}) } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(String(payload.detail || `Request failed (${response.status})`));
  return payload;
}

function text(id: string, value: unknown): void {
  byId(id).textContent = String(value ?? "—");
}

function empty(list: HTMLElement, message: string): void {
  list.replaceChildren();
  const item = document.createElement("li");
  item.className = "twin-empty";
  item.textContent = message;
  list.append(item);
}

function row(title: string, meta: string, badge?: string): HTMLLIElement {
  const item = document.createElement("li");
  item.className = "twin-row";
  const head = document.createElement("div");
  head.className = "twin-row-head";
  const heading = document.createElement("div");
  heading.className = "twin-row-title";
  heading.textContent = title;
  head.append(heading);
  if (badge) {
    const label = document.createElement("span");
    label.className = `twin-badge ${badge === "submission_unknown" ? "unknown" : badge === "partially_filled" ? "partial" : ""}`;
    label.textContent = badge.replaceAll("_", " ");
    head.append(label);
  }
  const detail = document.createElement("div");
  detail.className = "twin-row-meta";
  detail.textContent = meta;
  item.append(head, detail);
  return item;
}

function setNotice(message: string, danger = false): void {
  const notice = byId("twinConnectionNotice");
  notice.textContent = message;
  notice.classList.toggle("danger", danger);
}

function renderStatus(): void {
  if (!twinStatus || !twinPortfolio) return;
  const accounts: Json[] = twinPortfolio.accounts || [];
  const cash = accounts.reduce((sum, account) => sum + Number(account.available_cash_for_reservation || 0), 0);
  const reserved = accounts.reduce((sum, account) => sum + Number(account.reserved_max_loss || 0), 0);
  text("twinCash", money(cash));
  text("twinReserved", money(reserved));
  text("twinAccounts", twinStatus.account_count ?? accounts.length);
  text("twinUnknown", twinStatus.unknown_commands ?? 0);

  const mode = byId("twinModePill");
  mode.textContent = `${String(twinStatus.mode || "shadow")} · ${twinStatus.new_exposure_allowed ? "ready" : "blocked"}`;
  mode.dataset.tone = twinStatus.new_exposure_allowed ? "good" : "danger";
  const pause = byId<HTMLButtonElement>("twinPauseBtn");
  pause.disabled = false;
  pause.textContent = twinStatus.paused ? "Resume autonomy" : "Pause autonomy";
  setNotice(
    twinStatus.paused
      ? `Autonomy paused since ${timestamp(twinStatus.pause?.changed_at)}. Existing orders remain visible for recovery.`
      : `Private state refreshed ${timestamp(twinStatus.generated_at)}.`,
    Boolean(twinStatus.paused || twinStatus.unknown_commands),
  );

  const blockers = byId("twinBlockerList");
  const reasons: string[] = twinStatus.blockers || [];
  blockers.replaceChildren(...reasons.map((reason) => row(
    reason.replaceAll("_", " "),
    reason === "submission_unknown"
      ? "An order acknowledgement is uncertain. New exposure stays blocked until reconciliation completes."
      : reason === "account_snapshot_stale"
        ? "At least one venue account snapshot is missing or stale."
        : reason === "readiness_not_met"
          ? "The current release has not passed its exact evidence gates."
          : reason === "no_active_mandate"
            ? "No approved, unexpired mandate currently grants authority."
            : "Owner control currently blocks new autonomous exposure.",
  )));
  if (!reasons.length) empty(blockers, "All current controls permit bounded new exposure.");

  activeMandate = (twinStatus.mandates || []).find((mandate: Json) => mandate.active) || null;
  renderMandateReview(activeMandate);
}

function renderPortfolio(): void {
  const list = byId("twinPortfolioList");
  const accounts: Json[] = twinPortfolio?.accounts || [];
  const readiness = new Map<string, Json>((twinReadiness?.accounts || []).map((item: Json) => [item.account_scope_id, item]));
  list.replaceChildren(...accounts.map((account) => {
    const gate = readiness.get(account.account_scope_id) || {};
    return row(
      `${account.venue} · ${account.environment} · ${account.account_scope_id}`,
      `${money(account.available_cash_for_reservation)} available · ${money(account.reserved_max_loss)} reserved risk · ${account.position_count ?? "—"} positions · snapshot ${account.snapshot_status} at ${timestamp(account.snapshot_received_at)} · readiness ${gate.status || "unavailable"}`,
      account.snapshot_status,
    );
  }));
  if (!accounts.length) empty(list, "No registered autonomous account scopes.");

  const select = byId<HTMLSelectElement>("twinMandateScope");
  const selected = select.value;
  select.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = accounts.length ? "Select an account scope" : "No account scope available";
  select.append(placeholder);
  for (const account of accounts) {
    const option = document.createElement("option");
    option.value = account.account_scope_id;
    option.dataset.environment = account.environment;
    option.textContent = `${account.venue} · ${account.environment} · ${account.account_scope_id}`;
    select.append(option);
  }
  if ([...select.options].some((option) => option.value === selected)) select.value = selected;
}

function renderPage(target: string, payload: Json, kind: "decision" | "command", append: boolean): void {
  const list = byId(target);
  const items: Json[] = payload.items || [];
  if (!append) list.replaceChildren();
  for (const item of items) {
    const title = kind === "decision"
      ? `${item.decision} · ${item.reason}`
      : `${item.state} · ${item.id}`;
    const meta = kind === "decision"
      ? `${item.strategy_version} · ${timestamp(item.created_at)}${item.intent ? ` · ${item.intent.action} ${item.intent.quantity} @ ${item.intent.limit_price}` : ""}`
      : `${timestamp(item.created_at)} · scope ${item.account_scope_id}${item.reason ? ` · ${item.reason}` : ""}`;
    const itemRow = row(title, meta, kind === "command" ? item.state : item.decision);
    if (kind === "command" && item.cancellable) {
      const button = document.createElement("button");
      button.className = "twin-action secondary";
      button.type = "button";
      button.textContent = "Request cancel";
      button.addEventListener("click", () => requestCancel(item));
      itemRow.querySelector(".twin-row-head")?.append(button);
    }
    list.append(itemRow);
  }
  if (!list.children.length) empty(list, kind === "decision" ? "No strategy decisions recorded." : "No order commands recorded.");
}

async function loadDesk(): Promise<void> {
  if (!localStorage.getItem("foresea_session")) {
    setNotice("Sign in to load private twin state.", true);
    return;
  }
  const refresh = byId<HTMLButtonElement>("twinRefreshBtn");
  refresh.disabled = true;
  setNotice("Loading durable account, mandate, strategy and command state…");
  try {
    const [status, portfolio, readiness, decisions, commands] = await Promise.all([
      api("/twin/status"), api("/twin/portfolio"), api("/twin/readiness"),
      api("/twin/decisions?limit=20"), api("/twin/commands?limit=20"),
    ]);
    twinStatus = status;
    twinPortfolio = portfolio;
    twinReadiness = readiness;
    decisionCursor = decisions.next_cursor || null;
    commandCursor = commands.next_cursor || null;
    renderStatus();
    renderPortfolio();
    renderPage("twinDecisionList", decisions, "decision", false);
    renderPage("twinCommandList", commands, "command", false);
    byId<HTMLButtonElement>("twinMoreDecisions").hidden = !decisionCursor;
    byId<HTMLButtonElement>("twinMoreCommands").hidden = !commandCursor;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Private twin state is unavailable.";
    setNotice(message, true);
    const mode = byId("twinModePill");
    mode.textContent = "Disconnected";
    mode.dataset.tone = "danger";
  } finally {
    refresh.disabled = false;
  }
}

async function loadMore(kind: "decision" | "command"): Promise<void> {
  const cursor = kind === "decision" ? decisionCursor : commandCursor;
  if (!cursor) return;
  const plural = kind === "decision" ? "decisions" : "commands";
  const payload = await api(`/twin/${plural}?limit=20&cursor=${encodeURIComponent(cursor)}`);
  renderPage(kind === "decision" ? "twinDecisionList" : "twinCommandList", payload, kind, true);
  if (kind === "decision") decisionCursor = payload.next_cursor || null;
  else commandCursor = payload.next_cursor || null;
  byId<HTMLButtonElement>(kind === "decision" ? "twinMoreDecisions" : "twinMoreCommands").hidden = !payload.next_cursor;
}

async function togglePause(): Promise<void> {
  if (!twinStatus) return;
  const paused = !Boolean(twinStatus.paused);
  const phrase = paused ? "PAUSE AUTONOMY" : "RESUME AUTONOMY";
  if (window.prompt(`Type ${phrase} to continue.`) !== phrase) return;
  await api("/twin/pause", {
    method: "POST",
    body: JSON.stringify({ paused, reason: paused ? "owner_kill_switch" : "owner_resume", idempotency_key: requestId("pause") }),
  });
  await loadDesk();
}

function selectedScope(): Json | null {
  const value = byId<HTMLSelectElement>("twinMandateScope").value;
  return (twinPortfolio?.accounts || []).find((account: Json) => account.account_scope_id === value) || null;
}

async function draftMandate(event: SubmitEvent): Promise<void> {
  event.preventDefault();
  const account = selectedScope();
  if (!account) return setNotice("Select a registered account scope.", true);
  const expires = new Date(byId<HTMLInputElement>("twinMandateExpiry").value);
  if (Number.isNaN(expires.getTime())) return setNotice("Choose a valid expiry.", true);
  const body = {
    client_request_id: requestId("mandate"),
    account_scope_id: account.account_scope_id,
    strategy_version: byId<HTMLInputElement>("twinMandateStrategy").value,
    expires_at: expires.toISOString(),
    live: account.environment === "live",
    allowed_actions: ["BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"],
    max_capital: byId<HTMLInputElement>("twinMandateCapital").value,
    max_loss: byId<HTMLInputElement>("twinMandateLoss").value,
    max_model_usd: byId<HTMLInputElement>("twinMandateModelUsd").value,
    max_model_tokens: Number(byId<HTMLInputElement>("twinMandateTokens").value),
    max_model_requests: Number(byId<HTMLInputElement>("twinMandateRequests").value),
  };
  try {
    activeMandate = await api("/twin/mandates", { method: "POST", body: JSON.stringify(body) });
    renderMandateReview(activeMandate);
    setNotice("Mandate drafted. Verify every value below before activation.");
  } catch (error) {
    setNotice(error instanceof Error ? error.message : "Mandate draft failed.", true);
  }
}

function renderMandateReview(mandate: Json | null): void {
  const review = byId("twinMandateReview");
  review.hidden = !mandate;
  if (!mandate) return;
  const mode = mandate.live ? "LIVE" : "SHADOW";
  text("twinReviewScope", `${mode} · ${mandate.venue} · ${mandate.account_scope_id}`);
  text("twinReviewDetails", `Expires ${timestamp(mandate.expires_at)} · capital ${money(mandate.max_capital)} · loss ${money(mandate.max_loss)} · model ${money(mandate.max_model_usd)} / ${mandate.max_model_tokens} tokens / ${mandate.max_model_requests} requests · authority ${mandate.authority_hash}`);
  byId<HTMLButtonElement>("twinApproveBtn").hidden = Boolean(mandate.approved_at || mandate.approved);
  byId<HTMLButtonElement>("twinRevokeBtn").hidden = Boolean(mandate.revoked);
}

async function approveMandate(): Promise<void> {
  if (!activeMandate) return;
  const phrase = `ACTIVATE ${activeMandate.live ? "LIVE" : "SHADOW"}`;
  if (window.prompt(`Review the exact scope, expiry and limits. Type ${phrase} to activate.`) !== phrase) return;
  activeMandate = await api(`/twin/mandates/${encodeURIComponent(activeMandate.id)}/approve`, {
    method: "POST",
    body: JSON.stringify({ expected_hash: activeMandate.authority_hash, idempotency_key: requestId("approve") }),
  });
  await loadDesk();
}

async function revokeMandate(): Promise<void> {
  if (!activeMandate) return;
  if (window.prompt("Type REVOKE AUTONOMY to remove this authority.") !== "REVOKE AUTONOMY") return;
  await api(`/twin/mandates/${encodeURIComponent(activeMandate.id)}/revoke`, {
    method: "POST", body: JSON.stringify({ idempotency_key: requestId("revoke") }),
  });
  activeMandate = null;
  await loadDesk();
}

async function requestCancel(command: Json): Promise<void> {
  const phrase = `CANCEL ${command.id}`;
  if (window.prompt(`Type ${phrase} to enqueue reconciliation and cancellation.`) !== phrase) return;
  await api(`/twin/commands/${encodeURIComponent(command.id)}/cancel`, {
    method: "POST", body: JSON.stringify({ idempotency_key: requestId("cancel") }),
  });
  await loadDesk();
}

function startTwinDesk(): void {
  byId("twinRefreshBtn").addEventListener("click", loadDesk);
  byId("twinPauseBtn").addEventListener("click", togglePause);
  byId("twinMoreDecisions").addEventListener("click", () => loadMore("decision"));
  byId("twinMoreCommands").addEventListener("click", () => loadMore("command"));
  byId("twinMandateForm").addEventListener("submit", (event) => draftMandate(event as SubmitEvent));
  byId("twinApproveBtn").addEventListener("click", approveMandate);
  byId("twinRevokeBtn").addEventListener("click", revokeMandate);
  const expiry = byId<HTMLInputElement>("twinMandateExpiry");
  const defaultExpiry = new Date(Date.now() + 24 * 60 * 60 * 1000);
  expiry.value = defaultExpiry.toISOString().slice(0, 16);
  loadDesk();
}

function initializeTwinDesk(): void {
  try {
    startTwinDesk();
    document.documentElement.dataset.twinDeskReady = "true";
  } catch (error) {
    document.documentElement.dataset.twinDeskReady = "false";
    const notice = document.getElementById("twinConnectionNotice");
    if (notice) {
      notice.textContent = error instanceof Error ? error.message : "Twin operator controls failed to initialize.";
      notice.classList.add("danger");
    }
  }
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", initializeTwinDesk, { once: true });
} else {
  initializeTwinDesk();
}
