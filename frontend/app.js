/* ANVIL frontend.
 *
 * Mirrors the approved three-zone workbench: setup on the left, a persistent
 * viewport in the centre that shows pipeline progress inline, and an activity
 * rail on the right carrying the batch queue, edit history, and sessions.
 *
 * All state that matters lives on the server; this file renders it and streams
 * progress over SSE.
 */

const API = "";

/* ================================================================
   CURSOR — unchanged from the approved mockup
   ================================================================ */
const curDot = document.getElementById("cur-dot");
const curRing = document.getElementById("cur-ring");
let mx = innerWidth / 2, my = innerHeight / 2, rx = mx, ry = my;

addEventListener("mousemove", (e) => {
  mx = e.clientX; my = e.clientY;
  curDot.style.left = mx + "px"; curDot.style.top = my + "px";
  const t = e.target;
  const interactive = t.closest(
    "button,.btn,.mode-tab,.switch,.mini-switch,.icon-btn,.style-opt,.style-select-btn," +
    ".session-item,.radio-opt,.optional-toggle,.tab,.run-toggle button,.q-actions button,a,[data-action]"
  );
  const textLike = t.closest("input,textarea,select");
  document.body.classList.toggle("cur-hover", !!interactive && !textLike);
  document.body.classList.toggle("cur-text", !!textLike);
});
addEventListener("mousedown", () => document.body.classList.add("cur-down"));
addEventListener("mouseup", () => document.body.classList.remove("cur-down"));
(function loop() {
  rx += (mx - rx) * 0.22; ry += (my - ry) * 0.22;
  curRing.style.left = rx + "px"; curRing.style.top = ry + "px";
  requestAnimationFrame(loop);
})();

/* ================================================================
   STATE
   ================================================================ */
const state = {
  modes: {},
  styles: [],
  proceduralUnsuited: [],
  mode: "text_to_image",
  styleId: "realistic",
  procedural: false,
  checklist: {},
  flags: {},
  uploadPath: null,
  uploadName: null,
  batchMode: false,
  queue: { items: [], counts: {}, running: false, resumable: false },
  sessionId: null,
  session: null,
  editHistory: [],
  busy: false,
  pendingConfirm: null,
  logLines: [],
  assist: { open: false, history: [], busy: false, aiFilled: new Set() },
  status: { data: null, fetchedAt: 0, timer: null },
  review: null,
  texture: { available: false, reason: "checking…" },
  remesh: { available: false, reason: "checking…" },
  railView: "projects",
};

const $ = (id) => document.getElementById(id);

/* ================================================================
   API helpers
   ================================================================ */
async function api(path, options = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try { body = await res.json(); } catch { /* empty body is fine */ }
  if (!res.ok) {
    throw new Error(body?.detail || `${res.status} ${res.statusText}`);
  }
  return body;
}

const post = (path, data) => api(path, { method: "POST", body: JSON.stringify(data || {}) });
const del = (path) => api(path, { method: "DELETE" });

/* ================================================================
   BOOT
   ================================================================ */
async function boot() {
  try {
    const [schemasRes, stylesRes, settingsRes, health] = await Promise.all([
      api("/api/schemas"),
      api("/api/styles"),
      api("/api/settings"),
      api("/api/health"),
    ]);

    state.modes = schemasRes.modes;
    state.styles = stylesRes.styles;
    state.proceduralUnsuited = stylesRes.procedural_unsuited;
    state.flags = defaultFlags();

    renderModeTabs();
    renderStyleMenu();
    renderFields();
    applyHealth(health);
    populateSettings(settingsRes, health);

    try {
      if (localStorage.getItem("anvil.railCollapsed") === "1") setRailCollapsed(true);
    } catch { /* private mode */ }
    setRailView(state.railView, { expand: false });

    await refreshQueue();
    await refreshSessions();
    connectEvents();
    setViewportEmpty();
    renderEditLog();
    updatePrimaryButton();
    await refreshStatus();
    startStatusPolling();
    await refreshTextureAvailability();
  } catch (err) {
    toast("Can't reach the ANVIL server", err.message, "error");
    $("status-pill").classList.add("offline");
    $("status-backend").textContent = "offline";
  }
}

function defaultFlags() {
  const out = {};
  (state.modes.image_upload?.flags || []).forEach((f) => { out[f.key] = f.default; });
  return out;
}

function applyHealth(health) {
  const pill = $("status-pill");
  pill.classList.remove("offline");
  $("status-backend").textContent = `${health.backends.model} · ${health.backends.llm.split(" ")[0]}`;
  $("gizmo-file").textContent = health.blender.blend_file.split(/[\\/]/).pop();

  if (health.gpu.warning) {
    toast(health.gpu.blocked ? "Generation blocked" : "Heads up", health.gpu.warning,
          health.gpu.blocked ? "error" : "warn");
  }
  if (!health.blender.available) {
    toast("Blender not configured", "Meshes will be generated to disk but not imported into a scene. Set BLENDER_PATH in .env.", "warn");
  }
}

/* ================================================================
   SETUP PANEL — mode tabs, style select, fields
   ================================================================ */
function renderModeTabs() {
  const wrap = $("mode-tabs");
  wrap.innerHTML = Object.entries(state.modes)
    .map(([id, def]) => `
      <div class="mode-tab${id === state.mode ? " active" : ""}" data-mode="${id}">
        <span>${def.label}</span><span class="sub">${def.hint}</span>
      </div>`)
    .join("");
  wrap.querySelectorAll(".mode-tab").forEach((el) =>
    el.addEventListener("click", () => setMode(el.dataset.mode)));
}

function setMode(mode) {
  state.mode = mode;
  state.checklist = {};
  state.uploadPath = null;
  state.uploadName = null;
  resetAssist();
  renderModeTabs();
  renderFields();
  updatePrimaryButton();
}

function renderStyleMenu() {
  const menu = $("style-menu");
  menu.innerHTML = state.styles
    .map((s) => `
      <div class="style-opt${s.id === state.styleId ? " sel" : ""}" data-style="${s.id}">
        ${s.label}<span class="tag">${s.target_tris.toLocaleString()} tris</span>
      </div>`)
    .join("");
  menu.querySelectorAll(".style-opt").forEach((el) =>
    el.addEventListener("click", () => pickStyle(el.dataset.style)));
  const current = state.styles.find((s) => s.id === state.styleId);
  $("style-select-label").textContent = current ? current.label : state.styleId;
}

function pickStyle(id) {
  state.styleId = id;
  $("style-select").classList.remove("open");
  renderStyleMenu();
  if (state.procedural && state.proceduralUnsuited.includes(id)) {
    toast("Style may not suit procedural mode",
          `${state.styles.find((s) => s.id === id)?.label} will look primitive as generated geometry.`, "warn");
  }
}

$("style-select-btn").addEventListener("click", () => $("style-select").classList.toggle("open"));
document.addEventListener("click", (e) => {
  if (!e.target.closest("#style-select")) $("style-select").classList.remove("open");
});

function renderFields() {
  const wrap = $("mode-fields");
  const def = state.modes[state.mode];
  if (!def) { wrap.innerHTML = ""; return; }

  if (state.mode === "image_upload") {
    wrap.innerHTML = `
      <div class="dropzone" id="dropzone">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 16V4m0 0L7 9m5-5l5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>
        Click to upload a reference image
      </div>
      <input type="file" id="file-input" accept="image/png,image/jpeg,image/webp" hidden>
      <div class="col-label" style="margin-bottom:8px;font-size:10.5px">Generation flags</div>
      ${def.flags.map((f) => renderFlag(f)).join("")}`;

    $("dropzone").addEventListener("click", () => $("file-input").click());
    $("file-input").addEventListener("change", handleUpload);
    wrap.querySelectorAll("[data-flag]").forEach(bindFlag);
    return;
  }

  let html = def.mandatory.map((f) => fieldHtml(f, true)).join("");
  if (def.mandatory.length && def.optional.length) {
    html += `<div class="optional-toggle" id="optional-toggle">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M9 6l6 6-6 6"/></svg>
      Optional fields (${def.optional.length})</div>
      <div class="optional-fields">${def.optional.map((f) => fieldHtml(f, false)).join("")}</div>`;
  } else {
    html += def.optional.map((f) => fieldHtml(f, false)).join("");
  }
  wrap.innerHTML = html;

  const toggle = $("optional-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      toggle.classList.toggle("open");
      toggle.nextElementSibling.classList.toggle("open");
    });
  }
  wrap.querySelectorAll("input[data-key]").forEach((input) => {
    input.addEventListener("input", () => {
      state.checklist[input.dataset.key] = input.value;
      const field = input.closest(".field");
      field.classList.toggle("filled", input.value.length > 0);
      // Typing over an assistant-filled value makes it the user's again.
      field.classList.remove("ai-filled");
      state.assist.aiFilled.delete(input.dataset.key);
      updatePrimaryButton();
    });
  });

  // Re-rendering the form drops the assistant's markers; restore them, and put
  // back any values it had already extracted.
  Object.entries(state.checklist).forEach(([key, value]) => {
    const input = wrap.querySelector(`input[data-key="${CSS.escape(key)}"]`);
    if (!input || !value) return;
    input.value = value;
    const field = input.closest(".field");
    field.classList.add("filled");
    if (state.assist.aiFilled.has(key)) field.classList.add("ai-filled");
  });
  syncAssistVisibility();
}

function fieldHtml(f, mandatory) {
  return `<div class="field">
    <label>${f.label}${mandatory ? '<span class="req">*</span>' : ""}</label>
    <input type="text" data-key="${f.key}" placeholder="${f.placeholder || ""}">
  </div>`;
}

/* ================================================================
   CONVERSATIONAL CHECKLIST ASSISTANT

   An alternative way into the same checklist, not a second one. Every reply
   writes through to state.checklist and to the visible inputs, so the form
   stays the single source of truth and a user can switch between talking and
   typing mid-way without losing either.
   ================================================================ */
function resetAssist() {
  state.assist.history = [];
  state.assist.busy = false;
  state.assist.aiFilled = new Set();
  const log = $("assist-log");
  if (log) log.innerHTML = "";
  syncAssistVisibility();
}

function syncAssistVisibility() {
  const el = $("assist");
  if (!el) return;
  // Image-upload mode has flags rather than a checklist — nothing to interview about.
  const def = state.modes[state.mode];
  const applicable = state.mode !== "image_upload"
    && !!def && (def.mandatory.length + def.optional.length) > 0;
  el.style.display = applicable ? "" : "none";
  el.classList.toggle("open", applicable && state.assist.open);
}

function toggleAssist() {
  state.assist.open = !state.assist.open;
  syncAssistVisibility();
  if (state.assist.open) $("assist-text")?.focus();
}

function pushAssistMsg(role, text, cls = "") {
  const log = $("assist-log");
  if (!log) return null;
  const div = document.createElement("div");
  div.className = `assist-msg ${role}${cls ? " " + cls : ""}`;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

/** Write assistant-extracted values into both the model and the visible inputs. */
function applyAssistFields(fields) {
  const keys = Object.keys(fields || {});
  keys.forEach((key) => {
    state.checklist[key] = fields[key];
    state.assist.aiFilled.add(key);
    const input = document.querySelector(`#mode-fields input[data-key="${CSS.escape(key)}"]`);
    if (input) {
      input.value = fields[key];
      const field = input.closest(".field");
      field?.classList.add("filled", "ai-filled");
    }
  });
  // An optional field the assistant filled may be hidden behind the collapse.
  if (keys.length) {
    const toggle = $("optional-toggle");
    const hidden = keys.some((k) =>
      document.querySelector(`.optional-fields input[data-key="${CSS.escape(k)}"]`));
    if (toggle && hidden && !toggle.classList.contains("open")) {
      toggle.classList.add("open");
      toggle.nextElementSibling.classList.add("open");
    }
  }
  updatePrimaryButton();
  return keys.length;
}

async function sendAssistMessage() {
  const input = $("assist-text");
  const text = (input.value || "").trim();
  if (!text || state.assist.busy) return;

  state.assist.busy = true;
  input.value = "";
  $("assist-send").disabled = true;
  pushAssistMsg("user", text);
  const pending = pushAssistMsg("bot", "thinking…", "pending");

  try {
    const res = await api("/api/checklist/chat", {
      method: "POST",
      body: JSON.stringify({
        mode: state.mode,
        message: text,
        checklist: state.checklist,
        history: state.assist.history,
      }),
    });

    const filled = applyAssistFields(res.fields);
    const reply = res.reply || res.next_question
      || (filled ? "Got it — updated the fields." : "Noted.");

    pending.remove();
    pushAssistMsg("bot", reply);
    // Only the conversation itself goes into history; field values travel
    // separately as the checklist, so the two can't drift apart.
    state.assist.history.push({ role: "user", content: text });
    state.assist.history.push({ role: "assistant", content: reply });

    if (filled) {
      const names = Object.keys(res.fields).length;
      toast("Checklist updated", `${names} field${names === 1 ? "" : "s"} filled from chat`, "ok");
    }
  } catch (err) {
    pending.remove();
    // The form still works without the LLM, so say what's still possible.
    pushAssistMsg("bot", `${err.message}\n\nYou can still fill the fields directly below.`, "err");
  } finally {
    state.assist.busy = false;
    $("assist-send").disabled = false;
    input.focus();
  }
}

function renderFlag(f) {
  if (f.type === "bool") {
    return `<div class="flag-row"><span class="label">${f.label}</span>
      <div class="mini-switch${f.default ? " on" : ""}" data-flag="${f.key}" data-type="bool"></div></div>`;
  }
  return `<div class="flag-row"><span class="label">${f.label}</span>
    <select data-flag="${f.key}" data-type="enum">
      ${f.options.map((o) => `<option${o === f.default ? " selected" : ""}>${o}</option>`).join("")}
    </select></div>`;
}

function bindFlag(el) {
  const key = el.dataset.flag;
  if (el.dataset.type === "bool") {
    el.addEventListener("click", () => {
      el.classList.toggle("on");
      state.flags[key] = el.classList.contains("on");
    });
  } else {
    el.addEventListener("change", () => { state.flags[key] = el.value; });
  }
}

async function handleUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch(API + "/api/upload", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Upload failed");
    state.uploadPath = body.path;
    state.uploadName = body.name;
    const dz = $("dropzone");
    dz.classList.add("has-file");
    dz.innerHTML = `<div class="thumb"><img src="${body.url}" alt=""></div>
      <div style="font-size:11.5px;color:var(--text-dim)">${body.name}</div>`;
    updatePrimaryButton();
  } catch (err) {
    toast("Upload failed", err.message, "error");
  }
}

function inputComplete() {
  const def = state.modes[state.mode];
  if (!def) return false;
  if (state.mode === "image_upload") return !!state.uploadPath;
  return def.mandatory.every((f) => (state.checklist[f.key] || "").trim().length > 0);
}

function updatePrimaryButton() {
  const btn = $("generate-btn");
  btn.textContent = state.batchMode ? "+ Add to queue" : "Generate";
  btn.disabled = !inputComplete() || state.busy || (state.batchMode && state.queue.running);
  updateGatedControls();
}

/* The server allows exactly one pipeline at a time. Reflecting that in the UI
 * beats letting the user click and only then discover it via a 409 toast. */
function updateGatedControls() {
  const gated = state.busy || state.queue.running;
  const hasObject = !!state.sessionId;
  $("edit-input").disabled = gated || !hasObject;
  $("edit-send-btn").disabled = gated || !hasObject;
  ["btn-rig", "btn-remesh", "btn-texture", "btn-export", "btn-undo"].forEach((id) => {
    const el = $(id);
    if (el) el.disabled = gated || !hasObject;
  });
  // Texturing and remeshing have prerequisites beyond "an object exists"; when they
  // aren't met the button stays disabled and explains why on hover rather than
  // failing after the click.
  const gatedAction = (id, cap, ready) => {
    const el = $(id);
    if (!el) return;
    if (hasObject && !gated && cap && !cap.available) {
      el.disabled = true;
      el.title = cap.reason;
    } else {
      el.title = ready;
    }
  };
  gatedAction("btn-texture", state.texture, "Paint PBR textures onto this mesh");
  gatedAction("btn-remesh", state.remesh, "Convert to quad topology (adds ~1 min)");
  if (gated && hasObject) {
    $("edit-input").placeholder = "a generation is running — edits are paused until it finishes";
  }
}

/* ---- procedural toggle ---- */
$("proc-switch").addEventListener("click", () => {
  if (state.procedural) {
    state.procedural = false;
    $("proc-switch").classList.remove("on");
    toast("Procedural mode disabled", "The 3D backend reloads on next use, not eagerly");
  } else {
    const style = state.styles.find((s) => s.id === state.styleId);
    const unsuited = state.proceduralUnsuited.includes(state.styleId);
    $("proc-warn-box").innerHTML = unsuited
      ? `Your current style — <b>${style.label}</b> — is one of the presets that will look primitive as generated geometry, not photorealistic or stylized. Consider a hard-surface preset instead.`
      : `Not suited for organic shapes, detailed or worn surfaces, characters, or the <b>Realistic</b>, <b>Hyper realistic</b>, <b>Anime cel-shaded</b>, or <b>Painterly</b> presets — these will look primitive, not photorealistic or stylized.`;
    $("dlg-proc").classList.add("show");
  }
});

$("proc-confirm").addEventListener("click", () => {
  state.procedural = true;
  $("proc-switch").classList.add("on");
  $("dlg-proc").classList.remove("show");
  toast("Procedural mode enabled", "3D-gen backend ejected from VRAM");
});

/* ================================================================
   SINGLE / BATCH
   ================================================================ */
$("run-single").addEventListener("click", () => setBatchMode(false));
$("run-batch").addEventListener("click", () => setBatchMode(true));

function setBatchMode(on) {
  state.batchMode = on;
  $("run-single").classList.toggle("active", !on);
  $("run-batch").classList.toggle("active", on);
  $("run-batch").classList.toggle("is-batch", on);
  $("queue-section").style.display = on ? "block" : "none";
  if (on) setRailView("activity");
  updatePrimaryButton();
}

/* ================================================================
   ACTIVITY RAIL
   ================================================================ */
$("assist-toggle").addEventListener("click", toggleAssist);
$("assist-send").addEventListener("click", sendAssistMessage);
$("assist-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); sendAssistMessage(); }
});

/* ================================================================
   NAV RAIL

   The icon strip selects which view the panel beside it shows. Clicking the
   active icon collapses the panel, which is the usual convention and saves a
   separate control for "hide". Collapsed state persists, because someone who
   wants the width back wants it back on every launch.
   ================================================================ */
const RAIL_VIEWS = ["projects", "activity", "status"];

function setRailView(name, { toggle = false, expand = true } = {}) {
  const shell = $("app-shell");
  const collapsed = shell.classList.contains("rail-collapsed");

  if (toggle && state.railView === name && !collapsed) {
    setRailCollapsed(true);
    return;
  }
  state.railView = name;
  RAIL_VIEWS.forEach((v) => {
    $(`view-${v}`)?.classList.toggle("active", v === name);
    document.querySelector(`.rail-btn[data-view="${v}"]`)?.classList.toggle("active", v === name);
  });
  // Selecting a view opens the panel to show it — except on boot, where doing so
  // would override a collapsed state the user deliberately saved.
  if (collapsed && expand) setRailCollapsed(false);

  if (name === "projects") refreshSessions();
  if (name === "status") refreshStatus();
}

function setRailCollapsed(collapsed) {
  // The class goes on the panel itself, not an ancestor: a rule keyed off an
  // ancestor class was silently never applied in testing, while a class on the
  // element always invalidates correctly.
  $("rail-panel").classList.toggle("is-collapsed", collapsed);
  $("app-shell").classList.toggle("rail-collapsed", collapsed);
  try { localStorage.setItem("anvil.railCollapsed", collapsed ? "1" : "0"); } catch { /* private mode */ }
  // three.js sizes to its container, so the viewport has to be told it grew.
  window.dispatchEvent(new Event("resize"));
}

document.querySelectorAll(".rail-btn[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => setRailView(btn.dataset.view, { toggle: true }));
});
$("rail-toggle").addEventListener("click", () => {
  setRailCollapsed(!$("app-shell").classList.contains("rail-collapsed"));
});
$("rail-settings").addEventListener("click", () => $("settings-panel").classList.add("show"));

/* ================================================================
   PIPELINE STATUS

   SSE already pushes each event as it happens. This polls the whole snapshot
   on a fixed interval as well, which covers what a stream cannot: a tab opened
   part-way through a run, a reconnect that missed events while it was down, and
   long silent steps — mesh generation emits nothing for minutes, and a display
   that only moves on events looks indistinguishable from a hang.
   ================================================================ */
const STATUS_POLL_MS = 10000;

function statusTabIsOpen() {
  // The status view is only "open" when its panel is showing AND not collapsed away.
  return $("view-status")?.classList.contains("active")
      && !$("app-shell").classList.contains("rail-collapsed");
}

async function refreshStatus() {
  try {
    state.status.data = await api("/api/status");
    state.status.fetchedAt = Date.now();
  } catch {
    state.status.data = null;      // rendered as "unreachable", not left stale
  }
  renderStatus();
}

function startStatusPolling() {
  if (state.status.timer) return;
  state.status.timer = setInterval(() => {
    // Poll while the panel is visible, and while a run is in flight regardless —
    // the tab's live dot has to stay honest even when you're looking elsewhere.
    if (statusTabIsOpen() || state.busy || state.queue.running) refreshStatus();
    else renderStatusAge();
  }, STATUS_POLL_MS);
}

function fmtDuration(s) {
  if (s === null || s === undefined) return "—";
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

function renderStatusAge() {
  const el = $("status-age");
  if (!el) return;
  if (!state.status.fetchedAt) { el.textContent = "—"; return; }
  const age = Math.round((Date.now() - state.status.fetchedAt) / 1000);
  el.textContent = age < 2 ? "just now" : `${age}s ago`;
}

function renderStatus() {
  const body = $("status-body");
  if (!body) return;
  const d = state.status.data;

  $("status-live-dot").style.display = d && d.phase === "running" ? "" : "none";
  renderStatusAge();

  if (!d) {
    body.innerHTML = `<div class="status-empty">Can't reach the server. The page will keep trying every ${STATUS_POLL_MS / 1000}s.</div>`;
    return;
  }

  const phaseWord = { idle: "Idle", running: "Running", done: "Finished", failed: "Failed" }[d.phase] || d.phase;
  const what = d.current ? d.current.label
    : d.phase === "idle" ? "Nothing in the pipeline"
    : d.error ? d.error : "All steps complete";
  const progress = d.steps_total ? `step ${d.steps_done}/${d.steps_total}` : "";

  let html = `
    <div class="status-phase ${d.phase}">
      <span class="dot"></span>
      <span class="what">${escapeHtml(what)}<small>${phaseWord}${progress ? " · " + progress : ""}${d.gate?.kind ? " · " + escapeHtml(d.gate.kind) : ""}</small></span>
      <span class="clock">${fmtDuration(d.elapsed_s)}</span>
    </div>`;

  if (d.steps?.length) {
    html += `<div class="status-steps">` + d.steps.map((s) => `
      <div class="status-step ${s.status}">
        <span class="idx">${s.status === "done" ? "✓" : s.index}</span>
        <span class="nm">${escapeHtml(s.label)}</span>
        <span class="dur">${fmtDuration(s.duration_s)}</span>
      </div>`).join("") + `</div>`;
  }

  const vramPct = d.vram?.used_gb && d.vram?.total_gb
    ? Math.min(100, Math.round((d.vram.used_gb / d.vram.total_gb) * 100)) : 0;

  html += `<dl class="status-meta">
    <dt>Resident</dt><dd>${d.resident?.key ? escapeHtml(d.resident.key) : "nothing loaded"}</dd>
    <dt>3D</dt><dd>${escapeHtml(d.backends?.model || "—")}</dd>
    <dt>Image</dt><dd>${escapeHtml(d.backends?.image || "—")}</dd>
    <dt>LLM</dt><dd>${escapeHtml(d.backends?.llm || "—")}</dd>
    <dt>Session</dt><dd>${d.session_id ? escapeHtml(d.session_id.slice(0, 8)) : "—"}</dd>
    <dt>VRAM</dt><dd>${d.vram?.used_gb ?? "—"} / ${d.vram?.total_gb ?? "—"} GB
      <div class="vram-bar"><i style="width:${vramPct}%"></i></div></dd>
  </dl>`;

  if (d.queue?.items?.length) {
    html += `<div class="col-label" style="margin:0 0 8px">Queue</div>
      <dl class="status-meta">
        <dt>Items</dt><dd>${d.queue.items.length}</dd>
        <dt>Running</dt><dd>${d.queue.running ? "yes" : "no"}</dd>
      </dl>`;
  }

  if (d.notes?.length) {
    html += `<div class="col-label" style="margin:0 0 8px">Latest</div>
      <div class="status-notes">` +
      d.notes.slice().reverse().map((n) =>
        `<div class="n ${n.level}">${escapeHtml(n.message)}</div>`).join("") + `</div>`;
  }

  body.innerHTML = html;
}

/* ================================================================
   PRIMARY ACTION
   ================================================================ */
$("generate-btn").addEventListener("click", () => {
  if (state.batchMode) addToQueue();
  else startGeneration();
});

function requestPayload() {
  return {
    mode: state.mode,
    style_id: state.styleId,
    procedural: state.procedural,
    checklist: state.checklist,
    flags: state.flags,
    image_path: state.uploadPath,
    // Single Text→Image runs stop on the concept image so it can be judged before
    // paying for 3D. Batch never does — nobody is watching to approve each one.
    preview_first: state.mode === "text_to_image" && !state.procedural && !state.batchMode,
  };
}

async function startGeneration() {
  if (!inputComplete()) return;
  state.busy = true;
  state.logLines = [];
  updatePrimaryButton();
  $("col-setup").classList.add("locked");
  setViewportGenerating("Running generation pipeline", []);
  try {
    const res = await post("/api/generate", requestPayload());
    state.sessionId = res.session_id;
  } catch (err) {
    state.busy = false;
    $("col-setup").classList.remove("locked");
    updatePrimaryButton();
    setViewportEmpty();
    toast("Couldn't start generation", err.message, "error");
  }
}

/* ================================================================
   QUEUE
   ================================================================ */
async function refreshQueue() {
  try {
    state.queue = await api("/api/batch");
    renderQueue();
  } catch { /* server not up yet */ }
}

async function addToQueue() {
  try {
    const payload = { ...requestPayload(), label: currentLabel() };
    const res = await post("/api/batch/add", payload);
    state.queue = res.queue;
    // Reset the input portion; keep mode + style as a convenience for the next item
    state.checklist = {};
    state.uploadPath = null;
    renderFields();
    updatePrimaryButton();
    renderQueue();
    toast("Added to queue", `${state.modes[state.mode].label} · ${res.item.label}`, "ok");
  } catch (err) {
    toast("Couldn't queue that", err.message, "error");
  }
}

function currentLabel() {
  if (state.mode === "image_upload") return state.uploadName || "(image)";
  return state.checklist.object_type || state.checklist.style_desc || "(untitled)";
}

function renderQueue() {
  const { items, counts, running, resumable } = state.queue;
  const listEl = $("queue-list");
  const pending = counts?.pending || 0;
  const done = counts?.done || 0;
  const failed = counts?.failed || 0;

  const active = pending + (counts?.running || 0);
  $("rail-activity-count").textContent = active ? String(active) : "";

  if (!items || !items.length) {
    listEl.innerHTML = `<div class="activity-empty">No items queued yet — fill in the setup panel and press "Add to queue".</div>`;
    $("queue-run-actions").style.display = "none";
    $("queue-summary-wrap").innerHTML = "";
    return;
  }

  listEl.innerHTML = items.map((item, i) => {
    const style = state.styles.find((s) => s.id === item.style_id);
    let actions = "";
    if (item.status === "failed") actions += `<button class="q-retry" data-action="retry" data-id="${item.queue_id}">Retry</button>`;
    if (item.status === "done") actions += `<button class="q-edit" data-action="open" data-id="${item.session_id}">Edit</button>`;
    if (item.status === "pending" && !running) actions += `<button data-action="remove" data-id="${item.queue_id}" aria-label="Remove"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6L6 18"/></svg></button>`;
    return `<div class="queue-item q-${item.status}">
      <div class="q-status">${queueIcon(item.status)}</div>
      <div class="q-info">
        <div class="name">${i + 1}. ${escapeHtml(item.label)}</div>
        <div class="meta">${state.modes[item.mode]?.label || item.mode} · ${style?.label || item.style_id}</div>
        ${item.error ? `<div class="err-msg">${escapeHtml(item.error)}</div>` : ""}
      </div>
      <div class="q-actions">${actions}</div>
    </div>`;
  }).join("");

  listEl.querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const { action, id } = btn.dataset;
      try {
        if (action === "retry") state.queue = await post(`/api/batch/item/${id}/retry`);
        if (action === "remove") state.queue = await del(`/api/batch/item/${id}`);
        if (action === "open") await loadSession(id);
        renderQueue();
      } catch (err) { toast("Queue action failed", err.message, "error"); }
    });
  });

  $("queue-summary-wrap").innerHTML = (done || failed)
    ? `<div class="q-summary"><span><b>${done}</b> done · <b>${failed}</b> failed · <b>${pending}</b> pending</span></div>`
    : "";

  const runActions = $("queue-run-actions");
  runActions.style.display = (pending > 0 || running) ? "flex" : "none";
  $("run-all-btn").style.display = running ? "none" : "inline-flex";
  $("run-all-btn").textContent = `Generate all (${pending})`;
  $("clear-queue-btn").disabled = running;
}

function queueIcon(status) {
  if (status === "done") return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M5 13l4 4L19 7"/></svg>`;
  if (status === "failed") return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M6 6l12 12M18 6L6 18"/></svg>`;
  return "";
}

$("run-all-btn").addEventListener("click", async () => {
  try {
    state.logLines = [];
    $("col-setup").classList.add("locked");
    await post("/api/batch/start");
  } catch (err) {
    $("col-setup").classList.remove("locked");
    toast("Couldn't start the batch", err.message, "error");
  }
});

$("clear-queue-btn").addEventListener("click", async () => {
  try { state.queue = await del("/api/batch"); renderQueue(); }
  catch (err) { toast("Couldn't clear the queue", err.message, "error"); }
});

/* ================================================================
   VIEWPORT
   ================================================================ */
let viewer = null;

/* ================================================================
   CONCEPT-IMAGE REVIEW

   Text→Image parks here after the picture and before the 3D half. The point is
   cost: the image takes ~30s and the mesh ~3 minutes, so judging the cheap half
   first avoids spending the expensive one on a picture that was never right.
   ================================================================ */
function setViewportReview(evt) {
  disposeViewer();
  state.review = { sessionId: evt.session_id, prompt: evt.prompt || "" };

  $("viewport-stage").innerHTML = `
    <div class="vp-review">
      <img class="review-img" id="review-img" alt="Concept image"
           src="/api/session/${encodeURIComponent(evt.session_id)}/concept?t=${Date.now()}">
    </div>`;
  $("viewport-title").innerHTML =
    `Concept image<span class="sub">approve it, or change the prompt and redraw</span>`;
  $("viewport-actions").style.display = "none";
  $("mesh-stats").style.display = "none";

  $("confirm-slot").innerHTML = `
    <div class="review-bar">
      <textarea id="review-prompt" rows="2"
                placeholder="prompt used for this image — edit and redraw">${escapeHtml(evt.prompt || "")}</textarea>
      <div class="review-actions">
        <button class="btn btn-primary" id="review-accept">Generate 3D →</button>
        <button class="btn" id="review-redraw">Redraw</button>
        <button class="btn btn-ghost" id="review-discard">Discard</button>
      </div>
    </div>`;

  $("review-accept").addEventListener("click", acceptConcept);
  $("review-redraw").addEventListener("click", redrawConcept);
  $("review-discard").addEventListener("click", () => {
    state.review = null;
    startNewProject({ keepInputs: true });
  });
  // Editing the prompt only matters for a redraw, so Enter there means redraw.
  $("review-prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); redrawConcept(); }
  });

  state.busy = false;
  $("col-setup").classList.remove("locked");
  updatePrimaryButton();
}

async function acceptConcept() {
  const id = state.review?.sessionId;
  if (!id) return;
  state.busy = true;
  updatePrimaryButton();
  $("col-setup").classList.add("locked");
  setViewportGenerating("Generating 3D model", []);
  try {
    await post(`/api/session/${encodeURIComponent(id)}/continue`, {});
  } catch (err) {
    state.busy = false;
    $("col-setup").classList.remove("locked");
    updatePrimaryButton();
    toast("Couldn't continue", err.message, "error");
  }
}

async function redrawConcept() {
  const id = state.review?.sessionId;
  if (!id) return;
  const prompt = $("review-prompt")?.value.trim() || null;
  state.busy = true;
  updatePrimaryButton();
  setViewportGenerating("Redrawing concept image", []);
  try {
    await post(`/api/session/${encodeURIComponent(id)}/image/regenerate`, { prompt });
  } catch (err) {
    state.busy = false;
    updatePrimaryButton();
    toast("Couldn't redraw", err.message, "error");
  }
}

function setViewportEmpty() {
  disposeViewer();
  $("viewport-stage").innerHTML = `
    <div class="vp-empty">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      <div class="t1">Nothing generated yet</div>
      <div class="t2">Fill in the setup panel on the left and press Generate — your object will appear here and drop into the Blender scene.</div>
    </div>`;
  $("viewport-title").innerHTML = `No object yet<span class="sub">set up a generation on the left</span>`;
  $("viewport-actions").style.display = "none";
  $("mesh-stats").style.display = "none";
  $("gizmo").style.display = "none";
  $("proc-badge").style.display = "none";
  $("edit-input").disabled = true;
  $("edit-send-btn").disabled = true;
  $("edit-input").placeholder = "generate an object first, then edit it here";
  $("confirm-slot").innerHTML = "";
}

function setViewportGenerating(subtitle, steps) {
  disposeViewer();
  $("viewport-stage").innerHTML = `
    <div class="vp-forge">
      <div class="forge-core"><div class="ring"></div><div class="ring2"></div><div class="glow"></div></div>
      <div class="forge-subtitle" id="forge-subtitle">${escapeHtml(subtitle)}</div>
      <div class="steps" id="forge-steps">${steps.map(stepHtml).join("")}</div>
      <div class="log-feed" id="log-feed"></div>
      ${state.queue.running ? `<div class="forge-stop"><button class="btn btn-ghost btn-sm" id="stop-batch-btn">Stop after current</button></div>` : ""}
    </div>`;
  $("viewport-title").innerHTML = `Generating<span class="sub">pipeline running — hang tight</span>`;
  $("viewport-actions").style.display = "none";
  $("mesh-stats").style.display = "none";
  $("edit-input").disabled = true;
  $("edit-send-btn").disabled = true;
  $("confirm-slot").innerHTML = "";

  const stopBtn = $("stop-batch-btn");
  if (stopBtn) {
    stopBtn.addEventListener("click", async () => {
      stopBtn.textContent = "Stopping…";
      stopBtn.disabled = true;
      try { await post("/api/batch/stop"); } catch { /* already stopping */ }
    });
  }
  renderLogFeed();
}

function stepHtml(name, i) {
  return `<div class="step" data-step="${name}">
    <div class="step-dot"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M5 13l4 4L19 7"/></svg></div>
    ${prettyStep(name)}</div>`;
}

const STEP_LABELS = {
  mode_select: "Select input mode",
  style_resolve: "Resolve style preset",
  checklist_fill: "Verify checklist",
  build_prompt: "Build prompt",
  image_generate: "Generate concept image",
  determine_3d_input: "Determine 3D input",
  model_generate: "Generate 3D mesh",
  postprocess: "Post-process mesh",
  import_to_blender: "Import to scene",
};
const prettyStep = (name) => STEP_LABELS[name] || name;

function ensureStep(name) {
  let el = document.querySelector(`.step[data-step="${name}"]`);
  if (!el && $("forge-steps")) {
    $("forge-steps").insertAdjacentHTML("beforeend", stepHtml(name));
    el = document.querySelector(`.step[data-step="${name}"]`);
  }
  return el;
}

function renderLogFeed() {
  const feed = $("log-feed");
  if (!feed) return;
  feed.innerHTML = state.logLines.slice(-40)
    .map((l) => `<div class="l-${l.level}">${escapeHtml(l.message)}</div>`).join("");
  feed.scrollTop = feed.scrollHeight;
}

async function showResult(sessionId) {
  state.sessionId = sessionId;
  const session = await api(`/api/session/${sessionId}`);
  state.session = session;
  state.editHistory = session.edit_history || [];

  const style = state.styles.find((s) => s.id === session.style_id);
  $("viewport-title").innerHTML =
    `Scene <em>${escapeHtml(style?.label || session.style_id)}</em><span class="sub">${escapeHtml(session.display_label || currentLabelFor(session))}</span>`;
  $("viewport-actions").style.display = "flex";
  $("gizmo").style.display = "block";
  $("edit-input").disabled = false;
  $("edit-send-btn").disabled = false;
  $("edit-input").placeholder = "make it bigger / paint it rust red / rig this character / give me a different design";

  const inScene = (session.final_mesh_path || "").startsWith("in_scene:");
  $("proc-badge").style.display = session.procedural ? "block" : "none";

  if (inScene) {
    disposeViewer();
    $("viewport-stage").innerHTML = `
      <div class="vp-empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/></svg>
        <div class="t1">Built directly in the Blender scene</div>
        <div class="t2">Procedural mode constructs geometry in-scene rather than producing a mesh file, so there's nothing to preview here. Open your .blend to see it.</div>
      </div>`;
  } else {
    await loadMeshIntoViewport(sessionId);
  }

  const stats = session.stats;
  if (stats) {
    $("mesh-stats").style.display = "block";
    $("mesh-stats").innerHTML =
      `<b>${stats.tris.toLocaleString()}</b> tris · <b>${stats.verts.toLocaleString()}</b> verts · style: <b>${escapeHtml(style?.label || "")}</b>` +
      (stats.unremeshed ? ` · <b>unremeshed</b>` : "");
  }

  if (session.generated_image_path) {
    $("viewport").insertAdjacentHTML("beforeend",
      `<div class="concept-thumb" id="concept-thumb"><img src="/api/session/${sessionId}/concept" alt="concept image"><div class="cap">step 5</div></div>`);
  }

  renderEditLog();
}

function currentLabelFor(session) {
  return session.checklist?.object_type || session.label || session.session_id;
}

async function loadSession(sessionId) {
  try {
    await showResult(sessionId);
    setRailView("projects");
    toast("Loaded", "Session opened in the viewport", "ok");
  } catch (err) {
    toast("Couldn't load that session", err.message, "error");
  }
}

$("btn-clear").addEventListener("click", () => startNewProject({ keepInputs: true }));

/* Start over. `keepInputs` distinguishes the viewport's clear button (drop the
 * result, keep what I typed) from New project (a genuinely blank slate).
 *
 * Detaching state.sessionId is the part that matters: while it is set, every edit,
 * rig, texture and undo targets that old session. Pressing Generate did make a new
 * one, but the previous object stayed on screen and the edit box still pointed at
 * the finished asset, so "new" and "continue editing the old one" were impossible
 * to tell apart. */
function startNewProject({ keepInputs = false } = {}) {
  state.sessionId = null;
  state.session = null;
  state.editHistory = [];
  state.logLines = [];

  if (!keepInputs) {
    state.checklist = {};
    state.uploadPath = null;
    state.uploadName = null;
    resetAssist();
    renderFields();
  }

  setViewportEmpty();
  renderEditLog();
  updatePrimaryButton();
  $("edit-input").value = "";
  $("confirm-slot").innerHTML = "";
  document.querySelectorAll(".session-item.current").forEach((el) => el.classList.remove("current"));
}

$("new-project-btn").addEventListener("click", () => {
  startNewProject();
  toast("New project", "Setup cleared — describe the next object", "ok");
});

/* ---- three.js GLB preview ---- */
async function loadMeshIntoViewport(sessionId) {
  disposeViewer();
  const stage = $("viewport-stage");
  stage.innerHTML = `<div class="vp-canvas-wrap"><canvas id="gl-canvas"></canvas></div>`;

  try {
    const THREE = await import("three");
    const { GLTFLoader } = await import("./vendor/GLTFLoader.js");
    const { OrbitControls } = await import("./vendor/OrbitControls.js");

    const canvas = $("gl-canvas");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.position.set(2.2, 1.6, 2.2);

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffddd5, 1.5);
    key.position.set(3, 5, 2);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xdd2f29, 0.8);
    rim.position.set(-3, 1, -2);
    scene.add(rim);

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.2;

    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(`/api/session/${sessionId}/mesh`);
    const model = gltf.scene;

    // Frame the model regardless of the scale it was generated at
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3()).length();
    const center = box.getCenter(new THREE.Vector3());
    model.position.sub(center);
    const scale = 1.8 / (size || 1);
    model.scale.setScalar(scale);
    scene.add(model);

    let running = true;
    function resize() {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (w && h && (canvas.width !== w || canvas.height !== h)) {
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
    }
    function tick() {
      if (!running) return;
      resize();
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    }
    tick();

    viewer = { dispose() { running = false; controls.dispose(); renderer.dispose(); } };
  } catch (err) {
    // three.js is an optional enhancement — the pipeline result is still valid
    // without it, so say so plainly rather than showing a broken canvas.
    stage.innerHTML = `
      <div class="vp-empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/></svg>
        <div class="t1">Mesh generated</div>
        <div class="t2">The 3D preview needs the bundled three.js files (run the setup script to fetch them). The mesh itself is on disk and in your Blender scene.</div>
      </div>`;
  }
}

function disposeViewer() {
  if (viewer) { viewer.dispose(); viewer = null; }
  $("concept-thumb")?.remove();
}

/* ================================================================
   EDIT LOOP
   ================================================================ */
$("edit-send-btn").addEventListener("click", submitEdit);
$("edit-input").addEventListener("keydown", (e) => { if (e.key === "Enter") submitEdit(); });

async function submitEdit(confirmed = false) {
  const input = $("edit-input");
  const instruction = confirmed ? state.pendingConfirm : input.value.trim();
  if (!instruction || !state.sessionId) return;
  if (!confirmed) input.value = "";

  $("confirm-slot").innerHTML = "";
  state.busy = true;
  const expensive = ["shape", "new_model", "rig"];

  try {
    const res = await post("/api/edit", {
      session_id: state.sessionId, instruction, confirmed,
    });

    if (res.needs_confirmation) {
      state.pendingConfirm = instruction;
      $("confirm-slot").innerHTML = `
        <div class="confirm-bar">
          <span class="msg">${escapeHtml(res.message)}</span>
          <button class="btn btn-sm btn-primary" id="confirm-yes">Go ahead</button>
          <button class="btn btn-sm btn-ghost" id="confirm-no">Cancel</button>
        </div>`;
      $("confirm-yes").addEventListener("click", () => submitEdit(true));
      $("confirm-no").addEventListener("click", () => { $("confirm-slot").innerHTML = ""; state.pendingConfirm = null; });
      state.busy = false;
      return;
    }

    if (res.capped) { toast("Regeneration cap reached", res.message, "warn"); state.busy = false; return; }

    toast(titleForCategory(res.category), res.message, "ok");
    await showResult(res.session_id || state.sessionId);
  } catch (err) {
    toast("Edit failed", err.message, "error");
  } finally {
    state.busy = false;
    state.pendingConfirm = null;
  }
}

function titleForCategory(category) {
  return {
    property: "Property edit applied",
    shape: "Mesh regenerated",
    new_model: "New asset generated",
    rig: "Rigging complete",
  }[category] || "Edit applied";
}

$("btn-undo").addEventListener("click", async () => {
  if (!state.sessionId) return;
  try {
    const res = await post("/api/undo", { session_id: state.sessionId, instruction: "" });
    toast(res.ok ? "Edit undone" : "Can't undo", res.message, res.ok ? "ok" : "warn");
    if (res.ok) await showResult(state.sessionId);
  } catch (err) { toast("Undo failed", err.message, "error"); }
});

$("btn-rig").addEventListener("click", async () => {
  if (!state.sessionId) return;
  setViewportGenerating("Rigging asset", ["Eject 3D backend", "Load UniRig", "Predict skeleton + weights", "Restore previous backend"]);
  try {
    const res = await post("/api/rig", { session_id: state.sessionId, instruction: "[Rig this]" });
    toast("Rigging complete", res.message, "ok");
  } catch (err) {
    toast("Rigging failed", err.message, "error");
  }
  await showResult(state.sessionId);
});

$("btn-texture").addEventListener("click", async () => {
  if (!state.sessionId) return;
  if (!state.texture.available) {
    toast("Texturing unavailable", state.texture.reason, "warn");
    return;
  }
  setViewportGenerating("Painting textures", [
    "Eject 3D backend",
    "Load paint model",
    "Render multi-view",
    "Bake PBR maps",
    "Restore previous backend",
  ]);
  try {
    const res = await post("/api/texture", { session_id: state.sessionId, instruction: "[Texture this]" });
    toast("Textures applied", "PBR maps baked onto the mesh", "ok");
  } catch (err) {
    toast("Texturing failed", err.message, "error");
  }
  await showResult(state.sessionId);
});

$("btn-remesh").addEventListener("click", async () => {
  if (!state.sessionId) return;
  if (!state.remesh.available) {
    toast("Remesh unavailable", state.remesh.reason, "warn");
    return;
  }
  setViewportGenerating("Quad remeshing", [
    "Export mesh",
    "Run AutoRemesher",
    "Decimate to style target",
  ]);
  try {
    const res = await post("/api/remesh", { session_id: state.sessionId, instruction: "[Remesh this]" });
    toast("Remeshed", `${res.faces_before.toLocaleString()} → ${res.faces_after.toLocaleString()} faces`, "ok");
  } catch (err) {
    toast("Remesh failed", err.message, "error");
  }
  await showResult(state.sessionId);
});

async function refreshTextureAvailability() {
  const [tex, rem] = await Promise.all([
    api("/api/texture/availability").catch(() => ({ available: false, reason: "Couldn't reach the server to check." })),
    api("/api/remesh/availability").catch(() => ({ available: false, reason: "Couldn't reach the server to check." })),
  ]);
  state.texture = tex;
  state.remesh = rem;
  updateGatedControls();
}

function renderEditLog() {
  const el = $("edit-log");
  if (!state.editHistory.length) {
    el.innerHTML = `<div class="activity-empty">Nothing yet — edits to the current object will appear here as you make them.</div>`;
    return;
  }
  el.innerHTML = state.editHistory.slice().reverse().map((e) => `
    <div class="edit-entry">
      <span class="edit-badge ${e.category}">${e.category.replace("_", " ")}</span>
      <span class="txt"><b>${escapeHtml(e.instruction)}</b></span>
    </div>`).join("");
}

/* ================================================================
   EXPORT
   ================================================================ */
$("btn-export").addEventListener("click", () => $("dlg-export").classList.add("show"));

document.querySelectorAll("#export-format .radio-opt, #export-target .radio-opt").forEach((el) => {
  el.addEventListener("click", () => {
    el.parentElement.querySelectorAll(".radio-opt").forEach((o) => o.classList.remove("sel"));
    el.classList.add("sel");
    if (el.parentElement.id === "export-target") {
      $("export-target-note").textContent = el.dataset.val === "unreal"
        ? "X forward / Z up, unit scale applied so it lands in Unreal at the right size."
        : "Blender-native conventions, -Z forward / Y up.";
    }
  });
});

$("export-confirm").addEventListener("click", async () => {
  const format = document.querySelector("#export-format .sel").dataset.val;
  const target = document.querySelector("#export-target .sel").dataset.val;
  $("dlg-export").classList.remove("show");
  try {
    const res = await post("/api/export", { session_id: state.sessionId, format, target });
    const filename = res.path.split(/[\\/]/).pop();
    toast("Export complete", `${filename} · ${(res.bytes / 1024).toFixed(0)} KB`, "ok");
    window.location.href = `/api/session/${state.sessionId}/download/${filename}`;
  } catch (err) {
    toast("Export failed", err.message, "error");
  }
});

/* ================================================================
   SESSIONS
   ================================================================ */
async function refreshSessions() {
  try {
    const [resumable, all] = await Promise.all([
      api("/api/sessions?resumable=true"),
      api("/api/sessions?resumable=false"),
    ]);
    renderSessions(resumable.sessions, all.sessions);
  } catch { /* offline */ }
}

function renderSessions(resumable, all) {
  const el = $("sessions-list");
  let html = "";

  if (state.queue.resumable) {
    const pending = state.queue.counts.pending || 0;
    html += `<div class="session-item" data-resume-batch="1">
      <div class="meta"><b>Batch queue</b>${pending} item${pending === 1 ? "" : "s"} still pending</div>
      <div class="when"><span>stopped mid-run</span><span class="load-link">Resume →</span></div>
    </div>`;
  }

  if (resumable.length) {
    html += `<div class="col-label" style="margin:14px 0 10px">Incomplete</div>`;
    html += resumable.map((s) => sessionItemHtml(s, true)).join("");
  }

  const finished = all.filter((s) => s.status === "done");
  if (finished.length) {
    html += `<div class="col-label" style="margin:14px 0 10px">Finished</div>`;
    html += finished.map((s) => sessionItemHtml(s, false)).join("");
  }

  el.innerHTML = html || `<div class="activity-empty">No sessions yet. Generate something and it'll show up here.</div>`;

  el.querySelectorAll("[data-session]").forEach((node) => {
    node.addEventListener("click", async () => {
      const id = node.dataset.session;
      if (node.dataset.resumable === "1") {
        try {
          state.logLines = [];
          setViewportGenerating("Resuming session", []);
          await post(`/api/session/${id}/resume`);
        } catch (err) { toast("Couldn't resume", err.message, "error"); setViewportEmpty(); }
      } else {
        await loadSession(id);
      }
    });
  });

  el.querySelector("[data-resume-batch]")?.addEventListener("click", async () => {
    setBatchMode(true);
    try { await post("/api/batch/start"); }
    catch (err) { toast("Couldn't resume the batch", err.message, "error"); }
  });
}

function sessionItemHtml(s, resumable) {
  const style = state.styles.find((x) => x.id === s.style_id);
  const when = relativeTime(s.last_modified);
  return `<div class="session-item" data-session="${s.session_id}" data-resumable="${resumable ? 1 : 0}">
    <div class="meta"><b>${state.modes[s.mode]?.label || s.mode}</b>${escapeHtml(style?.label || s.style_id)}${
      resumable ? ` · step ${s.current_step} of ${s.total_steps || 10}` : ""}</div>
    <div class="when"><span>${when}</span><span class="load-link">${resumable ? "Resume" : "Load"} →</span></div>
  </div>`;
}

function relativeTime(ts) {
  const secs = Date.now() / 1000 - ts;
  if (secs < 60) return "just now";
  if (secs < 3600) return `${Math.floor(secs / 60)} min ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)} hours ago`;
  return `${Math.floor(secs / 86400)} days ago`;
}

/* ================================================================
   SETTINGS
   ================================================================ */
$("btn-settings").addEventListener("click", () => $("settings-panel").classList.add("show"));
$("settings-close").addEventListener("click", () => $("settings-panel").classList.remove("show"));
$("set-vision").addEventListener("click", () => $("set-vision").classList.toggle("on"));

function populateSettings(settingsRes, health) {
  const s = settingsRes.settings;
  $("set-backend").innerHTML = settingsRes.model_backends
    .map((b) => `<option value="${b.id}"${b.id === s.model_backend ? " selected" : ""}>${b.label}</option>`).join("");
  $("set-image").value = s.image_backend;
  $("set-llm").value = s.llm_provider;
  $("set-model").value = s.llm_model;
  $("set-endpoint").value = s.llm_endpoint;
  $("set-blender").value = s.blender_path || "";
  $("set-vision").classList.toggle("on", s.llm_vision_capable);
  syncProviderFields();

  const gpu = health.gpu;
  $("gpu-info").innerHTML = gpu.cuda_available
    ? `${escapeHtml(gpu.device_name || "CUDA device")}<br>${gpu.total_vram_gb} GB VRAM`
    : `No CUDA GPU detected<br>mock backends available`;
}

$("set-llm").addEventListener("change", syncProviderFields);

function syncProviderFields() {
  const provider = $("set-llm").value;
  const isLocal = provider === "lm_studio";
  $("vision-field").style.display = isLocal ? "block" : "none";
  $("apikey-note").style.display = isLocal ? "none" : "block";
  $("set-endpoint").parentElement.style.display = isLocal ? "block" : "none";
}

$("settings-save").addEventListener("click", async () => {
  const status = $("settings-status");
  status.textContent = "Applying — ejecting the current backend first…";
  try {
    const res = await post("/api/settings", {
      model_backend: $("set-backend").value,
      image_backend: $("set-image").value,
      llm_provider: $("set-llm").value,
      llm_model: $("set-model").value,
      llm_endpoint: $("set-endpoint").value,
      llm_vision_capable: $("set-vision").classList.contains("on"),
      blender_path: $("set-blender").value,
    });
    status.textContent = res.changed.length ? `Applied: ${res.changed.join(", ")}` : "Nothing changed.";
    const health = await api("/api/health");
    applyHealth(health);
    toast("Settings applied", res.changed.length ? "Previous backend ejected from VRAM" : "No changes", "ok");
  } catch (err) {
    status.textContent = `Failed: ${err.message}`;
  }
});

$("settings-test").addEventListener("click", async () => {
  const status = $("settings-status");
  status.textContent = "Testing the reasoning LLM…";
  try {
    const res = await api("/api/llm/health");
    status.textContent = res.ok
      ? `Reachable — ${res.provider} (${res.model}) replied "${res.reply}"`
      : `Not reachable: ${res.error}`;
  } catch (err) {
    status.textContent = `Test failed: ${err.message}`;
  }
});

/* ================================================================
   DIALOG CLOSERS
   ================================================================ */
document.querySelectorAll("[data-close]").forEach((btn) =>
  btn.addEventListener("click", () => $(btn.dataset.close).classList.remove("show")));
document.querySelectorAll(".dbackdrop").forEach((bd) =>
  bd.addEventListener("click", (e) => { if (e.target === bd) bd.classList.remove("show"); }));
$("settings-panel").addEventListener("click", (e) => {
  if (e.target === $("settings-panel")) $("settings-panel").classList.remove("show");
});

/* ================================================================
   SSE — live pipeline progress
   ================================================================ */
function connectEvents() {
  const source = new EventSource("/api/events");

  source.onmessage = async (msg) => {
    let evt;
    try { evt = JSON.parse(msg.data); } catch { return; }
    handleEvent(evt);
  };

  source.onerror = () => {
    $("status-pill").classList.add("offline");
    $("status-backend").textContent = "reconnecting…";
  };
  source.onopen = () => {
    $("status-pill").classList.remove("offline");
  };
}

async function handleEvent(evt) {
  // The 10s poll is the floor, not the ceiling: when an event does arrive, the
  // panel should reflect it immediately rather than waiting out the interval.
  if (statusTabIsOpen()
      && ["step_start", "step_done", "session_done", "session_failed", "backend_ready", "backend_ejected"].includes(evt.type)) {
    refreshStatus();
  }

  switch (evt.type) {
    case "step_start": {
      const el = ensureStep(evt.step);
      el?.classList.add("active");
      break;
    }
    case "step_done": {
      const el = ensureStep(evt.step);
      el?.classList.remove("active");
      el?.classList.add("done");
      break;
    }
    case "log": {
      state.logLines.push({ message: evt.message, level: evt.level || "info" });
      renderLogFeed();
      break;
    }
    case "backend_loading":
      $("status-pill").classList.add("busy");
      $("status-backend").textContent = `loading ${evt.key}`;
      break;
    case "backend_ready":
      $("status-pill").classList.remove("busy");
      $("status-backend").textContent = evt.key;
      break;
    case "backend_ejected":
      state.logLines.push({ message: `ejected ${evt.key} from VRAM`, level: "info" });
      renderLogFeed();
      break;
    case "session_paused": {
      state.sessionId = evt.session_id;
      setViewportReview(evt);
      break;
    }
    case "session_done": {
      state.busy = false;
      $("col-setup").classList.remove("locked");
      updatePrimaryButton();
      if (!state.queue.running) {
        await showResult(evt.session_id);
        refreshSessions();
      }
      break;
    }
    case "session_failed": {
      state.busy = false;
      $("col-setup").classList.remove("locked");
      updatePrimaryButton();
      if (!state.queue.running) {
        setViewportEmpty();
        toast("Generation failed", evt.error, "error");
      }
      break;
    }
    case "queue_changed":
      state.queue = { ...state.queue, ...evt };
      renderQueue();
      updatePrimaryButton();
      break;
    case "batch_item_start":
      setViewportGenerating(`Item ${evt.index} of ${evt.total} — ${evt.label}`, []);
      break;
    case "batch_item_done":
      if (evt.status === "failed") toast("Queue item failed", evt.error || "", "error");
      break;
    case "batch_finished": {
      $("col-setup").classList.remove("locked");
      const done = evt.counts?.done || 0;
      const failed = evt.counts?.failed || 0;
      const pending = evt.counts?.pending || 0;
      state.queue = { ...state.queue, ...evt };
      renderQueue();
      updatePrimaryButton();
      refreshSessions();
      if (pending > 0) {
        toast("Batch stopped", `${done} done · ${failed} failed · ${pending} still pending`, "warn");
      } else {
        toast("Batch complete", `${done} object${done === 1 ? "" : "s"} generated${failed ? `, ${failed} failed` : ""}`, "ok");
      }
      const lastDone = (evt.items || []).filter((i) => i.status === "done").pop();
      if (lastDone?.session_id) await showResult(lastDone.session_id);
      else setViewportEmpty();
      break;
    }
    case "error":
      toast("Something went wrong", evt.message, "error");
      break;
  }
}

/* ================================================================
   TOASTS + utils
   ================================================================ */
function toast(title, sub, kind = "") {
  const wrap = $("toast-wrap");
  const el = document.createElement("div");
  el.className = `toast${kind ? ` t-${kind}` : ""}`;
  el.innerHTML = `<b>${escapeHtml(title)}</b>${sub ? `<span>${escapeHtml(sub)}</span>` : ""}`;
  wrap.appendChild(el);
  setTimeout(() => {
    el.classList.add("out");
    setTimeout(() => el.remove(), 220);
  }, kind === "error" ? 7000 : 4000);
}

function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

boot();
