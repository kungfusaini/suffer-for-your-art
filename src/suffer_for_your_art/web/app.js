"use strict";

const R = {
  runList: document.getElementById("run-list"),
  refresh: document.getElementById("refresh-btn"),
  overlay: document.getElementById("canvas-overlay"),
  status: document.getElementById("status-text"),
  frame: document.getElementById("frame-img"),
  title: document.getElementById("run-title"),
  prompt: document.getElementById("run-prompt"),
  quote: document.getElementById("artist-quote"),
  rationaleText: document.getElementById("artist-rationale-text"),
  contextToggle: document.getElementById("context-toggle"),
  contextViewer: document.getElementById("context-viewer"),
  contextClose: document.getElementById("context-close"),
  contextProvenance: document.getElementById("context-provenance"),
  contextPrev: document.getElementById("context-prev"),
  contextNext: document.getElementById("context-next"),
  contextRound: document.getElementById("context-round"),
  contextRoundCount: document.getElementById("context-round-count"),
  contextStatus: document.getElementById("context-status"),
  contextContent: document.getElementById("context-content"),
  contextCanvas: document.getElementById("context-canvas"),
  contextMovesTab: document.getElementById("context-tab-moves"),
  contextRecentMap: document.getElementById("context-recent-map"),
  contextMoveList: document.getElementById("context-move-list"),
  contextStrokesBefore: document.getElementById("context-strokes-before"),
  contextBatchSize: document.getElementById("context-batch-size"),
  contextPromptVersion: document.getElementById("context-prompt-version"),
  contextCanvasHash: document.getElementById("context-canvas-hash"),
  contextRequestHash: document.getElementById("context-request-hash"),
  contextSystemSection: document.getElementById("context-system-section"),
  contextSystem: document.getElementById("context-system"),
  contextUser: document.getElementById("context-user"),
  contextCanvasMessage: document.getElementById("context-canvas-message"),
  contextRecentMessageSection: document.getElementById("context-recent-message-section"),
  contextRecentMessage: document.getElementById("context-recent-message"),
  contextSchema: document.getElementById("context-schema"),
  contextTabs: Array.from(document.querySelectorAll("[data-context-tab]")),
  contextPanels: Array.from(document.querySelectorAll("[data-context-panel]")),
  contextRawLoad: document.getElementById("context-raw-load"),
  contextRaw: document.getElementById("context-raw"),
  slider: document.getElementById("frame-slider"),
  count: document.getElementById("frame-counter"),
  restart: document.getElementById("restart-btn"),
  prev: document.getElementById("prev-btn"),
  play: document.getElementById("play-btn"),
  next: document.getElementById("next-btn"),
  speeds: Array.from(document.querySelectorAll(".speed-btn")),
};
const M = {
  runs: [],
  selected: "",
  total: 0,
  at: 0,
  loading: false,
  playing: false,
  speed: 1,
  token: 0,
  frameUrl: "",
  requestedAt: 0,
  seekPending: null,
  seekPromise: null,
  scrubbing: false,
  detail: null,
  contextOpen: false,
  contextLoading: false,
  contextRound: 1,
  contextTotal: 0,
  contextToken: 0,
  contextTab: "overview",
  rawLoadedRound: 0,
};

function message(text, kind = "info") {
  R.status.textContent = text;
  if (text) {
    R.overlay.hidden = false;
    R.overlay.classList.toggle("error", kind === "error");
  } else {
    R.overlay.hidden = true;
    R.overlay.classList.remove("error");
  }
}

function sliderFromState() {
  R.slider.value = String(M.at);
  R.slider.max = String(M.total);
  R.count.textContent = `${M.at} / ${M.total}`;
}

function contextControls() {
  const available = Boolean(M.selected && M.detail?.mode === "paint" && M.detail?.model);
  R.contextToggle.disabled = M.loading || !available;
  const blocked = M.contextLoading || !M.contextOpen || M.contextTotal < 1;
  R.contextRound.disabled = blocked;
  R.contextPrev.disabled = blocked || M.contextRound <= 1;
  R.contextNext.disabled = blocked || M.contextRound >= M.contextTotal;
}

function controls() {
  const hasRun = Boolean(M.selected);
  R.refresh.disabled = M.loading;
  R.slider.disabled = !hasRun || M.total === 0;
  R.restart.disabled = M.loading || !hasRun;
  R.prev.disabled = M.loading || !hasRun || M.at <= 0;
  R.next.disabled = M.loading || !hasRun || M.at >= M.total;
  R.play.disabled = !hasRun || M.total === 0 || (M.loading && !M.playing);
  R.speeds.forEach((b) => (b.disabled = M.loading || !hasRun));
  contextControls();
}

function setPlayLabel() {
  R.play.textContent = M.playing ? "Pause" : "Play";
  R.play.setAttribute("aria-pressed", String(M.playing));
}

function setActiveRun(id) {
  for (const item of R.runList.querySelectorAll(".run-item")) {
    item.classList.toggle("is-active", item.dataset.runId === id);
  }
}

function setMeta(run, detail) {
  const hasWork = Boolean(detail?.prompt);
  const title = hasWork ? (detail?.title || run.title || "Untitled work") : "No work selected";
  R.title.textContent = title;
  R.prompt.textContent = typeof detail?.prompt === "string" ? detail.prompt.trim() : "";
  R.rationaleText.textContent = typeof detail?.initial_vision === "string"
    ? detail.initial_vision.trim()
    : "";
  R.quote.textContent = typeof detail?.final_description === "string"
    ? detail.final_description.trim()
    : "";
}

function contextMessage(text, kind = "info") {
  R.contextStatus.textContent = text;
  R.contextStatus.classList.toggle("error", kind === "error");
}

function setContextTab(name, focus = false) {
  const available = R.contextTabs.filter((tab) => !tab.hidden && !tab.disabled);
  const next = available.some((tab) => tab.dataset.contextTab === name) ? name : "overview";
  M.contextTab = next;
  R.contextTabs.forEach((tab) => {
    const active = !tab.hidden && tab.dataset.contextTab === next;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focus) tab.focus();
  });
  R.contextPanels.forEach((panel) => {
    panel.hidden = panel.dataset.contextPanel !== next;
  });
}

function resetContext() {
  M.contextOpen = false;
  M.contextLoading = false;
  M.contextRound = 1;
  M.contextTotal = 0;
  M.contextToken += 1;
  M.rawLoadedRound = 0;
  R.contextToggle.textContent = "Inspect provenance";
  R.contextToggle.setAttribute("aria-expanded", "false");
  R.contextViewer.hidden = true;
  R.contextContent.hidden = true;
  R.contextRound.min = "1";
  R.contextRound.max = "1";
  R.contextRound.value = "1";
  R.contextRoundCount.textContent = "1 / 1";
  R.contextCanvas.removeAttribute("src");
  R.contextMovesTab.hidden = true;
  R.contextRecentMap.removeAttribute("src");
  R.contextMoveList.textContent = "";
  R.contextRecentMessageSection.hidden = true;
  R.contextRecentMessage.textContent = "";
  R.contextRaw.textContent = "Select “Load request” to fetch the exact body.";
  R.contextRawLoad.textContent = "Load request";
  R.contextRawLoad.disabled = false;
  R.contextProvenance.textContent = "";
  setContextTab("overview");
  contextMessage("");
}

function setContextOpen(open) {
  M.contextOpen = Boolean(open);
  R.contextViewer.hidden = !M.contextOpen;
  R.contextToggle.textContent = M.contextOpen ? "Hide provenance" : "Inspect provenance";
  R.contextToggle.setAttribute("aria-expanded", String(M.contextOpen));
  if (!M.contextOpen) {
    M.contextToken += 1;
    M.contextLoading = false;
    R.contextToggle.focus();
  } else {
    R.contextViewer.scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
    R.contextClose.focus();
  }
  contextControls();
}

function messageText(value) {
  if (typeof value === "string") return value;
  if (!Array.isArray(value)) return "";
  return value
    .filter((part) => part && part.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("\n");
}

function renderContext(data) {
  M.contextRound = Number(data.round);
  M.contextTotal = Number(data.event_count);
  R.contextRound.min = "1";
  R.contextRound.max = String(M.contextTotal);
  R.contextRound.value = String(M.contextRound);
  R.contextRoundCount.textContent = `${M.contextRound} / ${M.contextTotal}`;

  const caveat = data.reconstruction?.caveat || "Reconstructed from persisted run provenance.";
  R.contextProvenance.textContent = caveat;
  R.contextCanvas.src = data.canvas_url;
  R.contextCanvas.alt = `Canvas shown to the model before round ${M.contextRound}`;
  R.contextStrokesBefore.textContent = String(data.strokes_before);
  R.contextBatchSize.textContent = String(data.required_batch_size);
  R.contextPromptVersion.textContent = data.prompt_template_version || "—";
  R.contextCanvasHash.textContent = data.canvas_sha256 || "—";
  R.contextRequestHash.textContent = data.raw_request_sha256 || "—";

  const messages = data.messages || {};
  const messageItems = Array.isArray(messages) ? messages : [];
  const labeled = (label) => messageItems.find((item) => item?.label === label)?.content;
  const systemText = messageText(labeled("system") ?? messages.system);
  R.contextSystem.textContent = systemText;
  R.contextSystemSection.hidden = !systemText;
  R.contextUser.textContent = messageText(
    labeled("context_action_contract") ?? messages.user_context ?? messages.canvas_user,
  );
  R.contextCanvasMessage.textContent = messageText(
    labeled("multimodal_canvas") ?? messages.canvas_message ?? messages.snapshot_user,
  );
  const recentMessage = messageText(labeled("recent_stroke_map"));
  const recentMoves = Array.isArray(data.recent_moves) ? data.recent_moves : [];
  const hasRecentFeedback = Boolean(data.recent_stroke_map_url && recentMoves.length);
  R.contextMovesTab.hidden = !hasRecentFeedback;
  R.contextRecentMessageSection.hidden = !recentMessage;
  R.contextRecentMessage.textContent = recentMessage;
  R.contextMoveList.textContent = hasRecentFeedback
    ? JSON.stringify(recentMoves, null, 2)
    : "";
  if (hasRecentFeedback) {
    R.contextRecentMap.src = data.recent_stroke_map_url;
    R.contextRecentMap.alt = `Numbered map of the strokes rendered immediately before round ${M.contextRound}`;
  } else {
    R.contextRecentMap.removeAttribute("src");
    if (M.contextTab === "moves") M.contextTab = "overview";
  }
  R.contextSchema.textContent = JSON.stringify(data.response_schema || {}, null, 2);
  R.contextRaw.textContent = "Select “Load request” to fetch the exact body.";
  R.contextRawLoad.textContent = "Load request";
  R.contextRawLoad.disabled = false;
  M.rawLoadedRound = 0;
  setContextTab(M.contextTab);
  R.contextContent.hidden = false;
}

async function loadContextRound(nextRound) {
  if (!M.selected || !M.contextOpen) return;
  const total = Math.max(1, M.contextTotal || Number(M.detail?.rounds_executed) || 1);
  const round = Math.max(1, Math.min(total, Number(nextRound)));
  if (!Number.isFinite(round)) return;

  const token = ++M.contextToken;
  const selected = M.selected;
  M.contextLoading = true;
  R.contextContent.hidden = true;
  contextMessage(`Loading model context for round ${round}…`);
  contextControls();

  try {
    const data = await apiGet(`/api/runs/${encodeURIComponent(selected)}/context?round=${round}`);
    if (token !== M.contextToken || selected !== M.selected) return;
    renderContext(data);
    contextMessage("");
  } catch (err) {
    if (token !== M.contextToken) return;
    contextMessage(`Context error: ${err.message}`, "error");
    R.contextContent.hidden = true;
  } finally {
    if (token === M.contextToken) {
      M.contextLoading = false;
      contextControls();
    }
  }
}

async function loadRawContext() {
  if (M.rawLoadedRound === M.contextRound || !M.selected || M.contextLoading) return;
  const selected = M.selected;
  const round = M.contextRound;
  const token = M.contextToken;
  R.contextRawLoad.disabled = true;
  R.contextRawLoad.textContent = "Loading…";
  R.contextRaw.textContent = "Loading exact request…";
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(selected)}/context?round=${round}&raw=1`);
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    const raw = await res.text();
    if (token !== M.contextToken || selected !== M.selected || round !== M.contextRound) return;
    R.contextRaw.textContent = raw;
    const hash = res.headers.get("X-Request-Sha256");
    if (hash) R.contextRequestHash.textContent = hash;
    M.rawLoadedRound = round;
    R.contextRawLoad.textContent = "Loaded";
  } catch (err) {
    if (token !== M.contextToken) return;
    R.contextRaw.textContent = `Unable to load raw request: ${err.message}`;
    R.contextRawLoad.textContent = "Retry";
  } finally {
    if (token === M.contextToken && M.rawLoadedRound !== round) {
      R.contextRawLoad.disabled = false;
    }
  }
}

function makeRunEl(run) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "run-item";
  btn.dataset.runId = run.run_id;

  const thumb = document.createElement("img");
  thumb.className = "run-thumb";
  thumb.loading = "lazy";
  thumb.src = run.thumbnail;
  thumb.alt = `${run.title} thumbnail`;

  const body = document.createElement("div");
  const displayTitle = run.title || run.prompt || "Untitled work";
  const title = document.createElement("h3");
  title.textContent = displayTitle;
  const prompt = document.createElement("p");
  prompt.textContent = run.prompt && run.prompt.trim().toLocaleLowerCase() !== displayTitle.trim().toLocaleLowerCase()
    ? run.prompt
    : "";
  const meta = document.createElement("p");
  const status = {
    done: "Complete",
    failed: "Failed",
    stopped: "Stopped",
    incomplete: "Incomplete",
  }[run.status];
  meta.textContent = [status, `${run.stroke_count} strokes`, run.model || "offline"].filter(Boolean).join(" · ");

  body.append(title, prompt, meta);
  btn.append(thumb, body);
  btn.addEventListener("click", () => { void selectRun(run.run_id); });
  return btn;
}

function renderRunState(text) {
  const state = document.createElement("p");
  state.className = "run-state";
  state.textContent = text;
  R.runList.replaceChildren(state);
}

function renderRuns() {
  R.runList.replaceChildren();
  if (!M.runs.length) {
    renderRunState("No paintings yet. Create a run, then use Refresh.");
    return;
  }
  for (const run of M.runs) R.runList.append(makeRunEl(run));
  setActiveRun(M.selected);
}

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) {
    let e = `Request failed: ${res.status}`;
    try { const data = await res.json(); if (data?.error) e = data.error; } catch {}
    throw new Error(e);
  }
  return res.json();
}

function setFrame(url) {
  if (M.frameUrl) URL.revokeObjectURL(M.frameUrl);
  M.frameUrl = url;
  R.frame.src = url;
}

function clearFrame() {
  if (M.frameUrl) URL.revokeObjectURL(M.frameUrl);
  M.frameUrl = "";
  R.frame.removeAttribute("src");
}

async function loadFrame(strokes) {
  const res = await fetch(`/api/runs/${encodeURIComponent(M.selected)}/frame?strokes=${strokes}`);
  if (!res.ok) {
    let e = `Frame failed: ${res.status}`;
    try { const data = await res.json(); if (data?.error) e = data.error; } catch {}
    throw new Error(e);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  await new Promise((resolve, reject) => {
    const clean = () => {
      R.frame.removeEventListener("load", onLoad);
      R.frame.removeEventListener("error", onError);
    };
    const onLoad = () => { clean(); resolve(); };
    const onError = () => { clean(); reject(new Error("Failed to decode image")); };
    R.frame.addEventListener("load", onLoad);
    R.frame.addEventListener("error", onError);
    setFrame(url);
  });
}

async function seekTo(next) {
  if (!M.selected) return;
  const n = Math.max(0, Math.min(M.total, Number(next)));
  if (!Number.isFinite(n) || n === M.at) return;

  M.loading = true;
  controls();
  try {
    await loadFrame(n);
    M.at = n;
    if (!M.scrubbing && n === M.requestedAt) sliderFromState();
    R.frame.alt = n === 0 ? "Blank canvas" : `Frame ${n}`;
    if (n === M.requestedAt) message("");
  } catch (err) {
    M.playing = false;
    setPlayLabel();
    if (n === M.requestedAt) message(`Playback error: ${err.message}`, "error");
  } finally {
    M.loading = false;
    controls();
  }
}

async function drainSeekQueue() {
  while (M.seekPending !== null) {
    const next = M.seekPending;
    M.seekPending = null;
    await seekTo(next);
  }
}

function requestSeek(next) {
  if (!M.selected) return Promise.resolve();
  const n = Math.max(0, Math.min(M.total, Number(next)));
  if (!Number.isFinite(n)) return Promise.resolve();
  M.requestedAt = n;
  M.seekPending = n;
  if (!M.seekPromise) {
    M.seekPromise = drainSeekQueue().finally(() => {
      M.seekPromise = null;
      if (M.seekPending !== null) void requestSeek(M.seekPending);
    });
  }
  return M.seekPromise;
}

function stopPlayback() {
  M.playing = false;
  M.token += 1;
  setPlayLabel();
}

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function stepDelay() {
  return (reduceMotion.matches ? 900 : 480) / M.speed;
}

async function playbackLoop(token) {
  while (M.playing && token === M.token && M.selected) {
    if (M.at >= M.total) {
      stopPlayback();
      break;
    }
    await requestSeek(M.at + 1);
    if (!M.playing || token !== M.token) break;
    if (M.at >= M.total) {
      stopPlayback();
      break;
    }
    await new Promise((r) => setTimeout(r, stepDelay()));
  }
}

function togglePlay() {
  if (M.playing) { stopPlayback(); return; }
  if (!M.selected || M.total === 0 || M.loading) return;

  const start = async () => {
    M.playing = true;
    M.token += 1;
    setPlayLabel();
    await playbackLoop(M.token);
  };

  if (M.at >= M.total) {
    void requestSeek(0).then(start);
  } else {
    start();
  }
}

function setSpeed(v) {
  M.speed = Number(v);
  R.speeds.forEach((btn) => {
    const act = Number(btn.dataset.speed) === M.speed;
    btn.classList.toggle("is-active", act);
    btn.setAttribute("aria-pressed", String(act));
  });
}

async function selectRun(id) {
  if (M.loading || id === M.selected) return;
  const run = M.runs.find((x) => x.run_id === id);
  if (!run) return;

  stopPlayback();
  clearFrame();
  resetContext();
  M.seekPending = null;
  cancelScrubPreview();
  M.detail = null;
  M.loading = true;
  controls();
  message("Loading run details…");

  try {
    const detail = await apiGet(`/api/runs/${encodeURIComponent(id)}`);
    M.selected = id;
    M.detail = detail;
    M.total = Number(detail.total_strokes || run.stroke_count || 0);
    M.contextTotal = Number(detail.rounds_executed || 0);
    M.at = M.total;
    M.requestedAt = M.at;
    sliderFromState();
    setMeta(run, detail);
    setActiveRun(id);
    await loadFrame(M.at);
    R.frame.alt = M.at === 0 ? "Blank canvas" : `Final frame after ${M.at} strokes`;
    message("");
  } catch (err) {
    message(`Failed to load run: ${err.message}`, "error");
    M.selected = "";
    M.detail = null;
    M.total = 0;
    M.at = 0;
    M.requestedAt = 0;
    sliderFromState();
    setActiveRun("");
    setMeta({ title: "No run selected" }, {});
    clearFrame();
  } finally {
    M.loading = false;
    controls();
  }
}

async function loadRuns() {
  const previousSelection = M.selected;
  let nextSelection = "";
  stopPlayback();
  message("Loading paintings…");
  M.loading = true;
  R.runList.setAttribute("aria-busy", "true");
  renderRunState("Loading works…");
  controls();
  M.runs = [];
  M.selected = "";
  M.detail = null;
  resetContext();
  M.total = 0;
  M.at = 0;
  M.requestedAt = 0;
  M.seekPending = null;
  cancelScrubPreview();
  sliderFromState();
  setMeta({ title: "No work selected" }, {});
  clearFrame();

  try {
    const payload = await apiGet("/api/runs");
    M.runs = Array.isArray(payload?.runs) ? payload.runs : [];
    nextSelection = M.runs.some((run) => run.run_id === previousSelection)
      ? previousSelection
      : (M.runs[0]?.run_id || "");
    renderRuns();
    message(M.runs.length ? "Loading selected work…" : "No paintings yet. Create a run, then use Refresh.");
  } catch (err) {
    M.runs = [];
    renderRunState("Couldn’t load the archive. Use Refresh to retry.");
    message(`Failed to load runs: ${err.message}`, "error");
  } finally {
    M.loading = false;
    R.runList.setAttribute("aria-busy", "false");
    controls();
  }

  if (nextSelection) await selectRun(nextSelection);
}

const SCRUB_INTERVAL_MS = 80;
let scrubTimer = null;
let scrubLatest = 0;

function flushScrubPreview() {
  scrubTimer = null;
  if (scrubLatest !== M.requestedAt) {
    void requestSeek(scrubLatest);
    scrubTimer = window.setTimeout(flushScrubPreview, SCRUB_INTERVAL_MS);
  }
}

function scheduleScrubPreview(next) {
  scrubLatest = next;
  if (scrubTimer !== null) return;
  void requestSeek(next);
  scrubTimer = window.setTimeout(flushScrubPreview, SCRUB_INTERVAL_MS);
}

function cancelScrubPreview() {
  if (scrubTimer !== null) window.clearTimeout(scrubTimer);
  scrubTimer = null;
  M.scrubbing = false;
}

function bind() {
  R.refresh.addEventListener("click", () => void loadRuns());
  R.slider.addEventListener("input", (event) => {
    if (!M.selected) return;
    if (M.playing) stopPlayback();
    M.scrubbing = true;
    const next = event.target.valueAsNumber;
    R.count.textContent = `${next} / ${M.total}`;
    scheduleScrubPreview(next);
  });
  R.slider.addEventListener("change", (event) => {
    if (!M.selected) return;
    if (M.playing) stopPlayback();
    M.scrubbing = false;
    if (scrubTimer !== null) window.clearTimeout(scrubTimer);
    scrubTimer = null;
    scrubLatest = event.target.valueAsNumber;
    void requestSeek(scrubLatest);
  });
  R.restart.addEventListener("click", () => { if (!M.loading && M.selected) void requestSeek(0); });
  R.prev.addEventListener("click", () => { if (!M.loading && M.selected) void requestSeek(M.at - 1); });
  R.next.addEventListener("click", () => { if (!M.loading && M.selected) void requestSeek(M.at + 1); });
  R.play.addEventListener("click", togglePlay);
  R.speeds.forEach((btn) => btn.addEventListener("click", () => setSpeed(btn.dataset.speed)));
  R.contextToggle.addEventListener("click", () => {
    if (M.contextOpen) {
      setContextOpen(false);
      return;
    }
    M.contextTotal = Number(M.detail?.rounds_executed || 0);
    setContextOpen(true);
    void loadContextRound(1);
  });
  R.contextClose.addEventListener("click", () => setContextOpen(false));
  R.contextPrev.addEventListener("click", () => void loadContextRound(M.contextRound - 1));
  R.contextNext.addEventListener("click", () => void loadContextRound(M.contextRound + 1));
  R.contextRound.addEventListener("change", (e) => void loadContextRound(e.target.valueAsNumber));
  R.contextRawLoad.addEventListener("click", () => void loadRawContext());
  R.contextTabs.forEach((tab) => {
    tab.addEventListener("click", () => setContextTab(tab.dataset.contextTab));
    tab.addEventListener("keydown", (event) => {
      const available = R.contextTabs.filter((item) => !item.hidden && !item.disabled);
      const index = available.indexOf(tab);
      const last = available.length - 1;
      let next = index;
      if (event.key === "ArrowRight") next = index === last ? 0 : index + 1;
      else if (event.key === "ArrowLeft") next = index === 0 ? last : index - 1;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = last;
      else return;
      event.preventDefault();
      setContextTab(available[next].dataset.contextTab, true);
    });
  });

  window.addEventListener("keydown", (e) => {
    const t = document.activeElement?.tagName;
    const editing = ["A", "BUTTON", "INPUT", "TEXTAREA", "SELECT"].includes(t) || document.activeElement?.isContentEditable;
    if (editing) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.code === "ArrowLeft") { e.preventDefault(); void requestSeek(M.at - 1); }
    if (e.code === "ArrowRight") { e.preventDefault(); void requestSeek(M.at + 1); }
  });
}

(function init() {
  resetContext();
  setSpeed(1);
  setPlayLabel();
  setActiveRun("");
  controls();
  sliderFromState();
  bind();
  void loadRuns();
})();
