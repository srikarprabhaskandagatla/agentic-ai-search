const STAGES = [
  ["planning", "Plan"],
  ["searching", "Search"],
  ["scraping", "Scrape"],
  ["extracting", "Extract"],
  ["resolving", "Resolve"],
  ["analyzing", "Gaps"],
  ["filling", "Fill"],
  ["ranking", "Rank"]
];
const FALLBACK_EXAMPLES = [
  "AI startups in healthcare",
  "top pizza places in Brooklyn",
  "open source database tools",
  "autonomous vehicle companies 2026",
  "no-code app building tools",
  "large language model providers"
];
const DEFAULT_API = "http://localhost:8000";
const TRUNCATE = 72;

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const label = col => col.replace(/_/g, " ");

const state = { result: null, controller: null, depth: 2 };

const SEARCH_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" stroke-width="2.2"/><path d="M16 16l4.5 4.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>';
const STOP_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/></svg>';

const SUN = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
const MOON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>';

function readStore(key) {
  try { return localStorage.getItem(key); } catch (e) { return null; }
}

function writeStore(key, value) {
  try { localStorage.setItem(key, value); } catch (e) {}
}

function currentTheme() {
  const set = document.documentElement.dataset.theme;
  if (set) return set;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  writeStore("ais-theme", theme);
  const btn = $("theme-toggle");
  btn.innerHTML = theme === "dark" ? SUN : MOON;
  btn.setAttribute("aria-label", theme === "dark" ? "Light theme" : "Dark theme");
  btn.title = theme === "dark" ? "Light" : "Dark";
}

function apiBase() {
  return readStore("apiBase") || DEFAULT_API;
}

function setApiOpen(open) {
  $("api-panel").hidden = !open;
  $("api-toggle").setAttribute("aria-expanded", String(open));
  if (open) {
    $("api-input").value = apiBase();
    $("api-input").focus();
  }
}

function buildSteps() {
  $("steps").innerHTML = STAGES.map(([id, name], i) =>
    `<li data-stage="${id}"><span class="dot">${i + 1}</span><span class="label">${name}</span></li>`
  ).join("");
}

function setStage(stageId) {
  const idx = stageId === "done" ? STAGES.length : STAGES.findIndex(([id]) => id === stageId);
  if (idx < 0) return;
  $("steps").querySelectorAll("li").forEach((li, i) => {
    li.classList.toggle("done", i < idx);
    li.classList.toggle("active", i === idx);
  });
}

function setBusy(busy) {
  const btn = $("search-btn");
  btn.innerHTML = busy ? STOP_ICON : SEARCH_ICON;
  btn.setAttribute("aria-label", busy ? "Stop" : "Search");
  btn.title = busy ? "Stop" : "Search";
  btn.classList.toggle("stop", busy);
  $("spinner").hidden = !busy;
}

function setDepth(depth) {
  state.depth = depth;
  $("depth").querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(Number(b.dataset.depth) === depth)));
}

function showError(message) {
  $("error-msg").textContent = message;
  $("error").hidden = false;
}

async function startSearch() {
  const query = $("query").value.trim();
  if (!query) return;
  const depth = state.depth;

  document.querySelector(".app").classList.add("ran");
  $("results").hidden = true;
  $("error").hidden = true;
  $("progress").hidden = false;
  buildSteps();
  $("bar-fill").style.width = "0%";
  $("status-msg").textContent = "Starting";
  $("status-detail").textContent = "";
  state.result = null;
  state.controller = new AbortController();
  setBusy(true);

  try {
    const resp = await fetch(`${apiBase()}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, search_depth: depth }),
      signal: state.controller.signal
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(typeof err.detail === "string" ? err.detail : "API error");
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      buf += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buf.split("\n");
      buf = lines.pop();
      lines.filter(line => line.startsWith("data: ")).forEach(line => handleEvent(JSON.parse(line.slice(6))));
      if (done) break;
    }
  } catch (err) {
    $("progress").hidden = true;
    if (err.name !== "AbortError") {
      showError(err instanceof TypeError ? `Could not reach the backend at ${apiBase()}. Is it running?` : err.message);
    }
  } finally {
    state.controller = null;
    setBusy(false);
  }
}

function handleEvent(msg) {
  if (msg.type === "progress") {
    setStage(msg.stage);
    $("bar-fill").style.width = `${Math.round(msg.progress * 100)}%`;
    $("status-msg").textContent = msg.message;
    $("status-detail").textContent = msg.detail || "";
  } else if (msg.type === "result") {
    state.result = msg.data;
    $("progress").hidden = true;
    renderResults(msg.data);
    $("results").hidden = false;
  } else if (msg.type === "error") {
    $("progress").hidden = true;
    showError(msg.message);
  }
}

function confLevel(conf) {
  return conf >= 0.85 ? "high" : conf >= 0.65 ? "mid" : "low";
}

function cellHtml(col, cell, entitySources) {
  if (!cell || cell.value === null || cell.value === undefined || cell.value === "") {
    return `<td class="${col === "name" ? "name" : ""}"><span class="null">-</span></td>`;
  }
  const val = String(cell.value);
  const conf = cell.confidence ?? 1;
  const sources = cell.sources || [];
  sources.forEach(src => {
    if (!entitySources.some(s => s.url === src.url)) entitySources.push(src);
  });
  const badges = sources.map(src => {
    const n = entitySources.findIndex(s => s.url === src.url) + 1;
    return `<button class="src" data-url="${esc(src.url)}" data-title="${esc(src.title || src.url)}" data-snippet="${esc(src.snippet || "")}" title="${esc(src.title || src.url)}" aria-label="Source ${n}">${n}</button>`;
  }).join("");
  const star = cell.llm_filled ? `<span class="star" title="From the model's memory, not a web page">*</span>` : "";
  const cut = col !== "name" && val.length > TRUNCATE;
  const shown = cut
    ? `${esc(val.slice(0, TRUNCATE))}...<button class="more" data-col="${esc(label(col))}" data-val="${esc(val)}">more</button>`
    : esc(val);
  return `<td class="${col === "name" ? "name" : ""}">
    <div class="cell">
      <span class="conf ${confLevel(conf)}" title="Confidence ${Math.round(conf * 100)}%"></span>
      <span class="val">${star}${shown}</span>
      ${badges ? `<span class="srcs">${badges}</span>` : ""}
    </div>
  </td>`;
}

function renderResults(data) {
  const { entity_type, columns, entities, sources_consulted, search_queries_used, rounds_completed, ranking } = data;
  $("results-title").textContent = `${entities.length} ${entity_type}`;
  $("results-meta").textContent = `${sources_consulted.length} sources, ${search_queries_used.length} search queries, ${rounds_completed} round${rounds_completed === 1 ? "" : "s"}`;

  const hasRanking = Boolean(ranking && ranking.total > 0 && ranking.labels && ranking.labels.length);
  $("pills").hidden = !hasRanking;
  $("pills").innerHTML = hasRanking
    ? `<span class="tag">Ranked by</span>` + ranking.labels.map(l => `<span class="pill">${esc(l)}</span>`).join("")
    : "";

  $("thead").innerHTML = `<tr>${hasRanking ? `<th class="match">Match</th>` : ""}${columns.map(c => `<th>${esc(label(c))}</th>`).join("")}<th class="count">Sources</th></tr>`;

  $("tbody").innerHTML = entities.map(entity => {
    const entitySources = [];
    let match = "";
    if (hasRanking) {
      const n = ranking.scores[entity.id] ?? 0;
      const share = n / ranking.total;
      const cls = share === 1 ? "full" : share >= 0.5 ? "part" : "none";
      match = `<td class="match"><span class="badge ${cls}" title="${n} of ${ranking.total} rules met">${n}/${ranking.total}</span></td>`;
    }
    const cells = columns.map(col => cellHtml(col, entity.cells[col], entitySources)).join("");
    return `<tr>${match}${cells}<td class="count">${entitySources.length}</td></tr>`;
  }).join("");

  const hasLlm = entities.some(e => Object.values(e.cells || {}).some(c => c.llm_filled));
  $("sources-note").textContent = "Click a number to see the page and the quote behind a value." +
    (hasLlm ? " Values with * come from the model's memory, not a web page." : "");
}

function toggleDetail(btn) {
  const row = btn.closest("tr");
  const next = row.nextElementSibling;
  if (next && next.classList.contains("detail")) {
    next.remove();
    return;
  }
  const span = row.children.length;
  const detail = document.createElement("tr");
  detail.className = "detail";
  detail.innerHTML = `<td colspan="${span}">
    <div class="detail-box">
      <div><p class="small-head">${esc(btn.dataset.col)}</p><p>${esc(btn.dataset.val)}</p></div>
      <button class="close detail-close" aria-label="Close">x</button>
    </div>
  </td>`;
  row.after(detail);
}

function openSource(btn) {
  const { url, title, snippet } = btn.dataset;
  let domain = url;
  try { domain = new URL(url).hostname; } catch (e) {}
  $("modal-body").innerHTML = `
    <h3>${esc(title || domain)}</h3>
    <a class="url" href="${esc(url)}" target="_blank" rel="noopener">${esc(url)}</a>
    ${snippet ? `<p class="quote">${esc(snippet)}</p>` : `<p class="muted" style="margin:12px 0">No quote was saved for this value.</p>`}
    <a class="primary" href="${esc(url)}" target="_blank" rel="noopener">Open page</a>`;
  $("modal").hidden = false;
  $("modal-close").focus();
}

function closeSource() {
  $("modal").hidden = true;
}

function download(content, filename, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function csvCell(v) {
  return `"${String(v).replace(/"/g, '""')}"`;
}

function exportJSON() {
  if (!state.result) return;
  download(JSON.stringify(state.result, null, 2), `agentic-search-${Date.now()}.json`, "application/json");
}

function exportCSV() {
  if (!state.result) return;
  const { columns, entities } = state.result;
  const header = [...columns, "sources"].map(csvCell).join(",");
  const rows = entities.map(e => {
    const urls = [...new Set(Object.values(e.cells || {}).flatMap(c => (c.sources || []).map(s => s.url)))];
    return [...columns.map(c => csvCell(e.cells[c]?.value ?? "")), csvCell(urls.join("; "))].join(",");
  });
  download([header, ...rows].join("\n"), `agentic-search-${Date.now()}.csv`, "text/csv");
}

function renderExamples(list) {
  $("examples").innerHTML = list.map(ex => `<li><button type="button" data-ex="${esc(ex)}">${esc(ex)}</button></li>`).join("");
}

async function loadExamples() {
  renderExamples(FALLBACK_EXAMPLES);
  try {
    const r = await fetch(`${apiBase()}/example-queries`);
    const data = await r.json();
    if (Array.isArray(data.examples) && data.examples.length) renderExamples(data.examples);
  } catch (e) {}
}

function init() {
  const saved = readStore("ais-theme");
  applyTheme(saved === "light" || saved === "dark" ? saved : currentTheme());
  setBusy(false);
  $("depth").addEventListener("click", e => {
    const b = e.target.closest("button[data-depth]");
    if (b) setDepth(Number(b.dataset.depth));
  });
  $("theme-toggle").addEventListener("click", () => applyTheme(currentTheme() === "dark" ? "light" : "dark"));

  /*
  $("api-toggle").addEventListener("click", () => setApiOpen($("api-panel").hidden));
  $("api-panel").addEventListener("submit", e => {
    e.preventDefault();
    const val = $("api-input").value.trim().replace(/\/$/, "");
    writeStore("apiBase", val || DEFAULT_API);
    $("api-note").textContent = `Saved. Using ${apiBase()}`;
    loadExamples();
  });
  document.addEventListener("click", e => {
    if (!$("api-panel").hidden && !$("api-panel").parentElement.contains(e.target)) setApiOpen(false);
  });
  */

  $("search-form").addEventListener("submit", e => {
    e.preventDefault();
    if (state.controller) state.controller.abort();
    else startSearch();
  });
  $("query").addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("search-form").requestSubmit();
    }
  });
  $("examples").addEventListener("click", e => {
    const b = e.target.closest("button[data-ex]");
    if (!b || state.controller) return;
    $("query").value = b.dataset.ex;
    startSearch();
  });

  $("tbody").addEventListener("click", e => {
    const src = e.target.closest(".src");
    if (src) return openSource(src);
    const more = e.target.closest(".more");
    if (more) return toggleDetail(more);
    const close = e.target.closest(".detail-close");
    if (close) close.closest("tr").remove();
  });

  $("modal-close").addEventListener("click", closeSource);
  $("modal").addEventListener("click", e => { if (e.target === $("modal")) closeSource(); });
  document.addEventListener("keydown", e => {
    if (e.key !== "Escape") return;
    if (!$("modal").hidden) closeSource();
    /* else if (!$("api-panel").hidden) { setApiOpen(false); $("api-toggle").focus(); } */
  });

  $("export-json").addEventListener("click", exportJSON);
  $("export-csv").addEventListener("click", exportCSV);

  loadExamples();
}

init();
