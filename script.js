// ── Theme ─────────────────────────────────────────────────────────────────────

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  const thumb = document.getElementById('tt');
  if (thumb) thumb.textContent = theme === 'dark' ? '🌙' : '☀️';
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

// Apply saved theme immediately (before paint) — default dark like portfolio
applyTheme(localStorage.getItem('theme') || 'dark');

// ── Config ──────────────────────────────────────────────────────────────────

function getApiBase() {
  return localStorage.getItem('apiBase') || 'http://localhost:8000';
}

function saveApiUrl(notify = false) {
  const val = document.getElementById('apiUrlInput').value.trim().replace(/\/$/, '');
  if (val) {
    localStorage.setItem('apiBase', val);
    if (notify) alert('Saved! API URL: ' + val);
  }
}

function toggleSettings() {
  const p = document.getElementById('settingsPanel');
  const isOpen = p.style.display !== 'none' && p.style.display !== '';
  p.style.display = isOpen ? 'none' : 'block';
  if (!isOpen) document.getElementById('apiUrlInput').value = getApiBase();
}

// ── Pipeline stages ──────────────────────────────────────────────────────────

const STAGES = [
  { id: 'planning',   label: 'Plan',    icon: '🧠' },
  { id: 'searching',  label: 'Search',  icon: '🔍' },
  { id: 'scraping',   label: 'Scrape',  icon: '📄' },
  { id: 'extracting', label: 'Extract', icon: '⚗️'  },
  { id: 'resolving',  label: 'Resolve', icon: '🔗' },
  { id: 'analyzing',  label: 'Analyse', icon: '📊' },
  { id: 'filling',    label: 'LLM Fill', icon: '🤖' },
];

function buildPipelineSteps() {
  const container = document.getElementById('pipelineSteps');
  container.innerHTML = '';
  STAGES.forEach((s, i) => {
    const step = document.createElement('div');
    step.id = `step-${s.id}`;
    step.className = 'flex flex-col items-center gap-1';
    step.innerHTML = `
      <div id="step-icon-${s.id}"
        class="w-9 h-9 rounded-full border-2 border-slate-200 bg-white flex items-center justify-center text-base transition-all duration-300"
        title="${s.label}">
        ${s.icon}
      </div>
      <span class="text-[10px] text-slate-400 font-medium" id="step-label-${s.id}">${s.label}</span>
    `;
    container.appendChild(step);

    if (i < STAGES.length - 1) {
      const conn = document.createElement('div');
      conn.id = `conn-${s.id}`;
      conn.className = 'step-connector self-start mt-4 mx-1';
      container.appendChild(conn);
    }
  });
}

function updateStep(stageId, state) {
  // state: 'active' | 'done' | 'idle'
  const icon = document.getElementById(`step-icon-${stageId}`);
  const label = document.getElementById(`step-label-${stageId}`);
  if (!icon) return;
  if (state === 'active') {
    icon.className = 'w-9 h-9 rounded-full border-2 border-blue-500 bg-blue-50 flex items-center justify-center text-base shadow-sm shadow-blue-100 scale-110 transition-all duration-300';
    label.className = 'text-[10px] text-blue-600 font-semibold';
  } else if (state === 'done') {
    icon.className = 'w-9 h-9 rounded-full border-2 border-green-400 bg-green-50 flex items-center justify-center text-base transition-all duration-300';
    label.className = 'text-[10px] text-green-600 font-medium';
    const connId = `conn-${stageId}`;
    const conn = document.getElementById(connId);
    if (conn) conn.classList.add('done');
  }
}

let currentStageIdx = -1;
function advanceToStage(stageId) {
  const idx = STAGES.findIndex(s => s.id === stageId);
  if (idx < 0) return;
  // Mark previous as done
  for (let i = 0; i < idx; i++) updateStep(STAGES[i].id, 'done');
  updateStep(stageId, 'active');
  currentStageIdx = idx;
}

// ── State ────────────────────────────────────────────────────────────────────

let lastResult = null;
let currentAbortController = null;

// ── Search ───────────────────────────────────────────────────────────────────

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); startSearch(); }
}

async function startSearch() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;
  const searchDepth = parseInt(document.getElementById('roundsSelect').value);

  // Reset UI
  hide('resultsSection');
  hide('errorBanner');
  show('progressSection');
  buildPipelineSteps();
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressMessage').textContent = 'Starting…';
  document.getElementById('progressDetail').textContent = '';
  setStopBtn();
  lastResult = null;

  currentAbortController = new AbortController();
  const url = `${getApiBase()}/api/search`;
  let searchComplete = false;

  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, search_depth: searchDepth }),
      signal: currentAbortController.signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'API error');
    }

    // Read SSE stream
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      buf += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const msg = JSON.parse(line.slice(6));
          if (msg.type === 'result' || msg.type === 'error') searchComplete = true;
          handleSSE(msg);
        }
      }
      if (done) break;
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      hide('progressSection');
    } else {
      showError(err.message);
    }
    searchComplete = true;
    resetSearchBtn();
  } finally {
    currentAbortController = null;
    if (!searchComplete) resetSearchBtn(); // safety net: stream closed without result/error
  }
}

function stopSearch() {
  if (currentAbortController) {
    currentAbortController.abort();
  }
}

function handleSSE(msg) {
  if (msg.type === 'progress') {
    advanceToStage(msg.stage);
    document.getElementById('progressFill').style.width = `${Math.round(msg.progress * 100)}%`;
    document.getElementById('progressMessage').textContent = msg.message;
    document.getElementById('progressDetail').textContent = msg.detail || '';

    if (msg.stage === 'done') {
      STAGES.forEach(s => updateStep(s.id, 'done'));
      document.getElementById('progressSpinner').style.display = 'none';
    }
  } else if (msg.type === 'result') {
    lastResult = msg.data;
    hide('progressSection');
    renderResults(msg.data);
    show('resultsSection');
    resetSearchBtn();
  } else if (msg.type === 'error') {
    hide('progressSection');
    showError(msg.message);
    resetSearchBtn();
  }
}

// ── Results rendering ────────────────────────────────────────────────────────

function renderResults(data) {
  const { query, entity_type, columns, entities, sources_consulted,
          search_queries_used, rounds_completed } = data;

  document.getElementById('resultsTitle').textContent =
    `${entities.length} ${entity_type}`;
  document.getElementById('resultsMeta').textContent =
    `${sources_consulted.length} sources consulted · ${search_queries_used.length} search queries · ${rounds_completed} round(s)`;
  const hasLLMFilled = entities.some(e =>
    Object.values(e.cells || {}).some(c => c.llm_filled)
  );
  document.getElementById('sourcesText').textContent =
    `Every highlighted value traces to a source. Click the blue badges (①②③) to see the exact excerpt.` +
    (hasLLMFilled ? '  * Starred values are LLM estimates and may be inaccurate.' : '');

  // Build column list: skip 'name' from columns since it's rendered first anyway
  const allCols = columns;

  // Header
  const thead = document.getElementById('tableHead');
  thead.innerHTML = `<tr>${allCols.map(c => `<th>${c.replace(/_/g, ' ')}</th>`).join('')}<th>Sources</th></tr>`;

  // Body
  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = '';

  entities.forEach((entity, entityIdx) => {
    const tr = document.createElement('tr');
    let allSources = [];

    allCols.forEach((col, colIdx) => {
      const td = document.createElement('td');
      const cell = entity.cells[col];

      if (!cell || cell.value === null || cell.value === undefined || cell.value === '') {
        td.innerHTML = `<span class="null-cell">-</span>`;
      } else {
        const val = String(cell.value);
        const conf = cell.confidence || 1;
        const confClass = conf >= 0.85 ? 'conf-high' : conf >= 0.65 ? 'conf-mid' : 'conf-low';
        const sources = cell.sources || [];
        const isLLMFilled = cell.llm_filled === true;

        // Collect all sources for the entity
        sources.forEach(src => {
          const exists = allSources.find(s => s.url === src.url);
          if (!exists) allSources.push(src);
        });

        // Build source badges referencing the entity-level source list
        const badges = sources.map(src => {
          const srcIdx = allSources.findIndex(s => s.url === src.url);
          return `<span class="src-badge"
            data-url="${escapeHtml(src.url)}"
            data-title="${escapeHtml(src.title || src.url)}"
            data-snippet="${escapeHtml(src.snippet || '')}"
            onclick="showSourceFromEl(this)"
            title="${escapeHtml(src.title || src.url)}">${srcIdx + 1}</span>`;
        }).join('');

        const starPrefix = isLLMFilled ? `<span class="llm-star" title="LLM estimate — may be inaccurate">*</span> ` : '';
        const TRUNCATE = 72;
        const isLong = col !== 'name' && (col === 'description' || val.length > TRUNCATE);
        const displayVal = isLong && val.length > TRUNCATE ? escapeHtml(val.slice(0, TRUNCATE)) + '…' : escapeHtml(val);

        td.className = col === 'name' ? 'name-cell' : '';

        if (isLong) {
          td.innerHTML = `
            <div class="flex items-start gap-1">
              <span class="conf-dot ${confClass}" title="Confidence: ${Math.round(conf*100)}%"></span>
              <span class="cell-expandable"
                data-col="${escapeHtml(col.replace(/_/g, ' '))}"
                data-val="${escapeHtml(val)}"
                onclick="expandCell(this)">
                ${starPrefix}${displayVal}<span class="show-more-hint">expand</span>
              </span>
              ${badges}
            </div>`;
        } else {
          td.innerHTML = `
            <div class="flex items-start gap-1">
              <span class="conf-dot ${confClass}" title="Confidence: ${Math.round(conf*100)}%"></span>
              ${starPrefix}<span title="${escapeHtml(val)}">${escapeHtml(val)}</span>
              ${badges}
            </div>`;
        }
      }
      tr.appendChild(td);
    });

    // Sources cell
    const srcTd = document.createElement('td');
    srcTd.innerHTML = `<span class="text-slate-400 text-xs mono">${allSources.length}</span>`;
    tr.appendChild(srcTd);

    tbody.appendChild(tr);
  });

  // Confidence legend (rendered once after the table)
  const existingLegend = document.getElementById('confLegend');
  if (existingLegend) existingLegend.remove();
  const legend = document.createElement('div');
  legend.id = 'confLegend';
  legend.className = 'conf-legend';
  legend.innerHTML = `
    <span class="conf-legend-label">Confidence:</span>
    <span class="conf-legend-item"><span class="conf-dot conf-high"></span> ≥ 85% — High</span>
    <span class="conf-legend-item"><span class="conf-dot conf-mid"></span> 65–84% — Medium</span>
    <span class="conf-legend-item"><span class="conf-dot conf-low"></span> &lt; 65% — Low</span>`;
  document.getElementById('sourcesFooter').after(legend);

  document.getElementById('resultsSection').classList.add('fade-in');
}

// ── Source modal ──────────────────────────────────────────────────────────────

function showSource(url, title, snippet) {
  const modal = document.getElementById('sourceModal');
  const content = document.getElementById('modalContent');

  const domain = (() => { try { return new URL(url).hostname; } catch { return url; } })();

  content.innerHTML = `
    <div class="mb-3">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Page</div>
      <div class="font-medium text-slate-800 text-sm">${escapeHtml(title || domain)}</div>
      <a href="${escapeHtml(url)}" target="_blank" rel="noopener"
        class="text-blue-600 hover:underline text-xs mono break-all">${escapeHtml(url)}</a>
    </div>
    ${snippet ? `
    <div class="mt-3">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Supporting Excerpt</div>
      <blockquote class="border-l-3 border-blue-400 pl-3 text-sm text-slate-700 italic leading-relaxed bg-blue-50 rounded-r-lg py-2 pr-3">
        "${escapeHtml(snippet)}"
      </blockquote>
    </div>` : ''}
    <div class="mt-4">
      <a href="${escapeHtml(url)}" target="_blank" rel="noopener"
        class="inline-flex items-center gap-1.5 text-xs bg-blue-600 text-white px-3 py-1.5 rounded-lg hover:bg-blue-700 transition">
        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
        </svg>
        Open source
      </a>
    </div>
  `;
  modal.classList.add('open');
}

function closeSourceModal() { document.getElementById('sourceModal').classList.remove('open'); }
function closeModal(e) { if (e.target === document.getElementById('sourceModal')) closeSourceModal(); }

// ── Export ───────────────────────────────────────────────────────────────────

function exportJSON() {
  if (!lastResult) return;
  download(JSON.stringify(lastResult, null, 2), `agentic-search-${Date.now()}.json`, 'application/json');
}

function exportCSV() {
  if (!lastResult) return;
  const { columns, entities } = lastResult;
  const header = [...columns, 'sources'].join(',');
  const rows = entities.map(e => [
    ...columns.map(c => csvCell(e.cells[c]?.value ?? '')),
    csvCell((e.cells[columns[0]]?.sources || []).map(s => s.url).join('; '))
  ].join(','));
  download([header, ...rows].join('\n'), `agentic-search-${Date.now()}.csv`, 'text/csv');
}

function csvCell(v) {
  const s = String(v).replace(/"/g, '""');
  return `"${s}"`;
}

function download(content, filename, mime) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = filename;
  a.click();
}

// ── Examples ─────────────────────────────────────────────────────────────────

async function loadExamples() {
  try {
    const r = await fetch(`${getApiBase()}/api/example-queries`);
    const data = await r.json();
    const container = document.getElementById('exampleContainer');
    data.examples.forEach(ex => {
      const btn = document.createElement('button');
      btn.textContent = ex;
      btn.className = 'ex-pill';
      btn.onclick = () => {
        document.getElementById('queryInput').value = ex;
        startSearch();
      };
      container.appendChild(btn);
    });
  } catch {
    // backend not reachable on load - that's fine
  }
}

// ── Utils ─────────────────────────────────────────────────────────────────────

function show(id) { document.getElementById(id).style.display = ''; }
function hide(id) { document.getElementById(id).style.display = 'none'; }

function showError(msg) {
  document.getElementById('errorMessage').textContent = msg;
  show('errorBanner');
}

function setStopBtn() {
  const btn = document.getElementById('searchBtn');
  btn.disabled = false;
  btn.onclick = stopSearch;
  btn.innerHTML = `
    <svg style="width:13px;height:13px;" viewBox="0 0 24 24" fill="currentColor">
      <rect x="5" y="5" width="14" height="14" rx="2"/>
    </svg>
    Stop`;
}

function resetSearchBtn() {
  const btn = document.getElementById('searchBtn');
  btn.disabled = false;
  btn.onclick = startSearch;
  btn.innerHTML = `
    <svg style="width:14px;height:14px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
    </svg>
    Search`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

// ── Inline cell expand ────────────────────────────────────────────────────────

function showSourceFromEl(el) {
  showSource(el.dataset.url, el.dataset.title, el.dataset.snippet);
}

function expandCell(triggerEl) {
  const colName = triggerEl.dataset.col || '';
  const fullText = triggerEl.dataset.val || '';
  const row = triggerEl.closest('tr');
  const colCount = row.closest('table').querySelectorAll('thead th').length || row.cells.length;

  // Toggle: if already open, collapse
  const next = row.nextElementSibling;
  if (next && next.classList.contains('expanded-detail-row')) {
    next.classList.remove('open');
    next.addEventListener('transitionend', () => next.remove(), { once: true });
    return;
  }

  const expandedRow = document.createElement('tr');
  expandedRow.className = 'expanded-detail-row';
  expandedRow.innerHTML = `
    <td colspan="${colCount}">
      <div class="expanded-detail-inner">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;">
          <div style="flex:1;">
            <div class="expanded-col-label">${escapeHtml(colName.replace(/_/g, ' '))}</div>
            <div class="expanded-detail-text">${escapeHtml(fullText)}</div>
          </div>
          <button class="expanded-close-btn" title="Collapse">
            <svg style="width:15px;height:15px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
      </div>
    </td>`;

  expandedRow.querySelector('.expanded-close-btn').addEventListener('click', () => {
    expandedRow.classList.remove('open');
    expandedRow.addEventListener('transitionend', () => expandedRow.remove(), { once: true });
  });

  row.after(expandedRow);
  expandedRow.getBoundingClientRect(); // force reflow so CSS transition fires
  expandedRow.classList.add('open');
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadExamples();
document.getElementById('apiUrlInput') && (document.getElementById('apiUrlInput').value = getApiBase());
