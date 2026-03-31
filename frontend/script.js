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
  p.classList.toggle('hidden');
  if (!p.classList.contains('hidden')) {
    document.getElementById('apiUrlInput').value = getApiBase();
  }
}

// ── Pipeline stages ──────────────────────────────────────────────────────────

const STAGES = [
  { id: 'planning',   label: 'Plan',    icon: '🧠' },
  { id: 'searching',  label: 'Search',  icon: '🔍' },
  { id: 'scraping',   label: 'Scrape',  icon: '📄' },
  { id: 'extracting', label: 'Extract', icon: '⚗️'  },
  { id: 'resolving',  label: 'Resolve', icon: '🔗' },
  { id: 'analyzing',  label: 'Analyse', icon: '📊' },
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

// ── Search ───────────────────────────────────────────────────────────────────

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); startSearch(); }
}

async function startSearch() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;
  const maxRounds = parseInt(document.getElementById('roundsSelect').value);

  // Reset UI
  hide('resultsSection');
  hide('errorBanner');
  show('progressSection');
  buildPipelineSteps();
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('progressMessage').textContent = 'Starting…';
  document.getElementById('progressDetail').textContent = '';
  document.getElementById('searchBtn').disabled = true;
  document.getElementById('searchBtn').innerHTML = `
    <div class="w-4 h-4 border-2 border-white border-t-blue-200 rounded-full animate-spin"></div>
    Searching…
  `;
  lastResult = null;

  const url = `${getApiBase()}/api/search`;
  let es;
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, max_rounds: maxRounds }),
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
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          handleSSE(JSON.parse(line.slice(6)));
        }
      }
    }
  } catch (err) {
    showError(err.message);
  } finally {
    resetSearchBtn();
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
  } else if (msg.type === 'error') {
    hide('progressSection');
    showError(msg.message);
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
  document.getElementById('sourcesText').textContent =
    `Every highlighted value traces to a source. Click the blue badges (①②③) to see the exact excerpt.`;

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

        // Collect all sources for the entity
        sources.forEach(src => {
          const exists = allSources.find(s => s.url === src.url);
          if (!exists) allSources.push(src);
        });

        // Build source badges referencing the entity-level source list
        const badges = sources.map(src => {
          const srcIdx = allSources.findIndex(s => s.url === src.url);
          const snippetEscaped = escapeHtml(src.snippet || '');
          const titleEscaped = escapeHtml(src.title || src.url);
          const urlEscaped = escapeHtml(src.url);
          return `<span class="src-badge" onclick="showSource('${urlEscaped}','${titleEscaped}','${snippetEscaped}')" title="${titleEscaped}">${srcIdx + 1}</span>`;
        }).join('');

        td.className = col === 'name' ? 'name-cell' : '';
        td.innerHTML = `
          <div class="flex items-start gap-1">
            <span class="conf-dot ${confClass}" title="Confidence: ${Math.round(conf*100)}%"></span>
            <span title="${escapeHtml(val)}">${escapeHtml(val)}</span>
            ${badges}
          </div>`;
      }
      tr.appendChild(td);
    });

    // Sources cell
    const srcTd = document.createElement('td');
    srcTd.innerHTML = `<span class="text-slate-400 text-xs mono">${allSources.length}</span>`;
    tr.appendChild(srcTd);

    tbody.appendChild(tr);
  });

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
      btn.className = 'text-xs bg-slate-100 hover:bg-blue-100 hover:text-blue-700 text-slate-600 px-3 py-1 rounded-full transition';
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

function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

function showError(msg) {
  document.getElementById('errorMessage').textContent = msg;
  show('errorBanner');
}

function resetSearchBtn() {
  const btn = document.getElementById('searchBtn');
  btn.disabled = false;
  btn.innerHTML = `
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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

// ── Init ──────────────────────────────────────────────────────────────────────
loadExamples();
document.getElementById('apiUrlInput') && (document.getElementById('apiUrlInput').value = getApiBase());
