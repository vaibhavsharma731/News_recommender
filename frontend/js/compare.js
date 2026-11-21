/**
 * compare.js — Dual Algorithm Side-by-Side Recommendation Comparison
 * Section 3: User Page Redesign - Side-by-Side Recommendation Comparison
 *
 * Features:
 * - Article search/selection
 * - Two algorithm picker dropdowns (A vs B)
 * - Parallel fetch of both recommendation sets
 * - Rank-shift badges, score delta highlighting, shared/unique article markers
 * - Latency comparison display
 */
const Compare = (() => {
  let activeArticle = null;
  let isRunning = false;

  const dom = {};

  function cacheDom() {
    dom.storyInput   = document.getElementById('compare-story-input');
    dom.storyClear   = document.getElementById('compare-story-clear');
    dom.storyDropdown = document.getElementById('compare-story-dropdown');
    dom.algoA        = document.getElementById('compare-algo-a');
    dom.algoB        = document.getElementById('compare-algo-b');
    dom.dsA          = document.getElementById('compare-dataset-a');
    dom.dsB          = document.getElementById('compare-dataset-b');
    dom.runBtn       = document.getElementById('compare-run-btn');
    dom.latencyBadge = document.getElementById('compare-latency-badge');
    dom.latencyText  = document.getElementById('compare-latency-text');
    dom.sourceBox    = document.getElementById('compare-source-box');
    dom.sourceTitle  = document.getElementById('compare-source-title');
    dom.sourceMeta   = document.getElementById('compare-source-meta');
    dom.columns      = document.getElementById('compare-columns');
    dom.resultsA     = document.getElementById('compare-results-a');
    dom.resultsB     = document.getElementById('compare-results-b');
    dom.headerA      = document.getElementById('compare-header-a');
    dom.headerB      = document.getElementById('compare-header-b');
    dom.latencyA     = document.getElementById('compare-latency-a');
    dom.latencyB     = document.getElementById('compare-latency-b');
    dom.emptyState   = document.getElementById('compare-empty');
  }

  function bindEvents() {
    if (!dom.storyInput) return;

    // Article search
    let searchTimer = null;
    dom.storyInput.addEventListener('input', () => {
      const q = dom.storyInput.value.trim();
      clearTimeout(searchTimer);
      if (dom.storyClear) dom.storyClear.classList.toggle('hidden', !q);
      if (!q) { hideDropdown(); return; }
      searchTimer = setTimeout(() => fetchSuggestions(q), 220);
    });

    if (dom.storyClear) {
      dom.storyClear.addEventListener('click', () => {
        dom.storyInput.value = '';
        dom.storyClear.classList.add('hidden');
        activeArticle = null;
        if (dom.runBtn) dom.runBtn.disabled = true;
        hideDropdown();
        resetUI();
      });
    }

    document.addEventListener('click', (e) => {
      if (!dom.storyDropdown) return;
      if (!dom.storyDropdown.contains(e.target) && e.target !== dom.storyInput) {
        hideDropdown();
      }
    });

    if (dom.runBtn) {
      dom.runBtn.addEventListener('click', runComparison);
    }
  }

  async function fetchSuggestions(q) {
    try {
      const data = await Api.articles({ page: 1, perPage: 8, search: q });
      const arts = data.articles || [];
      showDropdown(arts);
    } catch (_) { hideDropdown(); }
  }

  function showDropdown(arts) {
    if (!dom.storyDropdown) return;
    if (!arts.length) { hideDropdown(); return; }
    dom.storyDropdown.innerHTML = arts.map(a => `
      <div class="story-finder-item" data-id="${escapeHtml(String(a.id || ''))}">
        <div class="story-finder-item-title">${escapeHtml(String(a.title || ''))}</div>
        <div class="story-finder-item-meta">
          ${escapeHtml(String(a.section || 'General'))} &middot;
          ${a.date ? new Date(a.date).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : 'Recent'}
        </div>
      </div>
    `).join('');
    dom.storyDropdown.classList.remove('hidden');
    dom.storyDropdown.querySelectorAll('.story-finder-item').forEach(el => {
      el.addEventListener('click', () => {
        const id = el.dataset.id;
        const title = el.querySelector('.story-finder-item-title').textContent;
        const meta  = el.querySelector('.story-finder-item-meta').textContent;
        selectArticle({ id, title, meta, section: '', date: '' });
        dom.storyInput.value = title;
        if (dom.storyClear) dom.storyClear.classList.remove('hidden');
        hideDropdown();
      });
    });
  }

  function hideDropdown() {
    if (dom.storyDropdown) dom.storyDropdown.classList.add('hidden');
  }

  function selectArticle(art) {
    activeArticle = art;
    if (dom.runBtn) dom.runBtn.disabled = false;

    // Show source box
    if (dom.sourceBox) {
      dom.sourceBox.style.display = 'block';
      if (dom.sourceTitle) dom.sourceTitle.textContent = art.title || '';
      if (dom.sourceMeta)  dom.sourceMeta.textContent  = art.meta  || art.section || '';
    }
    resetResults();
  }

  function resetResults() {
    if (dom.columns)    { dom.columns.style.display = 'none'; dom.columns.style.gridTemplateColumns = ''; }
    if (dom.emptyState) dom.emptyState.style.display = 'block';
    if (dom.latencyBadge) dom.latencyBadge.style.display = 'none';
  }

  function resetUI() {
    if (dom.sourceBox)   dom.sourceBox.style.display = 'none';
    if (dom.columns)     { dom.columns.style.display = 'none'; }
    if (dom.emptyState)  dom.emptyState.style.display = 'block';
    if (dom.latencyBadge) dom.latencyBadge.style.display = 'none';
  }

  const ALGO_LABELS = {
    'hybrid+rerank': '⚡ Precomputed Hybrid + Reranker',
    'hybrid':        '🛰️ Live Hybrid (No Reranker)',
    'semantic':      '🧠 Semantic Dense Only',
    'bm25':          '📖 BM25 Lexical Only',
  };

  async function runComparison() {
    if (!activeArticle || isRunning) return;
    const methodA = dom.algoA?.value || 'hybrid+rerank';
    const methodB = dom.algoB?.value || 'hybrid';
    const dsA = dom.dsA?.value || 'default';
    const dsB = dom.dsB?.value || 'default';

    if (methodA === methodB && dsA === dsB) {
      alert('Please select either different algorithms or different datasets to compare.');
      return;
    }

    isRunning = true;
    if (dom.runBtn) {
      dom.runBtn.disabled = true;
      dom.runBtn.innerHTML = '<span style="opacity:0.6;">⏳ Running comparison…</span>';
    }

    const labelA = `${ALGO_LABELS[methodA] || methodA} [${dsA}]`;
    const labelB = `${ALGO_LABELS[methodB] || methodB} [${dsB}]`;
    if (dom.headerA) dom.headerA.textContent = labelA;
    if (dom.headerB) dom.headerB.textContent = labelB;

    if (dom.resultsA) dom.resultsA.innerHTML = renderLoading();
    if (dom.resultsB) dom.resultsB.innerHTML = renderLoading();

    if (dom.columns)   { dom.columns.style.display = 'grid'; dom.columns.style.gridTemplateColumns = '1fr 1fr'; }
    if (dom.emptyState) dom.emptyState.style.display = 'none';

    const artId = String(activeArticle.id);

    try {
      const t0 = performance.now();
      const [resA, resB] = await Promise.all([
        fetchRecs(artId, methodA, dsA),
        fetchRecs(artId, methodB, dsB),
      ]);
      const wallMs = Math.round(performance.now() - t0);

      const idSetA = new Set((resA.recs || []).map(r => r.id));
      const idSetB = new Set((resB.recs || []).map(r => r.id));

      if (dom.resultsA) dom.resultsA.innerHTML = renderPane(resA.recs || [], idSetB, 'a', resA.ms);
      if (dom.resultsB) dom.resultsB.innerHTML = renderPane(resB.recs || [], idSetA, 'b', resB.ms);

      if (dom.latencyA) dom.latencyA.textContent = `${resA.ms} ms`;
      if (dom.latencyB) dom.latencyB.textContent = `${resB.ms} ms`;

      if (dom.latencyBadge) {
        dom.latencyBadge.style.display = 'flex';
        const faster = resA.ms < resB.ms ? `System A is faster (+${resB.ms - resA.ms} ms)` : (resB.ms < resA.ms ? `System B is faster (+${resA.ms - resB.ms} ms)` : 'Both equal');
        dom.latencyText.textContent = `Wall: ${wallMs} ms | A: ${resA.ms} ms, B: ${resB.ms} ms → ${faster}`;
      }
    } catch (err) {
      const errHtml = `<div class="empty-state" style="padding:24px;"><div class="empty-icon">⚠️</div><p>${escapeHtml(err.message)}</p></div>`;
      if (dom.resultsA) dom.resultsA.innerHTML = errHtml;
      if (dom.resultsB) dom.resultsB.innerHTML = errHtml;
    } finally {
      isRunning = false;
      if (dom.runBtn) {
        dom.runBtn.disabled = false;
        dom.runBtn.innerHTML = '⚖️ Run Comparison';
      }
    }
  }

  async function fetchRecs(articleId, method, dataset = 'default') {
    const t0 = performance.now();
    let recs = [];
    if (method === 'hybrid+rerank') {
      // Precomputed path
      const data = await Api.precomputedRecs(articleId, 10, dataset);
      recs = (data.recommendations || []).map((r, i) => ({
        id:    r.id || r.recommended_article_id || '',
        rank:  i + 1,
        title: r.title || '',
        section: r.section || '',
        score:   r.score || r.confidence || 0,
        source:  'precomputed',
      }));
    } else {
      const data = await Api.recommend({ articleId, method, dataset, topK: 10, useBoosts: true, diversify: true, calibrate: true });
      recs = (data.recommendations || []).map((r, i) => ({
        id:    (r.article || {}).id || '',
        rank:  r.rank || (i + 1),
        title: (r.article || {}).title || '',
        section: (r.article || {}).section || '',
        score:   r.score || 0,
        source: 'live',
      }));
    }
    const ms = Math.round(performance.now() - t0);
    return { recs, ms };
  }

  function renderLoading() {
    return Array.from({ length: 5 }, (_, i) => `
      <div style="padding:12px 14px;border-bottom:1px solid var(--border-subtle);">
        <div class="skeleton" style="height:14px;width:80%;border-radius:4px;margin-bottom:6px;"></div>
        <div class="skeleton" style="height:11px;width:50%;border-radius:4px;"></div>
      </div>`).join('');
  }

  function renderPane(recs, otherIdSet, side, ms) {
    if (!recs.length) {
      return `<div class="empty-state" style="padding:24px;"><div class="empty-icon">🔍</div><p>No recommendations returned.</p></div>`;
    }

    const color = side === 'a' ? '#6366f1' : '#f97316';

    return recs.map((r, idx) => {
      const isShared  = otherIdSet.has(r.id);
      const scoreStr  = typeof r.score === 'number' ? (r.score * 100).toFixed(1) + '%' : '—';
      const sharedBadge = isShared
        ? `<span style="font-size:0.68rem;font-weight:700;padding:2px 6px;border-radius:4px;background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.3);">✓ Both</span>`
        : `<span style="font-size:0.68rem;font-weight:700;padding:2px 6px;border-radius:4px;background:rgba(239,68,68,0.1);color:#ef4444;border:1px solid rgba(239,68,68,0.25);">Unique</span>`;

      return `
        <div class="compare-rec-row" style="padding:10px 14px;border-bottom:1px solid var(--border-subtle);display:flex;align-items:flex-start;gap:10px;">
          <div style="min-width:28px;height:28px;border-radius:50%;background:${color}22;border:1.5px solid ${color}55;display:flex;align-items:center;justify-content:center;font-size:0.78rem;font-weight:800;color:${color};">${idx + 1}</div>
          <div style="flex:1;min-width:0;">
            <div style="font-size:0.83rem;font-weight:600;color:var(--text-primary);line-height:1.4;margin-bottom:3px;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;">${escapeHtml(r.title)}</div>
            <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
              <span style="font-size:0.72rem;color:var(--text-muted);">${escapeHtml(r.section || 'General')}</span>
              <span style="font-size:0.72rem;font-weight:700;color:${color};">${scoreStr}</span>
              ${sharedBadge}
            </div>
          </div>
        </div>`;
    }).join('');
  }

  function escapeHtml(s) {
    return String(s || '').replace(/[&<>"']/g, m => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[m]));
  }

  async function populateDatasets() {
    try {
      const res = await Api.listDatasets();
      const datasets = res.datasets || [];
      const opts = datasets.map(d => `<option value="${escapeHtml(d.id)}">${escapeHtml(d.name)} (${escapeHtml(d.pipeline_type || 'precomputed')})</option>`).join('');
      if (dom.dsA && opts) {
        const curA = dom.dsA.value;
        dom.dsA.innerHTML = opts;
        if (curA) dom.dsA.value = curA;
      }
      if (dom.dsB && opts) {
        const curB = dom.dsB.value;
        dom.dsB.innerHTML = opts;
        if (curB) dom.dsB.value = curB;
      }
    } catch (_) {}
  }

  function init() {
    cacheDom();
    bindEvents();
    populateDatasets();
  }

  return { init, populateDatasets };
})();

window.Compare = Compare;
