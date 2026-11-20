/**
 * hybrid_lab.js — Live Hybrid Recommendation Lab (No Cross-Encoder)
 * 
 * Interactive evaluation of pure in-memory Hybrid recommendations:
 *   - Global article search across all 4,177+ database articles (PostgreSQL ILIKE query)
 *   - Story selection without auto-computation (user clicks "Run Live Hybrid")
 *   - Interactive Platt Calibration toggle (ON/OFF) to observe latency difference in ms
 *   - Stage-by-stage timing HUD (prep, BM25, Dense vector scan, blending, domain boosts, MMR, calibration)
 *   - Real-time confidence calibration comparison
 */

const HybridLab = (() => {
  let activeArticle = null;
  let allArticles = [];
  let currentSearchResults = [];
  let isEvaluating = false;
  let hasEvaluatedOnce = false;
  let currentCandidateStrategy = 'quota'; // 'quota' (12/5/3) or 'pure_rrf' (20 slots)

  // Color map for timing breakdown segments
  const STAGE_COLORS = {
    prep: '#06b6d4',
    retrieval: '#10b981',
    blend: '#8b5cf6',
    boosts: '#f59e0b',
    mmr: '#ec4899',
    calibration: '#3b82f6',
  };

  /** Initialize the Hybrid Lab controller */
  async function init() {
    initSearchBox();
    initControls();
    await loadInitialArticles();
  }

  /** Load a diverse pool of articles for quick selection pills */
  async function loadInitialArticles() {
    try {
      const data = await Api.articles({ perPage: 40 });
      allArticles = data.articles || [];
      renderSamplePills();
      
      // Auto-select the first article as default head story WITHOUT auto-running recommendations
      if (allArticles.length > 0 && !activeArticle) {
        selectArticle(allArticles[0], false);
      }
    } catch (e) {
      console.warn('[HybridLab] Failed to load sample articles:', e);
    }
  }

  /** Render sample story pills across diverse sections */
  function renderSamplePills() {
    const pillsContainer = document.getElementById('hybrid-sample-pills');
    if (!pillsContainer || !allArticles.length) return;

    // Pick diverse stories across different sections
    const seenSecs = new Set();
    const samples = [];
    for (const art of allArticles) {
      const sec = art.section || 'General';
      if (!seenSecs.has(sec) && samples.length < 7) {
        seenSecs.add(sec);
        samples.push(art);
      }
    }
    if (samples.length < 5) {
      samples.push(...allArticles.slice(0, 5 - samples.length));
    }

    pillsContainer.innerHTML = `
      <span style="font-size:0.76rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;">Quick Test Stories:</span>
      ${samples.map((art, idx) => `
        <button type="button" class="hybrid-pill ${idx === 0 ? 'active' : ''}" data-id="${art.id}" title="${art.title}">
          <span style="opacity:0.7;">${art.section || 'Story'}</span>: ${art.title.slice(0, 32)}…
        </button>
      `).join('')}
    `;

    // Attach click handlers to sample pills
    pillsContainer.querySelectorAll('.hybrid-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        pillsContainer.querySelectorAll('.hybrid-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const art = allArticles.find(a => String(a.id) === String(btn.dataset.id));
        if (art) {
          const input = document.getElementById('hybrid-story-input');
          if (input) input.value = art.title;
          // Set head article but DO NOT auto-evaluate (wait for Run button)
          selectArticle(art, false);
        }
      });
    });
  }

  /** Set up the global database article search dropdown */
  function initSearchBox() {
    const input = document.getElementById('hybrid-story-input');
    const dropdown = document.getElementById('hybrid-story-dropdown');
    const clearBtn = document.getElementById('hybrid-story-clear');
    if (!input || !dropdown) return;

    let debounceTimer = null;

    input.addEventListener('input', () => {
      const q = input.value.trim();
      if (clearBtn) clearBtn.classList.toggle('hidden', !q);
      clearTimeout(debounceTimer);
      if (!q) {
        dropdown.classList.add('hidden');
        return;
      }

      // Query database across all 4,177 articles via backend ILIKE search
      debounceTimer = setTimeout(async () => {
        dropdown.innerHTML = '<div class="story-finder-item-empty" style="padding:12px;color:var(--text-muted);"><span class="spinner-sm"></span> Searching database for matching articles...</div>';
        dropdown.classList.remove('hidden');

        try {
          const data = await Api.articles({ search: q, perPage: 15 });
          currentSearchResults = data.articles || [];

          if (!currentSearchResults.length) {
            dropdown.innerHTML = `<div class="story-finder-item-empty" style="padding:12px;color:var(--text-muted);">No articles found matching "<strong>${escapeHtml(q)}</strong>".<br/><span style="font-size:0.75rem;">(You can still click 'Run Live Hybrid' to evaluate as a free-text search query)</span></div>`;
            dropdown.classList.remove('hidden');
            return;
          }

          dropdown.innerHTML = currentSearchResults.map(art => {
            const sec = (art.section || 'General').toUpperCase();
            return `
              <div class="story-finder-item" data-id="${art.id}">
                <div class="story-finder-item-content">
                  <div class="story-finder-item-title">${escapeHtml(art.title)}</div>
                  <div class="story-finder-item-meta">
                    <span class="story-finder-item-sec">${sec}</span>
                    <span>📅 ${art.date || 'Recent'}</span>
                    <span style="font-size:0.72rem;opacity:0.75;">👁️ ${(art.page_views || art.pageViews || 0).toLocaleString()} views</span>
                  </div>
                </div>
              </div>
            `;
          }).join('');
          dropdown.classList.remove('hidden');

          dropdown.querySelectorAll('.story-finder-item').forEach(item => {
            item.addEventListener('click', () => {
              const art = currentSearchResults.find(a => String(a.id) === String(item.dataset.id));
              if (art) {
                input.value = art.title;
                dropdown.classList.add('hidden');
                // Select article without auto-triggering recommendation (user clicks Run button)
                selectArticle(art, false);
              }
            });
          });
        } catch (err) {
          console.error('[HybridLab] Search failed:', err);
          dropdown.innerHTML = `<div class="story-finder-item-empty" style="padding:10px;color:var(--color-danger);">Search error: ${err.message}</div>`;
        }
      }, 250);
    });

    // Enter key triggers evaluation directly
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        dropdown.classList.add('hidden');
        runCurrentSelection();
      }
    });

    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        input.value = '';
        clearBtn.classList.add('hidden');
        dropdown.classList.add('hidden');
      });
    }

    document.addEventListener('click', (e) => {
      if (!e.target.closest('#hybrid-story-box') && !e.target.closest('#hybrid-story-dropdown') && !e.target.closest('#hybrid-story-input')) {
        dropdown.classList.add('hidden');
      }
    });
  }

  /** Set up button controls and parameter toggles */
  function initControls() {
    const btnRecompute = document.getElementById('btn-recompute-hybrid');
    if (btnRecompute) {
      btnRecompute.addEventListener('click', () => {
        runCurrentSelection();
      });
    }

    // Toggle Platt Calibration indicator in controls
    const toggleCalib = document.getElementById('hybrid-toggle-calibration');
    if (toggleCalib) {
      toggleCalib.addEventListener('change', () => {
        const isCalib = toggleCalib.checked;
        const noteEl = document.getElementById('calib-status-indicator');
        if (noteEl) {
          noteEl.textContent = isCalib ? 'Platt: ON' : 'Platt: OFF (Raw Scores)';
          noteEl.style.color = isCalib ? '#10b981' : '#eab308';
        }
      });
    }

    // Candidate Strategy Switcher (Quota Plan vs Pure 20 RRF Slots)
    const btnQuota = document.getElementById('btn-strat-quota');
    const btnRrf = document.getElementById('btn-strat-rrf');
    const updateStrategyUI = (strat) => {
      currentCandidateStrategy = strat;
      if (btnQuota && btnRrf) {
        if (strat === 'quota') {
          btnQuota.style.background = 'var(--color-primary)';
          btnQuota.style.color = '#fff';
          btnRrf.style.background = 'transparent';
          btnRrf.style.color = 'var(--text-secondary)';
        } else {
          btnRrf.style.background = 'var(--color-primary)';
          btnRrf.style.color = '#fff';
          btnQuota.style.background = 'transparent';
          btnQuota.style.color = 'var(--text-secondary)';
        }
      }
      const badge = document.getElementById('hud-candidate-strategy-badge');
      if (badge) {
        if (strat === 'quota') {
          badge.textContent = '🎯 Quota (12 / 5 / 3)';
          badge.style.background = 'rgba(236,72,153,0.15)';
          badge.style.color = '#ec4899';
          badge.style.borderColor = 'rgba(236,72,153,0.35)';
        } else {
          badge.textContent = '🔀 Pure RRF (20 Slots)';
          badge.style.background = 'rgba(168,85,247,0.15)';
          badge.style.color = '#c084fc';
          badge.style.borderColor = 'rgba(168,85,247,0.35)';
        }
      }
    };

    if (btnQuota) {
      btnQuota.addEventListener('click', () => {
        if (currentCandidateStrategy !== 'quota') {
          updateStrategyUI('quota');
          if (hasEvaluatedOnce) runCurrentSelection();
        }
      });
    }
    if (btnRrf) {
      btnRrf.addEventListener('click', () => {
        if (currentCandidateStrategy !== 'pure_rrf') {
          updateStrategyUI('pure_rrf');
          if (hasEvaluatedOnce) runCurrentSelection();
        }
      });
    }
  }

  /** Run recommendation on whatever is currently selected or typed */
  function runCurrentSelection() {
    const input = document.getElementById('hybrid-story-input');
    const inputVal = input ? input.value.trim() : '';

    if (activeArticle && (!inputVal || inputVal.toLowerCase() === (activeArticle.title || '').toLowerCase())) {
      evaluate(activeArticle.id, null);
    } else if (inputVal) {
      evaluate(null, inputVal);
    } else if (activeArticle) {
      evaluate(activeArticle.id, null);
    } else {
      alert('Please select an article or type a search query first.');
    }
  }

  /** Select an article as the active source input */
  function selectArticle(article, triggerEval = false) {
    activeArticle = article;

    const titleEl = document.getElementById('hybrid-source-title');
    const secEl = document.getElementById('hybrid-source-section');
    const dateEl = document.getElementById('hybrid-source-date');
    const snippetEl = document.getElementById('hybrid-source-snippet');

    if (titleEl) titleEl.textContent = article.title;
    if (secEl) {
      secEl.textContent = (article.section || 'General').toUpperCase();
      const col = typeof sectionColor === 'function' ? sectionColor(article.section) : '#6366f1';
      secEl.style.borderColor = `${col}44`;
      secEl.style.color = col;
    }
    if (dateEl) dateEl.textContent = `📅 ${article.date || 'Recent'}`;
    if (snippetEl) snippetEl.textContent = article.snippet || article.description || '';

    // Synchronize the search box text
    const searchInput = document.getElementById('hybrid-story-input');
    const searchClear = document.getElementById('hybrid-story-clear');
    if (searchInput) searchInput.value = article.title || '';
    if (searchClear) searchClear.classList.remove('hidden');

    // Update prompt on HUD container or ready state
    const readyNotice = document.getElementById('hybrid-ready-notice');
    if (readyNotice) {
      readyNotice.textContent = `Story selected: "${article.title.slice(0, 45)}...". Click 'Run Live Hybrid' to evaluate.`;
    }

    // Visual feedback: highlight and smoothly scroll to the active source box
    const sourceBox = document.getElementById('hybrid-source-box');
    if (sourceBox) {
      sourceBox.classList.add('widget-highlighted');
      setTimeout(() => sourceBox.classList.remove('widget-highlighted'), 2600);
      sourceBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    if (triggerEval) {
      evaluate(article.id, null);
    }
  }

  /** Run the Live Pure Hybrid recommendation evaluation */
  async function evaluate(articleId, queryText = null) {
    if (isEvaluating) return;
    if (!articleId && !queryText) return;
    isEvaluating = true;

    const hudContainer = document.getElementById('hybrid-hud-container');
    const resultsGrid = document.getElementById('hybrid-results-grid');
    const resultsCount = document.getElementById('hybrid-results-count');
    const btnRecompute = document.getElementById('btn-recompute-hybrid');

    if (hudContainer) hudContainer.classList.remove('hidden');
    if (btnRecompute) {
      btnRecompute.disabled = true;
      btnRecompute.innerHTML = '<span>⏳</span> Computing...';
    }
    if (resultsGrid) {
      if (typeof showSpinner === 'function') {
        showSpinner(resultsGrid, 3);
      } else {
        resultsGrid.innerHTML = '<div style="padding:30px;text-align:center;color:var(--text-muted)"><span class="spinner-sm"></span> Computing live hybrid recommendations in memory...</div>';
      }
    }

    const useBoosts = document.getElementById('hybrid-toggle-boosts')?.checked ?? true;
    const diversify = document.getElementById('hybrid-toggle-diversify')?.checked ?? true;
    const calibrate = document.getElementById('hybrid-toggle-calibration')?.checked ?? true;
    const topK = parseInt(document.getElementById('hybrid-select-topk')?.value || '10', 10);

    // Capture source article for passing to card renderer
    let querySourceArt = null;
    if (queryText && !articleId) {
      const titleEl = document.getElementById('hybrid-source-title');
      const secEl = document.getElementById('hybrid-source-section');
      const dateEl = document.getElementById('hybrid-source-date');
      const snippetEl = document.getElementById('hybrid-source-snippet');
      if (titleEl) titleEl.textContent = `🔍 Free-Text Query: "${queryText}"`;
      if (secEl) {
        secEl.textContent = 'QUERY SEARCH';
        secEl.style.color = '#38bdf8';
      }
      if (dateEl) dateEl.textContent = 'Live Query';
      if (snippetEl) snippetEl.textContent = 'Ad-hoc BM25 query normalization + BGE dense vector query.';
      // Synthetic source art for evaluator panels
      querySourceArt = { title: `🔍 "${queryText}"`, section: 'Query Search', date: null };
    }

    const clientT0 = performance.now();
    try {
      const data = await Api.recommend({
        articleId: articleId ? String(articleId) : null,
        query: queryText ? String(queryText) : null,
        method: 'hybrid', // PURE HYBRID (No Cross-Encoder)
        useBoosts,
        diversify,
        calibrate,
        topK,
        candidateStrategy: currentCandidateStrategy,
      });

      const clientTotalMs = Math.round((performance.now() - clientT0) * 10) / 10;
      hasEvaluatedOnce = true;

      // Extract detailed server timings
      const timings = data.timings || {};
      const totalMs = timings.total_ms || clientTotalMs;

      // Render the HUD Metrics & Visual Bar
      renderTimingHUD(timings, totalMs, clientTotalMs, calibrate);

      // Resolve source article: prefer API response, then active article, then query synthetic
      const sourceArticle = data.source_article || activeArticle || querySourceArt || null;

      // Render the Recommendations using Evaluator-style evidence cards
      const recs = data.recommendations || [];
      if (resultsCount) resultsCount.textContent = `${recs.length} stories`;
      renderHybridRecommendations(recs, resultsGrid, calibrate, sourceArticle);

    } catch (err) {
      console.error('[HybridLab] Evaluation failed:', err);
      if (resultsGrid) {
        resultsGrid.innerHTML = `
          <div class="empty-state" style="grid-column: 1 / -1">
            <div class="empty-icon">⚠️</div>
            <h3>Live Hybrid Evaluation Failed</h3>
            <p>${escapeHtml(err.message)}</p>
          </div>
        `;
      }
    } finally {
      isEvaluating = false;
      if (btnRecompute) {
        btnRecompute.disabled = false;
        btnRecompute.innerHTML = '<span>⚡</span> Run Live Hybrid';
      }
    }
  }

  /** Render the Latency Dashboard, Performance Summary Strip, and Visual Timing Bar */
  function renderTimingHUD(timings, totalMs, clientTotalMs, isCalibOn) {
    const totalEl = document.getElementById('hud-total-ms');
    if (totalEl) {
      let resBadge = '';
      if (timings.resource) {
        const r = timings.resource;
        const ram = r.ram_mb ? `${Math.round(r.ram_mb)} MB RAM` : '';
        const cpu = (r.cpu_percent !== null && r.cpu_percent !== undefined) ? `${r.cpu_percent.toFixed(1)}% CPU` : '';
        const parts = [ram, cpu].filter(Boolean).join(' | ');
        if (parts) {
          resBadge = `<span class="badge" style="background:rgba(168,85,247,0.15);color:#c084fc;border:1px solid rgba(168,85,247,0.35);font-size:0.75rem;">⚙️ ${parts}</span>`;
        }
      }
      const calibNote = isCalibOn 
        ? `<span class="badge" style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.35);font-size:0.75rem;">Platt: ON</span>`
        : `<span class="badge" style="background:rgba(234,179,8,0.15);color:#eab308;border:1px solid rgba(234,179,8,0.35);font-size:0.75rem;">Platt: OFF (Raw Scores)</span>`;
      totalEl.innerHTML = `⏱️ <strong>${totalMs} <span style="font-size:0.85rem;color:#38bdf8;font-weight:700;">ms</span></strong> <span style="font-size:0.75rem;font-weight:400;color:var(--text-muted);">(Network Roundtrip: ${clientTotalMs} ms)</span> ${calibNote} ${resBadge}`;
    }

    // Capacity & Performance Summary Strip
    const perfThroughput = document.getElementById('hud-perf-throughput');
    if (perfThroughput) {
      const rps = timings.throughput_rps || (totalMs > 0 ? Math.round(1000 / totalMs * 10) / 10 : 40);
      perfThroughput.textContent = `~${rps} req/s`;
    }
    const perfCorpus = document.getElementById('hud-perf-corpus');
    if (perfCorpus) {
      const corpusCount = timings.corpus_size || (allArticles.length || 4177);
      perfCorpus.textContent = `${corpusCount.toLocaleString()} docs`;
    }
    const perfPool = document.getElementById('hud-perf-pool');
    if (perfPool) {
      const poolCount = timings.candidate_count || 20;
      const injected = timings.recency_injected ? ` (+${timings.recency_injected} recency)` : '';
      perfPool.textContent = `${poolCount} items${injected}`;
    }
    const perfRtt = document.getElementById('hud-perf-rtt');
    if (perfRtt) {
      perfRtt.textContent = `${clientTotalMs} ms`;
    }
    const perfOverhead = document.getElementById('hud-perf-overhead');
    if (perfOverhead) {
      const overhead = Math.max(0, Math.round((clientTotalMs - totalMs) * 10) / 10);
      perfOverhead.textContent = `${overhead.toFixed(1)} ms`;
    }

    // Metric cards: bold number + prominent uppercase ms badge
    const setMs = (id, val, isCalib = false) => {
      const el = document.getElementById(id);
      if (!el) return;
      if (isCalib && !isCalibOn) {
        el.innerHTML = `<span class="ms-num">0.00</span><span class="ms-unit off">OFF</span>`;
      } else {
        const numStr = (val != null ? Number(val) : 0).toFixed(2);
        el.innerHTML = `<span class="ms-num">${numStr}</span><span class="ms-unit">ms</span>`;
      }
    };

    const setSubMs = (id, val) => {
      const el = document.getElementById(id);
      if (!el) return;
      const numStr = (val != null ? Number(val) : 0).toFixed(2);
      el.innerHTML = `${numStr}<span class="ms-unit-sub">ms</span>`;
    };

    setMs('stage-prep-ms', timings.prep_ms);
    setMs('stage-retrieval-ms', timings.retrieval_ms);
    setSubMs('stage-dense-ms', timings.dense_scan_ms != null ? timings.dense_scan_ms : timings.dense_ms);
    setSubMs('stage-query-embed-ms', timings.query_embed_ms || 0.0);
    setSubMs('stage-bm25-ms', timings.bm25_ms);
    setMs('stage-blend-ms', timings.blend_ms);
    setMs('stage-boosts-ms', timings.boosts_ms);
    setMs('stage-mmr-ms', timings.mmr_ms);
    setMs('stage-calib-ms', timings.calibration_ms, true);

    // Visual proportional timing bar
    const bar = document.getElementById('hud-timing-bar');
    if (!bar) return;

    const stages = [
      { key: 'prep', name: 'Query Prep', ms: timings.prep_ms || 0.1 },
      { key: 'retrieval', name: 'Retrieval Scan', ms: timings.retrieval_ms || 0.5 },
      { key: 'blend', name: 'Hybrid Blending', ms: timings.blend_ms || 0.1 },
      { key: 'boosts', name: 'Domain Boosts', ms: timings.boosts_ms || 0.2 },
      { key: 'mmr', name: 'MMR Diversity', ms: timings.mmr_ms || 0.2 },
      { key: 'calibration', name: isCalibOn ? 'Platt Calibration & Evidence' : 'Calibration (Disabled)', ms: isCalibOn ? (timings.calibration_ms || 1.0) : 0.05 },
    ];

    const sumMs = stages.reduce((acc, s) => acc + s.ms, 0) || 1.0;
    bar.innerHTML = stages.map(s => {
      const pct = Math.max(2, Math.round((s.ms / sumMs) * 100));
      return `
        <div class="timing-segment timing-seg-${s.key}" 
             style="width:${pct}%;" 
             title="${s.name}: ${s.ms.toFixed(2)} ms (${pct}%)">
        </div>
      `;
    }).join('');
  }

  /** Build a clean compact news card: thumbnail + title + date + Inspect button */
  function buildHybridCard(rec, idx, isCalibOn, sourceArt) {
    const art = rec.article || rec;
    const title = art.title || 'Untitled Article';
    const image = art.image || '';
    const url = art.url || '';
    const section = (art.section || 'General').toUpperCase();
    const rank = rec.rank || (idx + 1);
    const rankRaw = rec.rank_raw || rank;
    const hasShift = rankRaw !== rank;
    const score = rec.score != null ? rec.score : null;

    const color = (typeof sectionColor === 'function') ? sectionColor(art.section) : '#eab308';
    const emoji = (typeof sectionEmoji === 'function') ? sectionEmoji(art.section) : '📰';

    // Format date nicely
    let dateStr = '';
    if (art.date) {
      try {
        const d = new Date(art.date);
        dateStr = d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
      } catch (e) { dateStr = String(art.date).slice(0, 10); }
    }

    // Thumbnail
    const thumbImg = image
      ? `<img src="${escapeHtml(image)}" alt="${escapeHtml(title)}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'hlc-fallback\\'>${emoji}</div>'" />`
      : `<div class="hlc-fallback">${emoji}</div>`;

    // Overlay Badges: Rank (with raw shift if MMR changed it)
    let rankTitle = `Rank #${rank}`;
    let rankLabel = `#${rank}`;
    if (hasShift) {
      rankLabel = `#${rank} <span class="rank-raw-shift">(#${rankRaw})</span>`;
      rankTitle = `MMR Rank #${rank} (Pure Relevance was #${rankRaw})`;
    }
    const rankBadge = `<span class="hybrid-card-rank-badge" title="${escapeHtml(rankTitle)}">${rankLabel}</span>`;

    // Confidence / Score badge
    let confBadge = '';
    if (isCalibOn && score != null) {
      const pct = Math.round(score * 100);
      const dotColor = score >= 0.65 ? '#4ade80' : score >= 0.35 ? '#fbbf24' : '#94a3b8';
      confBadge = `<span class="hlc-conf-pill" style="background:rgba(15,23,42,0.85);border:1px solid ${dotColor}44;color:${dotColor};" title="Confidence: ${pct}%"><span class="hlc-conf-dot" style="background:${dotColor};"></span>${pct}%</span>`;
    }

    // Section badge at bottom of thumb
    const secStrip = `
      <div class="card-section-strip">
        <span class="badge badge-section" style="border-color:${color}55;color:${color};background:rgba(15,23,42,0.88);backdrop-filter:blur(4px);font-weight:700;">${escapeHtml(section)}</span>
      </div>
    `;

    // Title — clicking opens the article
    const titleHTML = url
      ? `<a class="hlc-title" href="${escapeHtml(url)}" target="_blank" rel="noopener" title="${escapeHtml(title)}">${escapeHtml(title)}</a>`
      : `<p class="hlc-title" title="${escapeHtml(title)}">${escapeHtml(title)}</p>`;

    // Date line
    const dateHTML = dateStr
      ? `<span class="hlc-date">📅 ${escapeHtml(dateStr)}</span>`
      : '';

    // Inspect button — sole action on the card
    const inspectBtn = `
      <button type="button" class="btn-hlc-inspect" title="Click to inspect the full evidence breakdown for this recommendation">
        🔍 Inspect
      </button>
    `;

    // Assemble card
    const card = document.createElement('article');
    card.className = 'hlc-card animate-scale';
    card.dataset.id = art.id || '';
    card.dataset.rank = rank;

    card.innerHTML = `
      <div class="hlc-thumb">
        ${thumbImg}
        <div class="hlc-thumb-overlay">
          ${rankBadge}
          ${confBadge}
        </div>
        ${secStrip}
      </div>
      <div class="hlc-body">
        ${titleHTML}
        ${dateHTML}
        ${inspectBtn}
      </div>
    `;

    // Connect Inspect button → Evaluator modal
    const evalBtn = card.querySelector('.btn-hlc-inspect');
    if (evalBtn) {
      evalBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (window.Evaluator && typeof Evaluator.showModal === 'function') {
          Evaluator.showModal(rec, sourceArt, art);
        }
      });
    }

    return card;
  }

  /** Render rich news recommendation cards for Live Hybrid recommendations */
  function renderHybridRecommendations(recs, container, isCalibOn, sourceArt) {
    if (!container) return;
    container.innerHTML = '';

    if (!recs || !recs.length) {
      container.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-icon">📭</div>
          <h3>No Recommendations Found</h3>
          <p>Try enabling domain boosts or selecting a different source article.</p>
        </div>
      `;
      return;
    }

    // ── Live Hybrid mode banner ──────────────────────────────────────────────
    const banner = document.createElement('div');
    banner.style.cssText = 'grid-column:1/-1; margin-bottom:8px;';
    const calibBadge = isCalibOn
      ? `<span style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.35);padding:2px 8px;border-radius:4px;font-size:0.76rem;font-weight:700;">Platt ON</span>`
      : `<span style="background:rgba(234,179,8,0.15);color:#eab308;border:1px solid rgba(234,179,8,0.35);padding:2px 8px;border-radius:4px;font-size:0.76rem;font-weight:700;">Platt OFF — Raw Scores</span>`;
    const stratBadge = currentCandidateStrategy === 'pure_rrf'
      ? `<span style="background:rgba(168,85,247,0.15);color:#c084fc;border:1px solid rgba(168,85,247,0.35);padding:2px 8px;border-radius:4px;font-size:0.76rem;font-weight:700;">🔀 Pure 20 RRF</span>`
      : `<span style="background:rgba(236,72,153,0.15);color:#ec4899;border:1px solid rgba(236,72,153,0.35);padding:2px 8px;border-radius:4px;font-size:0.76rem;font-weight:700;">🎯 Quota (12/5/3)</span>`;
    banner.innerHTML = `
      <div style="font-size:0.78rem;color:var(--text-muted);display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:4px 0;">
        <span style="font-weight:700;color:var(--color-primary);">⚡ Live Hybrid Recommendations</span>
        <span style="opacity:0.5;">·</span>
        ${stratBadge}
        <span style="opacity:0.5;">·</span>
        ${calibBadge}
        <span style="opacity:0.5;">·</span>
        <span style="color:var(--text-primary);font-weight:600;">${recs.length} stories</span>
        <span style="opacity:0.5;">·</span>
        <span style="font-size:0.74rem;color:#c084fc;">Click <strong>🔬 Eval</strong> on any card to view its complete score journey</span>
      </div>
    `;
    container.appendChild(banner);

    // ── Render one rich news card per recommendation ────────────────────────
    recs.forEach((rec, idx) => {
      const card = buildHybridCard(rec, idx, isCalibOn, sourceArt);
      container.appendChild(card);
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  /** Called when the user clicks the Hybrid Lab tab */
  function onTabActivated() {
    if (!hasEvaluatedOnce && activeArticle) {
      // Do not auto-evaluate: leave it ready for user to click "Run Live Hybrid"
      console.log('[HybridLab] Tab opened. Active story:', activeArticle.title);
    }
  }

  return {
    init,
    selectArticle,
    evaluate,
    onTabActivated,
  };
})();

// Explicitly attach to window for cross-module coordination with Tabs and Randomizer
window.HybridLab = HybridLab;
