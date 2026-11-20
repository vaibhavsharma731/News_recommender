/**
 * app.js — Main application orchestrator.
 * Coordinates themes, sidebar, controls, initial load, and event listeners.
 */

document.addEventListener('DOMContentLoaded', async () => {

  // ─── 1. THEME TOGGLE ────────────────────────────────────────────────────────
  const themeToggle = document.getElementById('theme-toggle');
  const themeIcon = document.getElementById('theme-icon');

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (themeIcon) {
      themeIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
    localStorage.setItem('theme', theme);
  }

  const savedTheme = localStorage.getItem('theme') || 'dark';
  applyTheme(savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      State.set({ theme: next });
    });
  }

  // ─── 2. MOBILE SIDEBAR TOGGLE ───────────────────────────────────────────────
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });

    // Close when clicking outside on mobile
    document.addEventListener('click', (e) => {
      if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target) && sidebar.classList.contains('open')) {
        sidebar.classList.remove('open');
      }
    });
  }

  // ─── 3. RECSYS CONTROLS BINDING ─────────────────────────────────────────────
  const readerDatasetFilter = document.getElementById('reader-dataset-filter');
  const readerDatasetBadge = document.getElementById('reader-dataset-badge');
  const topKSlider = document.getElementById('top-k-slider');
  const topKValue = document.getElementById('top-k-value');
  const sectionFilter = document.getElementById('section-filter');
  const methodSelect = document.getElementById('method-select');
  const useBoostsToggle = document.getElementById('use-boosts');
  const useDiversityToggle = document.getElementById('use-diversity');

  if (readerDatasetFilter) {
    readerDatasetFilter.addEventListener('change', async (e) => {
      const ds = e.target.value;
      State.set({ activeDataset: ds });
      if (readerDatasetBadge) {
        readerDatasetBadge.textContent = ds === 'default' ? 'Production' : ds;
      }
      if (window.Tabs && typeof window.Tabs.clearArticlesCache === 'function') {
        window.Tabs.clearArticlesCache();
        window.Tabs.reloadCurrentTab();
      }
      if (window.shuffleRandomizer) {
        window.shuffleRandomizer();
      }
      if (typeof HybridLab !== 'undefined' && typeof HybridLab.init === 'function') {
        try { await HybridLab.init(); } catch (_) {}
      }
    });
  }

  if (topKSlider && topKValue) {
    topKSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      topKValue.textContent = val;
      State.set({ topK: val });
    });
    topKSlider.addEventListener('change', () => {
      triggerActiveTabReload();
    });
  }

  if (sectionFilter) {
    sectionFilter.addEventListener('change', (e) => {
      const sec = e.target.value;
      State.set({ activeSection: sec });
      const pillContainer = document.getElementById('reader-category-pills');
      if (pillContainer) {
        pillContainer.querySelectorAll('.cat-pill').forEach(btn => {
          btn.classList.toggle('active', btn.dataset.section === sec);
        });
      }
      triggerActiveTabReload();
    });
  }

  if (methodSelect) {
    methodSelect.addEventListener('change', (e) => {
      State.set({ method: e.target.value });
      triggerActiveTabReload();
    });
  }

  if (useBoostsToggle) {
    useBoostsToggle.addEventListener('change', (e) => {
      State.set({ useBoosts: e.target.checked });
      triggerActiveTabReload();
    });
  }

  if (useDiversityToggle) {
    useDiversityToggle.addEventListener('change', (e) => {
      State.set({ diversify: e.target.checked });
      triggerActiveTabReload();
    });
  }

  // ─── 4. GLOBAL SEARCH BAR IN NAVBAR ────────────────────────────────────────
  const globalSearchInput = document.getElementById('global-search-input');
  const searchClearBtn = document.getElementById('search-clear-btn');

  if (globalSearchInput) {
    globalSearchInput.addEventListener('input', (e) => {
      if (searchClearBtn) {
        searchClearBtn.classList.toggle('hidden', !e.target.value);
      }
    });

    globalSearchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const query = globalSearchInput.value.trim();
        if (query) {
          Tabs.performSearch(query);
        }
      }
    });
  }

  if (searchClearBtn && globalSearchInput) {
    searchClearBtn.addEventListener('click', () => {
      globalSearchInput.value = '';
      searchClearBtn.classList.add('hidden');
      globalSearchInput.focus();
    });
  }

  // ─── 5. RANDOMIZER BAR (RIGHT PANEL) ─────────────────────────────────────
  const randomizerFeed = document.getElementById('randomizer-feed');
  const randomizerShuffleBtn = document.getElementById('randomizer-shuffle-btn');
  const randomizerCountEl = document.getElementById('randomizer-count');
  const RANDOMIZER_SIZE = 6; // how many random articles to show at a time

  function buildRandomizerWidget(article) {
    const emoji = sectionEmoji(article.section);
    const color = sectionColor(article.section);
    const thumb = article.image
      ? `<img src="${article.image}" alt="" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'randomizer-widget-thumb-fallback\\'>${emoji}</div>'" />`
      : `<div class="randomizer-widget-thumb-fallback">${emoji}</div>`;

    const widget = document.createElement('div');
    widget.className = 'randomizer-widget';
    widget.title = article.title;
    widget.innerHTML = `
      <div class="randomizer-widget-thumb">
        ${thumb}
        <div class="randomizer-widget-section" style="color:${color}">${(article.section || 'General').toUpperCase()}</div>
      </div>
      <div class="randomizer-widget-body">
        <div class="randomizer-widget-title">${article.title}</div>
        <div class="randomizer-widget-meta">
          <span>📅 ${article.date || 'Recent'}</span>
        </div>
      </div>`;

    // On click, route according to the active tab:
    // If on Live Hybrid slide -> add article to the live hybrid slide and wait for user to click Run Live Hybrid
    // If on Precomputation slide -> show article in the similar articles feed with precomputed recommendations
    widget.addEventListener('click', () => {
      const activePanel = document.querySelector('.tab-panel.active');
      const activeTab = (activePanel ? activePanel.id.replace('tab-', '') : null) 
        || (window.State ? window.State.get().activeTab : 'similar');

      if (activeTab === 'hybrid-lab' && window.HybridLab && typeof window.HybridLab.selectArticle === 'function') {
        window.HybridLab.selectArticle(article, false);
      } else {
        if (window.Tabs && typeof window.Tabs.selectArticle === 'function') {
          window.Tabs.selectArticle(article);
        }
      }
    });

    return widget;
  }

  function renderRandomizerSkeletons() {
    if (!randomizerFeed) return;
    randomizerFeed.innerHTML = '';
    for (let i = 0; i < RANDOMIZER_SIZE; i++) {
      const skel = document.createElement('div');
      skel.className = 'randomizer-skeleton';
      skel.innerHTML = `
        <div class="randomizer-skeleton-thumb"></div>
        <div class="randomizer-skeleton-body">
          <div class="randomizer-skeleton-line"></div>
          <div class="randomizer-skeleton-line"></div>
        </div>`;
      randomizerFeed.appendChild(skel);
    }
  }

  /** Fisher-Yates shuffle to pick N random articles from the cache */
  function pickRandom(arr, n) {
    const copy = arr.slice();
    const result = [];
    for (let i = 0; i < n && copy.length > 0; i++) {
      const idx = Math.floor(Math.random() * copy.length);
      result.push(copy.splice(idx, 1)[0]);
    }
    return result;
  }

  async function shuffleRandomizer() {
    if (!randomizerFeed) return;

    // Add animation class to button
    if (randomizerShuffleBtn) {
      randomizerShuffleBtn.classList.add('shuffling');
      setTimeout(() => randomizerShuffleBtn.classList.remove('shuffling'), 600);
    }

    // Show skeletons briefly for visual feedback
    renderRandomizerSkeletons();

    // Wait for articles to be loaded
    let articles = [];
    try {
      // Reuse the Tabs article cache if available
      if (window.Tabs && window.Tabs._allArticlesCache) {
        articles = window.Tabs._allArticlesCache;
      }
      if (!articles || !articles.length) {
        const data = await Api.articles({ page: 1, perPage: 100, sort: 'confidence' });
        articles = data.articles || [];
      }
    } catch (e) {
      console.warn('Randomizer: could not load articles', e);
    }

    if (!articles.length) {
      randomizerFeed.innerHTML = `
        <div class="randomizer-empty">
          <span>📰</span>
          <p>No articles available yet. Ingest data from the Editor Studio.</p>
        </div>`;
      return;
    }

    // Update count
    if (randomizerCountEl) {
      randomizerCountEl.textContent = `${articles.length} stories`;
    }

    // Pick random articles
    const picks = pickRandom(articles, RANDOMIZER_SIZE);

    // Small delay for skeleton-to-widget transition
    await new Promise(r => setTimeout(r, 200));

    randomizerFeed.innerHTML = '';
    picks.forEach(art => {
      randomizerFeed.appendChild(buildRandomizerWidget(art));
    });
  }

  // Shuffle button event
  if (randomizerShuffleBtn) {
    randomizerShuffleBtn.addEventListener('click', shuffleRandomizer);
  }

  // Expose shuffleRandomizer so it can be called after data loads
  window.shuffleRandomizer = shuffleRandomizer;

  // ─── 6. HELPER: RELOAD ACTIVE TAB ──────────────────────────────────────────
  function triggerActiveTabReload() {
    Tabs.reloadCurrentTab();
  }

  // ─── 7. SYSTEM INITIALIZATION & API BOOTSTRAP ─────────────────────────────
  const loadingOverlay = document.getElementById('loading-overlay');
  const appContainer = document.getElementById('app');
  const loadingText = document.getElementById('loading-text');

  try {
    if (loadingText) loadingText.textContent = 'Connecting to Recommendation Engine...';

    // 1. Fetch Engine Status with resilient retry
    // 'let' not 'const': checkReaderDatabaseState() below refreshes this in place.
    let statusData = null;
    for (let attempt = 1; attempt <= 4; attempt++) {
      try {
        if (loadingText && attempt > 1) {
          loadingText.textContent = `Connecting to API backend (attempt ${attempt}/4)...`;
        }
        statusData = await Api.status();
        if (statusData) break;
      } catch (e) {
        if (attempt === 4) throw e;
        await new Promise(r => setTimeout(r, 1200));
      }
    }
    State.set({ engineInfo: statusData });

    const engineNameEl = document.getElementById('engine-name');
    const totalArticlesEl = document.getElementById('total-articles');
    if (engineNameEl) engineNameEl.textContent = statusData.engine || 'RecSys v1.0';
    if (totalArticlesEl) totalArticlesEl.textContent = `${(statusData.total_articles || 0).toLocaleString()} stories`;

    // 2. Fetch Sections & Categories
    if (loadingText) loadingText.textContent = 'Loading news categories...';
    const sectionsData = await Api.sections();
    const sections = sectionsData.sections || [];
    State.set({ allSections: sections });

    // Populate Sidebar Category Dropdown
    if (sectionFilter) {
      sections.forEach(sec => {
        const opt = document.createElement('option');
        opt.value = sec;
        opt.textContent = sec.charAt(0).toUpperCase() + sec.slice(1);
        sectionFilter.appendChild(opt);
      });
    }

    // Category filtering is now handled by the sidebar dropdown only.
    // The right panel is the Randomizer Bar (populated after Tabs init).

    // 3. Initialize Hero Carousel
    if (loadingText) loadingText.textContent = 'Loading top trending stories...';
    try { await Hero.init(); } catch(e) { console.warn('Hero init skipped:', e.message); }

    // 4. Initialize Tabs
    Tabs.init();
    if (typeof Tabs.populateCategoryPills === 'function') {
      Tabs.populateCategoryPills(sections);
    }
    if (typeof HybridLab !== 'undefined' && typeof HybridLab.init === 'function') {
      await HybridLab.init();
    }

    // 5. Initialize Editor & PostgreSQL Pipeline Hub (Editor is a const, always available)
    await Editor.init();

    // 5B. Initialize Monitoring & Performance Hub
    if (typeof Monitoring !== 'undefined' && typeof Monitoring.init === 'function') {
      await Monitoring.init();
    }

    // 5C. Initialize Side-by-Side Algorithm Comparison
    if (typeof Compare !== 'undefined' && typeof Compare.init === 'function') {
      Compare.init();
    }

    // 6. Role Switcher Controller (Editor vs Reader vs Monitor)
    const roleEditorBtn = document.getElementById('role-btn-editor');
    const roleReaderBtn = document.getElementById('role-btn-reader');
    const roleMonitorBtn = document.getElementById('role-btn-monitor');
    const editorView = document.getElementById('editor-view');
    const readerView = document.getElementById('reader-view');
    const monitorView = document.getElementById('monitoring-view');
    const readerEmptyBanner = document.getElementById('reader-empty-banner');
    const btnEmptyGotoEditor = document.getElementById('btn-empty-goto-editor');


    async function checkReaderDatabaseState() {
      try {
        const curPl = (typeof State !== 'undefined' ? State.get().modelVersion : '384d') || '384d';
        statusData = await Api.status(curPl);
        State.set({ engineInfo: statusData });
        if (engineNameEl) engineNameEl.textContent = statusData.engine || 'RecSys v1.0';
        if (totalArticlesEl) {
          totalArticlesEl.textContent = `${(statusData.total_articles || 0).toLocaleString()} stories`;
        }
      } catch (e) {
        console.warn('Could not refresh engine status:', e);
      }

      const isDbEmpty = !statusData || statusData.database_empty || statusData.total_articles === 0;
      if (readerEmptyBanner) {
        readerEmptyBanner.classList.toggle('hidden', !isDbEmpty);
      }
      const heroSection = document.getElementById('hero-section');
      const tabsNav = document.getElementById('tabs-nav');
      const similarPanel = document.getElementById('tab-similar');
      if (heroSection) heroSection.style.display = isDbEmpty ? 'none' : '';
      if (tabsNav) tabsNav.style.display = isDbEmpty ? 'none' : '';
      if (similarPanel) similarPanel.style.display = isDbEmpty ? 'none' : '';
      return !isDbEmpty;
    }

    window.checkReaderDatabaseState = checkReaderDatabaseState;

    async function setAppRole(role) {
      const isEditor = role === 'editor';
      const isReader = role === 'reader';
      const isMonitor = role === 'monitor';

      if (roleEditorBtn) roleEditorBtn.classList.toggle('active', isEditor);
      if (roleReaderBtn) roleReaderBtn.classList.toggle('active', isReader);
      if (roleMonitorBtn) roleMonitorBtn.classList.toggle('active', isMonitor);

      if (editorView) editorView.classList.toggle('hidden', !isEditor);
      if (readerView) readerView.classList.toggle('hidden', !isReader);
      if (monitorView) monitorView.classList.toggle('hidden', !isMonitor);
      localStorage.setItem('app_role', role);

      if (isMonitor) {
        if (typeof Monitoring !== 'undefined') Monitoring.activate();
      } else {
        if (typeof Monitoring !== 'undefined') Monitoring.deactivate();
      }

      if (isEditor) {
        Editor.refreshDbHealth();
      } else if (isReader) {
        const hasArticles = await checkReaderDatabaseState();
        if (hasArticles) {
          try { await Hero.init(); } catch (e) { console.warn('Hero init skipped:', e.message); }
          Tabs.populateArticlePicker();
          // Populate the randomizer bar after a short delay so articles cache is ready
          setTimeout(() => shuffleRandomizer(), 800);
        }
      }
    }

    window.setAppRole = setAppRole;
    window.refreshReaderAfterIngest = async () => {
      const hasArticles = await checkReaderDatabaseState();
      if (hasArticles) {
        try { await Hero.init(); } catch (e) { console.warn('Hero init skipped:', e.message); }
        Tabs.populateArticlePicker();
        setTimeout(() => shuffleRandomizer(), 800);
      }
    };

    if (roleEditorBtn) roleEditorBtn.addEventListener('click', () => setAppRole('editor'));
    if (roleReaderBtn) roleReaderBtn.addEventListener('click', () => setAppRole('reader'));
    if (roleMonitorBtn) roleMonitorBtn.addEventListener('click', () => setAppRole('monitor'));
    if (btnEmptyGotoEditor) btnEmptyGotoEditor.addEventListener('click', () => setAppRole('editor'));

    // Default to editor if database is empty, otherwise restore saved role
    const initialRole = (!statusData || statusData.database_empty || statusData.total_articles === 0)
      ? 'editor'
      : (localStorage.getItem('app_role') || 'editor');
    setAppRole(initialRole);

    // Reveal App
    if (loadingOverlay) loadingOverlay.classList.add('fade-out');
    if (appContainer) appContainer.classList.remove('hidden');

    setTimeout(() => {
      if (loadingOverlay) loadingOverlay.remove();
    }, 400);


  } catch (err) {
    console.error('Failed to initialize app with backend:', err);
    // Only a genuine network failure means the API server is unreachable. Any
    // other exception (e.g. a JavaScript ReferenceError in a frontend module)
    // used to be reported as "Backend offline", which sent debugging in the
    // wrong direction. Distinguish the two so the real cause is visible.
    const msg = String(err && err.message ? err.message : err);
    const isNetworkError = (err instanceof TypeError) &&
      /fetch|network|load failed|failed to fetch/i.test(msg);
    const title = isNetworkError ? 'Could not connect to API backend' : 'Frontend failed to initialize';
    const hint = isNetworkError
      ? 'Ensure the server is running: <code>python api/server.py</code>'
      : 'The API responded, but a frontend script threw an error. Open the browser console for details.';

    if (loadingText) {
      loadingText.innerHTML = `
        <span style="color:#ef4444;font-weight:600">${title}</span><br/>
        <small style="color:var(--text-muted);font-size:0.8rem">${msg || 'Unknown error'}</small><br/>
        <small style="color:var(--text-muted);font-size:0.75rem">${hint}</small><br/>
        <button onclick="location.reload()" style="margin-top:12px;padding:6px 14px;background:#3b82f6;color:white;border:none;border-radius:6px;cursor:pointer;font-size:0.85rem">🔄 Retry Now</button>
      `;
    }

    setTimeout(() => {
      if (loadingOverlay) loadingOverlay.classList.add('fade-out');
      if (appContainer) appContainer.classList.remove('hidden');
      showToast(
        isNetworkError
          ? 'Backend offline. Please start the API server with: python api/server.py'
          : `Frontend initialization error: ${msg}`,
        'warn',
        7000
      );
    }, 2000);
  }
});
