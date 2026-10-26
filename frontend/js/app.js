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
  const topKSlider = document.getElementById('top-k-slider');
  const topKValue = document.getElementById('top-k-value');
  const sectionFilter = document.getElementById('section-filter');
  const methodSelect = document.getElementById('method-select');
  const useBoostsToggle = document.getElementById('use-boosts');
  const useDiversityToggle = document.getElementById('use-diversity');

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
      State.set({ activeSection: e.target.value });
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

    // On click, scroll to or create the widget in the main feed
    widget.addEventListener('click', () => {
      const feed = document.getElementById('article-feed');
      if (!feed) return;

      // Try find existing widget in feed
      const existing = feed.querySelector(`.feed-widget[data-id="${article.id}"]`);
      if (existing) {
        existing.scrollIntoView({ behavior: 'smooth', block: 'center' });
        existing.classList.add('widget-highlighted');
        setTimeout(() => existing.classList.remove('widget-highlighted'), 2600);
      } else {
        // Use Tabs to prepend the widget
        if (window.Tabs && window.Tabs.buildFeed) {
          const storyFinderInput = document.getElementById('story-finder-input');
          if (storyFinderInput) {
            storyFinderInput.value = article.title;
            storyFinderInput.dispatchEvent(new Event('input'));
            // Simulate selecting first match after a brief delay
            setTimeout(() => {
              const firstItem = document.querySelector('.story-finder-item');
              if (firstItem) firstItem.click();
            }, 300);
          }
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

    // 1. Fetch Engine Status
    // 'let' not 'const': checkReaderDatabaseState() below refreshes this in place.
    let statusData = await Api.status();
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

    // 5. Initialize Editor & PostgreSQL Pipeline Hub (Editor is a const, always available)
    await Editor.init();

    // 6. Role Switcher Controller (Editor vs Reader)
    const roleEditorBtn = document.getElementById('role-btn-editor');
    const roleReaderBtn = document.getElementById('role-btn-reader');
    const editorView = document.getElementById('editor-view');
    const readerView = document.getElementById('reader-view');
    const readerEmptyBanner = document.getElementById('reader-empty-banner');
    const btnEmptyGotoEditor = document.getElementById('btn-empty-goto-editor');

    async function checkReaderDatabaseState() {
      try {
        statusData = await Api.status();
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

    async function setAppRole(role) {
      const isEditor = role === 'editor';
      if (roleEditorBtn) roleEditorBtn.classList.toggle('active', isEditor);
      if (roleReaderBtn) roleReaderBtn.classList.toggle('active', !isEditor);
      if (editorView) editorView.classList.toggle('hidden', !isEditor);
      if (readerView) readerView.classList.toggle('hidden', isEditor);
      localStorage.setItem('app_role', role);

      if (isEditor) {
        Editor.refreshDbHealth();
      } else {
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
    if (loadingText) {
      loadingText.innerHTML = `
        <span style="color:#ef4444">Could not connect to API backend</span><br/>
        <small style="color:var(--text-muted);font-size:0.8rem">Ensure the server is running at <code>http://localhost:5000</code></small>
      `;
    }

    setTimeout(() => {
      if (loadingOverlay) loadingOverlay.classList.add('fade-out');
      if (appContainer) appContainer.classList.remove('hidden');
      showToast('Backend offline. Please start the API server.', 'warn', 6000);
    }, 1500);
  }
});
