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

  // ─── 3. LOAD SAVED USER HISTORY ─────────────────────────────────────────────
  State.loadHistory();
  refreshHistoryUI();

  // ─── 4. RECSYS CONTROLS BINDING ─────────────────────────────────────────────
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
      const sec = e.target.value;
      State.set({ activeSection: sec });
      syncCategoryPills(sec);
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
      State.set({ useDiversity: e.target.checked });
      triggerActiveTabReload();
    });
  }

  // ─── 5. GLOBAL SEARCH BAR IN NAVBAR ────────────────────────────────────────
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

  // ─── 6. CATEGORIES & RIGHT PANEL PILLS ─────────────────────────────────────
  const categoryPillsContainer = document.getElementById('category-pills');
  const viewAllBtn = document.getElementById('rp-view-all');

  function syncCategoryPills(activeSec) {
    if (!categoryPillsContainer) return;
    categoryPillsContainer.querySelectorAll('.cat-pill').forEach(pill => {
      pill.classList.toggle('active', (pill.dataset.section || '') === (activeSec || ''));
    });
  }

  if (viewAllBtn) {
    viewAllBtn.addEventListener('click', () => {
      State.set({ activeSection: '' });
      if (sectionFilter) sectionFilter.value = '';
      syncCategoryPills('');
      triggerActiveTabReload();
    });
  }

  // ─── 7. HISTORY CLEAR & REFRESH BUTTONS ─────────────────────────────────────
  const clearHistoryBtn = document.getElementById('clear-history-btn');
  const rpClearBtn = document.getElementById('rp-clear-btn');
  const refreshBriefBtn = document.getElementById('refresh-brief-btn');

  function handleClearHistory() {
    State.clearHistory();
    State.saveHistory();
    refreshHistoryUI();
    showToast('Reading history cleared. Feed reset.', 'info');
    triggerActiveTabReload();
  }

  if (clearHistoryBtn) clearHistoryBtn.addEventListener('click', handleClearHistory);
  if (rpClearBtn) rpClearBtn.addEventListener('click', handleClearHistory);

  if (refreshBriefBtn) {
    refreshBriefBtn.addEventListener('click', () => {
      showToast('Refreshing recommendations…', 'info');
      Tabs.loadForYou();
    });
  }

  // ─── 8. HELPER: RELOAD ACTIVE TAB ──────────────────────────────────────────
  function triggerActiveTabReload() {
    const { activeTab } = State.get();
    if (activeTab === 'foryou') {
      Tabs.loadForYou();
    } else if (activeTab === 'search') {
      const q = document.getElementById('search-query-input')?.value;
      if (q) Tabs.performSearch(q);
    }
  }

  // ─── 9. STATE SUBSCRIPTION ─────────────────────────────────────────────────
  let prevHistoryLen = State.get().userHistory.length;
  State.subscribe((newState) => {
    if (newState.userHistory.length !== prevHistoryLen) {
      prevHistoryLen = newState.userHistory.length;
      if (newState.activeTab === 'foryou') {
        Tabs.loadForYou();
      }
    }
  });

  // ─── 10. SYSTEM INITIALIZATION & API BOOTSTRAP ─────────────────────────────
  const loadingOverlay = document.getElementById('loading-overlay');
  const appContainer = document.getElementById('app');
  const loadingText = document.getElementById('loading-text');

  try {
    if (loadingText) loadingText.textContent = 'Connecting to AI Engine...';

    // 1. Fetch Engine Status
    const statusData = await Api.status();
    State.set({ engineInfo: statusData });

    const engineNameEl = document.getElementById('engine-name');
    const totalArticlesEl = document.getElementById('total-articles');
    if (engineNameEl) engineNameEl.textContent = statusData.engine || 'RecSys v1.0';
    if (totalArticlesEl) totalArticlesEl.textContent = `${statusData.total_articles.toLocaleString()} stories`;

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

    // Populate Right-Panel Category Pills
    if (categoryPillsContainer) {
      buildCategoryPills(sections, categoryPillsContainer, (selectedSection) => {
        State.set({ activeSection: selectedSection });
        if (sectionFilter) sectionFilter.value = selectedSection;
        triggerActiveTabReload();
      });
    }

    // 3. Initialize Hero Carousel
    if (loadingText) loadingText.textContent = 'Loading top trending stories...';
    await Hero.init();

    // 4. Initialize Tabs
    Tabs.init();

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
        <span style="color:#ef4444">Could not connect to Flask API backend</span><br/>
        <small style="color:var(--text-muted);font-size:0.8rem">Ensure Flask is running at <code>http://localhost:5000</code></small>
      `;
    }

    // Even if backend fails, reveal app shell so user can see UI
    setTimeout(() => {
      if (loadingOverlay) loadingOverlay.classList.add('fade-out');
      if (appContainer) appContainer.classList.remove('hidden');
      showToast('Backend offline. Please start the Flask API server.', 'warn', 6000);
    }, 1500);
  }
});
