/**
 * tabs.js — Tab switching and tab-specific recommendation flows.
 * Handles "For You", "Similar Articles", and "AI Search".
 */

const Tabs = (() => {
  let allArticlesCache = [];

  // ─── TAB NAVIGATION ─────────────────────────────────────────────────────────
  function switchTab(tabName) {
    const nav = document.getElementById('tabs-nav');
    if (!nav) return;

    nav.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    document.querySelectorAll('.tab-panel').forEach(panel => {
      panel.classList.toggle('active', panel.id === `tab-${tabName}`);
    });

    State.set({ activeTab: tabName });

    if (tabName === 'foryou') {
      loadForYou();
    } else if (tabName === 'similar') {
      if (allArticlesCache.length === 0) {
        populateArticlePicker();
      }
    }
  }

  // ─── FOR YOU TAB ────────────────────────────────────────────────────────────
  async function loadForYou() {
    const grid = document.getElementById('foryou-grid');
    const moreGrid = document.getElementById('more-stories-grid');
    if (!grid) return;

    showSpinner(grid, 4);

    const { userHistory, topK, method, useBoosts, diversify, activeSection } = State.get();

    try {
      const data = await Api.recommend({
        userHistory: userHistory.length > 0 ? userHistory : null,
        topK,
        method,
        useBoosts,
        diversify,
        section: activeSection || null,
      });

      renderNotes(grid, data.notes || [], data.low_confidence || false);
      renderCards(grid, data.recommendations || []);

      // Also populate the "More Stories" secondary grid
      if (moreGrid) {
        loadMoreStories(activeSection);
      }
    } catch (err) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1">
          <div class="empty-icon">⚠️</div>
          <h3>Failed to load recommendations</h3>
          <p>${err.message || 'Make sure the backend server is running.'}</p>
        </div>`;
    }
  }

  async function loadMoreStories(section = '') {
    const moreGrid = document.getElementById('more-stories-grid');
    if (!moreGrid) return;

    try {
      const data = await Api.articles({
        page: 1,
        perPage: 6,
        section: section || '',
        sort: 'date',
      });
      renderCards(moreGrid, (data.articles || []).map(a => ({ article: a })), { compact: true });
    } catch (_) {
      // silently ignore secondary grid error
    }
  }

  // ─── SIMILAR ARTICLES TAB ───────────────────────────────────────────────────
  async function populateArticlePicker() {
    const picker = document.getElementById('article-picker');
    if (!picker) return;

    try {
      const data = await Api.articles({ page: 1, perPage: 100, sort: 'date' });
      allArticlesCache = data.articles || [];

      picker.innerHTML = '<option value="">-- Choose an article to find similar stories --</option>';
      allArticlesCache.forEach(art => {
        const opt = document.createElement('option');
        opt.value = art.id;
        opt.textContent = `[${art.section || 'General'}] ${art.title.slice(0, 90)}`;
        picker.appendChild(opt);
      });
    } catch (err) {
      picker.innerHTML = '<option value="">Could not load articles list</option>';
    }
  }

  async function onArticleSelected(articleId) {
    const preview = document.getElementById('selected-article-preview');
    const resultsGrid = document.getElementById('similar-results-grid');
    if (!articleId) {
      if (preview) {
        preview.classList.add('hidden');
        preview.innerHTML = '';
      }
      if (resultsGrid) resultsGrid.innerHTML = '';
      return;
    }

    const article = allArticlesCache.find(a => a.id === articleId);
    if (article && preview) {
      preview.classList.remove('hidden');
      const emoji = sectionEmoji(article.section);
      const thumb = article.image
        ? `<img src="${article.image}" alt="${article.title}" onerror="this.outerHTML='<div class=\\'rp-history-thumb-fallback\\'>${emoji}</div>'" />`
        : `<div class="rp-history-thumb-fallback">${emoji}</div>`;

      preview.innerHTML = `
        ${thumb}
        <div class="selected-preview-body">
          <div class="selected-preview-section">${article.section || 'General'}</div>
          <div class="selected-preview-title">${article.title}</div>
          <div class="selected-preview-date">Published: ${article.date || 'Recent'}</div>
        </div>`;
    }

    if (resultsGrid) {
      showSpinner(resultsGrid, 4);
      const { topK, method, useBoosts, diversify, activeSection } = State.get();

      try {
        const data = await Api.recommend({
          articleId,
          topK,
          method,
          useBoosts,
          diversify,
          section: activeSection || null,
        });

        renderNotes(resultsGrid, data.notes || [], data.low_confidence || false);
        renderCards(resultsGrid, data.recommendations || []);
      } catch (err) {
        resultsGrid.innerHTML = `
          <div class="empty-state" style="grid-column: 1 / -1">
            <div class="empty-icon">⚠️</div>
            <h3>Error finding similar articles</h3>
            <p>${err.message}</p>
          </div>`;
      }
    }
  }

  // ─── SEARCH TAB ─────────────────────────────────────────────────────────────
  async function performSearch(query) {
    if (!query || !query.trim()) return;
    query = query.trim();

    // Ensure we are on search tab
    switchTab('search');

    const searchInput = document.getElementById('search-query-input');
    if (searchInput) searchInput.value = query;

    const globalInput = document.getElementById('global-search-input');
    if (globalInput) globalInput.value = query;

    const resultsContainer = document.getElementById('search-results-container');
    const resultsGrid = document.getElementById('search-results-grid');
    const resultMeta = document.getElementById('search-result-meta');
    const emptyState = document.getElementById('search-empty');

    if (resultsContainer) resultsContainer.classList.remove('hidden');
    if (emptyState) emptyState.classList.add('hidden');
    if (resultsGrid) showSpinner(resultsGrid, 4);

    const { topK, method, useBoosts, diversify, activeSection } = State.get();

    try {
      const data = await Api.recommend({
        query,
        topK,
        method,
        useBoosts,
        diversify,
        section: activeSection || null,
      });

      if (resultMeta) renderResultMeta(resultMeta, data);
      if (resultsGrid) {
        renderNotes(resultsGrid, data.notes || [], data.low_confidence || false);
        const recs = data.recommendations || [];
        if (recs.length === 0) {
          if (resultsContainer) resultsContainer.classList.add('hidden');
          if (emptyState) emptyState.classList.remove('hidden');
        } else {
          renderCards(resultsGrid, recs);
        }
      }
    } catch (err) {
      if (resultsGrid) {
        resultsGrid.innerHTML = `
          <div class="empty-state" style="grid-column: 1 / -1">
            <div class="empty-icon">⚠️</div>
            <h3>Search request failed</h3>
            <p>${err.message}</p>
          </div>`;
      }
    }
  }

  // ─── INITIALIZATION ─────────────────────────────────────────────────────────
  function init() {
    // Tab buttons
    const nav = document.getElementById('tabs-nav');
    if (nav) {
      nav.addEventListener('click', (e) => {
        const btn = e.target.closest('.tab-btn');
        if (btn && btn.dataset.tab) {
          switchTab(btn.dataset.tab);
        }
      });
    }

    // Article picker in Similar tab
    const picker = document.getElementById('article-picker');
    if (picker) {
      picker.addEventListener('change', (e) => {
        onArticleSelected(e.target.value);
      });
    }

    // Search button and input Enter key in Search tab
    const searchBtn = document.getElementById('search-btn');
    const searchInput = document.getElementById('search-query-input');

    if (searchBtn && searchInput) {
      searchBtn.addEventListener('click', () => {
        performSearch(searchInput.value);
      });
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          performSearch(searchInput.value);
        }
      });
    }

    // Quick pills in Search tab
    const pills = document.querySelectorAll('#quick-pills .pill');
    pills.forEach(pill => {
      pill.addEventListener('click', () => {
        const q = pill.dataset.q;
        if (q) performSearch(q);
      });
    });

    // Populate article picker early in the background
    populateArticlePicker();

    // Initial load of default tab (For You)
    loadForYou();
  }

  return { init, switchTab, loadForYou, performSearch, populateArticlePicker };
})();
