/**
 * ui.js — Shared UI helpers: cards, toasts, history list, badges.
 */

// ─── Section colours ────────────────────────────────────────────────────────
const SECTION_COLORS = {
  politics:     '#f59e0b',
  sports:       '#22c55e',
  technology:   '#38bdf8',
  tech:         '#38bdf8',
  business:     '#a78bfa',
  health:       '#fb7185',
  entertainment:'#f472b6',
  environment:  '#4ade80',
  education:    '#60a5fa',
  india:        '#fb923c',
  world:        '#a78bfa',
  default:      '#94a3b8',
};

function sectionColor(section = '') {
  const key = section.toLowerCase().split(' ')[0];
  return SECTION_COLORS[key] || SECTION_COLORS.default;
}

function sectionEmoji(section = '') {
  const map = {
    politics:'🗳️', sports:'🏅', technology:'💻', tech:'💻',
    business:'💼', health:'🩺', entertainment:'🎬', environment:'🌿',
    education:'📚', india:'🇮🇳', world:'🌍', default:'📰'
  };
  const key = section.toLowerCase().split(' ')[0];
  return map[key] || map.default;
}

// ─── Toast ──────────────────────────────────────────────────────────────────
function showToast(message, type = 'success', duration = 3500) {
  const container = document.getElementById('toast-container');
  const icons = { success: '✅', info: 'ℹ️', warn: '⚠️' };
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || '📢'}</span>
    <span class="toast-msg">${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('toast-out');
    toast.addEventListener('animationend', () => toast.remove());
  }, duration);
}

// ─── Article Card ────────────────────────────────────────────────────────────
function buildArticleCard(article, { score = null, reason = null, compact = false } = {}) {
  const card = document.createElement('div');
  card.className = 'article-card animate-scale';
  card.dataset.id = article.id;

  const color   = sectionColor(article.section);
  const emoji   = sectionEmoji(article.section);
  const imgSrc  = article.image || '';
  const history = State.get().userHistory;
  const isRead  = history.includes(article.id);

  const scoreHTML = score !== null
    ? `<span class="badge badge-score">Match: ${Math.round(score * 100)}%</span>` : '';
  const liveHTML  = article.is_live
    ? `<span class="badge badge-live">🔴 LIVE</span>` : '';
  const sectionHTML = article.section
    ? `<span class="badge badge-section" style="border-color:${color}22;color:${color}">${article.section.toUpperCase()}</span>` : '';
  const subHTML = article.subsection
    ? `<span class="badge badge-section">${article.subsection.toUpperCase()}</span>` : '';

  const reasonHTML = reason
    ? `<div class="card-reason">💡 ${reason}</div>` : '';

  const thumbHTML = imgSrc
    ? `<img src="${imgSrc}" alt="${article.title}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'card-thumb-fallback\\'>${emoji}</div>'" />`
    : `<div class="card-thumb-fallback">${emoji}</div>`;

  card.innerHTML = `
    <div class="card-thumb">
      ${thumbHTML}
      <div class="card-section-strip">${sectionHTML}${subHTML}</div>
    </div>
    <div class="card-body">
      <div class="card-badges">${liveHTML}${scoreHTML}</div>
      <div class="card-title">${article.title}</div>
      <div class="card-snippet">${article.snippet || ''}</div>
      ${reasonHTML}
    </div>
    <div class="card-footer">
      <div class="card-meta-info">
        <span>📅 ${article.date || ''}</span>
      </div>
      <div class="card-actions">
        <button class="card-read-btn ${isRead ? 'reading' : ''}" data-id="${article.id}">
          ${isRead ? '✓ Read' : '📖 Read Story'}
        </button>
        ${article.url ? `<a href="${article.url}" target="_blank" rel="noopener" class="card-bookmark" title="Open full article">🔗</a>` : ''}
      </div>
    </div>`;

  // Read button handler
  card.querySelector('.card-read-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    const added = State.addToHistory(article.id);
    State.saveHistory();
    if (added) {
      showToast(`"${article.title.slice(0, 60)}…" added to history. Feed updating!`, 'success');
      e.currentTarget.textContent = '✓ Read';
      e.currentTarget.classList.add('reading');
      refreshHistoryUI();
    }
  });

  return card;
}

// ─── Render a grid of cards ──────────────────────────────────────────────────
function renderCards(container, items, { compact = false } = {}) {
  container.innerHTML = '';
  container.classList.add('stagger-children');
  if (!items || items.length === 0) {
    container.innerHTML = `<div class="empty-state" style="grid-column:1/-1">
      <div class="empty-icon">📰</div><h3>No articles found</h3><p>Try adjusting your filters or search terms.</p>
    </div>`;
    return;
  }
  items.forEach(item => {
    const art  = item.article || item;
    const card = buildArticleCard(art, {
      score:   item.score  ?? null,
      reason:  item.reason ?? null,
      compact,
    });
    container.appendChild(card);
  });
}

// ─── Notes / Banners ─────────────────────────────────────────────────────────
function renderNotes(container, notes = [], lowConfidence = false) {
  const existing = container.querySelectorAll('.note-banner');
  existing.forEach(el => el.remove());

  if (lowConfidence) {
    const el = document.createElement('div');
    el.className = 'note-banner warn';
    el.innerHTML = '⚡ No exact match found. Showing nearest contextual matches.';
    container.prepend(el);
  }
  notes.forEach(note => {
    const el = document.createElement('div');
    const isSuccess = note.includes('Auto-corrected');
    el.className = `note-banner ${isSuccess ? 'success' : 'info'}`;
    el.innerHTML = isSuccess ? `🔍 Smart Search: ${note}` : `ℹ️ ${note}`;
    container.prepend(el);
  });
}

// ─── Result Meta Bar ──────────────────────────────────────────────────────────
function renderResultMeta(container, result) {
  container.innerHTML = `
    <span>🔢 <strong>${result.recommendations.length}</strong> results</span>
    <span>🧠 Engine: <strong>${result.engine || ''}</strong></span>
    <span class="badge badge-method">${result.method}</span>
    ${result.best_relevance ? `<span>Best match: <strong>${Math.round(result.best_relevance*100)}%</strong></span>` : ''}
  `;
}

// ─── History UI (both sidebar + right panel) ─────────────────────────────────
function refreshHistoryUI() {
  const { userHistory } = State.get();

  // Sidebar
  const sideCount  = document.getElementById('history-count');
  const sideList   = document.getElementById('history-list');
  if (sideCount) sideCount.textContent = `${userHistory.length} article${userHistory.length !== 1 ? 's' : ''} read this session`;
  if (sideList) {
    if (userHistory.length === 0) {
      sideList.innerHTML = '<p class="empty-history">Click "Read Story" on any article to personalise your feed.</p>';
    } else {
      sideList.innerHTML = '';
      const recent = [...userHistory].reverse().slice(0, 5);
      recent.forEach(id => {
        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `<div class="history-thumb">📰</div><span class="history-title">${id}</span>`;
        sideList.appendChild(item);
      });
    }
  }

  // Right Panel
  const rpCount = document.getElementById('rp-history-count');
  const rpList  = document.getElementById('rp-history-list');
  if (rpCount) rpCount.textContent = `${userHistory.length} article${userHistory.length !== 1 ? 's' : ''}`;
  if (rpList) {
    if (userHistory.length === 0) {
      rpList.innerHTML = '<p class="rp-empty">No articles read yet.</p>';
    } else {
      rpList.innerHTML = '';
      const recent = [...userHistory].reverse().slice(0, 5);
      recent.forEach(id => {
        const item = document.createElement('div');
        item.className = 'rp-history-item';
        item.innerHTML = `
          <div class="rp-history-thumb"><div class="rp-history-thumb-fallback">📰</div></div>
          <div class="rp-history-body">
            <div class="rp-history-title">${id}</div>
            <div class="rp-history-time">Just now</div>
          </div>`;
        rpList.appendChild(item);
      });
    }
  }

  // Update AI Brief text
  const briefText = document.getElementById('ai-brief-text');
  if (briefText) {
    if (userHistory.length === 0) {
      briefText.textContent = 'Read a few articles to activate your personalised AI brief.';
    } else {
      briefText.textContent = `Based on ${userHistory.length} article${userHistory.length !== 1 ? 's' : ''} you've read, we've curated your top picks.`;
    }
  }
}

// ─── Category pills builder ───────────────────────────────────────────────────
function buildCategoryPills(sections, container, onChange) {
  container.innerHTML = '';
  const allBtn = document.createElement('button');
  allBtn.className = 'cat-pill active';
  allBtn.dataset.section = '';
  allBtn.textContent = 'All';
  allBtn.addEventListener('click', () => {
    container.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
    allBtn.classList.add('active');
    onChange('');
  });
  container.appendChild(allBtn);

  sections.forEach(sec => {
    const btn = document.createElement('button');
    btn.className = 'cat-pill';
    btn.dataset.section = sec;
    btn.textContent = sec;
    btn.style.setProperty('--cat-color', sectionColor(sec));
    btn.addEventListener('click', () => {
      container.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      onChange(sec);
    });
    container.appendChild(btn);
  });
}

// ─── Spinner helper ──────────────────────────────────────────────────────────
function showSpinner(container, cols = 4) {
  container.innerHTML = `
    <div class="skeleton-grid" style="grid-template-columns:repeat(${cols},1fr)">
      ${Array(cols).fill('<div class="skeleton skeleton-card"></div>').join('')}
    </div>`;
}
