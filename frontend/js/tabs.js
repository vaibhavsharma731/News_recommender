/**
 * tabs.js — Tab switching and recommendation flows.
 *
 * "Similar Articles" renders a scrolling FEED of "halo" widgets: each widget has
 * ONE source story in the centre with its recommendations arranged around it
 * (top / left / right / bottom). Widgets load lazily as you scroll.
 * "Semantic Search" stays a query → results grid.
 *
 * Every widget carries a source chip so it is obvious whether the answer came
 * from the precomputed PostgreSQL table (⚡ hybrid+rerank) or the live in-memory
 * engine (🛰️ bm25 / semantic / hybrid).
 *
 * The Recommendations Count slider (1–10) is sent as top_k, so every widget shows
 * exactly that many recommendations — served from precompute by default.
 */

const Tabs = (() => {
  let allArticlesCache = [];
  let feedObserver = null;   // lazy-loads a widget's recommendations near viewport
  let scrollObserver = null; // loads the NEXT batch when the bottom sentinel shows
  let feedQueue = [];        // source articles still waiting to be rendered
  let feedRendered = 0;      // how many halo widgets are currently in the DOM
  let pinnedArticleForFeed = null; // article pinned to the top of feed (e.g. from Discover Random or Story Finder)

  const FEED_CHUNK = 15;     // widgets appended per batch as you scroll
  const FEED_PAGE  = 100;    // API page size when pulling the full article list
  const SLOT_ORDER = ['top', 'bottom', 'left', 'right'];  // assignment order

  /** Escape API/DB strings before they touch innerHTML (article titles are free text). */
  const escHtml = (s) => String(s == null ? '' : s).replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

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

    if (tabName === 'similar') {
      const targetDs = State.get().activePrecompDataset || 'default';
      State.set({ activeDataset: targetDs });
      buildFeed();
    } else if (tabName === 'hybrid-lab') {
      const targetDs = State.get().activeLiveDataset || 'default';
      State.set({ activeDataset: targetDs });
      if (typeof HybridLab !== 'undefined' && typeof HybridLab.onTabActivated === 'function') {
        HybridLab.onTabActivated();
      }
    } else if (tabName === 'compare') {
      // Compare tab is self-contained; no extra activation needed beyond switching the panel.
      // Compare.init() already ran at startup.
    } else if (tabName === 'editor') {
      if (window.Editor) {
        Editor.refreshDbHealth();
      }
    }
  }


  // ─── SIMILAR ARTICLES FEED ──────────────────────────────────────────────────

  /** A coloured chip telling the reader where a result came from. */
  function sourceChip(source) {
    const pre = source === 'postgresql';
    const label = pre ? '⚡ Precomputed' : '🛰️ Live';
    const title = pre
      ? 'Served from the precomputed PostgreSQL table (sub-5ms)'
      : 'Computed live by the in-memory retrieval + rerank engine';
    return `<span class="source-chip source-chip-${pre ? 'pre' : 'live'}" title="${title}">${label}</span>`;
  }

  /**
   * Pull EVERY article, not just the first page. The feed is unlimited, so we
   * page through the API (100 at a time) until the whole table is loaded, then
   * render it progressively as the reader scrolls.
   *
   * sort='confidence' ranks stories by the average score of their precomputed
   * recommendations, so widgets whose recommendations are the strongest
   * (70%, 80% …) appear before weak ones (10%, 20% …).
   */
  async function loadAllArticles(force = false) {
    if (allArticlesCache.length && !force) return allArticlesCache;

    const acc = [];
    let page = 1;
    let total = Infinity;
    while (acc.length < total && page <= 500) {
      const data = await Api.articles({ page, perPage: FEED_PAGE, sort: 'confidence' });
      total = data.total || 0;
      const arts = data.articles || [];
      if (!arts.length) break;
      acc.push(...arts);
      if (arts.length < FEED_PAGE) break;   // last page reached
      page += 1;
    }
    allArticlesCache = acc;
    return allArticlesCache;
  }

  const skelTile = () => '<div class="skeleton orbit-tile orbit-tile-skel"></div>';

  /** Format an article's upload (ingest) time, e.g. "14 Oct, 13:04". */
  function uploadStamp(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    return d.toLocaleString(undefined, {
      day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
    });
  }

  /** Build a halo widget: centre story + 4 empty recommendation slots. */
  function buildFeedWidget(article) {
    const widget = document.createElement('article');
    widget.className = 'feed-widget';
    widget.dataset.id = article.id;
    widget._article = article;

    const color = sectionColor(article.section);
    const emoji = sectionEmoji(article.section);
    const thumb = article.image
      ? `<img src="${article.image}" alt="" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'orbit-main-fallback\\'>${emoji}</div>'" />`
      : `<div class="orbit-main-fallback">${emoji}</div>`;
    const liveBadge = article.is_live ? '<span class="badge badge-live">🔴 LIVE</span>' : '';
    const readLink = article.url
      ? `<a class="feed-read-link" href="${article.url}" target="_blank" rel="noopener">📖 Read story</a>`
      : '';

    widget.innerHTML = `
      <div class="feed-widget-head">
        <span class="feed-recs-title">✦ Around this story</span>
        <span class="feed-head-right">
          <button type="button" class="feed-mmr-btn active" data-action="toggle-mmr" title="Toggle MMR diversity on/off (instant 0ms)">
            <span class="mmr-icon">🔀</span> <span class="mmr-label">MMR: ON</span>
          </button>
          <span class="feed-conf-slot" data-slot="conf"></span>
          <span class="feed-source-slot" data-slot="source"></span>
        </span>
      </div>
      <div class="orbit-stage">
        <div class="orbit-slot orbit-slot-top" data-slot="top">${skelTile()}</div>
        <div class="orbit-mid">
          <div class="orbit-slot orbit-slot-left" data-slot="left">${skelTile()}</div>
          <div class="orbit-main">
            <div class="orbit-main-thumb">${thumb}</div>
            <div class="orbit-main-body">
              <div class="orbit-main-section" style="color:${color}">${(article.section || 'General').toUpperCase()} ${liveBadge}</div>
              <h3 class="orbit-main-title">${article.title}</h3>
              <div class="orbit-main-meta">
                <span>📅 ${article.date || 'Recent'}</span>
                ${article.created_at ? `<span class="feed-upload" title="Uploaded to the system ${uploadStamp(article.created_at)}">🕘 ${uploadStamp(article.created_at)}</span>` : ''}
                ${readLink}
              </div>
            </div>
          </div>
          <div class="orbit-slot orbit-slot-right" data-slot="right">${skelTile()}</div>
        </div>
        <div class="orbit-slot orbit-slot-bottom" data-slot="bottom">${skelTile()}</div>
      </div>`;

    const mmrBtn = widget.querySelector('[data-action="toggle-mmr"]');
    if (mmrBtn) {
      mmrBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const currentMmr = widget._useMmr !== false;
        widget._useMmr = !currentMmr;
        if (widget._cachedData) {
          renderWidgetRecs(widget, widget._cachedData, widget._useMmr);
        }
      });
    }

    return widget;
  }

  /** One recommendation tile that sits around the centre story. */
  function buildOrbitTile(rec, useMmr = true, sourceArticle = null) {
    const art   = Object.assign({}, rec.article || rec);
    if (art.rank == null && rec.rank != null) art.rank = rec.rank;
    if (art.rank_raw == null && rec.rank_raw != null) art.rank_raw = rec.rank_raw;
    if (art.score == null && rec.score != null) art.score = rec.score;
    if (art.rerank_score == null && rec.rerank_score != null) art.rerank_score = rec.rerank_score;
    const emoji = sectionEmoji(art.section);
    const thumb = art.image
      ? `<img src="${art.image}" alt="" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'orbit-tile-fallback\\'>${emoji}</div>'" />`
      : `<div class="orbit-tile-fallback">${emoji}</div>`;
    // Evidence extraction
    const rawReason = rec.reason || art.reason || '';
    let ev = null;
    if (rawReason && typeof rawReason === 'string' && rawReason.trim().startsWith('{')) {
      try { ev = JSON.parse(rawReason); } catch (e) { ev = null; }
    }

    const confBand = (ev && ev.confidence_band) || (rec.score != null ? (rec.score >= 0.65 ? 'HIGH' : rec.score >= 0.35 ? 'MEDIUM' : 'LOW') : 'MEDIUM');
    const scoreVal = rec.score != null ? Math.round(rec.score * 100) : null;
    const score = confBand === 'HIGH'
      ? `<span class="orbit-score" style="background:rgba(34,197,94,0.2);color:#4ade80;border:1px solid rgba(34,197,94,0.4);" title="High Confidence Band">🟢 High</span>`
      : confBand === 'MEDIUM'
      ? `<span class="orbit-score" style="background:rgba(245,158,11,0.2);color:#fbbf24;border:1px solid rgba(245,158,11,0.4);" title="Medium Confidence Band">🟡 Med</span>`
      : `<span class="orbit-score" style="background:rgba(148,163,184,0.2);color:#cbd5e1;border:1px solid rgba(148,163,184,0.4);" title="Low Confidence Band">⚪ Low</span>`;

    const rawRank = rec.rank_raw || rec.rank || 1;
    const mmrRank = rec.rank || 1;
    const hasShift = rawRank !== mmrRank;

    let rankLabel, rankTitle;
    if (useMmr) {
      if (hasShift && mmrRank > rawRank) {
        rankLabel = `#${mmrRank} <span style="opacity:0.85;font-size:0.56rem;">(#${rawRank})</span>`;
        rankTitle = `MMR Diversified Rank: #${mmrRank} (Demoted from Raw #${rawRank} due to near-duplicate redundancy with top stories)`;
      } else if (hasShift && mmrRank < rawRank) {
        rankLabel = `#${mmrRank} <span style="opacity:0.85;font-size:0.56rem;">(#${rawRank})</span>`;
        rankTitle = `MMR Diversified Rank: #${mmrRank} (Promoted from Raw #${rawRank} to introduce fresh angle)`;
      } else {
        rankLabel = `#${mmrRank}`;
        rankTitle = `MMR Diversified Rank: #${mmrRank} (Matches raw relevance)`;
      }
    } else {
      rankLabel = `Raw #${rawRank}`;
      rankTitle = `Pure Relevance Rank: #${rawRank} (Confidence: ${confBand}, sorted without MMR diversity)`;
    }
    const rankChip = `<span class="orbit-rank-chip" title="${rankTitle}">${rankLabel}</span>`;

    let evidenceHTML = '';
    if (ev) {
      const tags = [];

      // 1. MMR Shift badge
      if (useMmr && hasShift) {
        if (mmrRank > rawRank) {
          tags.push(`<span class="orbit-badge orbit-badge-mmr-demote" title="MMR diversity demoted this article from #${rawRank} to #${mmrRank} to prevent duplicate stories">🔀 Redundancy -${mmrRank - rawRank}</span>`);
        } else {
          tags.push(`<span class="orbit-badge orbit-badge-mmr-promo" title="MMR diversity promoted this article from #${rawRank} to #${mmrRank} for fresh angle">✨ Diversity +${rawRank - mmrRank}</span>`);
        }
      } else if (!useMmr) {
        tags.push(`<span class="orbit-badge orbit-badge-raw" title="Pure relevance ordering (no MMR penalty applied)">🎯 Raw #${rawRank}</span>`);
      }

      // 2. Real shared entities only (Fix a: show "none shared" if empty)
      if (ev.entities && ev.entities.length) {
        tags.push(`<span class="orbit-badge" title="Shared entity: ${ev.entities.join(', ')}">📍 ${ev.entities[0]}</span>`);
      } else {
        tags.push(`<span class="orbit-badge" style="opacity:0.65;" title="No shared named entities detected">📍 none shared</span>`);
      }

      // 3. Real BM25 match
      const topTerm = (ev.bm25_terms && ev.bm25_terms[0]) || null;
      if (topTerm) {
        tags.push(`<span class="orbit-badge" title="Top BM25 term: ${topTerm.term} (tf ${topTerm.tf}, idf ${topTerm.idf}, contributed +${topTerm.contribution})">🏷️ ${topTerm.term}</span>`);
      }

      // 4. Real embedding cosine alignment band (Fix c: never raw %)
      if (ev.cosine_band) {
        tags.push(`<span class="orbit-badge" title="${ev.cosine_display || ev.cosine_band}">🧭 ${ev.cosine_band.replace(' Semantic Alignment', '')}</span>`);
      } else if (typeof ev.cosine === 'number') {
        tags.push(`<span class="orbit-badge" title="Cosine similarity: ${ev.cosine.toFixed(4)}">🧭 Cosine ${ev.cosine.toFixed(2)}</span>`);
      }

      // 5. Signal agreement (Fix b: top-5 rule, "X of 3")
      const agreeCount = ev.agreement_count ?? 1;
      const maxSig = ev.max_signals || 3;
      const agreeLabel = ev.agreement_label || `${agreeCount}/${maxSig}`;
      if (agreeCount >= 2) {
        tags.push(`<span class="orbit-badge" title="Signals agreed in top-5">⚡ ${agreeLabel} Agreed</span>`);
      }

      // 6. Inspect Evaluation Breakdown
      tags.push(`<span class="orbit-badge orbit-badge-eval" title="Inspect complete grounded evaluation" style="cursor:pointer;background:rgba(168,85,247,0.2);color:#c084fc;border:1px solid rgba(168,85,247,0.4);font-weight:700;">🔬 Eval</span>`);

      const tagsHTML = tags.length
        ? `<div class="orbit-tile-badges">${tags.join('')}</div>`
        : '';

      const whyText = ev.verdict?.template_why || ev.template_why || ev.summary || '';
      const summaryHTML = whyText
        ? `<div class="orbit-tile-reason" title="${whyText.replace(/"/g, '&quot;')}">💡 ${whyText}</div>`
        : '';
      evidenceHTML = `${tagsHTML}${summaryHTML}`;
    } else if (rawReason && !rawReason.startsWith('Rank #')) {
      evidenceHTML = `<div class="orbit-tile-reason" title="${rawReason}">💡 ${rawReason}</div>`;
    }

    const tile = document.createElement('div');
    tile.className = 'orbit-tile';
    tile.innerHTML = `
      <div class="orbit-tile-thumb">${thumb}${rankChip}${score}</div>
      <div class="orbit-tile-body">
        <div class="orbit-tile-section">${(art.section || 'General')}</div>
        <div class="orbit-tile-title" title="${art.title}">${art.title}</div>
        ${evidenceHTML}
        ${art.created_at ? `<div class="orbit-tile-meta" title="Uploaded to the system ${uploadStamp(art.created_at)}">🕘 ${uploadStamp(art.created_at)}</div>` : ''}
      </div>`;

    if (art.url) {
      tile.classList.add('clickable');
      tile.title = 'Open full article';
      tile.addEventListener('click', () => window.open(art.url, '_blank', 'noopener'));
    }

    // Connect Evaluation Breakdown modal on click of Eval badge or Reason text
    tile._sourceArticle = sourceArticle;
    tile._rec = rec;
    tile._useMmr = useMmr;

    const evalTrigger = tile.querySelector('.orbit-badge-eval');
    const reasonEl = tile.querySelector('.orbit-tile-reason');
    const handleOpenEval = (e) => {
      e.stopPropagation();
      const parentWidget = tile.closest('.feed-widget');
      const sourceArt = tile._sourceArticle || parentWidget?._article || parentWidget?._cachedData?.source_article || parentWidget?._cachedData?.story || null;
      if (window.Evaluator) {
        window.Evaluator.showModal(rec, sourceArt, art);
      }
    };
    if (evalTrigger) evalTrigger.addEventListener('click', handleOpenEval);
    if (reasonEl) {
      reasonEl.style.cursor = 'pointer';
      reasonEl.addEventListener('click', handleOpenEval);
    }

    return tile;
  }

  /**
   * Decide how many recommendation tiles go on each side of the centre story.
   *
   * Few tiles  -> all on top/bottom rows (a tidy 2-row cluster, avoiding the
   *               half-empty side columns that used to leave big blank gaps).
   * Many tiles -> left/right get balanced PAIRS (always equal, so the halo stays
   *               symmetric) and the remainder fills the top/bottom rows.
   */
  function planHaloSlots(n) {
    if (n <= 0) return { top: 0, bottom: 0, left: 0, right: 0 };
    if (n <= 6) {
      const top = Math.ceil(n / 2);
      return { top, bottom: n - top, left: 0, right: 0 };
    }
    const side = n >= 9 ? 2 : 1;
    const rest = n - 2 * side;
    const top = Math.ceil(rest / 2);
    return { top, bottom: rest - top, left: side, right: side };
  }

  /** Technique 4: a separate "Trending in <section>" block for a thin related list. */
  function renderFallback(widget, fallback) {
    widget.querySelectorAll('.feed-fallback').forEach((el) => el.remove());
    const stage = widget.querySelector('.orbit-stage');
    const items = (fallback && fallback.items) || [];
    if (!stage || !items.length) return;

    const box = document.createElement('div');
    box.className = 'feed-fallback';
    box.innerHTML = `
      <div class="feed-fallback-head">
        <span class="feed-fallback-title">${escHtml(fallback.heading)}</span>
        <span class="feed-fallback-note">${escHtml(fallback.note)}</span>
      </div>
      <div class="feed-fallback-grid">
        ${items.map((art) => `
          <a class="feed-fallback-item"${art.url ? ` href="${escHtml(art.url)}" target="_blank" rel="noopener"` : ''}>
            <span class="feed-fallback-sec">${escHtml((art.section || 'General').toUpperCase())}</span>
            <span class="feed-fallback-hl">${escHtml(art.title)}</span>
            <span class="feed-fallback-meta">${escHtml(art.date || '')}${art.pageViews ? ` · ${Number(art.pageViews).toLocaleString()} views` : ''}</span>
          </a>`).join('')}
      </div>`;
    stage.insertAdjacentElement('afterend', box);
  }

  /** Distribute the returned recommendations around the centre. */
  function renderWidgetRecs(widget, data, useMmr) {
    widget._cachedData = data;
    if (data && data.source_article) {
      widget._article = data.source_article;
    }
    if (useMmr === undefined) {
      useMmr = widget._useMmr !== undefined ? widget._useMmr : (State.get().diversify !== false);
    }
    widget._useMmr = useMmr;

    const mmrBtn = widget.querySelector('[data-action="toggle-mmr"]');
    if (mmrBtn) {
      if (useMmr) {
        mmrBtn.className = 'feed-mmr-btn active';
        mmrBtn.innerHTML = '<span class="mmr-icon">🔀</span> <span class="mmr-label">MMR: ON</span>';
        mmrBtn.title = 'MMR Diversity ON: near-duplicate wire stories suppressed. Click to switch to Pure Relevance (RAW).';
      } else {
        mmrBtn.className = 'feed-mmr-btn inactive';
        mmrBtn.innerHTML = '<span class="mmr-icon">🎯</span> <span class="mmr-label">RAW RANK</span>';
        mmrBtn.title = 'Pure Relevance RAW: strict score order without MMR penalty. Click to switch to MMR Diversity.';
      }
    }

    const chipSlot = widget.querySelector('[data-slot="source"]');
    if (chipSlot) chipSlot.innerHTML = sourceChip(data.source);

    // Average confidence across this story's recommendations — the value the feed
    // is ordered by, shown so readers can see why a widget ranks where it does.
    const confSlot = widget.querySelector('[data-slot="conf"]');
    if (confSlot) {
      const scores = (data.recommendations || [])
        .map(r => r.score)
        .filter(s => typeof s === 'number' && !Number.isNaN(s));
      if (scores.length) {
        const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
        confSlot.innerHTML = `<span class="feed-conf-chip" title="Average confidence across this story's recommendations">${Math.round(avg * 100)}% avg</span>`;
      } else {
        confSlot.innerHTML = '';
      }
    }

    const slots = {
      top:    widget.querySelector('[data-slot="top"]'),
      bottom: widget.querySelector('[data-slot="bottom"]'),
      left:   widget.querySelector('[data-slot="left"]'),
      right:  widget.querySelector('[data-slot="right"]'),
    };
    Object.values(slots).forEach(s => { if (s) s.innerHTML = ''; });

    let recs = (data.recommendations || [])
      .filter(r => String((r.article || r).id) !== String(widget.dataset.id))
      .slice();

    if (!recs.length) {
      const stage = widget.querySelector('.orbit-stage');
      if (stage) stage.insertAdjacentHTML('beforeend',
        '<div class="feed-recs-empty">No close matches found for this story.</div>');
      renderFallback(widget, data.fallback);
      return;
    }

    // Sort according to active MMR mode:
    if (useMmr) {
      // MMR Diversified: by rank (1, 2, 3...)
      recs.sort((a, b) => (a.rank || 0) - (b.rank || 0));
    } else {
      // Pure Relevance without MMR: by rank_raw if available, else by score descending
      recs.sort((a, b) => {
        if (a.rank_raw != null && b.rank_raw != null) return a.rank_raw - b.rank_raw;
        return (b.score || 0) - (a.score || 0);
      });
    }

    // Spread the recommendations around the centre using the plan above.
    const plan = planHaloSlots(recs.length);
    let assign = 0;
    SLOT_ORDER.forEach(name => {
      const slot = slots[name];
      if (!slot) return;
      for (let k = 0; k < plan[name] && assign < recs.length; k++) {
        slot.appendChild(buildOrbitTile(recs[assign++], useMmr, widget._article));
      }
    });

    renderFallback(widget, data.fallback);
  }

  /** Fetch + fill a single halo widget (called when it scrolls into view). */
  async function loadWidgetRecommendations(widget) {
    if (widget.dataset.loaded === 'true') return;
    widget.dataset.loaded = 'true';

    const articleId = widget.dataset.id;
    const { topK, method, useBoosts, diversify, modelVersion } = State.get();
    try {
      // No section is sent on purpose: keeping the default similar-article request
      // means method=hybrid+rerank is served from the precomputed table. The source
      // list itself is filtered by category client-side.
      const precompDs = State.get().activePrecompDataset || State.get().activeDataset || 'default';
      const data = await Api.recommend({
        articleId,
        topK,
        method,
        dataset: precompDs,
        useBoosts,
        diversify,
        modelVersion: modelVersion || '384d',
      });
      renderWidgetRecs(widget, data);
    } catch (err) {
      const stage = widget.querySelector('.orbit-stage');
      if (stage) {
        stage.querySelectorAll('.orbit-slot').forEach(s => { s.innerHTML = ''; });
        const empty = document.createElement('div');
        empty.className = 'feed-recs-empty';
        empty.textContent = `⚠️ ${err.message}`;
        stage.appendChild(empty);
      }
    }
  }

  /** Lazy-load widgets only as they approach the viewport. Reuses one observer. */
  function observeWidgets(widgets) {
    if (!feedObserver) {
      feedObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            feedObserver.unobserve(entry.target);
            loadWidgetRecommendations(entry.target);
          }
        });
      }, { root: null, rootMargin: '400px 0px', threshold: 0.01 });
    }
    const nodes = widgets || document.querySelectorAll('#article-feed .feed-widget');
    nodes.forEach(w => feedObserver.observe(w));
  }

  function skeletonFeed(n = 2) {
    return Array(n).fill(`
      <div class="feed-widget feed-widget-skeleton">
        <div class="skeleton" style="height:18px;width:190px;border-radius:8px;margin-bottom:16px"></div>
        <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center">
          <div class="skeleton orbit-tile-skel" style="width:168px"></div>
          <div class="skeleton" style="height:250px;width:310px;border-radius:16px"></div>
          <div class="skeleton orbit-tile-skel" style="width:168px"></div>
        </div>
      </div>`).join('');
  }

  /** Sentinel state: "more" while batches remain, "done" at the end. */
  function setSentinel(state) {
    const sentinel = document.getElementById('article-feed-sentinel');
    if (!sentinel) return;
    if (state === 'more') {
      sentinel.classList.remove('done');
      sentinel.innerHTML = '<span class="feed-sentinel-text">Scroll for more stories…</span>';
    }
  }

  function markFeedEnd() {
    if (scrollObserver) { scrollObserver.disconnect(); scrollObserver = null; }
    const sentinel = document.getElementById('article-feed-sentinel');
    if (!sentinel) return;
    sentinel.classList.add('done');
    sentinel.innerHTML = feedRendered
      ? `<span class="feed-sentinel-text">✓ You've reached the end — ${feedRendered} stories shown.</span>`
      : '';
  }

  /** Append the next batch of halo widgets (infinite scroll). */
  function appendFeedChunk() {
    const feed = document.getElementById('article-feed');
    if (!feed) return;
    if (!feedQueue.length) { markFeedEnd(); return; }

    const batch = feedQueue.splice(0, FEED_CHUNK);
    const added = [];
    const frag = document.createDocumentFragment();
    batch.forEach(art => {
      const w = buildFeedWidget(art);
      frag.appendChild(w);
      added.push(w);
    });
    feed.appendChild(frag);
    feedRendered += batch.length;

    observeWidgets(added);
    if (feedQueue.length) setSentinel('more');
    else markFeedEnd();
  }

  /** Watch the bottom sentinel and pull the next batch when it scrolls into view. */
  function observeSentinel() {
    const sentinel = document.getElementById('article-feed-sentinel');
    if (!sentinel || !feedQueue.length) return;
    if (scrollObserver) scrollObserver.disconnect();
    scrollObserver = new IntersectionObserver((entries) => {
      if (entries.some(e => e.isIntersecting)) appendFeedChunk();
    }, { root: null, rootMargin: '600px 0px', threshold: 0 });
    scrollObserver.observe(sentinel);
  }

  /** (Re)build the whole feed from the current controls/state — unlimited. */
  async function buildFeed() {
    const feed = document.getElementById('article-feed');
    if (!feed) return;

    if (feedObserver) { feedObserver.disconnect(); feedObserver = null; }
    if (scrollObserver) { scrollObserver.disconnect(); scrollObserver = null; }
    feedQueue = [];
    feedRendered = 0;

    feed.innerHTML = skeletonFeed(2);
    setSentinel('more');

    let articles;
    try {
      articles = await loadAllArticles(false);
    } catch (err) {
      feed.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚠️</div>
          <h3>Could not load articles</h3>
          <p>${err.message}</p>
        </div>`;
      markFeedEnd();
      return;
    }

    const banner = document.getElementById('reader-empty-banner');
    if (!articles.length) {
      if (banner) banner.classList.remove('hidden');
      feed.innerHTML = '';
      markFeedEnd();
      return;
    } else {
      if (banner) banner.classList.add('hidden');
    }

    const { activeSection } = State.get();
    let list = articles;
    if (activeSection) {
      list = articles.filter(a => (a.section || '') === activeSection);
    }
    // No cap — every matching story is queued and rendered as the reader scrolls.

    if (!list.length && !pinnedArticleForFeed) {
      feed.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">📰</div>
          <h3>No articles in this category</h3>
          <p>Try clearing the category filter.</p>
        </div>`;
      markFeedEnd();
      return;
    }

    if (pinnedArticleForFeed) {
      // Ensure the pinned story is positioned as the very first item in the feed
      list = [pinnedArticleForFeed, ...list.filter(a => String(a.id) !== String(pinnedArticleForFeed.id))];
    }

    feed.innerHTML = '';
    feedQueue = list.slice();
    appendFeedChunk();
    observeSentinel();

    if (pinnedArticleForFeed) {
      const pinnedWidget = feed.querySelector(`.feed-widget[data-id="${pinnedArticleForFeed.id}"]`);
      if (pinnedWidget) {
        loadWidgetRecommendations(pinnedWidget);
        pinnedWidget.scrollIntoView({ behavior: 'smooth', block: 'start' });
        pinnedWidget.classList.add('widget-highlighted');
        setTimeout(() => pinnedWidget.classList.remove('widget-highlighted'), 2600);
      }
      pinnedArticleForFeed = null;
    }
  }

  // Kept for backward compatibility: app.js calls this after ingestion / role switch.
  async function populateArticlePicker() {
    allArticlesCache = [];
    return buildFeed();
  }


  // ─── SEARCH TAB ─────────────────────────────────────────────────────────────
  async function performSearch(query) {
    if (!query || !query.trim()) return;
    query = query.trim();

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

    const startTime = performance.now();
    try {
      const data = await Api.recommend({
        query,
        topK,
        method,
        useBoosts,
        diversify,
        section: activeSection || null,
      });

      const elapsedMs = Math.round(performance.now() - startTime);
      data.latency_ms = elapsedMs;

      if (resultMeta) renderResultMeta(resultMeta, data);
      if (resultsGrid) {
        const recs = data.recommendations || [];
        if (recs.length === 0) {
          if (resultsContainer) resultsContainer.classList.add('hidden');
          if (emptyState) emptyState.classList.remove('hidden');
        } else {
          // renderCards clears the container, so notes must be added AFTER it --
          // otherwise the banners are wiped the moment the cards render.
          renderCards(resultsGrid, recs);
          renderNotes(resultsGrid, data.notes || [], data.low_confidence || false);
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

  function selectArticle(article) {
    if (!article) return;
    const { activeTab } = State.get();
    if (activeTab !== 'similar') {
      pinnedArticleForFeed = article;
      switchTab('similar');
      return;
    }

    const input = document.getElementById('story-finder-input');
    const clearBtn = document.getElementById('story-finder-clear');
    const dropdown = document.getElementById('story-finder-dropdown');
    if (input) input.value = article.title || '';
    if (clearBtn) clearBtn.classList.remove('hidden');
    if (dropdown) dropdown.classList.add('hidden');

    const feed = document.getElementById('article-feed');
    if (!feed) return;

    // Check if widget is already in DOM
    let existingWidget = feed.querySelector(`.feed-widget[data-id="${article.id}"]`);
    if (existingWidget) {
      feed.prepend(existingWidget);
      loadWidgetRecommendations(existingWidget);
      existingWidget.scrollIntoView({ behavior: 'smooth', block: 'start' });
      existingWidget.classList.add('widget-highlighted');
      setTimeout(() => existingWidget.classList.remove('widget-highlighted'), 2600);
    } else {
      // Prepend directly to top of feed so user sees it immediately
      const newWidget = buildFeedWidget(article);
      feed.prepend(newWidget);
      loadWidgetRecommendations(newWidget);
      newWidget.scrollIntoView({ behavior: 'smooth', block: 'start' });
      newWidget.classList.add('widget-highlighted');
      setTimeout(() => newWidget.classList.remove('widget-highlighted'), 2600);
    }
  }

  // ─── QUICK STORY FINDER (Item-to-Item Fast Title Search & Selection) ──────────
  function initStoryFinder() {
    const box = document.getElementById('story-finder-box');
    const input = document.getElementById('story-finder-input');
    const clearBtn = document.getElementById('story-finder-clear');
    const dropdown = document.getElementById('story-finder-dropdown');
    if (!box || !input || !dropdown) return;

    let selectedIndex = -1;
    let currentMatches = [];

    function renderMatches(matches) {
      currentMatches = matches;
      selectedIndex = -1;
      if (!matches.length) {
        dropdown.innerHTML = '<div class="story-finder-item-empty">No matching articles found in database.</div>';
        dropdown.classList.remove('hidden');
        return;
      }
      dropdown.innerHTML = matches.map((art, idx) => {
        const sec = (art.section || 'General').toUpperCase();
        return `
          <div class="story-finder-item" data-idx="${idx}" data-id="${art.id}">
            <div class="story-finder-item-content">
              <div class="story-finder-item-title" title="${art.title}">${art.title}</div>
              <div class="story-finder-item-meta">
                <span class="story-finder-item-sec">${sec}</span>
                ${art.date ? `<span>📅 ${art.date}</span>` : ''}
              </div>
            </div>
            <span style="font-size:0.75rem;opacity:0.6;">➔</span>
          </div>`;
      }).join('');
      dropdown.classList.remove('hidden');
    }

    async function searchTitles(query) {
      const q = (query || '').trim().toLowerCase();
      const articles = await loadAllArticles(false);
      if (!articles || !articles.length) return;

      if (!q) {
        // Show top 12 recommended stories when clicking without typing
        renderMatches(articles.slice(0, 12));
        return;
      }

      const qTerms = q.split(/\s+/).filter(Boolean);
      const matches = articles.filter(a => {
        const titleLow = (a.title || '').toLowerCase();
        const secLow = (a.section || '').toLowerCase();
        return qTerms.every(term => titleLow.includes(term) || secLow.includes(term));
      }).slice(0, 15);

      renderMatches(matches);
    }

    input.addEventListener('focus', () => {
      searchTitles(input.value);
    });

    input.addEventListener('input', () => {
      if (clearBtn) {
        if (input.value) clearBtn.classList.remove('hidden');
        else clearBtn.classList.add('hidden');
      }
      searchTitles(input.value);
    });

    dropdown.addEventListener('click', (e) => {
      const item = e.target.closest('.story-finder-item');
      if (item) {
        const idx = parseInt(item.dataset.idx, 10);
        if (currentMatches[idx]) {
          selectArticle(currentMatches[idx]);
        }
      }
    });

    input.addEventListener('keydown', (e) => {
      const items = dropdown.querySelectorAll('.story-finder-item');
      if (dropdown.classList.contains('hidden') || !items.length) {
        if (e.key === 'ArrowDown') {
          searchTitles(input.value);
        }
        return;
      }

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        selectedIndex = (selectedIndex + 1) % items.length;
        items.forEach((it, i) => it.classList.toggle('active', i === selectedIndex));
        items[selectedIndex]?.scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        selectedIndex = (selectedIndex - 1 + items.length) % items.length;
        items.forEach((it, i) => it.classList.toggle('active', i === selectedIndex));
        items[selectedIndex]?.scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (selectedIndex >= 0 && currentMatches[selectedIndex]) {
          selectArticle(currentMatches[selectedIndex]);
        } else if (currentMatches.length > 0) {
          selectArticle(currentMatches[0]);
        }
      } else if (e.key === 'Escape') {
        dropdown.classList.add('hidden');
      }
    });

    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        input.value = '';
        clearBtn.classList.add('hidden');
        dropdown.classList.add('hidden');
        buildFeed();
      });
    }

    document.addEventListener('click', (e) => {
      if (!box.contains(e.target)) {
        dropdown.classList.add('hidden');
      }
    });
  }

  // ─── READER CATEGORY FILTER BAR & MODEL SWITCHER ───────────────────────────
  // Defined at module scope (not nested inside initModelSwitcher) because it is
  // also exported on the public Tabs object and called from app.js. Referencing
  // it from the return block while it was nested caused a ReferenceError that
  // aborted the whole tabs.js IIFE, leaving `Tabs` undefined.
  async function applyModelSwitch(version = '384d') {
    State.set({ modelVersion: '384d' });
    if (typeof window.checkReaderDatabaseState === 'function') {
      await window.checkReaderDatabaseState();
    }
    allArticlesCache = [];
    buildFeed();
  }

  function initModelSwitcher() {
    // Model switcher retired — single standard 384D pipeline active
  }

  function selectCategory(sec) {
    State.set({ activeSection: sec });
    const pillContainer = document.getElementById('reader-category-pills');
    if (pillContainer) {
      pillContainer.querySelectorAll('.cat-pill').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.section === sec);
      });
    }
    const sectionFilter = document.getElementById('section-filter');
    if (sectionFilter && sectionFilter.value !== sec) {
      sectionFilter.value = sec;
    }
    buildFeed();
  }

  function populateCategoryPills(sections) {
    const container = document.getElementById('reader-category-pills');
    if (!container || !sections || !sections.length) return;
    container.innerHTML = '';

    // 'All Stories' pill
    const allBtn = document.createElement('button');
    allBtn.type = 'button';
    allBtn.className = 'cat-pill' + (!State.get().activeSection ? ' active' : '');
    allBtn.dataset.section = '';
    allBtn.textContent = '🌟 All Stories';
    allBtn.addEventListener('click', () => selectCategory(''));
    container.appendChild(allBtn);

    // Primary priority sections first
    const primary = ['india', 'business', 'entertainment', 'sports', 'technology', 'explained', 'world', 'opinion', 'cities', 'lifestyle'];
    const sorted = [
      ...primary.filter(s => sections.includes(s)),
      ...sections.filter(s => !primary.includes(s))
    ];

    sorted.forEach(sec => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'cat-pill' + (State.get().activeSection === sec ? ' active' : '');
      btn.dataset.section = sec;
      const emoji = typeof sectionEmoji === 'function' ? sectionEmoji(sec) : '📁';
      btn.textContent = `${emoji} ${sec.charAt(0).toUpperCase() + sec.slice(1)}`;
      btn.addEventListener('click', () => selectCategory(sec));
      container.appendChild(btn);
    });
  }

  // ─── INITIALIZATION ─────────────────────────────────────────────────────────
  function init() {
    const nav = document.getElementById('tabs-nav');
    if (nav) {
      nav.addEventListener('click', (e) => {
        const btn = e.target.closest('.tab-btn');
        if (btn && btn.dataset.tab) {
          switchTab(btn.dataset.tab);
        }
      });
    }

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

    const pills = document.querySelectorAll('#quick-pills .pill');
    pills.forEach(pill => {
      pill.addEventListener('click', () => {
        const q = pill.dataset.q;
        if (q) performSearch(q);
      });
    });

    initStoryFinder();
    initModelSwitcher();
    buildFeed();
  }

  function reloadCurrentTab() {
    const { activeTab } = State.get();
    if (activeTab === 'similar') {
      buildFeed();
    } else if (activeTab === 'search') {
      const q = document.getElementById('search-query-input')?.value;
      if (q) performSearch(q);
    }
  }

  function clearArticlesCache() {
    allArticlesCache = [];
  }

  return {
    init, switchTab, performSearch, populateArticlePicker, buildFeed, reloadCurrentTab,
    populateCategoryPills, selectCategory, applyModelSwitch, selectArticle, clearArticlesCache,
    // Exposed so a widget can be rendered directly (the editor preview and the
    // render checks build one halo without booting the whole feed).
    buildFeedWidget, renderWidgetRecs,
    get _allArticlesCache() { return allArticlesCache; },
  };
})();

// Expose to window so app.js can detect and call Tabs methods
window.Tabs = Tabs;
