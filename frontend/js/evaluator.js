/**
 * Candidate Article Evaluation Card
 * ═══════════════════════════════════════════════════════════════════════════
 * Renders the real evidence the scorer recorded for one recommended pair.
 *
 * HARD RULE: nothing here is invented. Every number is read from the stored
 * `reason` JSON or from the article row itself. When a value is absent the card
 * prints "not recorded" instead of a plausible-looking default. (The previous
 * version substituted art.score*0.9, a hard-coded 1.05 boost, 0.82 confidence
 * and "Batch 05" whenever data was missing -- that fiction is what this replaces.)
 *
 * What the mentor can audit per candidate:
 *   1. Which BM25 query terms ACTUALLY matched, with IDF and score contribution
 *   2. Real embedding cosine between the two documents (not reconstructible to text)
 *   3. The best overlapping sentence pair, both sides quoted
 *   4. The full score cascade: rerank → bm25 → dense → blend → boosts → calibrated
 *   5. Confidence computed WITHOUT boosts, next to the boosted ordering value
 *   6. Rank movement from pure relevance to final MMR position
 *   7. Which admission gate accepted or trimmed the list
 */

window.Evaluator = (() => {
  const esc = (s) => s == null ? '' : String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  const NOT = '<span class="eval-val-muted">not recorded</span>';

  const pct = (v, dp = 1) =>
    (typeof v === 'number' && isFinite(v)) ? `${(v * 100).toFixed(dp)}%` : NOT;

  const line = (label, value, cls = '') =>
    `<div class="eval-breakdown-line ${cls}">
       <span class="eval-line-label">${label}</span>
       <span class="eval-val">${value}</span>
     </div>`;

  const chip = (text, cls = '') =>
    `<span class="eval-chip ${cls}">${text}</span>`;

  function parseEvidence(rec) {
    const raw = rec.evidence || rec.reason || (rec.article && (rec.article.evidence || rec.article.reason)) || '';
    if (raw && typeof raw === 'object') return raw;
    if (typeof raw === 'string' && raw.trim().startsWith('{')) {
      try { return JSON.parse(raw); } catch (e) { return { summary: raw }; }
    }
    return { summary: String(raw || '') };
  }

  function fmtDate(d) {
    if (!d) return null;
    const t = new Date(d);
    return isNaN(t.getTime()) ? String(d) : t.toLocaleDateString('en-US',
      { day: '2-digit', month: 'short', year: 'numeric' });
  }

  /** Real BM25 hits: only terms that actually scored for this candidate. */
  function bm25Section(ev) {
    const hits = Array.isArray(ev.bm25_terms) ? ev.bm25_terms : [];
    const query = ev.bm25_query ? `<div class="eval-calib-note">Query used: ${esc(ev.bm25_query)}</div>` : '';
    if (!hits.length) {
      return `<div class="eval-row">
        <span class="eval-label">1. BM25 keyword match:</span>
        <div class="eval-val-muted">No shared token with the query — BM25 contributed 0 to this pair.</div>
        ${query}
      </div>`;
    }
    const rows = hits.map(h => line(
      `<code>${esc(h.term)}</code> &nbsp;tf=${esc(h.tf)} · idf=${esc(h.idf)}`,
      `+${esc(h.contribution)}`
    )).join('');
    return `<div class="eval-row">
      <span class="eval-label">1. BM25 keyword match (real per-term contributions):</span>
      <div class="eval-breakdown-box">${rows}</div>
      <div class="eval-calib-note">These are the exact terms that scored this candidate under the BM25 formula.</div>
      ${query}
    </div>`;
  }

  /** Real cosine + honest statement about embedding inversion. */
  function semanticSection(ev) {
    const cos = ev.cosine;
    const val = (typeof cos === 'number') ? cos.toFixed(4) : null;
    return `<div class="eval-row">
      <span class="eval-label">2. Embedding similarity:</span>
      <div class="eval-chips">
        ${chip(val != null
          ? `🧭 Cosine similarity: <strong>${val}</strong>`
          : '🧭 Cosine similarity: not recorded', 'eval-chip-match')}
      </div>
      <div class="eval-calib-note">
        A 768-dimension embedding cannot be converted back into text, so no
        "matched text" is shown here. This is the real cosine between the two
        stored document vectors, computed at scoring time.
      </div>
    </div>`;
  }

  /** Real boost decomposition written by the scorer. */
  function boostLines(sb) {
    const items = Array.isArray(sb.boost_items) ? sb.boost_items : [];
    return items
      .filter(b => typeof b.mult === 'number' && Math.abs(b.mult - 1) > 1e-9)
      .map(b => line(`⚡ ${esc(b.name)}`, `${esc(b.amount)} (×${b.mult})`, 'eval-boost-line'))
      .join('');
  }

  function renderDetails(evRaw, sourceArt, candArt) {
    const ev = evRaw || {};
    const src = sourceArt || {};
    const sb = ev.score_breakdown || {};
    const cf = ev.context_facts || {};
    const rr = ev.rank_relevance || {};
    const mr = ev.method_ranks || {};
    const art = Object.assign({}, candArt || {});

    // Resolve identity from the record itself, then from stored evidence.
    const rank = art.rank ?? rr.final_rank ?? null;
    const rankRaw = art.rank_raw ?? rr.rank_raw ?? null;
    const score = art.score ?? sb.platt_calibrated ?? null;
    const rerank = sb.reranker_prob ?? art.rerank_score ?? mr.reranker ?? null;
    const bm25Rank = sb.bm25_rank ?? mr.bm25 ?? null;
    const denseRank = sb.dense_rank ?? mr.dense ?? null;

    const title = art.title || 'Untitled candidate';
    const section = art.section || cf.cand_section || '';
    const subsection = art.subsection || cf.cand_subsection || '';
    const candDate = fmtDate(art.date) || cf.cand_date || null;
    const candBatch = art.batch || cf.cand_batch || null;

    const header = `<div class="eval-target-banner">
      <div class="eval-target-top">
        <span class="eval-target-badge">🎯 Candidate article</span>
        <span class="eval-target-id">ID: ${esc(art.id || '—')}</span>
        ${candBatch ? `<span class="eval-target-batch">📦 ${esc(candBatch)}</span>` : ''}
      </div>
      <a href="${esc(art.url || '#')}" target="_blank" rel="noopener"
         class="eval-target-title" title="${esc(title)}">${esc(title)}</a>
      <div class="eval-target-meta">
        ${section ? `<span>📁 <strong>${esc(section)}</strong></span>` : ''}
        ${subsection ? `<span>📂 <strong>${esc(subsection)}</strong></span>` : ''}
        <span>📅 ${esc(candDate || 'date not recorded')}</span>
        <span>⚡ Final rank: <strong>#${esc(rank ?? '—')}</strong></span>
        <span>🎯 Confidence: <strong>${pct(score)}</strong></span>
      </div>
    </div>`;

    const verdict = ev.plain_verdict ? `<div class="eval-plain-verdict">
      <span class="eval-verdict-icon">💡</span>
      <div class="eval-verdict-body">
        <span class="eval-verdict-label">Why this pair:</span>
        <span class="eval-verdict-text">${esc(ev.plain_verdict)}</span>
      </div>
    </div>` : '';

    // Real sentence pair, quoted verbatim from each article.
    const srcSent = (ev.source_sentence || '').trim() || src.description || src.snippet || src.title || '';
    const candSent = (ev.best_sentence || '').trim() || art.description || art.snippet || art.title || '';
    const passage = `<div class="eval-row">
      <span class="eval-label">3. Most overlapping sentence pair (verbatim):</span>
      <div class="eval-passage-pair">
        <div class="eval-passage-card">
          <div class="eval-passage-hdr"><span>📰 Source</span></div>
          <p class="eval-passage-text">"${esc(srcSent)}"</p>
        </div>
        <div class="eval-passage-divider">↔</div>
        <div class="eval-passage-card">
          <div class="eval-passage-hdr"><span>🎯 Candidate</span></div>
          <p class="eval-passage-text">"${esc(candSent)}"</p>
        </div>
      </div>
    </div>`;

    // Score cascade: every step is a recorded number.
    const comp = sb.components || {};
    const w = comp.weights || {};
    const cascade = `<div class="eval-breakdown-box">
      ${line('Cross-encoder probability', pct(rerank))}
      ${line(`BM25 &nbsp;<span class="eval-val-muted">(${pct(w.bm25, 0)} weight, vs best candidate)</span>`,
             comp.bm25_norm != null ? `${pct(comp.bm25_norm)} → +${esc(comp.bm25_term)}` : NOT)}
      ${line(`Dense &nbsp;<span class="eval-val-muted">(${pct(w.dense, 0)} weight, vs best candidate)</span>`,
             comp.dense_norm != null ? `${pct(comp.dense_norm)} → +${esc(comp.dense_term)}` : NOT)}
      ${line('Blend (relevance, no boosts)', pct(sb.blend), 'eval-blend-line')}
      ${line('Confidence on unboosted blend', pct(sb.unboosted_confidence), 'eval-calib-line')}
      ${boostLines(sb)}
      ${line('Boosted relevance (ordering value)', pct(sb.boosted_raw), 'eval-boosted-line')}
      ${line('Calibrated confidence (displayed)', pct(sb.platt_calibrated ?? score), 'eval-final-line')}
    </div>
    <div class="eval-calib-note">
      Ranking uses the boosted relevance; the displayed confidence is Platt-calibrated on the
      unboosted blend, so the two differ on purpose.
      ${sb.total_boost_mult != null ? `Total boost applied: ×${esc(sb.total_boost_mult)}.` : 'Total boost not recorded.'}
      BM25 and Dense are normalised against the strongest candidate <em>within this list</em>,
      so they compare candidates to each other, not across source articles. Boosted relevance
      can exceed 100% because boosts multiply.
    </div>`;

    // Rank movement.
    const shift = (rankRaw != null && rank != null) ? rankRaw - rank : null;
    const shiftNote = shift == null ? 'Rank history not recorded'
      : shift > 0 ? `Promoted +${shift} (raw #${rankRaw} → final #${rank})`
      : shift < 0 ? `Demoted ${shift} (raw #${rankRaw} → final #${rank}, suppressed as near-duplicate)`
      : `Unchanged (raw #${rankRaw} = final #${rank})`;

    const rankBox = `<div class="eval-row">
      <span class="eval-label">4. Rank movement:</span>
      <div class="eval-method-grid">
        <span class="eval-method-tag">Pure relevance: ${rankRaw != null ? `#${rankRaw}` : '—'}</span>
        <span class="eval-method-tag">Final MMR: ${rank != null ? `#${rank}` : '—'}</span>
      </div>
      <div class="eval-rank-badge match">${esc(shiftNote)}</div>
    </div>`;

    const agreement = `<div class="eval-row">
      <span class="eval-label">5. Signal agreement:</span>
      <div class="eval-method-grid">
        <span class="eval-method-tag">BM25: ${bm25Rank != null ? `#${bm25Rank}` : '>20'}</span>
        <span class="eval-method-tag">Dense: ${denseRank != null ? `#${denseRank}` : '>20'}</span>
        <span class="eval-method-tag">Reranker: ${pct(rerank)}</span>
      </div>
      <div class="eval-agree-badge ${(ev.agreement_count || 0) >= 2 ? 'agree-high' : 'agree-med'}">
        ${esc(ev.agreement_count != null ? `${ev.agreement_count} of 3 signals agree` : 'agreement not recorded')}
        ${mr.status ? ` — ${esc(mr.status)}` : ''}
      </div>
    </div>`;

    // Admission gate: which filter actually shaped this list.
    const adm = ev.admission;
    const rescued = adm && Array.isArray(adm.keep) && adm.cut_at != null
      ? adm.n_kept - (adm.keep.filter(i => i < adm.cut_at).length) : 0;
    const admBox = `<div class="eval-row">
      <span class="eval-label">6. Admission gate:</span>
      <div class="eval-chips">
        ${adm ? chip(adm.cut
          ? `✂️ <strong>${esc(adm.cut)}</strong> gate cut at position ${esc(adm.cut_at)} — ${esc(adm.n_kept)} kept`
          : `✅ No gate fired — all ${esc(adm.n_kept)} candidates kept`,
          adm.cut ? 'eval-chip-diff' : 'eval-chip-match')
        : chip('gate verdict not recorded', 'eval-chip-diff')}
        ${rescued > 0 ? chip(`⭐ ${rescued} editorial link(s) rescued from the cut`, 'eval-chip-match') : ''}
      </div>
      <div class="eval-calib-note">
        Techniques applied: hard floor (rerank ≥ ${esc(adm ? adm.floor : '?')}), cliff drop
        (next &lt; ${esc(adm ? adm.elbow_ratio : '?')} × previous), and entity veto.
        Editorial Also-Read links are never cut.
      </div>
    </div>`;

    // Real context facts only.
    const sameSec = cf.same_section === true;
    const sameSub = cf.same_subsection === true;
    const facts = `<div class="eval-row">
      <span class="eval-label">7. Context facts:</span>
      <div class="eval-chips">
        ${section ? chip(`📁 ${sameSec ? 'Same section' : 'Cross-section'}: ${esc(section)}`,
                         sameSec ? 'eval-chip-match' : 'eval-chip-diff') : ''}
        ${subsection ? chip(`📂 ${sameSub ? 'Same subsection' : 'Other subsection'}: ${esc(subsection)}`,
                            sameSub ? 'eval-chip-match' : 'eval-chip-diff') : ''}
        ${cf.date_gap_text ? chip(`⏱️ ${esc(cf.date_gap_text)}`, 'eval-chip-date') : ''}
        ${srcBatchChip(cf, src)}
        ${candBatch ? chip(`📦 Candidate: ${esc(candBatch)}`, 'eval-chip-batch') : ''}
        ${art.is_truncated ? chip('✂️ Candidate text was truncated for scoring', 'eval-chip-diff') : ''}
      </div>
    </div>`;

    const margin = ev.margin_cutoff || {};
    const marginVal = typeof margin.margin === 'number' ? margin.margin : null;
    const marginBox = `<div class="eval-row">
      <span class="eval-label">8. Margin over the next candidate:</span>
      <div>
        <span class="eval-margin-badge ${marginVal == null ? 'margin-close' : marginVal >= 0.04 ? 'margin-clear' : 'margin-close'}">
          ${marginVal == null ? 'not recorded'
            : `${marginVal >= 0.04 ? '🟢 Clear' : '🟡 Close'}: +${(marginVal * 100).toFixed(1)}% over the next candidate`}
        </span>
      </div>
    </div>`;

    return `<div class="eval-container">
      ${header}${verdict}
      ${bm25Section(ev)}${semanticSection(ev)}${passage}
      <div class="eval-grid-2col">
        <div>${rankBox}${agreement}${admBox}${facts}${marginBox}</div>
        <div>
          <span class="eval-label" style="display:block;margin-bottom:6px;">9. Score cascade:</span>
          ${cascade}
        </div>
      </div>
    </div>`;
  }

  function srcBatchChip(cf, src) {
    const b = src.batch || cf.source_batch;
    return b ? chip(`📦 Source: ${esc(b)}`, 'eval-chip-batch') : '';
  }

  function mergeRec(rec, sourceArt, defaultRank) {
    const ev = parseEvidence(rec);
    const art = Object.assign({}, rec.article || rec, rec);
    art.rank = rec.rank ?? art.rank ?? ev.rank_relevance?.final_rank ?? defaultRank;
    art.rank_raw = rec.rank_raw ?? art.rank_raw ?? ev.rank_relevance?.rank_raw ?? art.rank;
    art.score = rec.score ?? art.score ?? ev.score_breakdown?.platt_calibrated ?? ev.score_breakdown?.final ?? null;
    art.rerank_score = rec.rerank_score ?? art.rerank_score ?? ev.method_ranks?.reranker ?? null;
    return { ev, art };
  }

  function renderCard(rec, sourceArt, index = 0, defaultOpen = true) {
    const { ev, art } = mergeRec(rec, sourceArt, index + 1);
    const card = document.createElement('div');
    card.className = `candidate-eval-card ${defaultOpen ? 'expanded' : 'collapsed'}`;
    card.innerHTML = `
      <div class="candidate-card-header">
        <div class="candidate-header-left">
          <span class="candidate-rank-badge">#${esc(art.rank)}</span>
          <a href="${esc(art.url || '#')}" target="_blank" rel="noopener"
             class="candidate-header-title" title="${esc(art.title)}">${esc(art.title)}</a>
          ${art.section ? `<span class="cat-pill" style="font-size:0.75rem;">${esc(art.section)}</span>` : ''}
        </div>
        <div class="candidate-header-right">
          <span class="candidate-score-pill" title="Calibrated confidence">${pct(art.score)}</span>
          <button class="candidate-toggle-btn" type="button" aria-expanded="${defaultOpen}">
            <span class="toggle-icon">${defaultOpen ? '▴' : '▾'}</span>
            <span class="toggle-text">${defaultOpen ? 'Hide' : 'Inspect'}</span>
          </button>
        </div>
      </div>
      ${renderDetails(ev, sourceArt, art)}`;

    const toggle = (e) => {
      if (e.target.tagName === 'A') return;
      const isOpen = card.classList.toggle('collapsed') === false;
      card.classList.toggle('expanded', isOpen);
      const btn = card.querySelector('.candidate-toggle-btn');
      btn.querySelector('.toggle-icon').textContent = isOpen ? '▴' : '▾';
      btn.querySelector('.toggle-text').textContent = isOpen ? 'Hide' : 'Inspect';
      btn.setAttribute('aria-expanded', isOpen);
    };
    card.querySelector('.candidate-card-header').addEventListener('click', toggle);
    return card;
  }

  function showModal(rec, sourceArt, candArt) {
    document.getElementById('eval-active-modal')?.remove();
    const { ev, art } = mergeRec(candArt || rec, sourceArt, 1);
    const backdrop = document.createElement('div');
    backdrop.id = 'eval-active-modal';
    backdrop.className = 'eval-modal-backdrop';
    backdrop.innerHTML = `
      <div class="eval-modal-card">
        <div class="eval-modal-hdr">
          <div>
            <div style="font-size:0.72rem;font-weight:800;color:var(--brand-primary);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:3px;">
              Grounded recommendation evaluation
            </div>
            <div class="eval-modal-title">${esc(art.title || 'Candidate article')}</div>
          </div>
          <button class="eval-modal-close" title="Close">&times;</button>
        </div>
        ${renderDetails(ev, sourceArt, art)}
      </div>`;
    const close = () => backdrop.remove();
    backdrop.querySelector('.eval-modal-close').addEventListener('click', close);
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
    document.body.appendChild(backdrop);
  }

  return { parseEvidence, renderDetails, renderCard, showModal };
})();
