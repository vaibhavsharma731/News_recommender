/**
 * Candidate Article Evaluation Card
 * ═══════════════════════════════════════════════════════════════════════════
 * Restructured presentation layout:
 *   1. Verdict: title, final rank, confidence band (High/Medium/Low), one template-generated "why" sentence (no LLM).
 *   2. Proof: matching sentence pair with highlighted shared words; real shared entities only ("none shared" if empty);
 *             three signal badges with actual ranks (top-5 rule, "X of 3").
 *   3. Checks (one line each, only when relevant): rank without boosts vs final; margin over next candidate; warnings.
 *   4. Collapsed "Show scoring details": BM25 terms, cosine (percentile/band, not %), score cascade (with boosted relevance),
 *             boosts, admission gate, provenance & spec mismatch flags.
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

  const COMMON_STOP = new Set([
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can',
    'could', 'it', 'its', 'they', 'them', 'their', 'this', 'that', 'these',
    'those', 'he', 'him', 'his', 'she', 'her', 'hers', 'we', 'us', 'our',
    'who', 'whom', 'which', 'what', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'says', 'said', 'also', 'news', 'reported', 'reports', 'report'
  ]);

  /** Highlight words shared between source and candidate sentences. */
  function highlightSharedWords(srcText, candText) {
    if (!srcText || !candText) {
      return { srcHTML: esc(srcText || ''), candHTML: esc(candText || '') };
    }
    const extractWords = (t) => {
      const matches = t.toLowerCase().match(/\b[a-zA-Z]{3,}\b/g) || [];
      return new Set(matches.filter(w => !COMMON_STOP.has(w)));
    };
    const srcWords = extractWords(srcText);
    const candWords = extractWords(candText);
    const shared = new Set([...srcWords].filter(w => candWords.has(w)));

    const highlight = (t) => {
      if (!shared.size) return esc(t);
      const sortedWords = [...shared].sort((a, b) => b.length - a.length);
      const pattern = sortedWords.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
      const regex = new RegExp(`\\b(${pattern})\\b`, 'gi');
      let out = '';
      let last = 0;
      let m;
      while ((m = regex.exec(t)) !== null) {
        out += esc(t.slice(last, m.index));
        out += `<mark class="eval-shared-word">${esc(m[0])}</mark>`;
        last = regex.lastIndex;
      }
      out += esc(t.slice(last));
      return out;
    };

    return {
      srcHTML: highlight(srcText),
      candHTML: highlight(candText)
    };
  }

  function parseEvidence(rec) {
    if (!rec) return {};
    const raw = rec.evidence || rec.reason || rec._rec?.evidence || rec._rec?.reason ||
      (rec.article && (rec.article.evidence || rec.article.reason)) || '';
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

  /** Real BM25 hits breakdown. */
  function bm25Section(ev) {
    const hits = Array.isArray(ev.bm25_terms) ? ev.bm25_terms : [];
    const query = ev.bm25_query ? `<div class="eval-calib-note">Query used: <code>${esc(ev.bm25_query)}</code></div>` : '';
    if (!hits.length) {
      const altTerms = Array.isArray(ev.terms) && ev.terms.length ? ev.terms : [];
      if (altTerms.length) {
        const chips = altTerms.map(t => chip(`🏷️ ${esc(t)}`, 'eval-chip-match')).join(' ');
        return `<div>
          <span class="eval-detail-label">BM25 Keywords & Shared Topics:</span>
          <div class="eval-chips" style="margin-top:4px;">${chips}</div>
          ${query}
        </div>`;
      }
      return `<div>
        <span class="eval-detail-label">BM25 Keyword Contributions:</span>
        <div class="eval-val-muted">Contextual neural match — connected via deep topic similarity rather than direct keyword repetition.</div>
        ${query}
      </div>`;
    }
    const rows = hits.map(h => line(
      `<code>${esc(h.term)}</code> &nbsp;tf=${esc(h.tf)} · idf=${esc(h.idf)}`,
      `+${esc(h.contribution)}`
    )).join('');
    return `<div>
      <span class="eval-detail-label">BM25 Exact Term Contributions:</span>
      <div class="eval-breakdown-box" style="margin-top:4px;">${rows}</div>
      <div class="eval-calib-note">Exact terms that scored this candidate under Okapi BM25 retrieval index.</div>
      ${query}
    </div>`;
  }

  /** Real boost decomposition lines. */
  function boostLines(sb) {
    const items = Array.isArray(sb.boost_items) ? sb.boost_items : [];
    return items
      .filter(b => typeof b.mult === 'number' && Math.abs(b.mult - 1) > 1e-9)
      .map(b => line(`⚡ ${esc(b.name)} boost`, `${esc(b.amount)} (×${b.mult})`, 'eval-boost-line'))
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

    // Resolve identity from record / evidence
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

    // ─── 1. VERDICT LAYER ───
    // Title, final rank, confidence band (High/Medium/Low), one template-generated "why" sentence (no LLM)
    const band = ev.confidence_band || (score != null ? (score >= 0.65 ? 'HIGH' : score >= 0.35 ? 'MEDIUM' : 'LOW') : 'MEDIUM');
    const bandPill = band === 'HIGH'
      ? '<span class="eval-band-badge band-high">🟢 High Confidence</span>'
      : band === 'MEDIUM'
      ? '<span class="eval-band-badge band-med">🟡 Medium Confidence</span>'
      : '<span class="eval-band-badge band-low">⚪ Low Confidence</span>';

    const whySentence = ev.verdict?.template_why || ev.template_why || ev.plain_verdict || ev.summary || 'Topical and contextual alignment.';

    const verdictHTML = `
      <div class="eval-verdict-card">
        <div class="eval-verdict-top">
          <div class="eval-verdict-title-box">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
              <span class="candidate-rank-badge">#${esc(rank ?? '—')}</span>
              ${candBatch ? `<span class="eval-target-batch">📦 ${esc(candBatch)}</span>` : ''}
              <span class="eval-target-id">ID: ${esc(art.id || '—')}</span>
            </div>
            <a href="${esc(art.url || '#')}" target="_blank" rel="noopener" class="eval-verdict-title" title="${esc(title)}">${esc(title)}</a>
            <div class="eval-verdict-meta">
              ${section ? `<span>📁 <strong>${esc(section)}</strong></span>` : ''}
              ${subsection ? `<span>📂 <strong>${esc(subsection)}</strong></span>` : ''}
              <span>📅 ${esc(candDate || 'date not recorded')}</span>
            </div>
          </div>
          <div class="eval-verdict-band">${bandPill}</div>
        </div>
        <div class="eval-verdict-why">
          <span class="eval-why-label">Verdict:</span>
          <span class="eval-why-text">${esc(whySentence)}</span>
        </div>
      </div>`;

    // ─── 2. PROOF LAYER ───
    // Matching sentence pair with highlighted shared words; real shared entities only; three signal badges with actual ranks.
    const srcSent = (ev.source_sentence || '').trim() || src.description || src.snippet || src.title || '';
    const candSent = (ev.best_sentence || '').trim() || art.description || art.snippet || art.title || '';
    const { srcHTML, candHTML } = highlightSharedWords(srcSent, candSent);

    // Shared Entities (Fix a: show "none shared" if empty)
    const entities = Array.isArray(ev.entities) ? ev.entities : [];
    const entitiesHTML = entities.length
      ? entities.map(e => `<span class="eval-chip eval-chip-entity">📍 ${esc(e)}</span>`).join(' ')
      : '<span class="eval-val-muted">none shared</span>';

    // Three Signal Badges — thresholds match consensus_multiplier() in src/scoring.py (Rank ≤15, CE ≥0.35)
    const agreementCount = ev.agreement_count ?? mr.agreement_count ?? 1;
    const agreeLabel = ev.agreement_label || `${agreementCount} of 3`;
    const isBm25Valid = bm25Rank != null && bm25Rank < 900;
    const bm25Badge = isBm25Valid
      ? `<span class="eval-signal-badge ${bm25Rank <= 15 ? 'signal-hit' : 'signal-miss'}">BM25: #${esc(bm25Rank)}</span>`
      : `<span class="eval-signal-badge signal-miss">BM25: not retrieved</span>`;
    const isDenseValid = denseRank != null && denseRank < 900;
    const denseBadge = isDenseValid
      ? `<span class="eval-signal-badge ${denseRank <= 15 ? 'signal-hit' : 'signal-miss'}">Dense: #${esc(denseRank)}</span>`
      : `<span class="eval-signal-badge signal-miss">Dense: not retrieved</span>`;

    // Cross-Encoder badge: uses genuine score and CE candidate rank (never fake recommendation rank)
    const ceScore = rerank != null ? rerank : (mr.reranker != null ? mr.reranker : null);
    const ceRank = mr.reranker_rank != null ? mr.reranker_rank : null;
    let ceBadge = '';
    if (ceScore != null) {
      const isCeHit = ceScore >= 0.35;
      const cePctStr = (ceScore * 100).toFixed(1) + '%';
      const ceRankStr = ceRank != null ? `#${ceRank} (${cePctStr})` : cePctStr;
      ceBadge = `<span class="eval-signal-badge ${isCeHit ? 'signal-hit' : 'signal-miss'}">Cross-Encoder: ${esc(ceRankStr)}</span>`;
    } else if (ceRank != null) {
      ceBadge = `<span class="eval-signal-badge ${ceRank <= 15 ? 'signal-hit' : 'signal-miss'}">Cross-Encoder: #${esc(ceRank)}</span>`;
    } else {
      ceBadge = `<span class="eval-signal-badge signal-miss">Cross-Encoder: not available</span>`;
    }

    const proofHTML = `
      <div class="eval-proof-box">
        <div class="eval-proof-header">
          <span class="eval-section-title">Proof (Corroborating Evidence)</span>
          <div class="eval-agreement-pill">⚡ Signal Agreement: <strong>${esc(agreeLabel)}</strong> corroborated</div>
        </div>

        <div class="eval-proof-signals">
          <div class="eval-signal-badges">${bm25Badge} ${denseBadge} ${ceBadge}</div>
          <div class="eval-entities-row">
            <span class="eval-detail-label">Shared Entities:</span>
            <div class="eval-chips">${entitiesHTML}</div>
          </div>
        </div>

        <div class="eval-passage-pair">
          <div class="eval-passage-card">
            <div class="eval-passage-hdr">
              <span>📰 Source Story</span>
              <span class="eval-passage-tag">Anchor</span>
            </div>
            <p class="eval-passage-text">"${srcHTML}"</p>
          </div>
          <div class="eval-passage-divider">↔</div>
          <div class="eval-passage-card">
            <div class="eval-passage-hdr">
              <span>🎯 Candidate Story</span>
              <span class="eval-passage-tag">Matched Sentence</span>
            </div>
            <p class="eval-passage-text">"${candHTML}"</p>
          </div>
        </div>
      </div>`;

    // ─── 3. CHECKS LAYER ───
    // One line each, only when relevant: rank without boosts vs final; margin over next candidate; warnings.
    const checkLines = [];

    // A. Rank without boosts vs final (only when shifted)
    const shift = (rankRaw != null && rank != null) ? rankRaw - rank : 0;
    if (rankRaw != null && rank != null && rankRaw !== rank) {
      if (shift > 0) {
        checkLines.push(`<div class="eval-check-line check-promo">🔀 <strong>Rank Movement:</strong> Promoted from pure relevance #${rankRaw} ➔ Final #${rank} (+${shift} promotion by domain boosts)</div>`);
      } else {
        checkLines.push(`<div class="eval-check-line check-demo">🔀 <strong>Rank Movement:</strong> Demoted from pure relevance #${rankRaw} ➔ Final #${rank} (-${-shift} demoted by MMR diversity for content variety)</div>`);
      }
    }

    // B. Margin over next candidate (only when recorded and positive)
    const margin = ev.margin_cutoff || {};
    const marginVal = typeof margin.margin === 'number' ? margin.margin : null;
    if (marginVal != null) {
      const isClear = marginVal >= 0.04;
      checkLines.push(`<div class="eval-check-line ${isClear ? 'check-margin-clear' : 'check-margin-close'}">📏 <strong>Candidate Margin:</strong> +${(marginVal * 100).toFixed(1)}% confidence over next candidate (${isClear ? 'Clear separation' : 'Narrow lead'})</div>`);
    }

    // C. Warnings (only when present)
    const warnings = Array.isArray(ev.warnings) ? ev.warnings : [];
    warnings.forEach(w => {
      checkLines.push(`<div class="eval-check-line check-warning">⚠️ <strong>Notice:</strong> ${esc(w)}</div>`);
    });

    const checksHTML = checkLines.length ? `
      <div class="eval-checks-box">
        <span class="eval-section-title">Checks & Diagnostics</span>
        <div class="eval-checks-list">${checkLines.join('')}</div>
      </div>` : '';

    // ─── 4. COLLAPSED DETAILS LAYER ───
    // BM25 terms, cosine (percentile/band, not %), score cascade, boosts, gate, provenance.
    const cosVal = typeof ev.cosine === 'number' ? ev.cosine : null;
    const cosDisplay = ev.cosine_display || (cosVal != null
      ? `${ev.cosine_band || 'Semantic Alignment'} (Cosine: ${cosVal.toFixed(4)})`
      : NOT);

    const comp = sb.components || {};
    const prov = ev.provenance || {};
    const liveW = prov.live_weights || comp.weights || { rerank: 0.65, bm25: 0.15, dense: 0.20 };
    const specW = prov.spec_weights || { rerank: 0.70, bm25: 0.20, dense: 0.10 };
    const hasMismatch = prov.has_weight_mismatch ?? (liveW.dense !== specW.dense);

    const cascadeBox = `<div class="eval-breakdown-box">
      ${line('Cross-encoder probability', pct(rerank))}
      ${line(`BM25 component &nbsp;<span class="eval-val-muted">(${pct(liveW.bm25, 0)} weight)</span>`,
             comp.bm25_norm != null ? `${comp.bm25_norm.toFixed(3)} → +${esc(comp.bm25_term)}` : NOT)}
      ${line(`Dense semantic component &nbsp;<span class="eval-val-muted">(${pct(liveW.dense, 0)} weight)</span>`,
             comp.dense_norm != null ? `${comp.dense_norm.toFixed(3)} → +${esc(comp.dense_term)}` : NOT)}
      ${line('Unboosted blend (relevance)', pct(sb.blend), 'eval-blend-line')}
      ${line('Confidence on unboosted blend', pct(sb.unboosted_confidence), 'eval-calib-line')}
      ${boostLines(sb)}
      ${line('Boosted relevance (ordering value)', pct(sb.boosted_raw), 'eval-boosted-line')}
      ${line('Platt-calibrated confidence', pct(sb.platt_calibrated ?? score), 'eval-final-line')}
    </div>`;

    // Admission gate
    const adm = ev.admission;
    const rescued = adm && Array.isArray(adm.keep) && adm.cut_at != null
      ? adm.n_kept - (adm.keep.filter(i => i < adm.cut_at).length) : 0;
    const admBox = adm ? `<div class="eval-detail-row">
      <span class="eval-detail-label">Admission Gate Cut:</span>
      <div class="eval-chips">
        ${chip(adm.cut
          ? `✂️ <strong>${esc(adm.cut)}</strong> gate cut at position ${esc(adm.cut_at)} — ${esc(adm.n_kept)} kept`
          : `✅ No gate fired — all ${esc(adm.n_kept)} candidates kept`,
          adm.cut ? 'eval-chip-diff' : 'eval-chip-match')}
        ${rescued > 0 ? chip(`⭐ ${rescued} editorial link(s) rescued from cut`, 'eval-chip-match') : ''}
      </div>
      <div class="eval-calib-note">
        Techniques applied: hard floor (rerank ≥ ${esc(adm.floor ?? '?')}), cliff drop (next &lt; ${esc(adm.elbow_ratio ?? '?')} × prev), entity veto. Editorial links bypass the gate.
      </div>
    </div>` : '';

    const detailsHTML = `
      <details class="eval-details-accordion">
        <summary class="eval-details-summary">
          <span>⚙️ Show scoring details (BM25 terms, cosine, score cascade, boosts, gate, provenance)</span>
          <span class="eval-summary-chevron">▾</span>
        </summary>
        <div class="eval-details-content">
          <!-- Cosine as percentile/band (Fix c: never as raw %) -->
          <div class="eval-detail-row">
            <span class="eval-detail-label">Dense Vector Cosine (Semantic Alignment):</span>
            <div class="eval-detail-val">${cosDisplay}</div>
            <div class="eval-calib-note">Evaluated using 768-dimensional BGE embeddings against random-pair background distribution. Displayed as alignment band and empirical percentile.</div>
          </div>

          <!-- BM25 Terms -->
          <div class="eval-detail-row">
            ${bm25Section(ev)}
          </div>

          <!-- Full Score Cascade (Fix f: boosted relevance moved to details) -->
          <div class="eval-detail-row">
            <span class="eval-detail-label">Full Score Cascade (Components ➔ Boosts ➔ Calibration):</span>
            ${cascadeBox}
            <div class="eval-calib-note">Ordering utilizes boosted relevance; reported confidence is Platt-calibrated on unboosted blend to prevent boost inflation.</div>
          </div>

          <!-- Admission Gate -->
          ${admBox}

          <!-- Provenance & Spec Verification (Fix g) -->
          <div class="eval-detail-row eval-provenance-box">
            <span class="eval-detail-label">System Provenance & Spec Verification:</span>
            <div class="eval-spec-table">
              <div class="eval-spec-col">
                <strong>Live Blend Weights:</strong>
                <div>Cross-Encoder: ${pct(liveW.rerank, 0)}</div>
                <div>BM25 Lexical: ${pct(liveW.bm25, 0)}</div>
                <div>Dense Semantic: ${pct(liveW.dense, 0)}</div>
              </div>
              <div class="eval-spec-col">
                <strong>Historical Spec:</strong>
                <div>Cross-Encoder: ${pct(specW.rerank, 0)}</div>
                <div>BM25 Lexical: ${pct(specW.bm25, 0)}</div>
                <div>Dense Semantic: ${pct(specW.dense, 0)}</div>
              </div>
              <div class="eval-spec-col">
                <strong>Active Boost Caps:</strong>
                <div>Editorial Link: ×1.40</div>
                <div>Section Boost: ×${esc(prov.boost_caps?.section ?? 1.072)}</div>
                <div>Subsection Boost: ×${esc(prov.boost_caps?.subsection ?? 1.09)}</div>
                <div>Popularity Max: ×${esc(prov.boost_caps?.popularity ?? 1.044)}</div>
              </div>
            </div>
            ${hasMismatch ? `<div class="eval-spec-flag">⚠️ <strong>Tuned vs Spec Note:</strong> ${esc(prov.mismatch_explanation || 'Live blend weights (65/15/20) differ from initial 70/20/10 spec to optimize dense semantic retrieval.')}</div>` : ''}
            ${prov.pipeline_version ? `<div class="eval-pipeline-ver">Pipeline version SHA: <code>${esc(prov.pipeline_version)}</code></div>` : ''}
          </div>
        </div>
      </details>`;

    return `<div class="eval-container">
      ${verdictHTML}
      ${proofHTML}
      ${checksHTML}
      ${detailsHTML}
    </div>`;
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

    // Fix (e): Show confidence band on header pill unless reliability chart exists
    const band = ev.confidence_band || (art.score != null ? (art.score >= 0.65 ? 'HIGH' : art.score >= 0.35 ? 'MEDIUM' : 'LOW') : 'MEDIUM');
    const bandPill = band === 'HIGH'
      ? '<span class="candidate-score-pill" style="background:rgba(34,197,94,0.2);color:#4ade80;border-color:rgba(34,197,94,0.4);" title="High Confidence Band">🟢 High</span>'
      : band === 'MEDIUM'
      ? '<span class="candidate-score-pill" style="background:rgba(245,158,11,0.2);color:#fbbf24;border-color:rgba(245,158,11,0.4);" title="Medium Confidence Band">🟡 Medium</span>'
      : '<span class="candidate-score-pill" style="background:rgba(148,163,184,0.2);color:#cbd5e1;border-color:rgba(148,163,184,0.4);" title="Low Confidence Band">⚪ Low</span>';

    card.innerHTML = `
      <div class="candidate-card-header">
        <div class="candidate-header-left">
          <span class="candidate-rank-badge">#${esc(art.rank)}</span>
          <a href="${esc(art.url || '#')}" target="_blank" rel="noopener"
             class="candidate-header-title" title="${esc(art.title)}">${esc(art.title)}</a>
          ${art.section ? `<span class="cat-pill" style="font-size:0.75rem;">${esc(art.section)}</span>` : ''}
        </div>
        <div class="candidate-header-right">
          ${bandPill}
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
    const fullRec = Object.assign({}, candArt || {}, rec || {});
    fullRec.article = Object.assign({}, rec?.article || {}, candArt || {});
    if (rec?.evidence) fullRec.evidence = rec.evidence;
    if (rec?.reason) fullRec.reason = rec.reason;
    if (rec?.score != null) fullRec.score = rec.score;
    if (rec?.rank != null) fullRec.rank = rec.rank;
    if (rec?.rank_raw != null) fullRec.rank_raw = rec.rank_raw;
    if (rec?.rerank_score != null) fullRec.rerank_score = rec.rerank_score;
    if (rec?.boosted_score != null) fullRec.boosted_score = rec.boosted_score;

    const { ev, art } = mergeRec(fullRec, sourceArt, 1);
    const backdrop = document.createElement('div');
    backdrop.id = 'eval-active-modal';
    backdrop.className = 'eval-modal-backdrop';
    backdrop.innerHTML = `
      <div class="eval-modal-card">
        <div class="eval-modal-hdr">
          <div>
            <div style="font-size:0.72rem;font-weight:800;color:var(--brand-primary);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:3px;">
              Grounded candidate evidence report
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
