/**
 * Recommendation Evidence Card — full technical score journey
 * ═══════════════════════════════════════════════════════════════════════════
 * Renders, from the STORED evidence JSON only:
 *
 *   BM25 + BGE Dense + Cross-Encoder
 *        → Raw Relevance (weighted blend)
 *        → Editorial & Business Boosts (multiplicative)
 *        → Signal Agreement Adjustment (additive)
 *        → Boosted Relevance Score
 *        → MMR Diversity selection
 *        → Final Rank
 *
 * Rules this file obeys:
 *   • It never invents, estimates or re-scores anything. Every displayed number
 *     comes from `evidence.score_breakdown`, `evidence.method_ranks`,
 *     `evidence.rank_relevance` or the record's own columns.
 *   • A missing value renders as "not recorded" — never as 0.
 *   • Derived values (the reconstructed agreement rule, the running boost
 *     product) are labelled as reconstructions and cross-checked against the
 *     stored result, with a visible warning when they disagree.
 *   • There is NO human-evaluation surface: no ratings, labels, notes,
 *     reviewed status, metrics dashboard or label storage.
 *
 * Thresholds mirror src/scoring.py exactly:
 *   rank ≤ 5 strong · ≤ 15 moderate · ≤ 25 weak · ≥ 900 / null = none
 *   CE ≥ 0.55 strong · ≥ 0.35 moderate · below = weak
 */
window.Evaluator = (() => {
  // ── Utilities ────────────────────────────────────────────────────────────
  const esc = (s) => s == null ? '' : String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  const NOT = '<span class="tec-muted">not recorded</span>';

  const isNum = (v) => typeof v === 'number' && isFinite(v);

  const num = (v) => {
    if (isNum(v)) return v;
    if (typeof v === 'string' && v.trim() !== '' && isFinite(Number(v))) return Number(v);
    return null;
  };

  const firstNum = (...vals) => {
    for (const v of vals) { const n = num(v); if (n !== null) return n; }
    return null;
  };

  const firstDefined = (...vals) => {
    for (const v of vals) if (v !== undefined && v !== null) return v;
    return null;
  };

  const intOrNull = (v) => {
    const n = num(v);
    return n === null ? null : Math.round(n);
  };

  const f = (v, dp = 4) => isNum(v) ? v.toFixed(dp) : null;

  const signed = (v, dp = 4) => isNum(v)
    ? `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(dp)}`
    : null;

  const pctSigned = (mult, dp = 1) => isNum(mult)
    ? `${mult >= 1 ? '+' : '−'}${Math.abs((mult - 1) * 100).toFixed(dp)}%`
    : null;

  const asArray = (v) => Array.isArray(v) ? v : [];

  const cssId = (s) => String(s == null ? '' : s).replace(/[^a-zA-Z0-9_-]/g, '_');

  const clamp01 = (v) => Math.max(0, Math.min(1, v));

  const fmtDate = (d) => {
    if (!d) return null;
    if (d instanceof Date && !isNaN(d.getTime())) {
      return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    }
    const raw = String(d);
    const t = new Date(raw);
    if (!isNaN(t.getTime())) {
      return t.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    }
    return raw;
  };

  const list = (items) => {
    const a = asArray(items).filter(Boolean);
    if (!a.length) return '';
    if (a.length === 1) return a[0];
    return `${a.slice(0, -1).join(', ')} and ${a[a.length - 1]}`;
  };

  const currentPipeline = () => {
    try {
      return (typeof State !== 'undefined' && State.get().modelVersion) || '384d';
    } catch (e) {
      return '384d';
    }
  };

  // ── Signal grading — mirrors src/scoring.py ──────────────────────────────
  // Two grade tables exist in the backend and they are NOT identical:
  //   • evaluate_signal_agreement() counts a signal when it is strong/moderate
  //     (rank ≤ 15, CE ≥ 0.35). Rank 16–25 does not count.
  //   • signal_agreement_adjustment() uses _signal_grade(), which adds a
  //     "weak" tier for rank 16–25 before falling back to "none".
  // Both are reproduced here so the reconstruction matches the stored value.
  const STRONG_RANK = 5, MOD_RANK = 15, WEAK_RANK = 25;
  const CE_STRONG = 0.55, CE_MOD = 0.35;

  function countGrade(rank) {
    if (rank === null || rank >= 900) return 'none';
    if (rank <= STRONG_RANK) return 'strong';
    if (rank <= MOD_RANK) return 'moderate';
    return 'none';
  }

  function adjGrade(rank) {
    if (rank === null || rank >= 900) return 'none';
    if (rank <= STRONG_RANK) return 'strong';
    if (rank <= MOD_RANK) return 'moderate';
    if (rank <= WEAK_RANK) return 'weak';
    return 'none';
  }

  function ceCountGrade(p) {
    if (p === null) return 'none';
    if (p >= CE_STRONG) return 'strong';
    if (p >= CE_MOD) return 'moderate';
    return 'none';
  }

  function ceAdjGrade(p) {
    if (p === null) return 'weak';
    if (p >= CE_STRONG) return 'strong';
    if (p >= CE_MOD) return 'moderate';
    return 'weak';
  }

  const GRADE_LABEL = { strong: 'Strong', moderate: 'Moderate', weak: 'Weak', none: 'Below threshold' };

  // ── Evidence parsing ─────────────────────────────────────────────────────
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

  function mergeRec(rec, sourceArt, defaultRank) {
    const ev = parseEvidence(rec);
    const art = Object.assign({}, rec.article || rec, rec);
    art.rank = rec.rank ?? art.rank ?? ev.rank_relevance?.final_rank ?? defaultRank;
    art.rank_raw = rec.rank_raw ?? art.rank_raw ?? ev.rank_relevance?.rank_raw ?? art.rank;
    art.score = firstNum(rec.score, art.score, ev.score_breakdown?.platt_calibrated, ev.score_breakdown?.final);
    art.boosted_score = firstNum(rec.boosted_score, art.boosted_score, ev.score_breakdown?.boosted_raw);
    art.rerank_score = firstNum(rec.rerank_score, art.rerank_score, ev.score_breakdown?.reranker_prob);
    return { ev, art };
  }

  // ── Stage A–F cascade reconstruction ─────────────────────────────────────
  const MULT_BOOST_ORDER = ['editorial', 'section', 'subsection', 'popularity', 'recency'];
  const MULT_BOOST_LABEL = {
    editorial: 'Editorial relationship',
    section: 'Section match',
    subsection: 'Subsection match',
    popularity: 'Popularity (page views)',
    recency: 'Recency (vs source date)'
  };

  function findBoost(boostItems, name) {
    return asArray(boostItems).find(b => String(b && b.name) === name) || null;
  }

  /**
   * Reconstruct the signal-agreement rule that produced the stored adjustment.
   * This is a mirror of src/scoring.py::signal_agreement_adjustment, used only
   * to EXPLAIN the number. The stored number always wins on screen.
   */
  function reconstructAgreement(bm25Rank, denseRank, ceProb, editorialApplied) {
    if (editorialApplied) {
      return { rule: 'Editorial "Also Read" link', expected: 0, note: 'Editorial pairs are neutral by design.' };
    }
    if (bm25Rank === null && denseRank === null && ceProb === null) {
      return { rule: 'No signal recorded', expected: 0, note: 'Nothing to agree on.' };
    }
    const b = adjGrade(bm25Rank);
    const d = adjGrade(denseRank);
    const g = ceAdjGrade(ceProb);
    const dAbsent = denseRank === null || denseRank > 20;
    const bAbsent = bm25Rank === null || bm25Rank > 20;

    if (g === 'absent' || g === 'none' || ceProb === null || ceProb === 0) {
      if (b === 'strong' && d === 'strong')
        return { rule: 'Dual retrieval agreement — BM25 strong + Dense strong', expected: 0.02, note: 'Both lexical and semantic retrieval ranked this candidate in their top-5 pool.' };
      if ((b === 'strong' && d === 'moderate') || (b === 'moderate' && d === 'strong'))
        return { rule: 'Dual retrieval agreement — strong + moderate', expected: 0.015, note: 'One retrieval signal is strong (top-5) and the other is moderate (top-20).' };
      if (b === 'moderate' && d === 'moderate')
        return { rule: 'Dual retrieval agreement — BM25 moderate + Dense moderate', expected: 0.01, note: 'Both retrieval signals placed this candidate in the top-20 pool.' };
      if ((b === 'strong' && (d === 'weak' || d === 'none')) || (d === 'strong' && (b === 'weak' || b === 'none')))
        return { rule: 'Single strong signal discovery', expected: 0.01, note: 'One retrieval branch placed this candidate in the top-5 pool.' };
      if ((b === 'moderate' && (d === 'weak' || d === 'none')) || (d === 'moderate' && (b === 'weak' || b === 'none')))
        return { rule: 'Single moderate signal discovery', expected: 0.005, note: 'One retrieval branch placed this candidate in the top-20 pool.' };
      if (b === 'weak' && d === 'weak')
        return { rule: 'Dual weak signals', expected: -0.005, note: 'Both signals placed this candidate beyond the top-20 pool.' };
      if ((b === 'weak' && d === 'none') || (d === 'weak' && b === 'none'))
        return { rule: 'Weak uncorroborated signal', expected: -0.01, note: 'Only a weak signal outside top-20 with the other signal absent.' };
      return { rule: 'Neutral dual retrieval', expected: 0.00, note: 'The agreement table produced a neutral adjustment.' };
    }

    if (b === 'strong' && d === 'strong' && g === 'strong')
      return { rule: 'Triple agreement — BM25 strong + Dense strong + CE strong', expected: 0.03, note: 'All three signals support the pair.' };
    if (b === 'none' && d === 'strong' && g === 'strong')
      return { rule: 'Strong semantic match — Dense strong + CE strong, no BM25', expected: 0.02, note: 'Semantic discovery is rewarded, not penalised for missing keywords.' };
    if (b === 'none' && d === 'moderate' && g === 'strong')
      return { rule: 'Moderate semantic match — Dense moderate + CE strong, no BM25', expected: 0.01, note: 'Weaker semantic retrieval, still CE-confirmed.' };
    if (b === 'strong' && d === 'strong' && g === 'moderate')
      return { rule: 'Strong retrieval agreement — BM25 strong + Dense strong + CE moderate', expected: 0.015, note: 'Retrieval agrees, CE is only moderately convinced.' };
    if (b === 'strong' && dAbsent && g === 'weak')
      return { rule: 'Keyword fluke — BM25 strong, Dense absent, CE weak', expected: -0.03, note: 'A lexical match the neural signals do not support.' };
    if (bAbsent && d === 'strong' && g === 'weak')
      return { rule: 'Embedding drift — Dense strong, BM25 absent, CE weak', expected: -0.02, note: 'Vector proximity the cross-encoder does not confirm.' };
    if (b === 'moderate' && dAbsent && g === 'weak')
      return { rule: 'Weak keyword match — BM25 moderate, Dense absent, CE weak', expected: -0.02, note: 'Partial lexical match with no neural support.' };
    if (bAbsent && d === 'moderate' && g === 'weak')
      return { rule: 'Weak semantic match — Dense moderate, BM25 absent, CE weak', expected: -0.015, note: 'Partial vector match with no neural support.' };
    return { rule: 'No rule fired — signals inconclusive', expected: 0, note: 'The agreement table produced a neutral adjustment.' };
  }

  function buildCascade(ev, art, meta) {
    const sb = ev.score_breakdown || {};
    const mr = ev.method_ranks || {};
    const comp = sb.components || {};
    const cf = ev.context_facts || {};
    const boostItems = asArray(sb.boost_items);

    // ---- Stage A: individual signals -------------------------------------
    const ceProb = firstNum(sb.reranker_prob, mr.reranker, art.rerank_score);
    const ceRank = intOrNull(firstNum(mr.reranker_rank));
    const hasCe = (ceProb !== null && ceProb > 0) || (ceRank !== null && ceRank < 900);
    const bm25Rank = intOrNull(firstNum(sb.bm25_rank, mr.bm25));
    const denseRank = intOrNull(firstNum(sb.dense_rank, mr.dense));
    const cosine = firstNum(ev.cosine);
    const cosinePct = firstNum(ev.cosine_percentile);

    const bm25Norm = firstNum(comp.bm25_norm);
    const denseNorm = firstNum(comp.dense_norm);
    const bm25Term = firstNum(comp.bm25_term);
    const denseTerm = firstNum(comp.dense_term);
    const rerankTerm = firstNum(comp.rerank_term);

    const weights = comp.weights || (meta && meta.weights) || {};
    const wRerank = firstNum(weights.rerank);
    const wBm25 = firstNum(weights.bm25);
    const wDense = firstNum(weights.dense);
    const weightsFromRecord = !!comp.weights;

    // ---- Stage B: raw relevance ------------------------------------------
    const blend = firstNum(sb.blend);
    const activeTerms = hasCe ? [rerankTerm, bm25Term, denseTerm] : [bm25Term, denseTerm];
    const termSum = activeTerms.every(v => v !== null)
      ? activeTerms.reduce((sum, val) => sum + val, 0) : null;
    const blendDelta = (blend !== null && termSum !== null) ? Math.abs(blend - termSum) : null;

    // ---- Stage C: multiplicative boosts ----------------------------------
    let running = blend;
    const boosts = MULT_BOOST_ORDER.map((name) => {
      const item = findBoost(boostItems, name);
      const mult = item ? num(item.mult) : null;
      const before = running;
      const after = (running !== null && mult !== null) ? running * mult : null;
      if (after !== null) running = after;

      const applied = mult !== null && (mult - 1) > 1e-4;
      let status, statusKey;
      if (applied) { statusKey = 'applied'; status = 'Applied'; }
      else if (name === 'editorial') { statusKey = 'inactive'; status = 'Not applied'; }
      else if ((name === 'section' || name === 'subsection') && !(name === 'section' ? cf.same_section : cf.same_subsection)) {
        statusKey = 'na'; status = 'Not applicable';
      }
      else if (name === 'popularity' && !(firstNum(art.pageViews) > 0)) { statusKey = 'na'; status = 'Not applicable'; }
      else { statusKey = 'neutral'; status = 'No change'; }

      return {
        name,
        label: MULT_BOOST_LABEL[name],
        mult,
        stored: !!item,
        applied,
        status,
        statusKey,
        reason: boostReason(name, { applied, cf, art, mult }),
        before,
        after
      };
    });

    const storedDomain = firstNum(findBoost(boostItems, 'domain_multiplier')?.mult, sb.domain_multiplier, sb.total_boost_mult);
    const productMult = boosts.every(b => b.mult !== null)
      ? boosts.reduce((acc, b) => acc * b.mult, 1) : null;
    const preAgreement = running; // blend × product of all five boosts

    // ---- Stage D: additive signal agreement ------------------------------
    const storedAdj = firstNum(
      findBoost(boostItems, 'agreement_adjustment')?.mult,
      sb.agreement_adjustment
    );
    const editorialApplied = !!boosts.find(b => b.name === 'editorial')?.applied;
    const recon = reconstructAgreement(bm25Rank, denseRank, ceProb, editorialApplied);
    const adjustment = storedAdj !== null ? storedAdj : null;
    const agreementCount = intOrNull(firstNum(ev.agreement_count, mr.agreement_count));
    const maxSignals = intOrNull(firstNum(ev.max_signals, sb.max_signals)) || (hasCe ? 3 : 2);
    const countGrades = {
      bm25: countGrade(bm25Rank),
      dense: countGrade(denseRank),
      crossEncoder: ceCountGrade(ceProb)
    };
    const adjGrades = {
      bm25: adjGrade(bm25Rank),
      dense: adjGrade(denseRank),
      crossEncoder: ceAdjGrade(ceProb)
    };
    const reconstructedCount = hasCe
      ? [countGrades.bm25, countGrades.dense, countGrades.crossEncoder].filter(g => g === 'strong' || g === 'moderate').length
      : [countGrades.bm25, countGrades.dense].filter(g => g === 'strong' || g === 'moderate').length;

    // ---- Stage E: boosted relevance --------------------------------------
    const boostedStored = firstNum(sb.boosted_raw, art.boosted_score);
    const boostedComputed = (preAgreement !== null && adjustment !== null)
      ? preAgreement + adjustment : null;
    const boostedDelta = (boostedStored !== null && boostedComputed !== null)
      ? Math.abs(boostedStored - boostedComputed) : null;

    // ---- Stage F: MMR + rank ---------------------------------------------
    const rr = ev.rank_relevance || {};
    const rankRaw = intOrNull(firstNum(rr.rank_raw, art.rank_raw, art.rank));
    const finalRank = intOrNull(firstNum(rr.final_rank, art.rank));
    const shift = (rankRaw !== null && finalRank !== null) ? rankRaw - finalRank : null;
    const mmrMeta = (meta && meta.mmr) || null;

    // ---- Verification warnings -------------------------------------------
    const warnings = [];
    if (blendDelta !== null && blendDelta > 0.0015) {
      warnings.push(`Weighted components sum to ${termSum.toFixed(4)} but the stored raw relevance is ${blend.toFixed(4)} (Δ ${blendDelta.toFixed(4)}). The stored value is shown.`);
    }
    if (storedDomain !== null && productMult !== null && Math.abs(storedDomain - productMult) > 0.0015) {
      warnings.push(`Boost product reconstructs to ×${productMult.toFixed(4)} but the stored domain multiplier is ×${storedDomain.toFixed(4)}. The stored value is shown.`);
    }
    if (storedAdj !== null && Math.abs(recon.expected - storedAdj) > 1e-6) {
      warnings.push(`The agreement rule "${recon.rule}" reconstructs ${signed(recon.expected, 3)} but the stored adjustment is ${signed(storedAdj, 3)}. The stored value is shown.`);
    }
    if (agreementCount !== null && reconstructedCount !== agreementCount) {
      warnings.push(`Stored agreement count is ${agreementCount} of ${maxSignals}; the displayed signal grades reconstruct ${reconstructedCount} of ${maxSignals}.`);
    }
    if (boostedDelta !== null && boostedDelta > 0.0015) {
      warnings.push(`Boosted score reconstructs to ${boostedComputed.toFixed(4)} but the stored value is ${boostedStored.toFixed(4)}. The stored value is shown.`);
    }

    return {
      // stage A
      ceProb, ceRank, bm25Rank, denseRank, cosine, cosinePct,
      bm25Norm, denseNorm, bm25Term, denseTerm, rerankTerm,
      wRerank, wBm25, wDense, weightsFromRecord,
      countGrades, adjGrades, agreementCount, reconstructedCount, maxSignals, hasCe,
      // stage B
      blend, termSum, blendDelta,
      // stage C
      boosts, productMult, storedDomain, preAgreement,
      // stage D
      adjustment, storedAdj, recon, editorialApplied,
      // stage E
      boostedStored, boostedComputed, boostedDelta,
      // stage F
      rankRaw, finalRank, shift, mmrMeta,
      // misc
      warnings,
      storedWarnings: asArray(ev.warnings),
      confidence: ev.confidence || {},
      band: ev.confidence_band || null
    };
  }

  function boostReason(name, ctx) {
    const { applied, cf, art, mult } = ctx;
    const srcSec = cf.source_section || '—';
    const candSec = cf.cand_section || art.section || '—';
    const gap = cf.date_gap_text || null;

    if (name === 'editorial') {
      return applied
        ? 'The editorial desk linked this story from the source article as an “Also Read” follow-up.'
        : 'No editorial “Also Read” link is recorded for this pair, so the editorial multiplier stays at ×1.000.';
    }
    if (name === 'section') {
      if (applied) return `Both articles are filed under “${candSec}”, so the section multiplier is active (×${mult.toFixed(3)}).`;
      return cf.same_section
        ? 'The sections match but the evaluated multiplier is ×1.000.'
        : `Sections differ (“${candSec}” vs “${srcSec}”), so no section multiplier applies.`;
    }
    if (name === 'subsection') {
      if (applied) return `Both articles share the subsection “${cf.cand_subsection || '—'}”.`;
      return cf.same_subsection
        ? 'The subsections match but the evaluated multiplier is ×1.000.'
        : 'The subsections differ, so no subsection multiplier applies.';
    }
    if (name === 'popularity') {
      const views = firstNum(art.pageViews);
      if (applied) return `Log-scaled from ${views !== null ? views.toLocaleString() : 'recorded'} page views against the corpus maximum.`;
      return views !== null
        ? `${views.toLocaleString()} page views produced no measurable multiplier.`
        : 'No page-view count is recorded for this candidate, so the popularity multiplier is ×1.000.';
    }
    if (name === 'recency') {
      if (applied) return `Published ${gap || 'close to the source article'}; the source-relative decay still returns a positive multiplier.`;
      return `Published ${gap || 'far from the source article'}; the decay has reduced the recency multiplier to ×1.000.`;
    }
    return '';
  }

  // ── Discovery classification ─────────────────────────────────────────────
  function classifyDiscovery(c) {
    const kw = c.countGrades.bm25 === 'strong' || c.countGrades.bm25 === 'moderate';
    const sem = c.countGrades.dense === 'strong' || c.countGrades.dense === 'moderate';
    const ce = c.countGrades.crossEncoder === 'strong' || c.countGrades.crossEncoder === 'moderate';
    const bm25Miss = c.bm25Rank === null;

    if (bm25Miss && (sem || ce)) {
      return {
        key: 'semantic',
        label: 'SEMANTIC DISCOVERY — LOW KEYWORD OVERLAP',
        text: 'BM25 did not retrieve this article. The dense vector space and/or the cross-encoder still place it close, so the recommendation comes from semantic understanding rather than shared vocabulary.'
      };
    }
    if (kw && (sem || ce)) {
      return {
        key: 'combined',
        label: 'COMBINED KEYWORD + SEMANTIC',
        text: 'BM25 retrieved the article on shared terms and at least one neural signal corroborates it.'
      };
    }
    if (kw) {
      return {
        key: 'keyword',
        label: 'KEYWORD-LED',
        text: 'BM25 retrieved the article on shared vocabulary. The neural signals did not reach the strong/moderate band.'
      };
    }
    if (ce) {
      return {
        key: 'contextual',
        label: 'CONTEXTUAL CROSS-ENCODER SUPPORT',
        text: 'The cross-encoder reads the pair as contextually related even though neither retrieval signal ranked it strongly.'
      };
    }
    if (sem) {
      return {
        key: 'semantic',
        label: 'SEMANTIC-LED',
        text: 'The dense vector space places the pair close; keyword and cross-encoder support is weak.'
      };
    }
    return {
      key: 'weak',
      label: 'WEAK SIGNAL SUPPORT',
      text: 'No signal reached the strong or moderate band. Treat this recommendation as low-confidence.'
    };
  }

  // ── Article comparison ───────────────────────────────────────────────────
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
        out += `<mark class="tec-shared-word">${esc(m[0])}</mark>`;
        last = regex.lastIndex;
      }
      out += esc(t.slice(last));
      return out;
    };

    return { srcHTML: highlight(srcText), candHTML: highlight(candText) };
  }

  function articleLead(art) {
    const desc = String(art?.description || '').trim();
    if (desc) return desc;
    return String(art?.snippet || '').trim();
  }

  function panelHTML(kind, art, excerptHTML, fallbackTitle) {
    const title = art?.title || fallbackTitle || 'Untitled article';
    const section = art?.section || '';
    const subsection = art?.subsection || '';
    const date = fmtDate(art?.date);
    const lead = articleLead(art);
    const url = art?.url || '';
    const batch = art?.batch || '';
    const meta = [
      section ? `<span>${esc(section)}${subsection ? ` · ${esc(subsection)}` : ''}</span>` : '',
      date ? `<span>${esc(date)}</span>` : '',
      batch ? `<span>${esc(batch)}</span>` : ''
    ].filter(Boolean).join('<span class="tec-meta-dot">·</span>');

    return `
      <div class="tec-compare-panel tec-compare-${kind}">
        <div class="tec-panel-kicker">${kind === 'source' ? 'Source article' : 'Recommended article'}</div>
        ${url
        ? `<a class="tec-panel-title" href="${esc(url)}" target="_blank" rel="noopener">${esc(title)}</a>`
        : `<div class="tec-panel-title">${esc(title)}</div>`}
        ${meta ? `<div class="tec-panel-meta">${meta}</div>` : ''}
        ${lead
        ? `<p class="tec-panel-lead">${esc(lead)}</p>`
        : `<p class="tec-panel-lead tec-panel-missing">No description is stored for this article.</p>`}
        ${excerptHTML
        ? `<blockquote class="tec-panel-excerpt">${excerptHTML}</blockquote>`
        : `<div class="tec-panel-excerpt tec-panel-missing">No matching content excerpt is recorded.</div>`}
      </div>`;
  }

  function renderComparison(ev, sourceArt, candArt) {
    const srcExcerpt = String(ev.source_sentence || '').trim();
    const candExcerpt = String(ev.best_sentence || '').trim();
    const { srcHTML, candHTML } = highlightSharedWords(srcExcerpt, candExcerpt);
    return `
      <section class="tec-compare" aria-label="Source and recommended article comparison">
        ${panelHTML('source', sourceArt, srcHTML, 'Source article')}
        ${panelHTML('candidate', candArt, candHTML, 'Recommended article')}
      </section>`;
  }

  // ── Why connected ────────────────────────────────────────────────────────
  function renderWhy(ev, candArt, c) {
    const entities = asArray(ev.entities);
    const terms = asArray(ev.terms);
    const bm25Terms = asArray(ev.bm25_terms).map(h => h && h.term).filter(Boolean);
    const editorial = c.boosts.find(b => b.name === 'editorial')?.applied;
    const disc = classifyDiscovery(c);

    const entityChips = entities.length
      ? entities.map(e => `<span class="tec-chip tec-chip-entity">${esc(e)}</span>`).join('')
      : '<span class="tec-muted">No shared named entities recorded.</span>';
    const termChips = (bm25Terms.length ? bm25Terms : terms).slice(0, 8).map(t =>
      `<span class="tec-chip tec-chip-term">${esc(t)}</span>`).join('') ||
      '<span class="tec-muted">No shared terms recorded.</span>';

    const basis = editorial
      ? `<span class="tec-basis tec-basis-editorial">Editorial evidence</span> Curated “Also Read” link recorded in the stored evidence.`
      : `<span class="tec-basis tec-basis-rule">Rule-based reading</span> Derived from the stored signal values, not from a verified editorial judgement.`;

    return `
      <section class="tec-why" aria-label="Why these articles are connected">
        <div class="tec-section-head">
          <h3 class="tec-section-title">Why these articles are connected</h3>
          <span class="tec-discovery tec-discovery-${disc.key}">${esc(disc.label)}</span>
        </div>
        <p class="tec-why-text">${esc(ev.summary || ev.template_why || 'No summary sentence is stored for this pair.')}</p>
        <p class="tec-discovery-text">${esc(disc.text)}</p>
        <div class="tec-why-row">
          <span class="tec-why-label">Shared entities</span>
          <div class="tec-chip-row">${entityChips}</div>
        </div>
        <div class="tec-why-row">
          <span class="tec-why-label">Shared terms</span>
          <div class="tec-chip-row">${termChips}</div>
        </div>
        <p class="tec-basis-line">${basis}</p>
      </section>`;
  }

  // ── Score flow (the visual spine) ────────────────────────────────────────
  function bar(value, scale) {
    if (!isNum(value) || !isNum(scale) || scale <= 0) return '';
    const w = Math.max(0, Math.min(100, (value / scale) * 100));
    return `<span class="tec-bar-fill" style="width:${w.toFixed(1)}%"></span>`;
  }

  function renderFlow(c) {
    const scale = Math.max(
      ...[c.blend, c.preAgreement, c.boostedStored, c.boostedComputed].filter(isNum),
      0.0001
    );

    const stage = (key, title, body, extra = '') => `
      <div class="tec-flow-stage tec-flow-${key}">
        <div class="tec-flow-rail"><span class="tec-flow-dot"></span></div>
        <div class="tec-flow-content">
          <div class="tec-flow-title">${title}</div>
          ${body}
          ${extra}
        </div>
      </div>`;

    const signalLine = (name, rank, valueText, grade) => `
      <div class="tec-flow-line">
        <span class="tec-flow-line-name">${esc(name)}</span>
        <span class="tec-flow-line-value">${valueText}</span>
        <span class="tec-grade tec-grade-${esc(grade)}">${esc(GRADE_LABEL[grade] || grade)}</span>
        <span class="tec-flow-line-rank">${rank !== null ? `rank #${rank}` : 'not retrieved'}</span>
      </div>`;

    const boostLines = c.boosts.map(b => `
      <div class="tec-flow-line">
        <span class="tec-flow-line-name">${esc(b.label)}</span>
        <span class="tec-flow-line-value">${b.mult !== null ? `×${b.mult.toFixed(4)}` : NOT}</span>
        <span class="tec-flow-line-delta ${b.applied ? 'is-up' : ''}">${b.mult !== null ? pctSigned(b.mult) : ''}</span>
        <span class="tec-flow-line-rank">${esc(b.status)}</span>
      </div>`).join('');

    return `
      <section class="tec-flow" aria-label="Score progression">
        <div class="tec-section-head">
          <h3 class="tec-section-title">Score progression</h3>
          <span class="tec-flow-note">Every value below is read from the stored evidence, in the order the pipeline applied it.</span>
        </div>

        ${stage('signals', 'A · Retrieval signals',
          signalLine('BM25 lexical', c.bm25Rank,
            c.bm25Norm !== null ? `norm ${c.bm25Norm.toFixed(4)}` : NOT, c.countGrades.bm25) +
          signalLine('BGE Dense semantic', c.denseRank,
            c.cosine !== null ? `raw cosine ${c.cosine.toFixed(4)}${c.denseNorm !== null ? ` (norm ${c.denseNorm.toFixed(4)})` : ''}` : NOT, c.countGrades.dense) +
          (c.hasCe ? signalLine('Cross-Encoder', c.ceRank,
            c.ceProb !== null ? `p ${c.ceProb.toFixed(4)}` : NOT, c.countGrades.crossEncoder) : ''))}

        ${stage('relevance', 'B · Raw relevance score', `
          <div class="tec-calc">${
            (c.hasCe
              ? [['Cross-Encoder (rerank prob)', c.ceProb, c.wRerank, c.rerankTerm],
                 ['BM25 (self-normalised)', c.bm25Norm, c.wBm25, c.bm25Term],
                 ['Dense (cosine percentile)', c.denseNorm, c.wDense, c.denseTerm]]
              : [['Dense semantic (rank-adjusted norm)', c.denseNorm, c.wDense, c.denseTerm],
                 ['BM25 lexical (self-normalised)', c.bm25Norm, c.wBm25, c.bm25Term]]
            ).map(([label, v, w, term]) => `
                <div class="tec-calc-line">
                  <span class="tec-calc-label">${esc(label)}</span>
                  <span class="tec-calc-expr">${v !== null ? v.toFixed(4) : '—'} × ${w !== null ? w.toFixed(2) : '—'}</span>
                  <span class="tec-calc-term">${term !== null ? `= ${term.toFixed(4)}` : NOT}</span>
                </div>`).join('')
          }</div>
          <div class="tec-score-line">
            <span class="tec-score-line-label">Raw relevance (stored blend)</span>
            <span class="tec-score-bar">${bar(c.blend, scale)}</span>
            <span class="tec-score-line-value">${c.blend !== null ? c.blend.toFixed(4) : NOT}</span>
          </div>`)}

        ${stage('boosts', 'C · Editorial &amp; business boosts', `
          <div class="tec-flow-lines">${boostLines}</div>
          <div class="tec-score-line">
            <span class="tec-score-line-label">Product of all boosts ${c.productMult !== null ? `(×${c.productMult.toFixed(4)})` : ''}</span>
            <span class="tec-score-bar">${bar(c.preAgreement, scale)}</span>
            <span class="tec-score-line-value">${c.preAgreement !== null ? c.preAgreement.toFixed(4) : NOT}</span>
          </div>`)}

        ${stage('agreement', 'D · Signal agreement adjustment', `
          <div class="tec-flow-line">
            <span class="tec-flow-line-name">${c.agreementCount !== null ? `${c.agreementCount} of ${c.maxSignals || 3} signals agree` : 'Agreement count'}</span>
            <span class="tec-flow-line-value">${c.adjustment !== null ? signed(c.adjustment) : NOT}</span>
            <span class="tec-flow-line-delta ${c.adjustment > 0 ? 'is-up' : c.adjustment < 0 ? 'is-down' : ''}">${c.adjustment !== null ? 'additive' : ''}</span>
            <span class="tec-flow-line-rank">${esc(c.recon.rule)}</span>
          </div>
          <div class="tec-score-line">
            <span class="tec-score-line-label">After additive adjustment</span>
            <span class="tec-score-bar">${bar(c.boostedStored, scale)}</span>
            <span class="tec-score-line-value">${c.boostedComputed !== null ? c.boostedComputed.toFixed(4) : NOT}</span>
          </div>`)}

        ${stage('boosted', 'E · Boosted relevance score', `
          <div class="tec-score-line tec-score-line-final">
            <span class="tec-score-line-label">Boosted score (the value MMR ranks on)</span>
            <span class="tec-score-bar">${bar(c.boostedStored, scale)}</span>
            <span class="tec-score-line-value">${c.boostedStored !== null ? c.boostedStored.toFixed(4) : NOT}</span>
          </div>`)}

        ${stage('mmr', 'F · MMR diversity &amp; final rank', `
          <div class="tec-flow-line">
            <span class="tec-flow-line-name">Boosted-relevance position</span>
            <span class="tec-flow-line-value">${c.rankRaw !== null ? `#${c.rankRaw}` : NOT}</span>
            <span class="tec-flow-line-delta">${c.shift !== null && c.shift !== 0 ? `${c.shift > 0 ? 'promoted' : 'demoted'} ${Math.abs(c.shift)}` : 'no movement'}</span>
            <span class="tec-flow-line-rank">pre-MMR order</span>
          </div>
          <div class="tec-flow-line">
            <span class="tec-flow-line-name">Final recommendation rank</span>
            <span class="tec-flow-line-value">${c.finalRank !== null ? `#${c.finalRank}` : NOT}</span>
            <span class="tec-flow-line-delta">${c.mmrMeta && c.mmrMeta.applied === false ? 'diversify off' : 'MMR selection'}</span>
            <span class="tec-flow-line-rank">served order</span>
          </div>`)}
      </section>`;
  }

  // ── Stage C table ────────────────────────────────────────────────────────
  function renderBoosts(c) {
    const rows = c.boosts.map(b => `
      <tr class="tec-boost-row tec-boost-${esc(b.statusKey)}">
        <td class="tec-boost-name">
          <span class="tec-boost-label">${esc(b.label)}</span>
          <span class="tec-boost-status tec-status-${esc(b.statusKey)}">${esc(b.status)}</span>
        </td>
        <td class="tec-boost-reason">${esc(b.reason)}</td>
        <td class="tec-boost-mult">${b.mult !== null ? `×${b.mult.toFixed(4)}` : NOT}</td>
        <td class="tec-boost-delta">${b.mult !== null ? pctSigned(b.mult) : NOT}</td>
        <td class="tec-boost-walk">${b.before !== null && b.after !== null
        ? `<span class="tec-walk-before">${b.before.toFixed(4)}</span><span class="tec-walk-arrow">→</span><span class="tec-walk-after">${b.after.toFixed(4)}</span>`
        : NOT
      }</td>
      </tr>`).join('');

    const appliedCount = c.boosts.filter(b => b.applied).length;
    const summary = appliedCount === 0
      ? 'No editorial or business boost was applied to this pair — the boosted score differs from raw relevance only by the signal-agreement adjustment.'
      : `${appliedCount} of 5 boosts applied. Boost multipliers are multiplicative, so the running “before → after” column is one valid ordering of the same product.`;

    return `
      <section class="tec-boosts" aria-label="Editorial and business boosts">
        <div class="tec-section-head">
          <h3 class="tec-section-title">Editorial &amp; business boosts</h3>
          <span class="tec-boost-product">${c.storedDomain !== null
        ? `Stored domain multiplier ×${c.storedDomain.toFixed(4)} (${pctSigned(c.storedDomain)})`
        : NOT
      }</span>
        </div>
        <p class="tec-section-note">${esc(summary)}</p>
        <div class="tec-table-wrap">
          <table class="tec-table tec-boost-table">
            <thead>
              <tr>
                <th>Boost</th><th>Why it applied / why it did not</th>
                <th>Multiplier</th><th>Change</th><th>Score before → after</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <div class="tec-formula">
          <span class="tec-formula-label">Applied formula</span>
          <code>boosted = (raw_relevance × editorial × section × subsection × popularity × recency) + agreement_adjustment</code>
        </div>
      </section>`;
  }

  // ── Stage D block ────────────────────────────────────────────────────────
  function renderAgreement(c) {
    const row = (name, rank, grade, adjG, detail) => `
      <div class="tec-agree-row tec-agree-${esc(grade)}">
        <span class="tec-agree-name">${esc(name)}</span>
        <span class="tec-agree-detail">${esc(detail)}</span>
        <span class="tec-grade tec-grade-${esc(grade)}">${esc(GRADE_LABEL[grade] || grade)}</span>
        <span class="tec-agree-counts">${grade === 'strong' || grade === 'moderate' ? 'counts' : 'does not count'}${adjG !== grade ? ` · adj. tier: ${esc(GRADE_LABEL[adjG] || adjG)}` : ''}</span>
      </div>`;

    const maxSignals = c.maxSignals || 3;
    const label = c.agreementCount === null ? 'not recorded'
      : c.agreementCount >= maxSignals ? 'Full agreement'
        : (c.agreementCount === 2 && maxSignals === 3) ? 'Partial agreement'
          : c.agreementCount === 1 ? 'Single-signal support'
            : 'No signal agreement';

    return `
      <section class="tec-agreement" aria-label="${maxSignals}-signal agreement">
        <div class="tec-section-head">
          <h3 class="tec-section-title">${maxSignals}-signal agreement</h3>
          <span class="tec-agree-badge">${c.agreementCount !== null ? `${c.agreementCount} of ${maxSignals} · ${esc(label)}` : NOT}</span>
        </div>
        <div class="tec-agree-list">
          ${row('BM25 lexical', c.bm25Rank, c.countGrades.bm25, c.adjGrades.bm25,
      c.bm25Rank !== null ? `rank #${c.bm25Rank}` : 'not in the retrieved set')}
          ${row('BGE Dense semantic', c.denseRank, c.countGrades.dense, c.adjGrades.dense,
        c.denseRank !== null ? `rank #${c.denseRank}` : 'not in the retrieved set')}
          ${c.hasCe !== false ? row('Cross-Encoder', c.ceRank, c.countGrades.crossEncoder, c.adjGrades.crossEncoder,
          c.ceProb !== null ? `probability ${c.ceProb.toFixed(4)}` : 'not scored') : row('Cross-Encoder', null, 'none', 'none', 'Not used (2-signal live pipeline)')}
        </div>
        <div class="tec-agree-calc">
          <div class="tec-agree-rule">
            <span class="tec-agree-rule-label">Rule that fired</span>
            <span class="tec-agree-rule-value">${esc(c.recon.rule)}</span>
            <span class="tec-agree-rule-note">${esc(c.recon.note)}</span>
          </div>
          <div class="tec-agree-math">
            <div><span>Score before agreement</span><strong>${c.preAgreement !== null ? c.preAgreement.toFixed(4) : NOT}</strong></div>
            <div><span>Additive adjustment</span><strong class="${c.adjustment > 0 ? 'is-up' : c.adjustment < 0 ? 'is-down' : ''}">${c.adjustment !== null ? signed(c.adjustment) : NOT}</strong></div>
            <div><span>Score after agreement</span><strong>${c.boostedComputed !== null ? c.boostedComputed.toFixed(4) : NOT}</strong></div>
          </div>
        </div>
        <p class="tec-section-note">
          This is an <strong>additive</strong> adjustment bounded to [−0.03, +0.03], not a multiplier.
          ${c.adjustment === 0 ? 'The stored adjustment is exactly 0.0000 — the agreement table produced a neutral result for this pair.' : ''}
        </p>
      </section>`;
  }

  // ── Stage F block ────────────────────────────────────────────────────────
  function renderMmr(c) {
    const m = c.mmrMeta;
    const lambda = m ? firstNum(m.lambda) : null;
    const theta = m ? firstNum(m.theta) : null;
    const topK = m ? firstNum(m.top_k) : null;
    const applied = m ? m.applied : null;

    let movement, movementKey;
    if (c.rankRaw === null || c.finalRank === null) {
      movement = 'Rank movement is not recorded for this pair.';
      movementKey = 'unknown';
    } else if (c.finalRank < c.rankRaw) {
      movement = `MMR moved this candidate up ${c.rankRaw - c.finalRank} place${c.rankRaw - c.finalRank === 1 ? '' : 's'}. At least one higher-scoring candidate was pushed below it, which happens when a candidate is too similar (cosine > θ) to an article already selected.`;
      movementKey = 'promoted';
    } else if (c.finalRank > c.rankRaw) {
      movement = `MMR moved this candidate down ${c.finalRank - c.rankRaw} place${c.finalRank - c.rankRaw === 1 ? '' : 's'}. That is consistent with a redundancy penalty: its similarity to an already-selected article exceeded θ.`;
      movementKey = 'demoted';
    } else {
      movement = 'No movement: the candidate sits at the same position in the boosted-relevance order and in the final served order.';
      movementKey = 'stable';
    }

    const caveat = applied === false
      ? 'This response was requested with diversify=false, so MMR did not order the list; the served order is the boosted-relevance order.'
      : 'The pipeline runs MMR only when the candidate pool is larger than top-K. The pool size and the per-step MMR value are not persisted, so “no movement” is an observed result, not proof that MMR was skipped.';

    return `
      <section class="tec-mmr" aria-label="MMR diversity and ranking">
        <div class="tec-section-head">
          <h3 class="tec-section-title">MMR diversity &amp; ranking</h3>
          <span class="tec-mmr-badge tec-mmr-${movementKey}">${esc(movementKey === 'stable' ? 'No reordering' : movementKey === 'unknown' ? 'Not recorded' : movementKey === 'promoted' ? 'Promoted by MMR' : 'Demoted by MMR')}</span>
        </div>
        <div class="tec-mmr-grid">
          <div class="tec-mmr-cell"><span>Score before MMR</span><strong>${c.boostedStored !== null ? c.boostedStored.toFixed(4) : NOT}</strong></div>
          <div class="tec-mmr-cell"><span>MMR lambda (λ)</span><strong>${lambda !== null ? lambda.toFixed(2) : NOT}</strong></div>
          <div class="tec-mmr-cell"><span>Similarity threshold (θ)</span><strong>${theta !== null ? theta.toFixed(2) : NOT}</strong></div>
          <div class="tec-mmr-cell"><span>Top-K cut</span><strong>${topK !== null ? topK : NOT}</strong></div>
          <div class="tec-mmr-cell"><span>Boosted-relevance rank</span><strong>${c.rankRaw !== null ? `#${c.rankRaw}` : NOT}</strong></div>
          <div class="tec-mmr-cell"><span>Final rank</span><strong>${c.finalRank !== null ? `#${c.finalRank}` : NOT}</strong></div>
        </div>
        <div class="tec-mmr-penalty">
          <span class="tec-mmr-penalty-label">Redundancy penalty</span>
          <span class="tec-mmr-penalty-value">${NOT}</span>
          <span class="tec-mmr-penalty-note">The pairwise similarity and the per-step MMR value are not stored per recommendation, so the exact penalty cannot be reconstructed. This is a genuinely unavailable historical value, not a zero.</span>
        </div>
        <p class="tec-mmr-text">${esc(movement)}</p>
        <div class="tec-formula">
          <span class="tec-formula-label">How MMR behaves</span>
          <code>value(i) = λ · boosted(i) − (1 − λ) · max(similarity(i, already selected)) when that similarity &gt; θ</code>
        </div>
        <p class="tec-section-note">${esc(caveat)} MMR changes <em>selection order only</em> — it never rewrites a stored score.</p>
      </section>`;
  }

  // ── Technical inspection (expandable) ────────────────────────────────────
  function techRow(label, value, cls = '') {
    return `<div class="tec-tech-row ${cls}"><span>${label}</span><span>${value}</span></div>`;
  }

  function renderTechnical(ev, candArt, c) {
    const prov = ev.provenance || {};
    const hits = asArray(ev.bm25_terms);
    const warnings = c.warnings.concat(c.storedWarnings);
    const conf = c.confidence || {};
    const caps = prov.boost_caps || {};
    const liveW = prov.live_weights || { rerank: c.wRerank, bm25: c.wBm25, dense: c.wDense };

    const bm25Rows = hits.length
      ? `<table class="tec-table tec-term-table">
           <thead><tr><th>Term</th><th>TF</th><th>IDF</th><th>Weight</th><th>Contribution</th></tr></thead>
           <tbody>${hits.map(h => `<tr>
             <td><code>${esc(h.term)}</code></td>
             <td>${esc(h.tf)}</td>
             <td>${esc(h.idf)}</td>
             <td>${esc(h.weight)}</td>
             <td>${h.contribution !== undefined && h.contribution !== null ? `+${esc(h.contribution)}` : NOT}</td>
           </tr>`).join('')}</tbody>
         </table>`
      : `<div class="tec-tech-empty">No BM25 term contributions are recorded for this pair.</div>`;

    const cascade = `
      <div class="tec-tech-table-wrap">
        ${c.hasCe ? techRow('Cross-Encoder probability', c.ceProb !== null ? c.ceProb.toFixed(4) : NOT) : ''}
        ${c.hasCe ? techRow('Cross-Encoder term', `${c.wRerank !== null ? c.wRerank.toFixed(2) : '—'} × ${c.ceProb !== null ? c.ceProb.toFixed(4) : '—'}`, 'is-sub') : ''}
        ${techRow('BM25 normalised', c.bm25Norm !== null ? c.bm25Norm.toFixed(4) : NOT)}
        ${techRow('BM25 term', `${c.wBm25 !== null ? c.wBm25.toFixed(2) : '—'} × ${c.bm25Norm !== null ? c.bm25Norm.toFixed(4) : '—'} = ${c.bm25Term !== null ? c.bm25Term.toFixed(4) : '—'}`, 'is-sub')}
        ${techRow(c.hasCe ? 'Dense cosine percentile' : 'Dense rank-adjusted norm', c.denseNorm !== null ? c.denseNorm.toFixed(4) : NOT)}
        ${techRow('Dense term', `${c.wDense !== null ? c.wDense.toFixed(2) : '—'} × ${c.denseNorm !== null ? c.denseNorm.toFixed(4) : '—'} = ${c.denseTerm !== null ? c.denseTerm.toFixed(4) : '—'}`, 'is-sub')}
        ${techRow('<strong>Raw relevance (stored)</strong>', c.blend !== null ? `<strong>${c.blend.toFixed(4)}</strong>` : NOT)}
        ${techRow('Sum of displayed terms', c.termSum !== null ? c.termSum.toFixed(4) : NOT, 'is-sub')}
        ${techRow('Domain multiplier (stored)', c.storedDomain !== null ? `×${c.storedDomain.toFixed(4)}` : NOT)}
        ${techRow('Boost product (reconstructed)', c.productMult !== null ? `×${c.productMult.toFixed(4)}` : NOT, 'is-sub')}
        ${techRow('Score before agreement', c.preAgreement !== null ? c.preAgreement.toFixed(4) : NOT)}
        ${techRow('Agreement adjustment (additive)', c.adjustment !== null ? signed(c.adjustment) : NOT)}
        ${techRow('<strong>Boosted score (stored)</strong>', c.boostedStored !== null ? `<strong>${c.boostedStored.toFixed(4)}</strong>` : NOT)}
        ${techRow('Boosted score (reconstructed)', c.boostedComputed !== null ? c.boostedComputed.toFixed(4) : NOT, 'is-sub')}
        ${techRow('Unboosted confidence (Platt)', ev.score_breakdown?.unboosted_confidence != null ? Number(ev.score_breakdown.unboosted_confidence).toFixed(4) : NOT)}
        ${techRow('Platt-calibrated confidence', ev.score_breakdown?.platt_calibrated != null ? Number(ev.score_breakdown.platt_calibrated).toFixed(4) : NOT)}
        ${techRow('Stored row score', candArt.score != null ? Number(candArt.score).toFixed(4) : NOT)}
      </div>`;


    const calibration = `
      <div class="tec-tech-block">
        <div class="tec-tech-label">Calibration &amp; confidence</div>
        <div class="tec-tech-table-wrap">
          ${techRow('Confidence band', c.band ? esc(c.band) : NOT)}
          ${techRow('Calibrated probability', conf.calibrated_prob != null ? Number(conf.calibrated_prob).toFixed(4) : NOT)}
          ${techRow('Platt a / b', conf.a != null || conf.b != null ? `${esc(conf.a ?? '—')} / ${esc(conf.b ?? '—')}` : NOT)}
          ${techRow('Margin over next candidate', conf.margin != null ? Number(conf.margin).toFixed(4) : NOT)}
          ${techRow('Agreement count used by calibrator', conf.agreement_count != null ? esc(`${conf.agreement_count} of ${conf.max_signals || c.maxSignals || 3}`) : NOT)}
        </div>
        <div class="tec-tech-note">Confidence is the Platt-scaled value of the unboosted relevance, not of the boosted score. Ordering never uses it.</div>
      </div>`;

    const rankMovement = `
      <div class="tec-tech-block">
        <div class="tec-tech-label">Rank movement diagnostics</div>
        <div class="tec-tech-table-wrap">
          ${techRow('Boosted-relevance rank (rank_raw)', c.rankRaw !== null ? `#${c.rankRaw}` : NOT)}
          ${techRow('Final served rank', c.finalRank !== null ? `#${c.finalRank}` : NOT)}
          ${techRow('Shift', c.shift !== null ? (c.shift === 0 ? '0' : `${c.shift > 0 ? '+' : '−'}${Math.abs(c.shift)}`) : NOT)}
          ${techRow('Stored explanation', esc(ev.rank_relevance?.description || 'not recorded'))}
        </div>
        <div class="tec-tech-note">rank_raw is the position in the <em>boosted</em> relevance order, so the shift is produced by the MMR pass only. The stored description can attribute it to “domain boosts”; the table above is the authoritative reading.</div>
      </div>`;

    const cosineBackground = `
      <div class="tec-tech-block">
        <div class="tec-tech-label">Cosine &amp; Dense Normalisation Details</div>
        <div class="tec-tech-table-wrap">
          ${techRow('Raw cosine similarity (dot product)', c.cosine !== null ? c.cosine.toFixed(4) : NOT)}
          ${techRow('Percentile vs random pairs', c.cosinePct !== null ? `~${Math.round(c.cosinePct)}th` : NOT)}
          ${techRow('Semantic alignment band', ev.cosine_band ? esc(ev.cosine_band) : NOT)}
          ${techRow(c.hasCe ? 'Normalised dense value (percentile)' : 'Normalised dense value (rank-adjusted)', c.denseNorm !== null ? c.denseNorm.toFixed(4) : NOT)}
        </div>
        <div class="tec-tech-note">${c.hasCe
          ? 'In the precomputed pipeline, the dense value in the blend is the empirical percentile of the cosine against a 50,000 random-pair background.'
          : 'In the live hybrid pipeline, the dense value in the blend is rank-adjusted normalisation (cosine percentile discounted by dense retrieval rank so #1 ranks higher than #50).'
        } Raw cosine (${c.cosine !== null ? c.cosine.toFixed(4) : '—'}) measures raw vector proximity, while normalised dense (${c.denseNorm !== null ? c.denseNorm.toFixed(4) : '—'}) is the bounded score that enters the linear blend.</div>
      </div>`;

    return `
      <details class="tec-technical">
        <summary>
          <span>Technical inspection — arithmetic, caps, provenance</span>
          <span class="tec-tech-chevron">▾</span>
        </summary>
        <div class="tec-tech-body">
          <div class="tec-tech-block">
            <div class="tec-tech-label">BM25 term contributions</div>
            ${bm25Rows}
            ${ev.bm25_query ? `<div class="tec-tech-note">Query: <code>${esc(ev.bm25_query)}</code></div>` : ''}
          </div>

          <div class="tec-tech-block">
            <div class="tec-tech-label">Full score arithmetic</div>
            ${cascade}
            <div class="tec-tech-note">${c.weightsFromRecord
        ? 'Weights are read from this record’s own stored component block.'
        : 'This record has no stored weight block; the live configuration weights are shown.'}</div>
          </div>

          ${calibration}
          ${rankMovement}
          ${cosineBackground}

          <div class="tec-tech-block">
            <div class="tec-tech-label">Boost metadata &amp; caps</div>
            <div class="tec-tech-table-wrap">
              ${techRow('Editorial cap', caps.editorial != null ? `×${Number(caps.editorial).toFixed(2)}` : NOT)}
              ${techRow('Section cap (focused / umbrella)', caps.section_focused != null ? `×${Number(caps.section_focused).toFixed(3)} / ×${Number(caps.section_umbrella ?? 1).toFixed(3)}` : NOT)}
              ${techRow('Subsection cap (focused / umbrella)', caps.subsection_focused != null ? `×${Number(caps.subsection_focused).toFixed(3)} / ×${Number(caps.subsection_umbrella ?? 1).toFixed(3)}` : NOT)}
              ${techRow('Popularity cap', caps.popularity != null ? `×${Number(caps.popularity).toFixed(3)}` : NOT)}
              ${techRow('Recency cap', caps.recency_max != null ? `×${Number(caps.recency_max).toFixed(3)}` : NOT)}
              ${techRow('Agreement bounds', '−0.03 … +0.03 (additive)')}
            </div>
          </div>

          ${warnings.length ? `
            <div class="tec-tech-block">
              <div class="tec-tech-label">Warnings &amp; reconstruction checks</div>
              <ul class="tec-tech-warnings">${warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul>
            </div>` : `
            <div class="tec-tech-block">
              <div class="tec-tech-label">Reconstruction checks</div>
              <div class="tec-tech-note">All reconstructed values agree with the stored values within 0.0015.</div>
            </div>`}

          <div class="tec-tech-block">
            <div class="tec-tech-label">Pipeline provenance</div>
            <div class="tec-tech-provenance">
              <div><strong>Live weights</strong><br>CE ${liveW.rerank != null ? Number(liveW.rerank).toFixed(2) : '—'} · BM25 ${liveW.bm25 != null ? Number(liveW.bm25).toFixed(2) : '—'} · Dense ${liveW.dense != null ? Number(liveW.dense).toFixed(2) : '—'}</div>
              <div><strong>Initial spec</strong><br>CE 0.70 · BM25 0.20 · Dense 0.10</div>
              <div><strong>MMR λ / θ</strong><br>${c.mmrMeta ? `${firstNum(c.mmrMeta.lambda)?.toFixed(2) ?? '—'} / ${firstNum(c.mmrMeta.theta)?.toFixed(2) ?? '—'}` : 'not recorded'}</div>
              <div><strong>Pipeline version</strong><br><code>${esc(prov.pipeline_version || 'not recorded')}</code></div>
            </div>
            ${prov.has_weight_mismatch ? `<div class="tec-tech-note">${esc(prov.mismatch_explanation || '')}</div>` : ''}
          </div>
        </div>
      </details>`;
  }

  // ── Card ─────────────────────────────────────────────────────────────────
  function expandCard(card) {
    if (!card || !card.classList.contains('collapsed')) return;
    card.classList.remove('collapsed');
    card.classList.add('expanded');
    const btn = card.querySelector('.tec-toggle-btn');
    if (btn) {
      const icon = btn.querySelector('.toggle-icon');
      const text = btn.querySelector('.toggle-text');
      if (icon) icon.textContent = '▴';
      if (text) text.textContent = 'Collapse';
      btn.setAttribute('aria-expanded', 'true');
    }
  }

  function renderCard(rec, sourceArt, index = 0, defaultOpen = true, opts = {}) {
    const { ev, art } = mergeRec(rec, sourceArt, index + 1);
    const pipeline = opts.pipeline || currentPipeline();
    const meta = opts.metadata || null;

    const c = buildCascade(ev, art, meta);
    const disc = classifyDiscovery(c);
    const candId = art.id || '';

    const band = ev.confidence_band || null;
    const bandClass = band === 'HIGH' ? 'band-high' : band === 'MEDIUM' ? 'band-med' : 'band-low';

    const card = document.createElement('article');
    card.className = `candidate-eval-card tec-card ${defaultOpen ? 'expanded' : 'collapsed'}`;
    card.id = `eval-card-${cssId(candId)}`;
    card.dataset.rank = art.rank;
    card.dataset.candId = candId;
    card.dataset.sourceId = sourceArt?.id || '';
    card.dataset.pipeline = pipeline;

    const section = art.section || '';
    const subsection = art.subsection || '';
    const date = fmtDate(art.date);
    const metaLine = [
      section ? `<span>${esc(section)}${subsection ? ` · ${esc(subsection)}` : ''}</span>` : '',
      date ? `<span>${esc(date)}</span>` : ''
    ].filter(Boolean).join('<span class="tec-meta-dot">·</span>');

    card.innerHTML = `
      <header class="candidate-card-header tec-card-header">
        <div class="candidate-header-left tec-card-heading">
          <span class="candidate-rank-badge">#${esc(art.rank ?? '—')}</span>
          <div class="tec-card-title-wrap">
            ${art.url
        ? `<a class="candidate-header-title tec-card-title" href="${esc(art.url)}" target="_blank" rel="noopener">${esc(art.title || 'Untitled candidate')}</a>`
        : `<span class="candidate-header-title tec-card-title">${esc(art.title || 'Untitled candidate')}</span>`}
            <div class="tec-card-meta">
              ${metaLine || '<span class="tec-muted">No section or date recorded</span>'}
              ${band ? `<span class="tec-confidence ${bandClass}">${esc(band === 'HIGH' ? 'High confidence' : band === 'MEDIUM' ? 'Medium confidence' : 'Low confidence')}</span>` : ''}
              <span class="tec-discovery tec-discovery-${disc.key}">${esc(disc.label)}</span>
            </div>
          </div>
        </div>
        <div class="candidate-header-right tec-card-actions">
          <div class="tec-head-scores">
            <div class="tec-head-score"><span>Raw</span><strong>${c.blend !== null ? c.blend.toFixed(4) : '—'}</strong></div>
            <div class="tec-head-score"><span>Boosted</span><strong>${c.boostedStored !== null ? c.boostedStored.toFixed(4) : '—'}</strong></div>
            <div class="tec-head-score"><span>Agreement</span><strong>${c.adjustment !== null ? signed(c.adjustment, 3) : '—'}</strong></div>
            <div class="tec-head-score"><span>Final</span><strong>#${esc(c.finalRank ?? art.rank ?? '—')}</strong></div>
          </div>
          <button class="candidate-toggle-btn tec-toggle-btn" type="button" aria-expanded="${defaultOpen}">
            <span class="toggle-icon">${defaultOpen ? '▴' : '▾'}</span>
            <span class="toggle-text">${defaultOpen ? 'Collapse' : 'Inspect'}</span>
          </button>
        </div>
      </header>
      <div class="eval-container tec-card-body">
        ${renderComparison(ev, sourceArt, art)}
        ${renderWhy(ev, art, c)}
        ${renderFlow(c)}
        ${renderBoosts(c)}
        ${renderAgreement(c)}
        ${renderMmr(c)}
        ${renderTechnical(ev, art, c)}
      </div>`;

    const header = card.querySelector('.tec-card-header');
    const toggle = (e) => {
      if (e.target.closest('a') || e.target.closest('button')) return;
      const isOpen = card.classList.toggle('collapsed') === false;
      card.classList.toggle('expanded', isOpen);
      const btn = card.querySelector('.tec-toggle-btn');
      if (btn) {
        const icon = btn.querySelector('.toggle-icon');
        const text = btn.querySelector('.toggle-text');
        if (icon) icon.textContent = isOpen ? '▴' : '▾';
        if (text) text.textContent = isOpen ? 'Collapse' : 'Inspect';
        btn.setAttribute('aria-expanded', String(isOpen));
      }
    };
    header.addEventListener('click', toggle);
    const toggleBtn = card.querySelector('.tec-toggle-btn');
    if (toggleBtn) toggleBtn.addEventListener('click', (e) => { e.stopPropagation(); toggle(e); });

    return card;
  }

  // ── Group ────────────────────────────────────────────────────────────────
  function sourcePanelHTML(sourceArt, total) {
    const title = sourceArt?.title || 'Source article';
    const section = sourceArt?.section || '';
    const subsection = sourceArt?.subsection || '';
    const date = fmtDate(sourceArt?.date);
    const url = sourceArt?.url || '';
    const lead = articleLead(sourceArt);
    const meta = [
      section ? `<span>${esc(section)}${subsection ? ` · ${esc(subsection)}` : ''}</span>` : '',
      date ? `<span>${esc(date)}</span>` : '',
      `<span>${total} precomputed recommendation${total === 1 ? '' : 's'}</span>`
    ].filter(Boolean).join('<span class="tec-meta-dot">·</span>');

    return `
      <section class="tec-source-panel" aria-label="Source article reference">
        <div class="tec-source-kicker">Evidence for recommendations of</div>
        ${url
        ? `<h2 class="tec-source-title"><a href="${esc(url)}" target="_blank" rel="noopener">${esc(title)}</a></h2>`
        : `<h2 class="tec-source-title">${esc(title)}</h2>`}
        <div class="tec-source-meta">${meta}</div>
        ${lead
        ? `<p class="tec-source-lead">${esc(lead)}</p>`
        : `<p class="tec-source-lead tec-panel-missing">No description is stored for this source article.</p>`}
      </section>`;
  }

  function configStripHTML(meta) {
    const w = (meta && meta.weights) || {};
    const m = (meta && meta.mmr) || {};
    const wLine = [w.rerank, w.bm25, w.dense].every(v => isNum(v))
      ? `CE ${w.rerank.toFixed(2)} · BM25 ${w.bm25.toFixed(2)} · Dense ${w.dense.toFixed(2)}`
      : 'not recorded';
    const mLine = isNum(m.lambda) && isNum(m.theta)
      ? `λ ${m.lambda.toFixed(2)} · θ ${m.theta.toFixed(2)}`
      : 'not recorded';

    return `
      <div class="tec-config" role="note">
        <span class="tec-config-item"><span>Raw relevance weights</span><strong>${esc(wLine)}</strong></span>
        <span class="tec-config-item"><span>MMR</span><strong>${esc(mLine)}</strong></span>
        <span class="tec-config-item"><span>Agreement</span><strong>additive, −0.03 … +0.03</strong></span>
        <span class="tec-config-item"><span>Boost order</span><strong>editorial → section → subsection → popularity → recency</strong></span>
      </div>`;
  }

  function renderReviewGroup(sourceArt, recs, opts = {}) {
    const pipeline = opts.pipeline || currentPipeline();
    const sourceId = sourceArt?.id || '';
    const recList = asArray(recs);
    const meta = opts.metadata || null;

    const root = document.createElement('section');
    root.className = 'tec-group';
    root.dataset.pipeline = pipeline;
    root.dataset.sourceId = sourceId;

    root.innerHTML = `
      ${sourcePanelHTML(sourceArt, recList.length)}
      ${configStripHTML(meta)}
      <div class="tec-toolbar">
        <div class="tec-rank-nav" role="navigation" aria-label="Jump to a recommendation">
          ${recList.map((rec, i) => {
      const rank = rec.rank ?? rec.rank_raw ?? (i + 1);
      const candId = rec.id || rec.recommended_article_id || rec.article?.id || `cand_${i}`;
      return `<button type="button" class="tec-rank-btn" data-rank="${esc(rank)}" data-cand="${cssId(candId)}" title="Jump to rank #${esc(rank)}">#${esc(rank)}</button>`;
    }).join('')}
        </div>
        <div class="tec-source-nav">
          <button type="button" class="tec-nav-btn" data-action="prev-source">← Previous source</button>
          <button type="button" class="tec-nav-btn" data-action="next-source">Next source →</button>
        </div>
      </div>
      <div class="tec-cards"></div>`;

    const cardsEl = root.querySelector('.tec-cards');
    recList.forEach((rec, idx) => {
      cardsEl.appendChild(renderCard(rec, sourceArt, idx, idx < (opts.openFirst ?? 2), {
        pipeline,
        metadata: meta
      }));
    });

    root.addEventListener('click', (e) => {
      const rankBtn = e.target.closest('.tec-rank-btn');
      if (rankBtn) {
        const card = root.querySelector(`#eval-card-${rankBtn.dataset.cand}`);
        if (card) {
          expandCard(card);
          card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        return;
      }

      const actionBtn = e.target.closest('[data-action]');
      if (!actionBtn) return;
      const action = actionBtn.dataset.action;
      if (action === 'prev-source' || action === 'next-source') {
        if (typeof opts.onNavigateSource === 'function') {
          opts.onNavigateSource(action === 'next-source' ? 1 : -1);
        }
      }
    });

    return root;
  }

  // ── Modal (Reader "Eval" badge) ──────────────────────────────────────────
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

    const pipeline = currentPipeline();
    const backdrop = document.createElement('div');
    backdrop.id = 'eval-active-modal';
    backdrop.className = 'eval-modal-backdrop tec-modal-backdrop';
    backdrop.innerHTML = `
      <div class="eval-modal-card tec-modal-card">
        <div class="eval-modal-hdr tec-modal-hdr">
          <div>
            <div class="tec-modal-kicker">Recommendation evidence</div>
            <div class="tec-modal-sub">Retrieval signals → raw relevance → boosts → agreement → MMR → final rank.</div>
          </div>
          <button class="eval-modal-close" title="Close">&times;</button>
        </div>
        <div class="tec-modal-body"></div>
      </div>`;

    const body = backdrop.querySelector('.tec-modal-body');
    body.appendChild(renderCard(fullRec, sourceArt, 0, true, { pipeline }));

    const close = () => backdrop.remove();
    backdrop.querySelector('.eval-modal-close').addEventListener('click', close);
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
    document.addEventListener('keydown', function onEsc(e) {
      if (e.key === 'Escape') { close(); document.removeEventListener('keydown', onEsc); }
    });
    document.body.appendChild(backdrop);
  }

  // ── Public API ───────────────────────────────────────────────────────────
  return {
    parseEvidence,
    renderDetails: renderTechnical,
    renderCard,
    renderReviewGroup,
    renderEvidenceGroup: renderReviewGroup,
    showModal,
    classifyDiscovery,
    buildCascade,
    reconstructAgreement,
    mergeRec
  };
})();
