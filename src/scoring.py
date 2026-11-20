"""
The four scoring steps that decide what a recommendation is worth.

One implementation, used by BOTH the ingestion pipeline (db/pipeline.py) and the
live path (src/recommender.py). They used to hold separate copies of the blend,
the boost table and MMR, and the copies had already drifted apart (ingested rows
and live rows disagreed on the dense normalisation and the MMR threshold).

Nothing here talks to a database or a model: given numbers, return numbers.
"""
import json
import math

import numpy as np

import config


def parse_ids(raw) -> list:
    """JSON string (DB form) or list (CSV form) of article IDs -> list of strings."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    return [str(x) for x in (raw if raw is not None else [])]


def json_ids(raw) -> str:
    """The stored form of a linked-ids value: always a JSON string."""
    return raw if isinstance(raw, str) else json.dumps(parse_ids(raw))


def linked_set(row) -> set:
    """Editor's "Also Read" article IDs of a row, as a string set.

    Accepts both the DB form (a JSON string) and the in-memory form (a list), and
    works on dicts as well as pandas rows. These links are editorial ground truth,
    so a lot of code asks for them.
    """
    return set(parse_ids(None if row is None else row.get("linked_ids")))


# Empirical cosine reference distribution (101 quantiles, 0% to 100% in 1% steps)
# calibrated against 50,000 random document pairs from the news corpus.
COSINE_QUANTILES = np.array([
    0.2578, 0.3355, 0.3515, 0.3609, 0.3683, 0.3742, 0.3801, 0.3847, 0.3889, 0.3925,
    0.3962, 0.3993, 0.4023, 0.4052, 0.4083, 0.4111, 0.4138, 0.4163, 0.4187, 0.4209,
    0.4230, 0.4252, 0.4272, 0.4291, 0.4311, 0.4329, 0.4347, 0.4366, 0.4384, 0.4403,
    0.4421, 0.4437, 0.4454, 0.4471, 0.4488, 0.4504, 0.4520, 0.4537, 0.4551, 0.4567,
    0.4583, 0.4599, 0.4613, 0.4627, 0.4642, 0.4658, 0.4673, 0.4689, 0.4705, 0.4720,
    0.4735, 0.4751, 0.4766, 0.4780, 0.4796, 0.4809, 0.4825, 0.4840, 0.4857, 0.4872,
    0.4888, 0.4903, 0.4920, 0.4936, 0.4952, 0.4969, 0.4984, 0.5001, 0.5020, 0.5036,
    0.5054, 0.5072, 0.5089, 0.5108, 0.5128, 0.5148, 0.5167, 0.5186, 0.5207, 0.5229,
    0.5255, 0.5279, 0.5304, 0.5329, 0.5355, 0.5383, 0.5411, 0.5444, 0.5479, 0.5517,
    0.5556, 0.5600, 0.5645, 0.5698, 0.5758, 0.5828, 0.5911, 0.6018, 0.6167, 0.6461,
    0.9428
], dtype=np.float32)


def cosine_percentile(cosines) -> np.ndarray:
    """Map raw cosine similarities to empirical percentiles [0.0, 1.0] against
    the random document-pair background distribution."""
    cosines = np.asarray(cosines, dtype=float)
    pcts = np.interp(
        cosines,
        COSINE_QUANTILES,
        np.linspace(0.0, 1.0, len(COSINE_QUANTILES)),
        left=0.0,
        right=1.0,
    )
    return np.clip(pcts, 0.0, 1.0)


def rank_adjusted_dense_norm(dense_vals, dense_rank_of: list = None) -> np.ndarray:
    """Rank-aware dense normalisation for the live 2-signal pipeline.

    Problem: top-50 dense candidates all score 0.96–0.99 cosine percentile —
    the percentile table is calibrated on random pairs, but every candidate here
    is already a semantic match. This flattens the dense signal to near-noise.

    Fix: blend cosine percentile with a log-rank discount so rank #1 is
    meaningfully better than rank #31.

    Formula:  rank_adj = cosine_pct * log_rank_weight
      where:  log_rank_weight = log(2) / log(rank + 1)   (rank #1 → 1.0, rank #10 → 0.53, rank #50 → 0.35)

    If dense_rank_of is None (ranks unknown), falls back to cosine_percentile.
    """
    cosines = np.asarray(dense_vals, dtype=float)
    pcts = cosine_percentile(cosines)

    if dense_rank_of is None or len(dense_rank_of) != len(cosines):
        return pcts

    # Build log-rank discount: rank 1→1.0, rank 2→0.63, rank 10→0.33, rank 50→0.19
    ranks = np.asarray(dense_rank_of, dtype=float)
    # Clamp unknown / very high ranks to 50
    ranks = np.where(ranks <= 0, 50.0, ranks)
    ranks = np.minimum(ranks, 50.0)
    log_weight = np.log(2.0) / np.log(ranks + 1.0)

    # Blend: 50% raw percentile + 50% rank-discounted percentile
    # This preserves absolute cosine signal while adding rank differentiation
    blended = 0.50 * pcts + 0.50 * (pcts * log_weight)
    return np.clip(blended, 0.0, 1.0)


def self_normalized_bm25(values, self_bm25: float = None) -> np.ndarray:
    """Scale BM25 scores by the source article's self-score into [0.0, 1.0].
    If self_bm25 is missing or zero (e.g. text search), falls back to max-scale."""
    values = np.where(np.isneginf(np.asarray(values, dtype=float)), 0.0, values)
    values = np.maximum(values, 0.0)
    if self_bm25 is not None and self_bm25 > 0:
        return np.clip(values / self_bm25, 0.0, 1.0)
    top = values.max() if values.size and values.max() > 0 else 1.0
    return np.clip(values / top, 0.0, 1.0)


def norm(values) -> np.ndarray:
    """Legacy helper: alias for self_normalized_bm25 with no self-score."""
    return self_normalized_bm25(values, self_bm25=None)


def blend(rerank_scores, bm25_scores, dense_scores, positions, self_bm25: float = None):
    """Fixed-scale v2 blend: raw cross-encoder probability, self-normalized BM25,
    and empirical cosine percentile against random-pair background distribution.

    Returns ``(relevance, bm25_norm, dense_norm)`` so the evaluation card can show
    the exact decomposition.
    """
    bm25_vals = [bm25_scores[p] for p in positions]
    dense_vals = [dense_scores[p] for p in positions]
    bm25_norm = self_normalized_bm25(bm25_vals, self_bm25)
    dense_norm = cosine_percentile(dense_vals)
    relevance = (config.WEIGHT_RERANK * np.asarray(rerank_scores, dtype=float)
                 + config.WEIGHT_BM25 * bm25_norm
                 + config.WEIGHT_DENSE * dense_norm)
    return relevance, bm25_norm, dense_norm


# --- Signal agreement tier thresholds ---
# Note: Ranks passed here are the original retrieval ranks (top-50) from BM25 and Dense
# search, preserving authentic retrieval strength.
_SIGNAL_STRONG_RANK = 5    # rank 1-5  = STRONG hit
_SIGNAL_MOD_RANK   = 15    # rank 6-15 = MODERATE hit
_SIGNAL_WEAK_RANK  = 25    # rank 16-25 = WEAK hit
CE_STRONG_THRESH = 0.55    # cross-encoder probability >= 0.55 = STRONG neural read
CE_MOD_THRESH    = 0.35    # cross-encoder probability >= 0.35 = MODERATE neural read


def _signal_grade(rank: int) -> str:
    """Return 'strong', 'moderate', 'weak' or 'none' for one retrieval rank (1..50)."""
    if rank is None or rank >= 900:
        return "none"
    if rank <= _SIGNAL_STRONG_RANK:
        return "strong"
    if rank <= _SIGNAL_MOD_RANK:
        return "moderate"
    if rank <= _SIGNAL_WEAK_RANK:
        return "weak"
    return "none"


def signal_agreement_adjustment(
    cand_row,
    source_row=None,
    bm25_rank: int = None,
    dense_rank: int = None,
    rerank_score: float = None,
) -> float:
    """
    Additive Signal Agreement Adjustment (Stage 2):

    Replaces the previous multiplicative consensus multiplier with a small,
    bounded additive adjustment [-0.03, +0.03] to the post-boost relevance score.

    Ensures strong semantic matches (Dense strong + CE strong) are not penalized
    for lacking BM25 keyword overlap, while maintaining penalties for confirmed
    keyword flukes and embedding drift.
    """
    # Editorial ground truth: always neutral
    if source_row is not None and str(cand_row.get("id")) in linked_set(source_row):
        return 0.0

    if bm25_rank is None and dense_rank is None and rerank_score is None:
        return 0.0

    b_r = bm25_rank if (bm25_rank is not None and bm25_rank < 900) else None
    d_r = dense_rank if (dense_rank is not None and dense_rank < 900) else None

    b_grade = _signal_grade(b_r)
    d_grade = _signal_grade(d_r)

    # Dual-Signal Retrieval Agreement (Live Hybrid Pipeline: BM25 + Dense without Cross-Encoder)
    if rerank_score is None:
        if b_grade == "strong" and d_grade == "strong":
            adj = 0.02
        elif (b_grade == "strong" and d_grade == "moderate") or (b_grade == "moderate" and d_grade == "strong"):
            adj = 0.015
        elif b_grade == "moderate" and d_grade == "moderate":
            adj = 0.01
        elif (b_grade == "strong" and d_grade in ("weak", "none")) or (d_grade == "strong" and b_grade in ("weak", "none")):
            adj = 0.01
        elif (b_grade == "moderate" and d_grade in ("weak", "none")) or (d_grade == "moderate" and b_grade in ("weak", "none")):
            adj = 0.005
        elif b_grade == "weak" and d_grade == "weak":
            adj = -0.005
        elif (b_grade == "weak" and d_grade == "none") or (d_grade == "weak" and b_grade == "none"):
            adj = -0.01
        else:
            adj = 0.00
        return float(np.clip(adj, -0.03, 0.03))

    # Triple-Signal Retrieval + Cross-Encoder Agreement (Precomputation / Full Rerank Pipeline)
    ce = float(rerank_score)
    if ce >= CE_STRONG_THRESH:
        ce_grade = "strong"
    elif ce >= CE_MOD_THRESH:
        ce_grade = "moderate"
    else:
        ce_grade = "weak"

    adj = 0.0

    # 1. Triple agreement: BM25 strong + Dense strong + CE strong -> +0.03
    if b_grade == "strong" and d_grade == "strong" and ce_grade == "strong":
        adj = 0.03

    # 2. Strong semantic match: Dense strong + CE strong, BM25 absent -> +0.02
    elif b_grade == "none" and d_grade == "strong" and ce_grade == "strong":
        adj = 0.02

    # 3. Moderate semantic match: Dense moderate + CE strong, BM25 absent -> +0.01
    elif b_grade == "none" and d_grade == "moderate" and ce_grade == "strong":
        adj = 0.01

    # 4. Strong retrieval agreement + moderate CE: BM25 strong + Dense strong + CE moderate -> +0.015
    elif b_grade == "strong" and d_grade == "strong" and ce_grade == "moderate":
        adj = 0.015

    # 5. Keyword fluke: BM25 strong + Dense absent + CE weak -> -0.03
    elif b_grade == "strong" and (d_r is None or d_r > 20) and ce_grade == "weak":
        adj = -0.03

    # 6. Embedding drift: Dense strong + BM25 absent + CE weak -> -0.02
    elif (b_r is None or b_r > 20) and d_grade == "strong" and ce_grade == "weak":
        adj = -0.02

    # 7. Secondary keyword/drift damping:
    elif b_grade == "moderate" and (d_r is None or d_r > 20) and ce_grade == "weak":
        adj = -0.02
    elif (b_r is None or b_r > 20) and d_grade == "moderate" and ce_grade == "weak":
        adj = -0.015

    # 8. All signals weak or inconclusive -> 0.00
    else:
        adj = 0.00

    return float(np.clip(adj, -0.03, 0.03))


def consensus_multiplier(
    cand_row,
    source_row=None,
    bm25_rank: int = None,
    dense_rank: int = None,
    rerank_score: float = None,
) -> float:
    """Legacy backward-compatibility shim: returns 1.0."""
    return 1.0


def boost_parts(
    source_row,
    cand_row,
    max_views: int,
    max_date,
    bm25_rank: int = None,
    dense_rank: int = None,
    rerank_score: float = None,
) -> dict:
    """Every domain boost applied to one candidate, plus their product and agreement adjustment.

    final_score = (base_relevance * domain_multiplier) + agreement_adjustment
    """
    parts = {"editorial": 1.0, "section": 1.0, "subsection": 1.0,
             "popularity": 1.0, "recency": 1.0}

    if source_row is not None:
        if str(cand_row.get("id")) in linked_set(source_row):
            parts["editorial"] = 1.40
        src_sec = str(source_row.get("section") or "").strip().lower()
        cand_sec = str(cand_row.get("section") or "").strip().lower()
        if src_sec and src_sec == cand_sec:
            # Explicit three-way lookup: umbrella → 1.020, focused → 1.100,
            # unknown future section → conservative umbrella-equivalent (1.020).
            _umbrella_set = getattr(config, "UMBRELLA_SECTIONS", set())
            _focused_set  = getattr(config, "FOCUSED_SECTIONS",  set())
            if src_sec in _umbrella_set:
                _sec_boost = getattr(config, "SECTION_BOOST_UMBRELLA", 1.020)
                _sub_boost = getattr(config, "SUBSECTION_BOOST_UMBRELLA", 1.020)
            elif src_sec in _focused_set:
                _sec_boost = getattr(config, "SECTION_BOOST_FOCUSED", 1.100)
                _sub_boost = getattr(config, "SUBSECTION_BOOST_FOCUSED", 1.050)
            else:
                # Unknown section: apply umbrella-equivalent dampening so an
                # unrecognised future feed never receives the full focused boost.
                _sec_boost = getattr(config, "SECTION_BOOST_UMBRELLA", 1.020)
                _sub_boost = getattr(config, "SUBSECTION_BOOST_UMBRELLA", 1.020)
            parts["section"] = _sec_boost
            src_sub = str(source_row.get("subsection") or "").strip().lower()
            cand_sub = str(cand_row.get("subsection") or "").strip().lower()
            if src_sub and src_sub == cand_sub:
                parts["subsection"] = _sub_boost

    # Popularity: log-scaled
    views = cand_row.get("page_views")
    if views is None:
        views = cand_row.get("pageViews") or 0
    if views > 0 and max_views > 0:
        parts["popularity"] = 1.0 + (config.POPULARITY_BOOST_MULT - 1.0) * (
            math.log1p(views) / math.log1p(max_views))

    # Recency: relative to the SOURCE article's date
    parts["recency"] = config.recency_boost(
        cand_row.get("date"),
        config.recency_ref_date(source_row.get("date") if source_row is not None else None, max_date),
        cand_row.get("section"),
        bool(cand_row.get("is_live", False)),
    )

    domain_mult = 1.0
    for value in parts.values():
        domain_mult *= value

    agreement_adj = signal_agreement_adjustment(
        cand_row,
        source_row=source_row,
        bm25_rank=bm25_rank,
        dense_rank=dense_rank,
        rerank_score=rerank_score,
    )

    parts["domain_multiplier"] = domain_mult
    parts["agreement_adjustment"] = agreement_adj
    parts["total"] = domain_mult
    parts["agreement"] = 1.0
    return parts


def mmr(scores, similarity, top_k: int, lam: float = config.MMR_LAMBDA,
        theta: float = config.MMR_THETA) -> list:
    """Maximal Marginal Relevance: relevance minus redundancy with what is already picked.

    Vectorized NumPy implementation: maintains max redundancy per remaining candidate
    against selected items with theta thresholding. Mathematically identical to the
    recursive definition while operating in O(k) NumPy operations.
    """
    scores_arr = np.asarray(scores, dtype=float)
    n = len(scores_arr)
    if n == 0 or top_k <= 0:
        return []

    sim_mat = np.asarray(similarity, dtype=float)
    if sim_mat.shape != (n, n):
        sim_mat = np.zeros((n, n), dtype=float)

    selected = []
    selected_mask = np.zeros(n, dtype=bool)
    max_redundancy = np.zeros(n, dtype=float)

    first = int(np.argmax(scores_arr))
    selected.append(first)
    selected_mask[first] = True

    target_k = min(top_k, n)
    for _ in range(1, target_k):
        last_selected = selected[-1]
        last_sims = sim_mat[:, last_selected]
        qualifying = np.where(last_sims > theta, last_sims, 0.0)
        max_redundancy = np.maximum(max_redundancy, qualifying)

        mmr_vals = lam * scores_arr - (1.0 - lam) * max_redundancy
        mmr_vals[selected_mask] = -np.inf

        best = int(np.argmax(mmr_vals))
        selected.append(best)
        selected_mask[best] = True

    return selected
