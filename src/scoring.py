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
# Ranks here are positions within the N_CANDIDATES (20) fusion pool the reranker
# actually sees -- not the full 50-article retrieval list.
# A rank of #1 or #5 means "both BM25 and Dense chose this as a top hit",
# which is far stronger evidence than being #22 out of 50.
_SIGNAL_STRONG_RANK = 5   # rank 1-5  within candidates  = STRONG keyword/semantic hit
_SIGNAL_MOD_RANK   = 15   # rank 6-15 within candidates  = MODERATE hit
# _SIGNAL_WEAK_RANK = 16-20: below moderate, still in the pool but barely
CE_STRONG_THRESH = 0.55   # cross-encoder probability >= 0.55 = STRONG neural read
CE_MOD_THRESH    = 0.35   # cross-encoder probability >= 0.35 = MODERATE neural read


def _signal_grade(rank: int) -> str:
    """Return 'strong', 'moderate', 'weak' or 'none' for one retrieval rank (1..50)."""
    if rank is None or rank >= 900:
        return "none"
    if rank <= _SIGNAL_STRONG_RANK:
        return "strong"
    if rank <= _SIGNAL_MOD_RANK:
        return "moderate"
    if rank <= 25:
        return "weak"
    return "none"


def consensus_multiplier(
    cand_row,
    source_row=None,
    bm25_rank: int = None,
    dense_rank: int = None,
    rerank_score: float = None,
) -> float:
    """
    Tiered Signal Consensus Multiplier (Change 18, 2026-10-28):

    Rewards precise multi-signal agreement to surface the genuinely best 3-5
    recommendations while suppressing keyword flukes and embedding drift.

    Ranks are from the top-50 retrieval candidates for each branch:
      STRONG   = rank 1–5   (top 10% retrieval hit)
      MODERATE = rank 6–15  (top 30% retrieval hit)
      WEAK     = rank 16–25 (top 50% retrieval hit)
      NONE     = rank > 25, or not retrieved in top 50 (None / 999)

    Cross-Encoder probability tiers:
      STRONG   = probability >= 0.55
      MODERATE = probability >= 0.35
      WEAK     = probability < 0.35

    Outcome table (monotonic and mathematically grounded):
      All 3 STRONG                           → ×1.15   (+15% triple consensus)
      2 STRONG + 1 MODERATE                  → ×1.10   (+10% high consensus)
      All 3 at least MODERATE                → ×1.08   (+8% full consensus)
      BM25 + Dense both MODERATE or better   → ×1.04   (+4% dual retrieval agreement)
      Dense + CE STRONG (BM25 absent)        → ×1.00   (neutral, genuine semantic match)
      Topic mismatch (BM25 strong, Dense>20) → ×0.75   (-25% damping e.g. sale vs lawsuit)
      Keyword fluke (BM25 strong, Dense>20, CE<0.55) → ×0.40 (-60% metaphoric quote e.g. jail)
      Embedding drift (Dense hit, BM25 none, CE<0.35) → ×0.35 (-65% unrelated crime on comedy)
      Only 1 signal qualifies at all         → ×0.80   (-20% unconfirmed single source)
      Otherwise (e.g. 2 moderate, 1 none)    → ×1.00   (neutral)
    """
    # Editorial 'Also Read' links are verified human ground-truth: always neutral, no penalty
    if source_row is not None and str(cand_row.get("id")) in linked_set(source_row):
        return 1.0

    if bm25_rank is None and dense_rank is None and rerank_score is None:
        return 1.0

    b_r = bm25_rank if (bm25_rank is not None and bm25_rank < 900) else None
    d_r = dense_rank if (dense_rank is not None and dense_rank < 900) else None
    ce = float(rerank_score) if rerank_score is not None else 0.0

    # Compute grade for each signal
    b_grade = _signal_grade(b_r)
    d_grade = _signal_grade(d_r)
    if ce >= CE_STRONG_THRESH:
        ce_grade = "strong"
    elif ce >= CE_MOD_THRESH:
        ce_grade = "moderate"
    else:
        ce_grade = "none"

    grades = [b_grade, d_grade, ce_grade]
    n_strong   = grades.count("strong")
    n_moderate = grades.count("moderate")
    n_qualify  = n_strong + n_moderate   # any grade better than "none"

    # --- Tiered multiplier lookup ---

    # 1. Triple consensus: all three signals are STRONG
    if n_strong == 3:
        return 1.15

    # 2. High consensus: all three qualify and at least two are STRONG
    if n_qualify == 3 and n_strong >= 2:
        return 1.10

    # 3. Full consensus: all three qualify at moderate or better
    if n_qualify == 3:
        return 1.08

    # 4. Dual retrieval agreement: both BM25 and Dense agree at moderate or better
    if b_grade in ("strong", "moderate") and d_grade in ("strong", "moderate"):
        return 1.04

    # 5. Semantic + Neural match: Dense vector and Cross-Encoder both confirm match,
    #    even though exact query keywords didn't overlap (BM25 absent)
    if b_grade == "none" and d_grade in ("strong", "moderate") and ce_grade == "strong":
        return 1.00
    if b_grade == "none" and d_grade == "strong" and ce_grade == "moderate":
        return 0.95

    # 6. Flukes, Topic Mismatches, and Embedding Drifts:
    # 6a. Keyword fluke: BM25 matched on metaphoric or casual words (e.g. Kapil Sharma 'jail'),
    #     Dense was absent/far (>20), and Cross-Encoder did NOT strongly corroborate
    if b_grade == "strong" and (d_r is None or d_r > 20) and ce_grade != "strong":
        return 0.40

    # 6b. Topic mismatch & Cross-domain fluke:
    #     BM25 matched words, but Dense vector strongly rejects the topic (d_r is None or > 20)
    if b_grade == "strong" and (d_r is None or d_r > 20) and ce_grade == "strong":
        # If candidate is from a different section, it's an unequivocal cross-domain keyword fluke
        # (e.g. Umesh Pal murder jail story matching Kapil Sharma comedy interview)
        if source_row is not None and source_row.get("section") and cand_row.get("section") and source_row.get("section") != cand_row.get("section"):
            return 0.30
        # Same section topic mismatch (e.g. Flipkart Sale matching Google Lawsuit, both in technology)
        return 0.75

    # 6c. Embedding drift: Dense retrieved an item, but BM25 is absent and Cross-Encoder
    #     rejects it (e.g. unrelated murder news appearing under comedy/entertainment)
    if (b_r is None or b_r > 20) and d_grade in ("strong", "moderate") and ce_grade == "none":
        return 0.35

    # 7. Unreliable single-signal candidate: only 1 signal qualifies at all
    if n_qualify <= 1:
        return 0.80

    # Default: two moderate signals — neutral
    return 1.00


def boost_parts(
    source_row,
    cand_row,
    max_views: int,
    max_date,
    bm25_rank: int = None,
    dense_rank: int = None,
    rerank_score: float = None,
) -> dict:
    """Every domain boost applied to one candidate, plus their product.

    ``source_row`` is None for a free-text search (there is no article to compare
    against). Works on dicts and on DataFrame rows: both ``page_views`` (DB) and
    ``pageViews`` (CSV) are accepted, so one function serves every caller.
    """
    parts = {"editorial": 1.0, "section": 1.0, "subsection": 1.0,
             "popularity": 1.0, "recency": 1.0, "agreement": 1.0}

    if source_row is not None:
        if str(cand_row.get("id")) in linked_set(source_row):
            parts["editorial"] = 1.40
        if source_row.get("section") and source_row.get("section") == cand_row.get("section"):
            parts["section"] = config.SECTION_BOOST_MULT
        if source_row.get("subsection") and source_row.get("subsection") == cand_row.get("subsection"):
            parts["subsection"] = config.SUBSECTION_BOOST_MULT

    # Popularity: log-scaled, so a viral story gets a nudge and not dominance.
    views = cand_row.get("page_views")
    if views is None:
        views = cand_row.get("pageViews") or 0
    if views > 0 and max_views > 0:
        parts["popularity"] = 1.0 + (config.POPULARITY_BOOST_MULT - 1.0) * (
            math.log1p(views) / math.log1p(max_views))

    # Recency: relative to the SOURCE article's date (config.BOOST_MODE), not the
    # newest row in the corpus.
    parts["recency"] = config.recency_boost(
        cand_row.get("date"),
        config.recency_ref_date(source_row.get("date") if source_row is not None else None, max_date),
        cand_row.get("section"),
        bool(cand_row.get("is_live", False)),
    )

    # Signal consensus: promote multi-signal winners, demote single-signal outliers
    parts["agreement"] = consensus_multiplier(
        cand_row,
        source_row=source_row,
        bm25_rank=bm25_rank,
        dense_rank=dense_rank,
        rerank_score=rerank_score,
    )

    total = 1.0
    for value in parts.values():
        total *= value
    parts["total"] = total
    return parts


def mmr(scores, similarity, top_k: int, lam: float = config.MMR_LAMBDA,
        theta: float = config.MMR_THETA) -> list:
    """Maximal Marginal Relevance: relevance minus redundancy with what is already picked.

    A similarity counts as redundancy only when it exceeds ``theta``, so ordinary
    topical closeness is never punished - only near-duplicate wire copy is.
    """
    selected, remaining = [], list(range(len(scores)))

    def value(i):
        redundant = [similarity[i][j] for j in selected if similarity[i][j] > theta]
        return lam * scores[i] - (1 - lam) * max(redundant, default=0.0)

    while remaining and len(selected) < top_k:
        best = max(remaining, key=value)
        selected.append(best)
        remaining.remove(best)
    return selected
