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


def norm(values) -> np.ndarray:
    """Scale so the best candidate is 1.0. All-zero (or all negative) -> zeros."""
    values = np.where(np.isneginf(np.asarray(values, dtype=float)), 0.0, values)
    top = values.max() if values.size and values.max() > 0 else 1.0
    return values / top


def blend(rerank_scores, bm25_scores, dense_scores, positions):
    """Weighted relevance for ``positions`` in a full-corpus score array.

    Returns ``(relevance, bm25_norm, dense_norm)`` so the evaluation card can show
    the decomposition of exactly the number that ranked the pair.
    """
    bm25_norm = norm([bm25_scores[p] for p in positions])
    dense_norm = np.clip(norm([dense_scores[p] for p in positions]), 0.0, 1.0)
    relevance = (config.WEIGHT_RERANK * np.asarray(rerank_scores, dtype=float)
                 + config.WEIGHT_BM25 * bm25_norm
                 + config.WEIGHT_DENSE * dense_norm)
    return relevance, bm25_norm, dense_norm


def boost_parts(source_row, cand_row, max_views: int, max_date) -> dict:
    """Every domain boost applied to one candidate, plus their product.

    ``source_row`` is None for a free-text search (there is no article to compare
    against). Works on dicts and on DataFrame rows: both ``page_views`` (DB) and
    ``pageViews`` (CSV) are accepted, so one function serves every caller.
    """
    parts = {"editorial": 1.0, "section": 1.0, "subsection": 1.0,
             "popularity": 1.0, "recency": 1.0}

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
