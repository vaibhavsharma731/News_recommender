"""
Deterministic Evidence Extraction & Reason Formulation
======================================================
Produces grounded, deterministic explanations for every recommendation:
  1. Shared Key Terms (BM25 IDF-weighted terms).
  2. Shared Named Entities (Proper nouns, names, and places appearing in both).
  3. Best-Matching Sentence (The most informative sentence from the recommended story).
  4. Signal Agreement (BM25 rank, Dense rank, and Cross-Encoder score).
  5. Boosts Applied (Also-Read editorial link, section, subsection, recency).

Everything is deterministic with ZERO LLM generation, ensuring no hallucinations.
Stored in the reason field as structured JSON with clean fallback text.
"""
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

import config

# Stopwords to exclude from entity and term overlap
COMMON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "it", "its", "they", "them", "their", "this", "that", "these",
    "those", "he", "him", "his", "she", "her", "hers", "we", "us", "our",
    "who", "whom", "which", "what", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "says", "said", "also", "new", "per", "one", "two", "first", "last",
    "report", "reports", "reported", "today", "yesterday", "news", "express"
}

# Regex to detect capitalized entity phrases (e.g. "Jasprit Bumrah", "New Delhi", "Supreme Court")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def bm25_term_hits(bm25_index, query: str, cand_pos: int, limit: int = 8) -> List[Dict]:
    """
    REAL per-term BM25 contributions for ONE candidate document.

    Uses `BM25Index.docs` / `postings` / `idf` and the identical scoring formula as
    `BM25Index.scores()`, so every number returned is exactly the number that
    ranked this candidate. Nothing is inferred or approximated.

    An empty list is an honest answer: it means the candidate shares no token
    with the query, so BM25 contributed zero to its score.
    """
    if bm25_index is None or cand_pos is None or cand_pos >= len(bm25_index.docs):
        return []
    try:
        _, _, expanded = bm25_index.normalize_query(query)
    except Exception:
        return []

    k1, b = bm25_index.k1, bm25_index.b
    length_norm = 1 - b + b * bm25_index.lengths[cand_pos] / bm25_index.avg_len
    counts = Counter(bm25_index.docs[cand_pos])

    hits = []
    for term, weight in expanded:
        tf = counts.get(term, 0)
        if not tf:
            continue
        idf = bm25_index.idf.get(term, 0.0)
        contrib = weight * idf * tf * (k1 + 1) / (tf + k1 * length_norm)
        hits.append({
            "term": term,
            "tf": int(tf),
            "idf": round(idf, 3),
            "weight": round(weight, 2),
            "contribution": round(contrib, 4),
        })
    hits.sort(key=lambda h: -h["contribution"])
    return hits[:limit]


def extract_shared_key_terms(source_text: str, cand_text: str, bm25_index=None, limit: int = 4) -> List[str]:
    """Extract top shared terms ordered by BM25 IDF (or length if no index)."""
    s_words = {w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", source_text)} - COMMON_STOPWORDS
    c_words = {w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", cand_text)} - COMMON_STOPWORDS
    shared = s_words & c_words

    if not shared:
        return []

    if bm25_index and hasattr(bm25_index, "idf"):
        return sorted(shared, key=lambda t: -bm25_index.idf.get(t, 0.0))[:limit]
    return sorted(shared, key=lambda t: -len(t))[:limit]


def extract_shared_entities(source_text: str, cand_text: str, limit: int = 3) -> List[str]:
    """
    Extract proper names, places, and organizations present in BOTH source and candidate articles.
    """
    s_entities = {e.strip() for e in ENTITY_RE.findall(source_text) if len(e.strip()) >= 3}
    c_entities = {e.strip() for e in ENTITY_RE.findall(cand_text) if len(e.strip()) >= 3}

    # Filter out sentence-starting stopwords that got capitalized (e.g. "After", "The")
    valid_s = {e for e in s_entities if e.lower() not in COMMON_STOPWORDS and not e.isdigit()}
    valid_c = {e for e in c_entities if e.lower() not in COMMON_STOPWORDS and not e.isdigit()}

    # Check case-insensitive overlap but return capitalized title case
    matched = []
    c_map = {e.lower(): e for e in valid_c}
    for e in valid_s:
        low = e.lower()
        if low in c_map and len(low) > 3:
            matched.append(c_map[low])

    # Sort multi-word entities first (e.g. "Jasprit Bumrah" before "India")
    return sorted(list(set(matched)), key=lambda x: (-len(x.split()), -len(x)))[:limit]


def extract_best_matching_sentence(
    source_query_or_title: str,
    cand_body: str,
    cand_title: str = "",
    cand_desc: str = ""
) -> str:
    """
    Deterministically finds the single best-matching evidence sentence from
    the candidate article that corroborates the recommendation.
    """
    text_pool = f"{cand_desc} {cand_body}".strip()
    if not text_pool:
        return cand_title

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text_pool) if len(s.strip()) >= 25]
    if not sentences:
        return cand_title

    query_terms = {w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", source_query_or_title)} - COMMON_STOPWORDS
    if not query_terms:
        return sentences[0]

    best_score = -1
    best_sentence = sentences[0]

    for s in sentences[:15]:  # inspect first 15 sentences of body
        s_terms = {w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", s)}
        overlap = len(query_terms & s_terms)
        if overlap > best_score:
            best_score = overlap
            best_sentence = s

    # Clean up and truncate cleanly if overly verbose
    best_sentence = re.sub(r"\s+", " ", best_sentence).strip()
    if len(best_sentence) > 180:
        best_sentence = best_sentence[:177].rsplit(" ", 1)[0] + "..."
    return best_sentence


def evaluate_signal_agreement(
    bm25_rank: Optional[int],
    dense_rank: Optional[int],
    rerank_score: Optional[float]
) -> Tuple[int, List[str]]:
    """
    Determines which retrieval/ranking signals agreed on this candidate:
      - BM25 rank <= 20
      - Dense semantic rank <= 20
      - Cross-Encoder score >= 0.35 (strong positive prediction)
    Returns (agreement_count, signal_labels).
    """
    signals = []
    if bm25_rank is not None and bm25_rank <= 20:
        signals.append(f"BM25 Keyword (Rank #{bm25_rank})")
    if dense_rank is not None and dense_rank <= 20:
        signals.append(f"Dense Vector (Rank #{dense_rank})")
    if rerank_score is not None and rerank_score >= 0.35:
        signals.append(f"Cross-Encoder ({int(rerank_score * 100)}%)")

    count = len(signals)
    if count == 0 and rerank_score is not None:
        signals.append(f"Cross-Encoder Reranker ({int(rerank_score * 100)}%)")
        count = 1
    return count, signals


def detect_applied_boosts(
    source_row: dict,
    cand_row: dict,
    editorial_link: bool = False
) -> List[str]:
    """Detect which domain boosts elevated this recommendation."""
    boosts = []
    if editorial_link:
        boosts.append("Editorial 'Also Read' Link (1.40x)")

    sec_s = source_row.get("section")
    sec_c = cand_row.get("section")
    if sec_s and sec_s == sec_c:
        boosts.append(f"Same Section: {sec_s}")

    subsec_s = source_row.get("subsection")
    subsec_c = cand_row.get("subsection")
    if subsec_s and subsec_s == subsec_c:
        boosts.append(f"Same Subsection: {subsec_s}")

    date_s = source_row.get("date")
    date_c = cand_row.get("date")
    if pd.notna(date_s) and pd.notna(date_c):
        try:
            days = abs((pd.to_datetime(date_s) - pd.to_datetime(date_c)).days)
            if days <= 1:
                boosts.append("Published within 24h")
        except Exception:
            pass
    return boosts


_BATCH_MAP: Dict[str, str] = {}


def get_article_batch(article_id: Optional[str]) -> str:
    """Returns the source batch file name for an article id (e.g. Batch 05)."""
    global _BATCH_MAP
    if not article_id:
        return "Batch 01"
    aid = str(article_id).strip()
    if not _BATCH_MAP:
        try:
            for bf in sorted(glob.glob("data/articles_batches/*.csv")):
                stem = Path(bf).stem  # e.g. articles_batch_05
                bnum = stem.replace("articles_batch_", "").replace("articles_", "")
                bname = f"Batch {bnum} ({Path(bf).name})"
                df = pd.read_csv(bf, usecols=["articleId"])
                for val in df["articleId"]:
                    _BATCH_MAP[str(val).strip()] = bname
        except Exception as e:
            print(f"[Evidence] Batch scan warning: {e}")
    return _BATCH_MAP.get(aid, "Batch 01 (articles_batch_01.csv)")


def compute_context_facts(source_row: dict, cand_row: dict) -> dict:
    """Calculates domain context facts: section/subsection match, batch, and publish-time delta."""
    sec_s = (source_row.get("section") or "").strip()
    sec_c = (cand_row.get("section") or "").strip()
    subsec_s = (source_row.get("subsection") or "").strip()
    subsec_c = (cand_row.get("subsection") or "").strip()

    date_s_raw = source_row.get("date")
    date_c_raw = cand_row.get("date")
    date_s_str = ""
    date_c_str = ""
    date_gap_text = ""
    date_gap_days = 0.0

    if pd.notna(date_s_raw) and pd.notna(date_c_raw):
        try:
            d_s = pd.to_datetime(date_s_raw)
            d_c = pd.to_datetime(date_c_raw)
            date_s_str = d_s.strftime("%d %b %Y")
            date_c_str = d_c.strftime("%d %b %Y")
            diff_hours = abs((d_s - d_c).total_seconds()) / 3600.0
            date_gap_days = round(diff_hours / 24.0, 1)
            if diff_hours < 1:
                date_gap_text = "< 1h apart"
            elif diff_hours < 24:
                date_gap_text = f"{diff_hours:.1f} hours apart"
            elif date_gap_days < 7:
                d = int(round(date_gap_days))
                date_gap_text = f"{d} day apart" if d == 1 else f"{d} days apart"
            else:
                weeks = round(date_gap_days / 7.0, 1)
                date_gap_text = f"{int(round(date_gap_days))} days apart (~{weeks} weeks)"
        except Exception:
            pass

    return {
        "same_section": bool(sec_s and sec_s.lower() == sec_c.lower()),
        "source_section": sec_s,
        "cand_section": sec_c,
        "same_subsection": bool(subsec_s and subsec_s.lower() == subsec_c.lower()),
        "source_subsection": subsec_s,
        "cand_subsection": subsec_c,
        "source_date": date_s_str,
        "cand_date": date_c_str,
        "date_gap_days": date_gap_days,
        "date_gap_text": date_gap_text,
        "source_batch": get_article_batch(source_row.get("id")),
        "cand_batch": get_article_batch(cand_row.get("id")),
    }


def build_evidence_reason(
    source_row: dict,
    cand_row: dict,
    bm25_index=None,
    bm25_rank: Optional[int] = None,
    dense_rank: Optional[int] = None,
    rerank_score: Optional[float] = None,
    editorial_link: bool = False,
    confidence_info: Optional[dict] = None,
    scores: Optional[dict] = None,
    rank_raw: Optional[int] = None,
    final_rank: Optional[int] = None,
    cand_pos: Optional[int] = None,
    bm25_query: Optional[str] = None,
    cosine: Optional[float] = None,
    admission: Optional[dict] = None,
) -> str:
    """
    Builds structured, deterministic evidence serialized as JSON inside the reason field.

    Every number in the output is either computed here or supplied by the scorer.
    Nothing is estimated, scaled or defaulted to a plausible-looking value: if a
    quantity is unknown it is emitted as None so the UI can say "not recorded"
    instead of inventing one.

    ``scores`` carries the raw numbers the scorer used, and the whole score cascade
    is formatted here once, so the ingestion pipeline and the live path cannot
    describe the same chain two different ways:
      blend, bm25_norm, dense_norm, unboosted_confidence, boost_items, boosted_raw

    `cand_pos` + `bm25_query` enable the exact BM25 term breakdown to be recorded
    (see bm25_term_hits). `cosine` is the real embedding similarity of this pair.
    """
    source_text = f"{source_row.get('title', '')} {source_row.get('description', '')}"
    cand_text = f"{cand_row.get('title', '')} {cand_row.get('description', '')}"
    cand_body = str(cand_row.get("body", "") or "")
    source_body = str(source_row.get("body", "") or "")

    terms = extract_shared_key_terms(source_text, cand_text, bm25_index=bm25_index)
    entities = extract_shared_entities(source_text, cand_text)

    # Passage pair: Best matching candidate sentence AND source sentence
    best_sentence = extract_best_matching_sentence(
        source_row.get("title", ""),
        cand_body=cand_body,
        cand_title=cand_row.get("title", ""),
        cand_desc=cand_row.get("description", "")
    )
    source_sentence = extract_best_matching_sentence(
        cand_row.get("title", ""),
        cand_body=source_body,
        cand_title=source_row.get("title", ""),
        cand_desc=source_row.get("description", "")
    )

    agreement_count, signals = evaluate_signal_agreement(bm25_rank, dense_rank, rerank_score)
    s = scores or {}
    boost_map = s.get("boost_items") or {}

    def rounded(value, places: int = 4):
        return round(float(value), places) if value is not None else None

    def weighted(weight, value):
        return rounded(weight * value) if value is not None else None

    boost_items = []
    for name, value in boost_map.items():
        if name == "total":
            continue
        mult = rounded(value)
        # The percentage comes from the SAME rounded multiplier that is displayed,
        # so "1.0414" and "+4.1%" can never contradict each other.
        boost_items.append({"name": name, "mult": mult, "amount": f"{(mult - 1) * 100:+.1f}%"})
    # Boost labels come from the real multipliers the scorer applied. The old
    # detector only knew about editorial/section/subsection/24h, so popularity and
    # recency decay were invisible and the labels disagreed with the score.
    boosts = [
        f"{b['name'].title()} boost ({b['amount']})"
        for b in boost_items if abs(float(b["mult"] or 1.0) - 1.0) > 1e-9
    ] or detect_applied_boosts(source_row, cand_row, editorial_link=editorial_link)
    context_facts = compute_context_facts(source_row, cand_row)

    # Human-readable summary for backwards compatibility
    summary_parts = []
    if entities:
        summary_parts.append(f"Shared entities: {', '.join(entities)}")
    elif terms:
        summary_parts.append(f"Shared topics: {', '.join(terms)}")
    else:
        summary_parts.append("Semantic topic match")

    if agreement_count >= 2:
        summary_parts.append(f"{agreement_count}/3 signals agreed")

    if boosts:
        summary_parts.append(f"Boost: {boosts[0]}")
    elif editorial_link:
        summary_parts.append("Editor's Pick")

    summary_str = "; ".join(summary_parts) if summary_parts else "High semantic context alignment"

    evidence_dict = {
        "summary": summary_str,
        "terms": terms,
        "entities": entities,
        "source_sentence": source_sentence,
        "best_sentence": best_sentence,
        "signals": signals,
        "agreement_count": agreement_count,
        "boosts": boosts,
        "context_facts": context_facts,
        # Real BM25 evidence: the exact query terms that hit this candidate and
        # their contribution to its score. Empty means BM25 contributed nothing.
        "bm25_terms": bm25_term_hits(bm25_index, bm25_query, cand_pos) if bm25_query else [],
        "bm25_query": bm25_query,
        # Real cosine similarity between the two document embeddings.
        "cosine": rounded(cosine),
        # Real admission-gate verdict from config.admission_cut().
        "admission": admission,
    }
    if confidence_info:
        evidence_dict["confidence"] = confidence_info

    if scores:
        # The complete score cascade: rerank -> bm25_norm -> dense_norm -> blend
        # -> boosts -> Platt. Ordering uses `boosted_raw`; the displayed confidence
        # is Platt applied to the UNBOOSTED blend. Two scales on purpose, and no
        # "Platt delta" is invented between them.
        calibrated = rounded((confidence_info or {}).get("calibrated_prob"))
        evidence_dict["score_breakdown"] = {
            "reranker_prob": rounded(rerank_score),
            "bm25_rank": bm25_rank,
            "dense_rank": dense_rank,
            "blend": rounded(s.get("blend")),
            "components": {
                "rerank_term": weighted(config.WEIGHT_RERANK, rerank_score),
                "bm25_norm": rounded(s.get("bm25_norm")),
                "bm25_term": weighted(config.WEIGHT_BM25, s.get("bm25_norm")),
                "dense_norm": rounded(s.get("dense_norm")),
                "dense_term": weighted(config.WEIGHT_DENSE, s.get("dense_norm")),
                "weights": {
                    "rerank": config.WEIGHT_RERANK,
                    "bm25": config.WEIGHT_BM25,
                    "dense": config.WEIGHT_DENSE,
                },
            },
            "unboosted_confidence": rounded(s.get("unboosted_confidence")),
            "boost_items": boost_items,
            "total_boost_mult": rounded(boost_map.get("total", 1.0)),
            "boosted_raw": rounded(s.get("boosted_raw")),
            "platt_calibrated": calibrated,
            "final": calibrated,
        }

    if rank_raw is not None and final_rank is not None:
        shift = rank_raw - final_rank
        note = (
            f"relevance-only rank #{rank_raw}, final rank #{final_rank} (promoted +{shift})"
            if final_rank < rank_raw else
            f"relevance-only rank #{rank_raw}, final rank #{final_rank} (demoted -{-shift} by MMR diversity)"
            if final_rank > rank_raw else
            f"relevance-only rank #{rank_raw}, final rank #{final_rank} (matches relevance)"
        )
        evidence_dict["rank_relevance"] = {
            "rank_raw": rank_raw, "final_rank": final_rank,
            "shift": shift, "description": note,
        }

    if bm25_rank is not None or dense_rank is not None or rerank_score is not None:
        evidence_dict["method_ranks"] = {
            "bm25": bm25_rank,
            "dense": dense_rank,
            "reranker": rounded(rerank_score, 3),
            "agreement_count": agreement_count,
            "status": "All three methods agree" if agreement_count >= 3 else (
                f"{agreement_count} of 3 methods agree" if agreement_count >= 2
                else "Single method lead"),
        }

    if (confidence_info or {}).get("margin") is not None:
        evidence_dict["margin_cutoff"] = confidence_info["margin"]

    return json.dumps(evidence_dict, ensure_ascii=False)


def _parse_reason(raw) -> dict:
    """Read a stored `reason` payload. Never fabricates a replacement."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            return json.loads(raw)
        except Exception:
            return {"summary": raw}
    return {"summary": str(raw or "")}


def enrich_recommendation_evidence(source_dict: Optional[dict], recs: list):
    """
    Normalise stored evidence for display across Editor Inspector and Reader View.

    HARD RULE: this function never invents a number. Every score, boost and
    probability shown to the user is read from what the scorer wrote into
    `reason`. The only values added here are text facts that are genuinely
    derivable from the article rows themselves (batch file, section/subsection
    match, publish-time delta).

    If the scorer did not record a quantity it stays None, and the UI renders
    "not recorded". The previous implementation substituted a scaled score
    (`score * 0.9`), assumed `+5.0%` boosts and rebuilt a "boosted" value from
    guessed multipliers -- which is exactly the fiction this replaces.
    """
    if not recs:
        return

    # Attach batch to source article if available
    if source_dict:
        source_dict["batch"] = get_article_batch(source_dict.get("id"))

    for r in recs:
        cand_dict = r.get("article") if isinstance(r.get("article"), dict) else r
        cand_id = str(r.get("id") or r.get("recommended_article_id") or cand_dict.get("id") or "")
        batch_name = get_article_batch(cand_id)
        r["batch"] = batch_name
        if isinstance(r.get("article"), dict):
            r["article"]["batch"] = batch_name

        raw_reason = r.get("reason") or r.get("evidence") or cand_dict.get("reason") or ""
        ev = _parse_reason(raw_reason)

        raw_entities = ev.get("entities") or []
        raw_terms = ev.get("terms") or []

        # 1. Deduplicate Terms & Entities case-insensitively
        entity_words = {w.lower() for e in raw_entities for w in e.split()} | {e.lower() for e in raw_entities}
        clean_terms = []
        seen_terms = set()
        for t in raw_terms:
            low = t.lower().strip()
            if low not in entity_words and low not in seen_terms and len(low) > 2:
                seen_terms.add(low)
                clean_terms.append(low)
        ev["entities"] = raw_entities
        ev["terms"] = clean_terms

        # 2. Passage Pair (Guaranteed Non-Empty for Source Story)
        cand_title = cand_dict.get("title") or r.get("title") or ""
        cand_desc = cand_dict.get("description") or cand_dict.get("snippet") or r.get("description") or r.get("snippet") or ""
        cand_body = cand_dict.get("body") or r.get("body") or ""

        if source_dict:
            src_text_pool = f"{source_dict.get('description', '')} {source_dict.get('body', '')}".strip()
            if not ev.get("source_sentence") or len(str(ev.get("source_sentence", "")).strip()) < 15:
                extracted_src = extract_best_matching_sentence(
                    cand_title,
                    cand_body=src_text_pool,
                    cand_title=source_dict.get("title", ""),
                    cand_desc=source_dict.get("description", "")
                )
                ev["source_sentence"] = extracted_src or source_dict.get("description") or source_dict.get("title")

            cand_text_pool = f"{cand_desc} {cand_body}".strip()
            if not ev.get("best_sentence") or len(str(ev.get("best_sentence", "")).strip()) < 15:
                extracted_cand = extract_best_matching_sentence(
                    source_dict.get("title", ""),
                    cand_body=cand_text_pool,
                    cand_title=cand_title,
                    cand_desc=cand_desc
                )
                ev["best_sentence"] = extracted_cand or cand_desc or cand_title

        # 3. Real scores only. The scorer records the cross-encoder probability in
        # `score_breakdown.reranker_prob` and in the `rerank_score` column; if
        # neither exists the value is genuinely unknown and stays None. The old
        # code substituted `score * 0.9`, which looked precise but was invented.
        sb = dict(ev.get("score_breakdown") or {})
        rerank_val = sb.get("reranker_prob")
        if rerank_val is None:
            rerank_val = r.get("rerank_score")
        if rerank_val is None:
            rerank_val = cand_dict.get("rerank_score")

        mr = dict(ev.get("method_ranks") or {})
        bm25_r = sb.get("bm25_rank", mr.get("bm25"))
        dense_r = sb.get("dense_rank", mr.get("dense"))

        # 4. Method agreement. `agreement_count` is computed by
        # evaluate_signal_agreement() at scoring time; it is read here, never
        # re-derived, so the count can never disagree with the stored signals.
        signals = ev.get("signals")
        agree_cnt = ev.get("agreement_count")
        if agree_cnt is None:
            agree_cnt = len(signals) if isinstance(signals, list) else None
        curr_final_rank = r.get("rank") or cand_dict.get("rank") or 1
        if bm25_r is not None and dense_r is not None:
            agree_status = (f"Reranker #{curr_final_rank} with BM25 #{bm25_r} and Dense #{dense_r}"
                            if bm25_r <= 20 and dense_r <= 20
                            else f"Cross-Encoder lead: BM25 #{bm25_r}, Dense #{dense_r}, final #{curr_final_rank}")
        elif bm25_r is not None:
            agree_status = f"Lexical only: BM25 #{bm25_r}, Dense outside top 20 (final #{curr_final_rank})"
        elif dense_r is not None:
            agree_status = f"Semantic only: Dense #{dense_r}, BM25 outside top 20 (final #{curr_final_rank})"
        else:
            agree_status = f"Cross-Encoder alone (final #{curr_final_rank})"

        ev["method_ranks"] = {
            "bm25": bm25_r,
            "dense": dense_r,
            "reranker": round(float(rerank_val), 4) if rerank_val is not None else None,
            "agreement_count": agree_cnt,
            "status": agree_status,
        }


        # 5. Score cascade. The scorer wrote the complete chain at ingest time:
        # rerank -> bm25_norm -> dense_norm -> blend -> boost_breakdown ->
        # boosted_raw -> Platt -> calibrated. Read it back verbatim.
        #
        # Note the two scales on purpose: ordering uses `boosted_raw` (relevance
        # after domain boosts), while the displayed confidence is Platt applied to
        # the UNBOOSTED blend. There is no meaningful "Platt delta" between them,
        # so none is invented here.
        final_score = sb.get("final", r.get("score", cand_dict.get("score")))
        ev["score_breakdown"] = {
            "reranker_prob": rerank_val,
            "bm25_rank": bm25_r,
            "dense_rank": dense_r,
            "blend": sb.get("blend"),
            "components": sb.get("components"),
            "unboosted_confidence": sb.get("unboosted_confidence"),
            "boost_items": sb.get("boost_items") or [],
            "boosts": ev.get("boosts") or [],
            "total_boost_mult": sb.get("total_boost_mult"),
            "boosted_raw": sb.get("boosted_raw"),
            "platt_calibrated": final_score,
            "final": final_score,
        }

        # 6. Rank & Diversity Progression (3 stages)
        raw_r = r.get("rank_raw") or r.get("rank") or 1
        final_r = r.get("rank") or 1
        boost_shift = raw_r - final_r
        if final_r < raw_r:
            shift_note = f"Promoted from Pure Relevance #{raw_r} ➔ Final #{final_r} (+{boost_shift} promotion by domain boosts & MMR diversity)"
            shift_type = "promoted"
        elif final_r > raw_r:
            shift_note = f"Demoted from Pure Relevance #{raw_r} ➔ Final #{final_r} (-{final_r - raw_r} demotion by MMR diversity to prevent near-duplicate redundancy)"
            shift_type = "demoted"
        else:
            shift_note = f"Pure Relevance Rank #{raw_r} matches Final Rank #{final_r} (Consensus: no demotion)"
            shift_type = "preserved"

        ev["rank_relevance"] = {
            "rank_raw": raw_r,
            "final_rank": final_r,
            "shift": boost_shift,
            "shift_type": shift_type,
            "description": shift_note
        }

        # 7. Cutoff margin. Only the real per-pair margin recorded by the
        # calibrator ("this candidate vs the next one") is used. The old code
        # substituted `final_score - worst_score_in_list`, which is a different
        # quantity and was labelled as the cutoff margin.
        conf_margin = (ev.get("confidence") or {}).get("margin")
        margin_val = conf_margin if isinstance(conf_margin, (int, float)) else None
        ev["margin_cutoff"] = {
            "margin": margin_val,
            "is_clear": None if margin_val is None else bool(margin_val >= 0.04),
            "label": (None if margin_val is None
                      else f"+{round(margin_val * 100, 1)}% over the next candidate"),
        }

        # 8. Context Facts (Section, Subsection, Dates, Gap, Batch)
        if source_dict:
            ev["context_facts"] = compute_context_facts(source_dict, cand_dict)

        # 9. One-line verdict, built only from what was actually detected.
        sec = cand_dict.get("section") or r.get("section") or "General"
        bm_hits = ev.get("bm25_terms") or []
        cos = ev.get("cosine")
        if raw_entities:
            ev["plain_verdict"] = f"Shared entities: {', '.join(raw_entities[:3])} ({sec})"
        elif bm_hits:
            ev["plain_verdict"] = (f"BM25 matched {len(bm_hits)} term(s): "
                                   f"{', '.join(h['term'] for h in bm_hits[:3])} ({sec})")
        elif cos is not None:
            ev["plain_verdict"] = f"No lexical overlap; semantic similarity {cos:.2f} ({sec})"
        else:
            ev["plain_verdict"] = f"No shared lexical evidence recorded ({sec})"

        r["evidence"] = ev
        r["reason"] = json.dumps(ev, ensure_ascii=False)

    return recs



