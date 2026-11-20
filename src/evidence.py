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

# Generic non-informative terms, days of week, months, and sentence-initial transitions to filter from NER
GENERIC_NEWS_ENTITIES = {
    # Days of week
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # Months
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    # Generic temporal & journalistic framing
    "today", "yesterday", "tomorrow", "tonight", "morning", "afternoon", "evening",
    "night", "week", "month", "year", "decade", "time", "day", "days",
    "india", "indian", "express", "indian express",
    "police", "court", "high court", "supreme court", "government", "govt",
    "ministry", "minister", "prime minister", "chief minister", "official", "officials",
    "state", "national", "international", "centre", "central", "department",
    "agency", "authorities", "authority", "spokesperson", "spokesman",
    "report", "reports", "reported", "reporting", "sources", "source", "news",
    "press", "bureau", "desk", "staff", "bench", "case", "order", "plea", "petition",
    "public", "people", "leader", "leaders", "party", "parties", "election",
    "elections", "assembly", "parliament", "lok sabha", "rajya sabha",
    # Common sentence-initial capitalized transition/discourse markers
    "however", "after", "before", "while", "meanwhile", "earlier", "according",
    "under", "over", "since", "despite", "although", "during", "following",
    "later", "first", "last", "next", "another", "several", "many", "some",
    "every", "each", "both", "few", "most", "all", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten", "furthermore",
    "moreover", "besides", "therefore", "thus", "hence", "consequently",
}

# Regex to detect capitalized entity phrases (e.g. "Jasprit Bumrah", "New Delhi", "Reserve Bank")
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

    k1 = getattr(bm25_index, "k1", 1.5)
    # Check if field-aware postings (BM25F)
    postings = getattr(bm25_index, "postings", {})
    postings_by_pos = getattr(bm25_index, "postings_by_pos", None)

    hits = []
    for term, weight in expanded:
        idf = bm25_index.idf.get(term, 0.0)
        if idf <= 0.0:
            continue
        
        # Look up precomputed tf_tilde for cand_pos: O(1) via postings_by_pos, fallback to O(postings) scan
        tf_tilde = None
        if postings_by_pos is not None:
            tf_tilde = postings_by_pos.get(term, {}).get(cand_pos)
        else:
            for pos, val in postings.get(term, []):
                if pos == cand_pos:
                    tf_tilde = val
                    break
        
        if tf_tilde is None or tf_tilde <= 0:
            continue

        contrib = weight * idf * (tf_tilde * (k1 + 1.0)) / (tf_tilde + k1)
        hits.append({
            "term": term,
            "tf": round(float(tf_tilde), 2),
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


def _extract_text_entities(text: str) -> set:
    """
    Extracts high-confidence named entities from article text:
    - Excludes sentence-initial words unless they also appear capitalized mid-sentence.
    - Excludes generic entities (weekdays, months, generic news terms, stopwords).
    - Preserves multi-word proper entities (e.g. 'Jasprit Bumrah', 'Arvind Kejriwal').
    """
    if not text:
        return set()

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    mid_capitalized_words = set()
    multi_word_entities = set()

    for sent in sentences:
        words = re.findall(r"\b[A-Za-z]+\b", sent)
        if not words:
            continue

        # Collect mid-sentence capitalized words (index > 0)
        for w in words[1:]:
            if len(w) >= 3 and w[0].isupper() and w[1:].islower():
                mid_capitalized_words.add(w.lower())

        # Find multi-word entity phrases in this sentence
        phrases = ENTITY_RE.findall(sent)
        for p in phrases:
            p_clean = p.strip()
            parts = p_clean.split()
            if len(parts) >= 2:
                # Discard if starts with generic transition like "After" or "During"
                if parts[0].lower() in GENERIC_NEWS_ENTITIES or parts[0].lower() in COMMON_STOPWORDS:
                    p_clean = " ".join(parts[1:])
                if len(p_clean.split()) >= 2 and p_clean.lower() not in GENERIC_NEWS_ENTITIES:
                    multi_word_entities.add(p_clean)

    # Valid single-word entities: must appear capitalized mid-sentence, not be generic or stopwords
    valid_singles = set()
    for w_low in mid_capitalized_words:
        if (
            w_low not in COMMON_STOPWORDS
            and w_low not in GENERIC_NEWS_ENTITIES
            and len(w_low) >= 3
            and not w_low.isdigit()
        ):
            valid_singles.add(w_low.title())

    # Filter multi-word entities
    valid_multi = set()
    for p in multi_word_entities:
        low = p.lower()
        if low not in GENERIC_NEWS_ENTITIES and not any(part.lower() in COMMON_STOPWORDS for part in p.split()):
            valid_multi.add(p)

    return valid_singles | valid_multi


def extract_shared_entities(source_text: str, cand_text: str, limit: int = 3) -> List[str]:
    """
    Extract proper names, places, and organizations present in BOTH source and candidate articles.
    Filters stopwords, sentence-initial words, and common non-informative entities.
    Returns empty list (displayed as 'none shared') if no real shared entities exist.
    """
    s_entities = _extract_text_entities(source_text)
    c_entities = _extract_text_entities(cand_text)

    # Case-insensitive intersection, returning canonical title-cased representation
    c_map = {e.lower(): e for e in c_entities}
    matched = []
    for e in s_entities:
        low = e.lower()
        if low in c_map and len(low) >= 3:
            matched.append(c_map[low])

    # Sort multi-word entities first (e.g. "Jasprit Bumrah" before "Mumbai")
    return sorted(list(set(matched)), key=lambda x: (-len(x.split()), -len(x)))[:limit]


def detect_template_match(source_sentence: str, cand_sentence: str) -> Optional[str]:
    """
    Detects if the best-matching sentence pair is a formulaic 'template match'
    (e.g. market closing figures, daily weather/horoscope, standard court disclaimers).
    """
    if not source_sentence or not cand_sentence:
        return None
    s_clean = re.sub(r"[^\w\s]", " ", source_sentence.lower())
    c_clean = re.sub(r"[^\w\s]", " ", cand_sentence.lower())
    s_words = {w for w in s_clean.split() if len(w) > 2 and w not in COMMON_STOPWORDS}
    c_words = {w for w in c_clean.split() if len(w) > 2 and w not in COMMON_STOPWORDS}
    if not s_words or not c_words:
        return None

    intersection = s_words & c_words
    union = s_words | c_words
    jaccard = len(intersection) / len(union) if union else 0.0
    min_len = min(len(s_words), len(c_words))
    overlap_ratio = len(intersection) / min_len if min_len > 0 else 0.0

    # Common journalistic boilerplates
    boilerplate_phrases = [
        "closed at", "shares of", "points higher", "points lower", "ended higher",
        "ended lower", "bench comprising", "bench of justices", "issued notice",
        "maximum temperature", "minimum temperature", "rain or thundershowers",
        "incident took place", "case has been registered", "police said",
        "according to officials", "sources said", "live updates", "express web desk"
    ]
    s_has_bp = any(bp in s_clean for bp in boilerplate_phrases)
    c_has_bp = any(bp in c_clean for bp in boilerplate_phrases)

    diff_words = (s_words ^ c_words)
    all_trivial_diff = all(w.isdigit() or w in GENERIC_NEWS_ENTITIES or len(w) <= 4 for w in diff_words)

    if (
        (overlap_ratio >= 0.50 and (s_has_bp and c_has_bp))
        or (all_trivial_diff and overlap_ratio >= 0.50)
        or (overlap_ratio >= 0.70 and (s_has_bp or c_has_bp))
        or (jaccard >= 0.65 and len(intersection) >= 5)
    ):
        return "Overlapping sentences follow standard syndicated template or boilerplate formula; verify contextual domain match."

    return None


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
    rerank_score: Optional[float],
    rerank_rank: Optional[int] = None,
) -> Tuple[int, List[str]]:
    """
    Determines which retrieval/ranking signals agreed on this candidate.
    Thresholds mirror the tiered system in consensus_multiplier() (scoring.py)
    so the displayed count always matches the boost that was actually applied.

    Tier definitions (within the 20 fusion candidates):
      BM25 / Dense rank 1-5    = STRONG hit
      BM25 / Dense rank 6-15   = MODERATE hit
      Cross-Encoder >= 0.55    = STRONG
      Cross-Encoder >= 0.35    = MODERATE

    A signal qualifies (counts toward agreement_count) when STRONG or MODERATE.
    Returns (agreement_count, signal_labels).
    """
    from src.scoring import _SIGNAL_STRONG_RANK, _SIGNAL_MOD_RANK, CE_STRONG_THRESH, CE_MOD_THRESH

    def _rank_grade(rank):
        if rank is None or rank >= 900:
            return "none"
        if rank <= _SIGNAL_STRONG_RANK:
            return "strong"
        if rank <= _SIGNAL_MOD_RANK:
            return "moderate"
        return "none"

    def _ce_grade(score, rank):
        if score is not None:
            if score >= CE_STRONG_THRESH:
                return "strong"
            if score >= CE_MOD_THRESH:
                return "moderate"
            return "none"
        if rank is not None and rank < 900:
            if rank <= _SIGNAL_STRONG_RANK:
                return "strong"
            if rank <= _SIGNAL_MOD_RANK:
                return "moderate"
        return "none"

    def _grade_label(grade):
        return {"strong": "★ Strong", "moderate": "◆ Moderate", "none": "✗ Below threshold"}[grade]

    b_r = bm25_rank if (bm25_rank is not None and bm25_rank < 900) else None
    d_r = dense_rank if (dense_rank is not None and dense_rank < 900) else None

    b_grade  = _rank_grade(b_r)
    d_grade  = _rank_grade(d_r)
    ce_grade = _ce_grade(rerank_score, rerank_rank)

    signals = []

    if b_r is not None:
        signals.append(f"BM25 Keyword (Rank #{b_r} — {_grade_label(b_grade)})")
    else:
        signals.append("BM25 Keyword (not in top-50 keyword results)")

    if d_r is not None:
        signals.append(f"Dense Vector (Rank #{d_r} — {_grade_label(d_grade)})")
    else:
        signals.append("Dense Vector (not in top-50 semantic results)")

    has_ce = (rerank_score is not None) or (rerank_rank is not None and rerank_rank < 900)
    if has_ce:
        if rerank_score is not None:
            prob_pct = f"{rerank_score * 100:.1f}%"
            rank_part = f"Rank #{rerank_rank}, " if (rerank_rank is not None and rerank_rank < 900) else ""
            signals.append(f"Cross-Encoder ({rank_part}{prob_pct} — {_grade_label(ce_grade)})")
        elif rerank_rank is not None and rerank_rank < 900:
            signals.append(f"Cross-Encoder (Rank #{rerank_rank} — {_grade_label(ce_grade)})")
        count = sum(g in ("strong", "moderate") for g in [b_grade, d_grade, ce_grade])
    else:
        signals.append("Cross-Encoder (Not used: 2-signal live pipeline)")
        count = sum(g in ("strong", "moderate") for g in [b_grade, d_grade])

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
    rerank_rank: Optional[int] = None,
    **kwargs,
) -> str:
    """
    Builds structured, deterministic evidence serialized as JSON inside the reason field.

    Restructured into 4 presentation layers:
      1. Verdict: title, final rank, confidence band (High/Medium/Low), template "why" sentence.
      2. Proof: matching sentence pair with shared words, real shared entities, 3 signal badges (top-5 rule).
      3. Checks: rank without boosts vs final, margin over next candidate, warnings (template match / truncation).
      4. Collapsed Details: BM25 terms, cosine percentile/band, score cascade, boosts, gate, provenance.
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

    # Signal agreement rule
    agreement_count, signals = evaluate_signal_agreement(
        bm25_rank, dense_rank, rerank_score, rerank_rank=rerank_rank
    )
    s = scores or {}
    boost_map = s.get("boost_items") or {}

    def rounded(value, places: int = 4):
        return round(float(value), places) if value is not None else None

    def weighted(weight, value):
        return rounded(weight * value) if value is not None else None

    boost_items = []
    _DOMAIN_BOOST_KEYS = ("editorial", "section", "subsection", "popularity", "recency")
    for name in _DOMAIN_BOOST_KEYS:
        if name in boost_map:
            mult = rounded(boost_map[name])
            boost_items.append({"name": name, "mult": mult, "amount": f"{(mult - 1) * 100:+.1f}%"})

    boosts = [
        f"{b['name'].title()} boost ({b['amount']})"
        for b in boost_items if abs(float(b["mult"] or 1.0) - 1.0) > 1e-9
    ] or detect_applied_boosts(source_row, cand_row, editorial_link=editorial_link)
    context_facts = compute_context_facts(source_row, cand_row)

    # Cosine mapping: band + percentile (never raw %)
    cosine_val = rounded(cosine)
    cosine_band = "Moderate Semantic Alignment"
    cosine_pctile = None
    if cosine_val is not None:
        if cosine_val >= 0.82:
            cosine_band = "Very High Semantic Alignment"
        elif cosine_val >= 0.72:
            cosine_band = "High Semantic Alignment"
        elif cosine_val >= 0.60:
            cosine_band = "Moderate Semantic Alignment"
        elif cosine_val >= 0.45:
            cosine_band = "Low Semantic Alignment"
        else:
            cosine_band = "Minimal Semantic Alignment"
        try:
            from src.scoring import cosine_percentile
            cosine_pctile = int(round(float(cosine_percentile([cosine_val])[0]) * 100))
        except Exception:
            cosine_pctile = None

    cosine_display = (
        f"{cosine_band} (Cosine: {cosine_val:.4f}, ~{cosine_pctile}th percentile)"
        if cosine_val is not None and cosine_pctile is not None
        else (f"{cosine_band} (Cosine: {cosine_val:.4f})" if cosine_val is not None else "Not recorded")
    )

    # Checks & Warnings
    warnings = []
    template_warning = detect_template_match(source_sentence, best_sentence)
    if template_warning:
        warnings.append(template_warning)
    if cand_row.get("is_truncated"):
        warnings.append("Candidate text was truncated to fit the token budget.")
    if rerank_score is not None and rerank_score < getattr(config, "MIN_RAW_RERANK_THRESHOLD", 0.12):
        warnings.append(f"Low raw Cross-Encoder confidence ({round(rerank_score, 3)} < {getattr(config, 'MIN_RAW_RERANK_THRESHOLD', 0.12)} threshold).")

    # Confidence Band (respects calibrator band incorporating multi-signal consensus)
    calibrated = rounded((confidence_info or {}).get("calibrated_prob"))
    conf_score = calibrated if calibrated is not None else rounded(rerank_score)
    if confidence_info and confidence_info.get("band"):
        conf_band = confidence_info["band"]
    elif conf_score is not None:
        conf_band = "HIGH" if conf_score >= 0.60 or (agreement_count >= 3 and conf_score >= 0.40) else ("MEDIUM" if conf_score >= 0.35 or (agreement_count >= 2 and conf_score >= 0.25) else "LOW")
    else:
        conf_band = "MEDIUM"

    # Template-generated one-line "why" sentence (deterministic, no LLM)
    cand_sec = cand_row.get("section") or "General"
    has_ce = (rerank_score is not None) or (rerank_rank is not None and rerank_rank < 900)
    max_signals = 3 if has_ce else 2

    if entities:
        template_why = f"Shares key entities ({', '.join(entities[:2])}) with {agreement_count} of {max_signals} signals corroborated ({cand_sec})."
    elif terms:
        template_why = f"Lexical topic overlap on '{', '.join(terms[:2])}' corroborated by {agreement_count} of {max_signals} signals ({cand_sec})."
    elif cosine_val is not None and cosine_val >= 0.72:
        template_why = f"{cosine_band} ({cand_sec})" + (" with Cross-Encoder lead." if has_ce else " (Dense semantic lead).")
    elif editorial_link:
        template_why = f"Curated editorial 'Also Read' link with verified contextual alignment ({cand_sec})."
    else:
        template_why = f"Contextual thematic alignment in {cand_sec}."

    # Spec Verification & Provenance (Fix g)
    if has_ce:
        live_w = {"rerank": config.WEIGHT_RERANK, "bm25": config.WEIGHT_BM25, "dense": config.WEIGHT_DENSE}
        spec_w = {"rerank": 0.70, "bm25": 0.20, "dense": 0.10}
        has_mismatch = (config.WEIGHT_RERANK != 0.70 or config.WEIGHT_BM25 != 0.20 or config.WEIGHT_DENSE != 0.10)
        mismatch_msg = f"Live weights ({config.WEIGHT_RERANK}/{config.WEIGHT_BM25}/{config.WEIGHT_DENSE}) from empirical tuning vs 0.70/0.20/0.10 initial spec."
    else:
        w_bm25_live = getattr(config, "LIVE_WEIGHT_BM25", 0.40)
        w_dense_live = getattr(config, "LIVE_WEIGHT_DENSE", 0.60)
        live_w = {"rerank": 0.0, "bm25": w_bm25_live, "dense": w_dense_live}
        spec_w = {"rerank": 0.0, "bm25": 0.40, "dense": 0.60}
        has_mismatch = False
        mismatch_msg = f"Live 2-signal retrieval weights (BM25 {w_bm25_live} / Dense {w_dense_live}) without Cross-Encoder."

    provenance = {
        "live_weights": live_w,
        "spec_weights": spec_w,
        "has_weight_mismatch": has_mismatch,
        "mismatch_explanation": mismatch_msg,
        "boost_caps": {
            "editorial": 1.40,
            "section_focused": getattr(config, "SECTION_BOOST_FOCUSED", 1.100),
            "section_umbrella": getattr(config, "SECTION_BOOST_UMBRELLA", 1.020),
            "subsection_focused": getattr(config, "SUBSECTION_BOOST_FOCUSED", 1.050),
            "subsection_umbrella": getattr(config, "SUBSECTION_BOOST_UMBRELLA", 1.020),
            "popularity": config.POPULARITY_BOOST_MULT,
            "recency_max": getattr(config, "RECENCY_MAX_BOOST_MULT", 1.104),
        },
        "pipeline_version": config.get_pipeline_version(),
    }

    evidence_dict = {
        "verdict": {
            "title": cand_row.get("title", ""),
            "final_rank": final_rank,
            "confidence_band": conf_band,
            "template_why": template_why,
        },
        "summary": template_why,
        "template_why": template_why,
        "confidence_band": conf_band,
        "terms": terms,
        "entities": entities,
        "source_sentence": source_sentence,
        "best_sentence": best_sentence,
        "signals": signals,
        "agreement_count": agreement_count,
        "agreement_label": f"{agreement_count} of {max_signals}",
        "max_signals": max_signals,
        "boosts": boosts,
        "context_facts": context_facts,
        "bm25_terms": bm25_term_hits(bm25_index, bm25_query, cand_pos) if bm25_query else [],
        "bm25_query": bm25_query,
        "cosine": cosine_val,
        "cosine_band": cosine_band,
        "cosine_percentile": cosine_pctile,
        "cosine_display": cosine_display,
        "warnings": warnings,
        "admission": admission,
        "provenance": provenance,
    }

    if confidence_info:
        evidence_dict["confidence"] = confidence_info

    if scores:
        from src.scoring import _signal_grade, CE_STRONG_THRESH, CE_MOD_THRESH
        b_grade = _signal_grade(bm25_rank)
        d_grade = _signal_grade(dense_rank)
        if has_ce and rerank_score is not None:
            ce_val = float(rerank_score)
            ce_grade = "strong" if ce_val >= CE_STRONG_THRESH else ("moderate" if ce_val >= CE_MOD_THRESH else "weak")
        elif has_ce:
            ce_grade = "moderate"
        else:
            ce_grade = "none"
        domain_mult = boost_map.get("domain_multiplier", boost_map.get("total", 1.0))
        agreement_adj = boost_map.get("agreement_adjustment", 0.0)

        evidence_dict["score_breakdown"] = {
            "reranker_prob": rounded(rerank_score),
            "bm25_rank": bm25_rank if (bm25_rank is not None and bm25_rank < 900) else None,
            "bm25_grade": b_grade,
            "dense_rank": dense_rank if (dense_rank is not None and dense_rank < 900) else None,
            "dense_grade": d_grade,
            "cross_encoder_grade": ce_grade,
            "max_signals": max_signals,
            "agreement_adjustment": rounded(agreement_adj, 4),
            "domain_multiplier": rounded(domain_mult, 4),
            "blend": rounded(s.get("blend")),
            "components": {
                "rerank_term": weighted(live_w["rerank"], rerank_score) if has_ce else None,
                "bm25_norm": rounded(s.get("bm25_norm")),
                "bm25_term": weighted(live_w["bm25"], s.get("bm25_norm")),
                "dense_norm": rounded(s.get("dense_norm")),
                "dense_term": weighted(live_w["dense"], s.get("dense_norm")),
                "weights": live_w,
            },
            "unboosted_confidence": rounded(s.get("unboosted_confidence")),
            "boost_items": boost_items,
            "total_boost_mult": rounded(domain_mult),
            "boosted_raw": rounded(s.get("boosted_raw")),
            "platt_calibrated": calibrated,
            "final": calibrated,
        }

    if rank_raw is not None and final_rank is not None:
        shift = rank_raw - final_rank
        note = (
            f"Pure relevance rank #{rank_raw} -> final rank #{final_rank} (promoted +{shift} by domain boosts)"
            if final_rank < rank_raw else
            f"Pure relevance rank #{rank_raw} -> final rank #{final_rank} (demoted -{-shift} by MMR diversity)"
            if final_rank > rank_raw else
            f"Pure relevance rank #{rank_raw} matches final rank #{final_rank} (no MMR demotion)"
        )
        evidence_dict["rank_relevance"] = {
            "rank_raw": rank_raw,
            "final_rank": final_rank,
            "shift": shift,
            "description": note,
        }

    if bm25_rank is not None or dense_rank is not None or rerank_score is not None or rerank_rank is not None:
        evidence_dict["method_ranks"] = {
            "bm25": bm25_rank if (bm25_rank is not None and bm25_rank < 900) else None,
            "dense": dense_rank if (dense_rank is not None and dense_rank < 900) else None,
            "reranker": rounded(rerank_score, 4),
            "reranker_rank": rerank_rank if (rerank_rank is not None and rerank_rank < 900) else None,
            "agreement_count": agreement_count,
            "agreement_label": f"{agreement_count} of {max_signals}",
            "status": f"{agreement_count} of {max_signals} signals corroborated",
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
    Ensures clean presentation:
      1. Verdict: final rank, confidence band, template why
      2. Proof: matching sentences, clean real entities, top-5 signal agreement
      3. Checks: rank shift, margin, template warnings
      4. Collapsed Details: BM25 terms, cosine band/percentile, cascade, boosts, gate, provenance
    """
    if not recs:
        return

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

        # 1. Clean & Deduplicate Terms & Entities
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

        # 2. Passage Pair (Guaranteed Non-Empty)
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

        # 3. Real scores
        sb = dict(ev.get("score_breakdown") or {})
        rerank_val = sb.get("reranker_prob")
        if rerank_val is None:
            rerank_val = r.get("rerank_score")
        if rerank_val is None:
            rerank_val = cand_dict.get("rerank_score")

        mr = dict(ev.get("method_ranks") or {})
        raw_b = sb.get("bm25_rank", mr.get("bm25"))
        raw_d = sb.get("dense_rank", mr.get("dense"))
        bm25_r = raw_b if (raw_b is not None and raw_b < 900) else None
        dense_r = raw_d if (raw_d is not None and raw_d < 900) else None
        curr_final_rank = r.get("rank") or cand_dict.get("rank") or 1

        # 4. Method agreement under signal consensus rule
        agree_cnt = ev.get("agreement_count")
        if agree_cnt is None:
            bm25_qual = bool(bm25_r is not None and bm25_r <= 15)
            dense_qual = bool(dense_r is not None and dense_r <= 15)
            rerank_qual = bool(rerank_val is not None and rerank_val >= 0.35)
            agree_cnt = sum([bm25_qual, dense_qual, rerank_qual])
        ev["agreement_count"] = agree_cnt
        ev["agreement_label"] = f"{agree_cnt} of 3"

        ev["method_ranks"] = {
            "bm25": bm25_r,
            "dense": dense_r,
            "reranker": round(float(rerank_val), 4) if rerank_val is not None else None,
            "reranker_rank": mr.get("reranker_rank"),
            "agreement_count": agree_cnt,
            "agreement_label": f"{agree_cnt} of 3",
            "status": f"{agree_cnt} of 3 signals corroborated",
        }

        # 5. Score cascade (ordering uses boosted_raw, moved to details)
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

        # 6. Rank movement
        raw_r = r.get("rank_raw") or r.get("rank") or 1
        final_r = r.get("rank") or 1
        boost_shift = raw_r - final_r
        if final_r < raw_r:
            shift_note = f"Pure relevance rank #{raw_r} -> final rank #{final_r} (promoted +{boost_shift} by domain boosts)"
            shift_type = "promoted"
        elif final_r > raw_r:
            shift_note = f"Pure relevance rank #{raw_r} -> final rank #{final_r} (demoted -{final_r - raw_r} by MMR diversity)"
            shift_type = "demoted"
        else:
            shift_note = f"Pure relevance rank #{raw_r} matches final rank #{final_r} (consensus)"
            shift_type = "preserved"

        ev["rank_relevance"] = {
            "rank_raw": raw_r,
            "final_rank": final_r,
            "shift": boost_shift,
            "shift_type": shift_type,
            "description": shift_note,
        }

        # 7. Cutoff margin
        conf_margin = (ev.get("confidence") or {}).get("margin")
        margin_val = conf_margin if isinstance(conf_margin, (int, float)) else None
        ev["margin_cutoff"] = {
            "margin": margin_val,
            "is_clear": None if margin_val is None else bool(margin_val >= 0.04),
            "label": (None if margin_val is None
                      else f"+{round(margin_val * 100, 1)}% over next candidate"),
        }

        # 8. Cosine band & percentile (Fix c)
        cos = ev.get("cosine")
        if cos is not None and isinstance(cos, (int, float)):
            if cos >= 0.82:
                c_band = "Very High Semantic Alignment"
            elif cos >= 0.72:
                c_band = "High Semantic Alignment"
            elif cos >= 0.60:
                c_band = "Moderate Semantic Alignment"
            elif cos >= 0.45:
                c_band = "Low Semantic Alignment"
            else:
                c_band = "Minimal Semantic Alignment"
            try:
                from src.scoring import cosine_percentile
                c_pctile = int(round(float(cosine_percentile([cos])[0]) * 100))
            except Exception:
                c_pctile = None
            ev["cosine_band"] = c_band
            ev["cosine_percentile"] = c_pctile
            ev["cosine_display"] = (
                f"{c_band} (Cosine: {cos:.4f}, ~{c_pctile}th percentile)"
                if c_pctile is not None else f"{c_band} (Cosine: {cos:.4f})"
            )

        # 9. Checks & Warnings (Fix d)
        warnings = list(ev.get("warnings") or [])
        src_s = ev.get("source_sentence") or ""
        best_s = ev.get("best_sentence") or ""
        t_warn = detect_template_match(src_s, best_s)
        if t_warn and t_warn not in warnings:
            warnings.append(t_warn)
        if cand_dict.get("is_truncated") and "Candidate text was truncated" not in str(warnings):
            warnings.append("Candidate text was truncated to fit the token budget.")
        ev["warnings"] = warnings

        # 10. Confidence Band & Template Why (Fix e)
        conf_obj = ev.get("confidence") or {}
        if conf_obj.get("band"):
            c_band = conf_obj["band"]
        else:
            f_score = float(final_score) if final_score is not None else None
            if f_score is not None:
                c_band = "HIGH" if f_score >= 0.60 or (agree_cnt >= 3 and f_score >= 0.40) else ("MEDIUM" if f_score >= 0.35 or (agree_cnt >= 2 and f_score >= 0.25) else "LOW")
            else:
                c_band = "MEDIUM"
        ev["confidence_band"] = c_band

        sec = cand_dict.get("section") or r.get("section") or "General"
        if raw_entities:
            why = f"Shares key entities ({', '.join(raw_entities[:2])}) with {agree_cnt} of 3 signals in top-5 ({sec})."
        elif clean_terms:
            why = f"Lexical topic overlap on '{', '.join(clean_terms[:2])}' corroborated by {agree_cnt} of 3 signals in top-5 ({sec})."
        elif ev.get("cosine_band"):
            why = f"{ev['cosine_band']} ({sec}) with Cross-Encoder lead."
        else:
            why = f"Contextual thematic alignment in {sec}."
        ev["template_why"] = ev.get("template_why") or why
        ev["summary"] = ev["template_why"]

        # 11. Provenance (Fix g)
        if "provenance" not in ev:
            has_ce_record = ev.get("score_breakdown", {}).get("cross_encoder_grade") not in ("none", None)
            if has_ce_record:
                live_w = {"rerank": config.WEIGHT_RERANK, "bm25": config.WEIGHT_BM25, "dense": config.WEIGHT_DENSE}
                spec_w = {"rerank": 0.70, "bm25": 0.20, "dense": 0.10}
                mismatch = (config.WEIGHT_RERANK != 0.70 or config.WEIGHT_BM25 != 0.20 or config.WEIGHT_DENSE != 0.10)
                mismatch_msg = f"Live weights ({config.WEIGHT_RERANK}/{config.WEIGHT_BM25}/{config.WEIGHT_DENSE}) from empirical tuning vs 0.70/0.20/0.10 initial spec."
            else:
                w_b = getattr(config, "LIVE_WEIGHT_BM25", 0.40)
                w_d = getattr(config, "LIVE_WEIGHT_DENSE", 0.60)
                live_w = {"rerank": 0.0, "bm25": w_b, "dense": w_d}
                spec_w = {"rerank": 0.0, "bm25": 0.40, "dense": 0.60}
                mismatch = False
                mismatch_msg = f"Live 2-signal retrieval weights (BM25 {w_b} / Dense {w_d}) without Cross-Encoder."
            ev["provenance"] = {
                "live_weights": live_w,
                "spec_weights": spec_w,
                "has_weight_mismatch": mismatch,
                "mismatch_explanation": mismatch_msg,
                "boost_caps": {
                    "editorial": 1.40,
                    "section_focused": getattr(config, "SECTION_BOOST_FOCUSED", 1.100),
                    "section_umbrella": getattr(config, "SECTION_BOOST_UMBRELLA", 1.020),
                    "subsection_focused": getattr(config, "SUBSECTION_BOOST_FOCUSED", 1.050),
                    "subsection_umbrella": getattr(config, "SUBSECTION_BOOST_UMBRELLA", 1.020),
                    "popularity": config.POPULARITY_BOOST_MULT,
                    "recency_max": getattr(config, "RECENCY_MAX_BOOST_MULT", 1.104),
                },
                "pipeline_version": config.get_pipeline_version(),
            }

        # 12. Context Facts
        if source_dict:
            ev["context_facts"] = compute_context_facts(source_dict, cand_dict)

        r["evidence"] = ev
        r["reason"] = json.dumps(ev, ensure_ascii=False)

    return recs



