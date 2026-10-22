"""
Step 3 - The reranker (cross-encoder).
Reads query + article TOGETHER and scores the match. Slower but more precise,
so it only runs on the best candidates.

Budget rules
------------
Total pair cap:  512 tokens (RERANK_MAX_LEN)
Special tokens:  4  (<s> query </s></s> passage </s> -- XLM-RoBERTa style)
Content budget:  508 tokens

Article-to-article mode (50/50 split):
  Each side gets (512 - 4) // 2 = 254 content tokens.
  Source article text_rerank  -- pre-budgeted at ingest time.
  Candidate article text_rerank -- pre-budgeted at ingest time.

Live search mode (query-first):
  Query tokens measured precisely; passage gets the remainder.
  Passage is query-centered: title whole, description whole (trimmed at word
  boundary if needed), then body sentences ordered around the first query hit.

The public entry points are:
  * build_reranker_pairs()  -- build and verify (query, passage) pairs
  * extract_query_centered_passage() -- token-budgeted passage for live search
"""
import re

import numpy as np
import pandas as pd

import config
from src.model_registry import get_cross_encoder

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Derived at module level so they are computed once from config.
_TOTAL_CAP = config.RERANK_MAX_LEN                   # 512
_PAIR_SPECIAL_TOKENS = config.RERANK_PAIR_SPECIAL_TOKENS  # 4
_CONTENT_BUDGET = _TOTAL_CAP - _PAIR_SPECIAL_TOKENS  # 508
# Per-side budget in article-to-article mode (exact 50/50)
_SIDE_BUDGET = _CONTENT_BUDGET // 2                  # 254


# ---------------------------------------------------------------------------
# Internal text helpers
# ---------------------------------------------------------------------------

def _join_text(a: str, b: str) -> str:
    """Join two text segments with correct punctuation spacing."""
    a, b = a.strip(), b.strip()
    if not a:
        return b
    if not b:
        return a
    sep = " " if a[-1] in ".!?:;\"'\u201d\u2019" else ". "
    return f"{a}{sep}{b}"


def _tok_len(text: str, tokenizer) -> int:
    if not text or not tokenizer:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _trim_to_budget(text: str, budget: int, tokenizer) -> str:
    """Trim text to at most `budget` tokens, stopping at the last whole sentence,
    or at a word boundary if even the first sentence exceeds the budget."""
    if not text:
        return ""
    if not tokenizer:
        # Rough fallback: ~1.3 tokens per word
        words = text.split()
        return " ".join(words[:max(1, int(budget / 1.3))])

    if _tok_len(text, tokenizer) <= budget:
        return text

    # Try sentence boundary first
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    chosen = []
    for s in sentences:
        candidate = " ".join(chosen + [s])
        if _tok_len(candidate, tokenizer) <= budget:
            chosen.append(s)
        else:
            break
    if chosen:
        return " ".join(chosen)

    # First sentence alone > budget: trim at word boundary via binary search
    words = (sentences[0] if sentences else text).split()
    low, high, best = 0, len(words), 0
    while low <= high:
        mid = (low + high) // 2
        if _tok_len(" ".join(words[:mid]), tokenizer) <= budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return " ".join(words[:best])


# ---------------------------------------------------------------------------
# Query-centered passage extraction (live search mode)
# ---------------------------------------------------------------------------

def extract_query_centered_passage(
    title: str,
    description: str,
    body: str,
    query_text: str,
    passage_budget: int,
    tokenizer,
) -> str:
    """
    Build a passage for live-search reranking that:
      1. Always starts with title (whole) and description (trimmed at word
         boundary only if title + description together exceed the budget).
      2. Finds the first sentence containing a query token.
      3. Expands outward (left then right) around that sentence until the
         passage budget is full.
      4. Falls back to lead sentences if no query match is found.
    
    Returns a string guaranteed to be <= passage_budget tokens.
    """
    title = str(title).strip() if pd.notna(title) else ""
    description = str(description).strip() if pd.notna(description) else ""
    body = str(body).strip() if pd.notna(body) else ""

    header = _join_text(title, description)
    header_toks = _tok_len(header, tokenizer)

    # If even title + description exceed budget, trim description at word boundary
    if header_toks > passage_budget:
        words = description.split()
        low, high, best = 0, len(words), 0
        while low <= high:
            mid = (low + high) // 2
            cand = _join_text(title, " ".join(words[:mid]))
            if _tok_len(cand, tokenizer) <= passage_budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return _join_text(title, " ".join(words[:best]))

    body_budget = max(0, passage_budget - header_toks)
    if body_budget <= 0 or not body:
        return header

    sentences = [s.strip() for s in SENTENCE_SPLIT.split(body) if s.strip()]
    if not sentences:
        return header

    # Find first sentence containing a query term
    query_words = {w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", query_text)}
    hit_idx = None
    if query_words:
        for idx, s in enumerate(sentences):
            s_words = {w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", s)}
            if s_words & query_words:
                hit_idx = idx
                break

    if hit_idx is None:
        # No query match: use lead sentences up to budget
        chosen = []
        for s in sentences:
            if _tok_len(_join_text(header, " ".join(chosen + [s])), tokenizer) <= passage_budget:
                chosen.append(s)
            else:
                break
        return _join_text(header, " ".join(chosen))

    # Anchor hit sentence, expand left then right
    s_hit = sentences[hit_idx]
    if _tok_len(_join_text(header, s_hit), tokenizer) > passage_budget:
        # Hit sentence itself too big — trim at word boundary
        words = s_hit.split()
        low, high, best = 0, len(words), 0
        while low <= high:
            mid = (low + high) // 2
            cand = _join_text(header, " ".join(words[:mid]))
            if _tok_len(cand, tokenizer) <= passage_budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return _join_text(header, " ".join(words[:best]))

    chosen = [s_hit]
    l_ptr, r_ptr = hit_idx - 1, hit_idx + 1
    while l_ptr >= 0 or r_ptr < len(sentences):
        expanded = False
        if l_ptr >= 0:
            cand = _join_text(header, " ".join([sentences[l_ptr]] + chosen))
            if _tok_len(cand, tokenizer) <= passage_budget:
                chosen.insert(0, sentences[l_ptr])
                l_ptr -= 1
                expanded = True
            else:
                l_ptr = -1
        if r_ptr < len(sentences):
            cand = _join_text(header, " ".join(chosen + [sentences[r_ptr]]))
            if _tok_len(cand, tokenizer) <= passage_budget:
                chosen.append(sentences[r_ptr])
                r_ptr += 1
                expanded = True
            else:
                r_ptr = len(sentences)
        if not expanded:
            break

    return _join_text(header, " ".join(chosen))


# ---------------------------------------------------------------------------
# Shared pair builder — used by recommender AND pipeline precompute
# ---------------------------------------------------------------------------

def build_reranker_pairs(
    query: str,
    documents: list,
    is_search: bool = False,
    tokenizer=None,
    articles_df=None,
) -> list:
    """
    Build (query, passage) pairs with guaranteed token budgets.

    Article-to-article (is_search=False):
      * query = source article's text_rerank (pre-budgeted to 254 tokens at ingest)
      * passage = candidate article's text_rerank (same budget)
      * If either side exceeds _SIDE_BUDGET (rounding or legacy data), hard-trim.

    Live search (is_search=True):
      * query is the raw search string (trimmed to at most 254 tokens)
      * passage_budget = 508 - q_tokens (remainder goes to passage)
      * Each document is expected to be a dict/Series with title, description, body keys.

    Always asserts total pair <= 512. Logs any violation before correcting it.
    Returns list of (q_text, d_text) string tuples in the same order as documents.
    """
    pairs = []
    over_limit = 0
    max_seen = 0

    if not is_search:
        # Article-to-article: trim both sides to SIDE_BUDGET
        q_str = str(query).strip()
        q_text = _trim_to_budget(q_str, _SIDE_BUDGET, tokenizer)

        for doc in documents:
            d_str = str(doc).strip() if not isinstance(doc, (dict, pd.Series)) else (
                str(doc.get("text_rerank") or _join_text(str(doc.get("title", "")), str(doc.get("description", ""))))
            )
            d_text = _trim_to_budget(d_str, _SIDE_BUDGET, tokenizer)

            if tokenizer:
                pair_toks = len(tokenizer(q_text, d_text, add_special_tokens=True)["input_ids"])
                max_seen = max(max_seen, pair_toks)
                if pair_toks > _TOTAL_CAP:
                    over_limit += 1
                    print(
                        f"[Reranker] VIOLATION: article-to-article pair has {pair_toks} tokens "
                        f"(cap {_TOTAL_CAP}). Hard-clamping passage side."
                    )
                    rem = max(1, _CONTENT_BUDGET - _tok_len(q_text, tokenizer))
                    d_text = _trim_to_budget(d_text, rem, tokenizer)

            pairs.append((q_text, d_text))
    else:
        # Live search: query-first dynamic split
        q_str = str(query).strip()
        q_text = _trim_to_budget(q_str, _SIDE_BUDGET, tokenizer)
        q_tokens = _tok_len(q_text, tokenizer) if tokenizer else max(1, len(q_text.split()))
        passage_budget = max(1, _CONTENT_BUDGET - q_tokens)

        for doc in documents:
            if isinstance(doc, (dict, pd.Series)):
                t = str(doc.get("title", "") or "")
                d = str(doc.get("description", "") or "")
                b = str(doc.get("body", "") or "")
                d_text = extract_query_centered_passage(t, d, b, q_str, passage_budget, tokenizer)
            else:
                d_text = _trim_to_budget(str(doc), passage_budget, tokenizer)

            if tokenizer:
                pair_toks = len(tokenizer(q_text, d_text, add_special_tokens=True)["input_ids"])
                max_seen = max(max_seen, pair_toks)
                if pair_toks > _TOTAL_CAP:
                    over_limit += 1
                    print(
                        f"[Reranker] VIOLATION: search pair has {pair_toks} tokens "
                        f"(cap {_TOTAL_CAP}). Hard-clamping passage."
                    )
                    rem = max(1, _CONTENT_BUDGET - q_tokens)
                    d_text = _trim_to_budget(d_text, rem, tokenizer)

            pairs.append((q_text, d_text))

    mode = "search" if is_search else "article-to-article"
    if over_limit:
        print(
            f"[Reranker] Budget summary ({mode}): {len(pairs)} pairs, "
            f"{over_limit} over-limit (corrected), max_pair={max_seen}/{_TOTAL_CAP}."
        )
    return pairs


# ---------------------------------------------------------------------------
# Reranker class
# ---------------------------------------------------------------------------

class Reranker:
    def __init__(self, model_name=None):
        from torch import nn
        self.model = get_cross_encoder(model_name)   # shared copy, loaded once per process
        self._identity = nn.Identity()
        # Reranker tokenizer for pair budgeting (lazy-loaded)
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from src.data_loader import get_reranker_tokenizer
                self._tokenizer = get_reranker_tokenizer()
            except Exception:
                self._tokenizer = None
        return self._tokenizer

    def score(self, query: str, documents: list, batch_size: int = 32) -> np.ndarray:
        """Return a relevance probability (0-1) for each document.
        
        The query and each document must already be pre-budgeted before calling.
        Use build_reranker_pairs() to prepare (query, passage) tuples.
        """
        if not documents:
            return np.array([])
        pairs = [(query, doc) for doc in documents]
        try:
            logits = np.asarray(
                self.model.predict(
                    pairs,
                    batch_size=batch_size,
                    activation_fn=self._identity,
                    show_progress_bar=False,
                ),
                dtype=float,
            )
        except TypeError:
            logits = np.asarray(
                self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False),
                dtype=float,
            )
            if logits.size and np.nanmax(np.abs(logits)) <= 1.0 + 1e-6:
                return np.clip(logits, 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def score_pairs(
        self,
        query: str,
        documents: list,
        is_search: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Budget-safe entry point. Builds pairs then scores them.
        
        Use this instead of score() whenever possible.
        is_search=True  -> live text query (query-first split)
        is_search=False -> article-to-article mode (50/50 split)
        """
        if not documents:
            return np.array([])
        tok = self._get_tokenizer()
        pairs = build_reranker_pairs(query, documents, is_search=is_search, tokenizer=tok)
        qs = [p[0] for p in pairs]
        ds = [p[1] for p in pairs]
        # All queries are the same — call score() with the list of passages
        # Use the first query string (they are all the same trimmed query)
        return self.score(qs[0], ds, batch_size=batch_size)