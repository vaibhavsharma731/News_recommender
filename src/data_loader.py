"""
Step 1 - Load the CSV and clean it. Every cleaning rule here comes from the EDA findings.

===========================================================================
THIS MODULE IS THE SINGLE AUTHORITATIVE DATA LOADER.
All consumers (EDA, evaluation, ingestion pipeline, API) must use these
functions.  Do NOT duplicate data-loading or cleaning logic elsewhere.
===========================================================================

Cleaning rules:
  1. drop duplicate articles (single policy: latest postPublishDate wins)
  2. normalise text: curly quotes -> plain quotes, \\r\\n and extra spaces -> one space
  3. remove links and e-mail addresses
  4. remove inline "ALSO READ <headline of another article>" blocks
     -> we also extract linked article IDs: editor choices become ground truth labels for evaluate.py
  5. remove boilerplate sentences repeated in many articles (disclaimers, quiz lines, etc.)
  6. shorten very long descriptions (video descriptions can be 1,000+ characters)
"""
import hashlib
import json
from pathlib import Path
import re
import time as _time
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import config

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
ALSO_READ_RE = re.compile(r"\balso\s+read\b[\s:|\-\u2013\u2014]*", re.I)
ALSO_READ_FALLBACK_WORDS = 14       # headline length used when the linked article is not in dataset
ALSO_READ_KEY_WORDS = 4             # first N words of a headline are used to match it
QUOTES = {ord(c): "'" for c in "\u2018\u2019\u201a\u201b"} | {ord(c): '"' for c in "\u201c\u201d\u201e"}


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def normalise_text(text) -> str:
    """Unicode clean-up: smart quotes become plain quotes, whitespace collapses."""
    text = unicodedata.normalize("NFKC", str(text) if pd.notna(text) else "")
    text = text.translate(QUOTES).replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def remove_links(text: str) -> str:
    return re.sub(r"\s+", " ", EMAIL_RE.sub(" ", URL_RE.sub(" ", text))).strip()


def _word(token: str) -> str:
    """Comparable form of a word: lowercase alphanumeric characters only."""
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _headline_index(titles, ids) -> dict:
    """first 4 words of every title -> (article id, number of words in the title)"""
    index = {}
    for article_id, title in zip(ids, titles):
        words = [w for w in map(_word, title.split()) if w]
        if len(words) >= ALSO_READ_KEY_WORDS:
            index[tuple(words[:ALSO_READ_KEY_WORDS])] = (article_id, len(words))
    return index


def strip_also_read(body: str, own_id, headlines: dict):
    """Cut every 'ALSO READ <headline>' block out of the body. Returns (clean body, linked article ids)."""
    pieces, linked, cursor = [], [], 0
    for match in ALSO_READ_RE.finditer(body):
        if match.start() < cursor:
            continue                                    # already inside a removed block
        pieces.append(body[cursor:match.start()])
        segment = body[match.end(): match.end() + 400]
        tokens = [(w, m.end()) for m in re.finditer(r"\S+", segment) if (w := _word(m.group()))]
        key = tuple(w for w, _ in tokens[:ALSO_READ_KEY_WORDS])
        article_id, n_words = headlines.get(key, (None, ALSO_READ_FALLBACK_WORDS))
        if article_id is not None and article_id != own_id:
            linked.append(article_id)
        cut = tokens[min(n_words, len(tokens)) - 1][1] if tokens else 0
        cursor = match.end() + cut
    pieces.append(body[cursor:])
    return re.sub(r"\s+", " ", " ".join(pieces)).strip(), linked


def _find_boilerplate(bodies) -> set:
    """Sentences that appear in many different articles are not about any specific topic."""
    counts = Counter()
    for body in bodies:
        counts.update({s.strip() for s in SENTENCE_SPLIT.split(body) if len(s.strip()) > 25})
    return {s for s, c in counts.items() if c >= config.BOILERPLATE_MIN_DOCS}


def _strip_boilerplate(body: str, boilerplate: set) -> str:
    return " ".join(s for s in SENTENCE_SPLIT.split(body) if s.strip() not in boilerplate).strip()


def _subsection(value) -> str:
    """'["Sports","Cricket"]' -> 'Cricket'   |   '["India"]' -> ''"""
    try:
        parts = json.loads(value)
    except (TypeError, ValueError):
        return ""
    return parts[-1] if len(parts) > 1 else ""


def _short_description(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


# ──────────────────────────────────────────────────────────────────────────────
# Token-budgeted text composition (using model-specific tokenizers)
# ──────────────────────────────────────────────────────────────────────────────

_EMBED_TOKENIZER = None
_RERANK_TOKENIZER = None


def get_embedder_tokenizer(model_name: Optional[str] = None):
    """Load and cache the HuggingFace tokenizer for the embedding model (BERT family)."""
    global _EMBED_TOKENIZER
    if _EMBED_TOKENIZER is None:
        model_name = model_name or getattr(config, "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        revision = getattr(config, "EMBEDDING_REVISION", None)
        try:
            from transformers import AutoTokenizer
            _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
            _EMBED_TOKENIZER.model_max_length = 100000  # Silence truncation warnings during budgeting
        except Exception as e:
            print(f"[DataLoader] Notice: Could not load embedder tokenizer ({e}).")
            _EMBED_TOKENIZER = None
    return _EMBED_TOKENIZER


def get_reranker_tokenizer(model_name: Optional[str] = None):
    """Load and cache the HuggingFace tokenizer for the reranker model (RoBERTa family)."""
    global _RERANK_TOKENIZER
    if _RERANK_TOKENIZER is None:
        model_name = model_name or getattr(config, "RERANKER_MODEL", "BAAI/bge-reranker-base")
        revision = getattr(config, "RERANKER_REVISION", None)
        try:
            from transformers import AutoTokenizer
            _RERANK_TOKENIZER = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
            _RERANK_TOKENIZER.model_max_length = 100000  # Silence truncation warnings during budgeting
        except Exception as e:
            print(f"[DataLoader] Notice: Could not load reranker tokenizer ({e}).")
            _RERANK_TOKENIZER = None
    return _RERANK_TOKENIZER


def get_tokenizer(model_name: Optional[str] = None):
    """Backward compatibility alias returning the primary embedder tokenizer."""
    return get_embedder_tokenizer(model_name)


def _join_text(a: str, b: str) -> str:
    """Join two text segments ensuring correct punctuation spacing."""
    a, b = a.strip(), b.strip()
    if not a:
        return b
    if not b:
        return a
    sep = " " if a[-1] in ".!?:;\"'”’" else ". "
    return f"{a}{sep}{b}"


def compose_sentence_budgeted_text(
    title: str,
    description: str,
    body: str,
    budget: int,
    tokenizer=None,
    query_prefix: str = "",
) -> Tuple[str, int, bool]:
    """
    Build a text representation budgeted strictly by model tokens:
      1. If query_prefix is present (e.g. BGE query prefix for short live queries),
         reserve space for its tokens against the budget.
      2. Title is kept whole.
      3. Description is kept whole; if title + description exceed budget,
         description is trimmed at a word boundary.
      4. Body sentences are added one by one until budget is full.
         Stops at the last whole sentence that fits.
      5. Returns (composed_text, token_count, is_truncated).
    """
    title = str(title).strip() if title and pd.notna(title) else ""
    description = str(description).strip() if description and pd.notna(description) else ""
    body = str(body).strip() if body and pd.notna(body) else ""

    prefix_len = 0
    if query_prefix and tokenizer:
        prefix_len = len(tokenizer.encode(query_prefix, add_special_tokens=False))
    effective_budget = max(1, budget - prefix_len)

    # Fallback when tokenizer is unavailable: ~1.3 tokens per word
    if tokenizer is None:
        t_words = title.split()
        d_words = description.split()
        t_est = int(len(t_words) * 1.3)
        rem = max(0, effective_budget - t_est)
        d_rem_words = max(0, int(rem / 1.3))
        if len(d_words) > d_rem_words:
            d_trimmed = " ".join(d_words[:d_rem_words])
            cur = _join_text(title, d_trimmed)
            return cur, int(len(cur.split()) * 1.3), True

        cur = _join_text(title, description)
        cur_toks = int(len(cur.split()) * 1.3)
        sents = [s.strip() for s in SENTENCE_SPLIT.split(body) if s.strip()]
        is_trunc = False
        for s in sents:
            nxt = _join_text(cur, s)
            nxt_toks = int(len(nxt.split()) * 1.3)
            if nxt_toks <= effective_budget:
                cur = nxt
                cur_toks = nxt_toks
            else:
                is_trunc = True
                break
        return cur, cur_toks, is_trunc

    # Model tokenizer path
    header = _join_text(title, description)
    header_enc = tokenizer.encode(header, add_special_tokens=False) if header else []

    if len(header_enc) > effective_budget:
        # Title plus description exceed budget: trim description at a word boundary
        words = description.split()
        low, high, best = 0, len(words), 0
        while low <= high:
            mid = (low + high) // 2
            cand_desc = " ".join(words[:mid])
            cand = _join_text(title, cand_desc)
            cand_len = len(tokenizer.encode(cand, add_special_tokens=False)) if cand else 0
            if cand_len <= effective_budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        trimmed_desc = " ".join(words[:best])
        cur = _join_text(title, trimmed_desc)
        cur_len = len(tokenizer.encode(cur, add_special_tokens=False)) if cur else 0
        return cur, cur_len, True

    cur = header
    cur_len = len(header_enc)
    sents = [s.strip() for s in SENTENCE_SPLIT.split(body) if s.strip()]
    is_trunc = False
    for s in sents:
        nxt = _join_text(cur, s)
        nxt_len = len(tokenizer.encode(nxt, add_special_tokens=False))
        if nxt_len <= effective_budget:
            cur = nxt
            cur_len = nxt_len
        else:
            is_trunc = True
            break

    return cur, cur_len, is_trunc


def build_sentence_budgeted_texts(
    titles: List[str],
    descriptions: List[str],
    bodies: List[str],
    budget: int,
    tokenizer=None,
    query_prefix: str = "",
) -> Tuple[List[str], List[int], List[bool]]:
    """
    Batch compose budgeted texts across articles:
    Returns (texts, token_counts, is_truncated_flags).
    """
    texts: List[str] = []
    counts: List[int] = []
    trunc_flags: List[bool] = []

    for t, d, b in zip(titles, descriptions, bodies):
        text, count, is_trunc = compose_sentence_budgeted_text(
            t, d, b, budget=budget, tokenizer=tokenizer, query_prefix=query_prefix
        )
        texts.append(text)
        counts.append(count)
        trunc_flags.append(is_trunc)

    return texts, counts, trunc_flags


def build_token_budgeted_texts(
    titles: List[str],
    descriptions: List[str],
    bodies: List[str],
    max_tokens: int,
    tokenizer=None,
    special_tokens_reserve: int = 2,
) -> List[str]:
    """Backward compatibility wrapper returning only text list."""
    budget = max(1, max_tokens - special_tokens_reserve)
    texts, _, _ = build_sentence_budgeted_texts(
        titles, descriptions, bodies, budget=budget, tokenizer=tokenizer
    )
    return texts



# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities for flexible column resolution and type coercion
# ──────────────────────────────────────────────────────────────────────────────

_ID_CANDIDATES = ["articleId", "id", "article_id", "ID"]
_DATE_CANDIDATES = ["postPublishDate", "date", "published_at", "publishDate"]
_TITLE_CANDIDATES = ["articleTitle", "title", "headline", "articleName"]


def _col(raw: pd.DataFrame, candidates: list, default="") -> pd.Series:
    """Return the first column from *candidates* that exists in *raw*, else a constant Series."""
    for c in candidates:
        if c in raw.columns:
            return raw[c]
    return pd.Series([default] * len(raw), index=raw.index)


def _as_bool(val) -> bool:
    """Robust boolean parsing for diverse CSV formats."""
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "y", "t"}


def _resolve_column(raw: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first column name from *candidates* that exists in *raw*."""
    for c in candidates:
        if c in raw.columns:
            return c
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Raw CSV loading
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_csv(paths=None) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files without any cleaning or dedup.

    This is the raw entry point used by EDA for pre-cleaning statistics and
    by ``load_articles()`` as input to ``clean_dataframe()``.
    """
    if paths is None:
        paths = getattr(config, "DATA_PATHS", [config.DATA_PATH])
    elif not isinstance(paths, (list, tuple)):
        paths = [paths]

    dfs = []
    for p in paths:
        p_obj = Path(p)
        if p_obj.exists():
            print(f"[DataLoader] Loading {p_obj.name}...")
            dfs.append(pd.read_csv(p_obj))
        else:
            print(f"[DataLoader] Warning: {p_obj} does not exist, skipping.")

    if not dfs:
        raise FileNotFoundError(f"No valid CSV dataset files found in {paths}")

    return pd.concat(dfs, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Deduplication — ONE policy for the whole project
# ──────────────────────────────────────────────────────────────────────────────

def _deduplicate(
    raw: pd.DataFrame,
    id_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Single duplicate policy for the entire project:

      **Latest postPublishDate wins.**
      Ties and missing dates fall back to last-in-file-order (preserving the
      previous ``keep='last'`` behaviour for articles without dates).

    Returns ``(deduped_df, stats)`` where *stats* contains:

      * ``duplicates_dropped`` -- total rows removed
      * ``content_conflicts`` -- how many duplicate IDs had *different* titles
    """
    n_before = len(raw)
    stats: Dict[str, Any] = {
        "total_before": n_before,
        "duplicates_dropped": 0,
        "content_conflicts": 0,
        "total_after": n_before,
    }

    dupes_mask = raw.duplicated(subset=id_col, keep=False)
    if not dupes_mask.any():
        return raw.reset_index(drop=True), stats

    # Count content conflicts (same ID, different title)
    title_col = _resolve_column(raw, _TITLE_CANDIDATES)
    if title_col:
        stats["content_conflicts"] = sum(
            1
            for _, g in raw[dupes_mask].groupby(id_col)
            if g[title_col].nunique() > 1
        )

    # Sort so the *winner* of each ID group comes first:
    #   primary key:   latest date first
    #   tiebreaker:    later position in file wins (same as old keep="last")
    date_col = _resolve_column(raw, _DATE_CANDIDATES)
    raw = raw.copy()
    raw["_dedup_order"] = range(len(raw))

    if date_col:
        raw["_dedup_date"] = pd.to_datetime(raw[date_col], utc=True, errors="coerce")
        raw = raw.sort_values(
            ["_dedup_date", "_dedup_order"],
            ascending=[False, False],
            na_position="last",
        )
        raw = raw.drop(columns=["_dedup_date"])
    else:
        # No date column — fall back to "last in file wins"
        raw = raw.sort_values("_dedup_order", ascending=False)

    raw = raw.drop(columns=["_dedup_order"])
    raw = raw.drop_duplicates(subset=id_col, keep="first")
    raw = raw.sort_index()          # restore original file order
    raw = raw.reset_index(drop=True)

    stats["duplicates_dropped"] = n_before - len(raw)
    stats["total_after"] = len(raw)
    return raw, stats


# ──────────────────────────────────────────────────────────────────────────────
# clean_dataframe — THE single authoritative cleaning function
# ──────────────────────────────────────────────────────────────────────────────

def clean_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalise a raw article DataFrame for ingestion or evaluation.

    This is the **single authoritative cleaning function** shared by every
    consumer (disk-based ``load_articles``, the ingestion pipeline, EDA, and
    evaluation).  Do NOT create parallel cleaning code elsewhere.

    Input
    -----
    A raw ``pd.DataFrame`` straight from ``pd.read_csv`` or ``load_raw_csv``.

    Output
    ------
    A cleaned DataFrame with **snake_case** column names for DB compatibility::

        id, title, description, section, subsection, date, url, image,
        is_live, body, linked_ids (JSON string), page_views, unique_users,
        text_embed, text_bm25, text_rerank
    """
    raw = raw_df.copy()

    # 1. Resolve ID column and deduplicate
    id_col_name = _resolve_column(raw, _ID_CANDIDATES)
    if id_col_name:
        raw, dedup_stats = _deduplicate(raw, id_col_name)
        ids = raw[id_col_name].astype(str).tolist()
    else:
        # Generate collision-safe synthetic IDs when no ID column exists
        raw = raw.reset_index(drop=True)
        prefix = hashlib.md5(str(_time.time()).encode()).hexdigest()[:6]
        ids = [f"art_{prefix}_{i + 1}" for i in range(len(raw))]
        dedup_stats = {
            "total_before": len(raw),
            "duplicates_dropped": 0,
            "content_conflicts": 0,
            "total_after": len(raw),
        }
        print(
            f"[DataLoader] [!] No ID column found. "
            f"Generated synthetic IDs with prefix '{prefix}'."
        )

    print(
        f"[DataLoader] Dedup: {dedup_stats['total_before']} rows -> "
        f"{dedup_stats['total_after']} unique articles "
        f"({dedup_stats['duplicates_dropped']} duplicates removed"
        + (
            f", {dedup_stats['content_conflicts']} had conflicting content"
            if dedup_stats["content_conflicts"]
            else ""
        )
        + ")."
    )

    # 2. Extract and clean text fields (flexible column names)
    titles = _col(raw, _TITLE_CANDIDATES).map(normalise_text)
    raw_desc = _col(raw, ["description", "snippet", "summary"])
    raw_bodies = _col(raw, ["body_raw", "body", "content", "text"]).fillna("")

    # If description is missing/empty, fall back to first 200 chars of body
    clean_desc = []
    for d, b in zip(raw_desc, raw_bodies):
        text = str(d).strip() if pd.notna(d) and str(d).strip() else str(b)[:200].strip()
        clean_desc.append(_short_description(remove_links(normalise_text(text))))

    # 3. Build headline index and extract "Also Read" links
    headlines = _headline_index(titles, ids)
    bodies, linked_ids = [], []
    for article_id, body in zip(ids, raw_bodies):
        body_clean = remove_links(normalise_text(body))
        body_clean, linked = strip_also_read(body_clean, article_id, headlines)
        bodies.append(body_clean)
        linked_ids.append(sorted(set(linked)))

    # 4. Remove boilerplate sentences
    boilerplate = _find_boilerplate(bodies)
    bodies = [_strip_boilerplate(b, boilerplate) for b in bodies]

    # 5. Resolve metadata columns flexibly
    sec_series = _col(raw, ["sectionName", "section1Slug", "section", "category"])
    subsec_series = (
        raw["section2Name"].map(_subsection)
        if "section2Name" in raw.columns
        else _col(raw, ["section2Slug", "subsection"])
    )
    pv_series = _col(raw, ["pageViews", "pageViews_latest", "pageViews_max", "views"], default=0)
    uu_series = _col(
        raw, ["uniqueUsers", "uniqueUsers_latest", "uniqueUsers_max", "users"], default=0
    )
    date_series = _col(raw, _DATE_CANDIDATES, default=None)

    # 6. Build standardised output (snake_case for DB compatibility)
    clean_df = pd.DataFrame({
        "id": ids,
        "title": titles.values,
        "description": clean_desc,
        "section": sec_series.fillna("").values,
        "subsection": subsec_series.fillna("").values,
        "date": pd.to_datetime(date_series, utc=True, errors="coerce"),
        "url": _col(raw, ["articleUrl", "url", "link"]).fillna("").values,
        "image": _col(raw, ["articleFeaturedImage", "image", "imageUrl", "img"]).fillna("").values,
        "is_live": _col(raw, ["liveBlog", "is_live"], default=False).map(_as_bool).values,
        "body": bodies,
        "linked_ids": [json.dumps(l) for l in linked_ids],
        "page_views": pd.to_numeric(pv_series, errors="coerce").fillna(0).astype(int).values,
        "unique_users": pd.to_numeric(uu_series, errors="coerce").fillna(0).astype(int).values,
    })

    # 7. Build text representations for the ML pipeline
    embed_budget = getattr(
        config,
        "EMBED_TOKEN_BUDGET",
        getattr(config, "EMBED_MAX_SEQ_LEN", 256) - getattr(config, "EMBED_SPECIAL_TOKENS", 2)
    )
    rerank_budget = getattr(
        config,
        "RERANK_TOKEN_BUDGET",
        getattr(config, "RERANK_PER_ARTICLE_TOKENS", 256) - 2
    )

    embed_tok = get_embedder_tokenizer()
    rerank_tok = get_reranker_tokenizer()

    embed_texts, embed_counts, embed_trunc = build_sentence_budgeted_texts(
        clean_df.title.tolist(),
        clean_df.description.tolist(),
        clean_df.body.tolist(),
        budget=embed_budget,
        tokenizer=embed_tok,
    )

    rerank_texts, rerank_counts, rerank_trunc = build_sentence_budgeted_texts(
        clean_df.title.tolist(),
        clean_df.description.tolist(),
        clean_df.body.tolist(),
        budget=rerank_budget,
        tokenizer=rerank_tok,
    )

    recipe_version = getattr(config, "TEXT_RECIPE_VERSION", "v2_sentence_token_budget")
    is_truncated_list = [bool(et or rt) for et, rt in zip(embed_trunc, rerank_trunc)]

    clean_df["text_embed"] = embed_texts
    clean_df["embed_token_count"] = embed_counts
    clean_df["text_bm25"] = [
        f"{t}. {t}. {t}. {d}. {b}"
        for t, d, b in zip(clean_df.title, clean_df.description, clean_df.body)
    ]
    clean_df["text_rerank"] = rerank_texts
    clean_df["rerank_token_count"] = rerank_counts
    clean_df["is_truncated"] = is_truncated_list
    clean_df["recipe_version"] = recipe_version

    # Summary statistics logging: no silent truncation, clear visibility
    trunc_count = sum(is_truncated_list)
    total_arts = len(clean_df)
    trunc_pct = (trunc_count / total_arts * 100) if total_arts else 0.0
    e_mean = float(np.mean(embed_counts)) if embed_counts else 0.0
    e_max = int(np.max(embed_counts)) if embed_counts else 0
    r_mean = float(np.mean(rerank_counts)) if rerank_counts else 0.0
    r_max = int(np.max(rerank_counts)) if rerank_counts else 0
    print(
        f"[DataLoader] Text Budgets (recipe={recipe_version}):\n"
        f"  * text_embed  (BERT, cap {embed_budget}): mean={e_mean:.1f}, max={e_max} tokens\n"
        f"  * text_rerank (RoBERTa, cap {rerank_budget}): mean={r_mean:.1f}, max={r_max} tokens\n"
        f"  * Truncated: {trunc_count}/{total_arts} ({trunc_pct:.1f}%) articles capped at whole sentence boundary."
    )
    return clean_df


# ──────────────────────────────────────────────────────────────────────────────
# load_articles — high-level disk-to-DataFrame API
# ──────────────────────────────────────────────────────────────────────────────

def load_articles(paths=None) -> pd.DataFrame:
    """
    Load article CSVs from disk, clean them, and return a DataFrame.

    This is the high-level entry point used by ``evaluate.py``,
    ``Recommender``, and the API server's CSV fallback path.

    Returns a DataFrame with backward-compatible column names:

      * ``pageViews``   (camelCase, not ``page_views``)
      * ``uniqueUsers`` (camelCase, not ``unique_users``)
      * ``linked_ids``  as **Python lists** (not JSON strings)
    """
    raw = load_raw_csv(paths)
    df = clean_dataframe(raw)

    # Backward compatibility for evaluate.py / recommender.py
    df = df.rename(columns={
        "page_views": "pageViews",
        "unique_users": "uniqueUsers",
    })
    df["linked_ids"] = df["linked_ids"].map(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    return df
