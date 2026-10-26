# 📖 Chapter 2: Data Loading & Text Hygiene — Deep Dive

> **File**: [`src/data_loader.py`](file:///d:/indian%20express/Recommendation%20system/news%20recommender%20base/src/data_loader.py) (650 lines)
> **Difficulty**: ⭐⭐ Medium → ⭐⭐⭐ Hard (Token budgeting & regex heuristics)
> **Time**: ~60-90 minutes for thorough study
> **Prerequisites**: [Chapter 1: `config.py`](file:///C:/Users/Dell/.gemini/antigravity-ide/brain/fe20be04-4258-48ca-9d2b-1eabffec3f27/chapter_01_config.md)

---

## 🧭 Why This File is the "Heart" of the Backend

In machine learning and recommendation systems, **Garbage In = Garbage Out**.
If you feed raw scraped news articles into a neural network:
- Curly quotes and weird symbols will corrupt character-level encodings.
- Links and email addresses will distract semantic embeddings.
- "ALSO READ: <Headline>" ads inside the article text will mislead topic matchers.
- Truncating text abruptly at 256 tokens in the middle of a sentence ruins cross-encoder reranking.
- Duplicate articles with slight timestamp differences will pollute recommendation slots.

[`src/data_loader.py`](file:///d:/indian%20express/Recommendation%20system/news%20recommender%20base/src/data_loader.py) is officially documented as:
> **THE SINGLE AUTHORITATIVE DATA LOADER**
> All consumers (EDA, evaluation, ingestion pipeline, API) must use these functions.
> Do NOT duplicate data-loading or cleaning logic elsewhere.

Let's dissect its 650 lines across **10 logical sections**.

---

## Section A: Imports, Constants & Regex Mastery (Lines 1-38)

```python
import hashlib                                                                   # Line 19
import json                                                                      # Line 20
from pathlib import Path                                                         # Line 21
import re                                                                        # Line 22
import time as _time                                                             # Line 23
import unicodedata                                                               # Line 24
from collections import Counter                                                  # Line 25
from typing import Any, Dict, List, Optional, Tuple                              # Line 26
import numpy as np                                                               # Line 27
import pandas as pd                                                              # Line 28
import config                                                                    # Line 29

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")                                   # Line 31
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)                             # Line 32
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")                           # Line 33
ALSO_READ_RE = re.compile(r"\balso\s+read\b[\s:|\-\u2013\u2014]*", re.I)       # Line 34
ALSO_READ_FALLBACK_WORDS = 14                                                   # Line 35
ALSO_READ_KEY_WORDS = 4                                                         # Line 36
QUOTES = {ord(c): "'" for c in "\u2018\u2019\u201a\u201b"} | {ord(c): '"' for c in "\u201c\u201d\u201e"}  # Line 37
```

### 🧠 Concepts to Master

#### 1. Why `re.compile()`?
If you write `re.sub(pattern, replacement, text)` inside a loop over 10,000 articles, Python has to parse and compile the regular expression into bytecode **10,000 times**.
`re.compile()` converts the regex into an immutable Pattern object **once** at import time. Inside loops, calling `PATTERN.sub(...)` runs directly in C at maximum speed.

#### 2. Advanced Regex: Positive Lookbehind `(?<=...)`
Look at Line 31:
```python
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
```
- Plain `[.!?]\s+` would match the punctuation mark **and** the space. Splitting on that would **delete the punctuation**!
  - Example: `"Hello world. How are you?"` → `["Hello world", "How are you?"]` (the `.` is lost!)
- `(?<=[.!?])` is a **Zero-Width Positive Lookbehind Assertion**.
  - It says: *"Match whitespace `\s+`, BUT ONLY IF it is immediately preceded by `.`, `!`, or `?`"*.
  - Because it is zero-width, the `.` is **not** part of the match!
  - Result: `["Hello world.", "How are you?"]` — the punctuation stays intact on the sentence!

#### 3. Unicode Characters in Regex
Look at Line 34: `[\s:|\-\u2013\u2014]*`
- `\u2013` is an **En-dash** (`–`, used for ranges like 2020–2026).
- `\u2014` is an **Em-dash** (`—`, used for editorial breaks like "Delhi — The Supreme Court...").
- News reporters and CMS systems constantly paste MS Word text containing en-dashes and em-dashes instead of standard ASCII hyphens `-`.

#### 4. `str.translate()` & Dictionary Union (`|`)
Look at Line 37:
```python
QUOTES = {ord(c): "'" for c in "\u2018\u2019\u201a\u201b"} | {ord(c): '"' for c in "\u201c\u201d\u201e"}
```
- `\u2018` and `\u2019` are curly single quotes (`‘` and `’`).
- `\u201c` and `\u201d` are curly double quotes (`“` and `”`).
- `ord(c)` returns the integer Unicode code point of a character (e.g., `ord('‘')` = `8216`).
- The `|` operator (introduced in Python 3.9) merges two dictionaries together!
- `str.translate(table)` is a built-in C-level method in Python. Passing a `{code_point: replacement_str}` dict replaces hundreds of characters across megabytes of text in microseconds — much faster than calling `.replace()` 8 times.

---

## Section B: Text Hygiene & Unicode Normalization (Lines 40-58)

```python
def first_words(text: str, n: int) -> str:                                       # Line 40
    return " ".join(text.split()[:n])                                            # Line 41


def normalise_text(text) -> str:                                                 # Line 44
    """Unicode clean-up: smart quotes become plain quotes, whitespace collapses."""
    text = unicodedata.normalize("NFKC", str(text) if pd.notna(text) else "")    # Line 46
    text = text.translate(QUOTES).replace("\u2013", "-").replace("\u2014", "-") # Line 47
    return re.sub(r"\s+", " ", text).strip()                                     # Line 48


def remove_links(text: str) -> str:                                              # Line 51
    return re.sub(r"\s+", " ", EMAIL_RE.sub(" ", URL_RE.sub(" ", text))).strip() # Line 52


def _word(token: str) -> str:                                                    # Line 55
    """Comparable form of a word: lowercase alphanumeric characters only."""
    return re.sub(r"[^a-z0-9]", "", token.lower())                               # Line 57
```

### 🧠 Deep Dive: `unicodedata.normalize("NFKC", text)`

What does `NFKC` stand for?
**Normalization Form Compatibility Composition**.

In real-world text, characters can look identical to humans but be encoded completely differently in memory:
1. **Ligatures**: In typography, `ﬁ` (one single character U+FB01) is often printed instead of `f` + `i` (two characters). NFKC decomposes `ﬁ` into standard ASCII `f` and `i`.
2. **Full-width characters**: Sometimes Asian keyboards type `ＡＢＣ` (full-width) instead of standard `ABC`. NFKC converts them to standard ASCII.
3. **Accented letters**: `é` can be encoded as a single code point (`\u00E9`) or as `e` followed by an accent mark (`e\u0301`). NFKC unifies them into a canonical single representation.

By running `unicodedata.normalize("NFKC", ...)` followed by `.translate(QUOTES)`, every article is guaranteed to speak pure standard ASCII punctuation before entering any neural model.

### Notice the Method Nesting in `remove_links`:
```python
re.sub(r"\s+", " ", EMAIL_RE.sub(" ", URL_RE.sub(" ", text))).strip()
```
It executes from the inside out:
1. `URL_RE.sub(" ", text)` replaces `http://...` or `www....` with a space `" "`
2. `EMAIL_RE.sub(" ", ...)` replaces `user@domain.com` with a space `" "`
3. `re.sub(r"\s+", " ", ...)` collapses multiple spaces into a single space
4. `.strip()` removes leading/trailing spaces

---

## Section C: The Editorial Link Extractor — "ALSO READ" (Lines 60-87)

This is one of the smartest domain-specific algorithms in the entire project.

### The Problem
Indian Express articles often have inline promotion blocks inserted right into the article body:
```text
The Supreme Court on Friday stayed the trial. ALSO READ: Manish Sisodia granted bail in excise policy case. In another development...
```
If we leave this in:
1. The article about the Supreme Court trial gets polluted with words about "Manish Sisodia" and "excise policy".
2. **Crucial insight**: The human editor chose to link Sisodia's case because it is editorially relevant! That editorial choice is **golden ground-truth data** for evaluating our recommender!

### The Solution
Strip the text from the body, but **record the linked article ID** to use as ground truth!

```python
def _headline_index(titles, ids) -> dict:                                        # Line 60
    """first 4 words of every title -> (article id, number of words in the title)"""
    index = {}
    for article_id, title in zip(ids, titles):
        words = [w for w in map(_word, title.split()) if w]                      # Line 64
        if len(words) >= ALSO_READ_KEY_WORDS:                                    # Line 65
            index[tuple(words[:ALSO_READ_KEY_WORDS])] = (article_id, len(words)) # Line 66
    return index
```

### 🧠 Advanced Python Concept: Tuple Keys in Dictionaries
Notice Line 66:
```python
index[tuple(words[:4])] = (article_id, len(words))
```
- In Python, **lists cannot be dictionary keys** because lists are mutable (they are unhashable).
- A **tuple** is immutable! So `("manish", "sisodia", "granted", "bail")` is hashable and can be a dictionary key.
- By storing the first 4 normalized words as a tuple key, looking up whether an "ALSO READ" headline exists takes **$O(1)$ constant time**!

Now let's examine the stripper:

```python
def strip_also_read(body: str, own_id, headlines: dict):
    """Cut every 'ALSO READ <headline>' block out of the body. Returns (clean body, linked article ids)."""
    pieces, linked, cursor = [], [], 0                                           # Line 72
    for match in ALSO_READ_RE.finditer(body):                                    # Line 73
        if match.start() < cursor:
            continue                                    # already inside a removed block
        pieces.append(body[cursor:match.start()])                                # Line 76
        segment = body[match.end(): match.end() + 400]                           # Line 77
        tokens = [(w, m.end()) for m in re.finditer(r"\S+", segment) if (w := _word(m.group()))] # Line 78
        key = tuple(w for w, _ in tokens[:ALSO_READ_KEY_WORDS])                  # Line 79
        article_id, n_words = headlines.get(key, (None, ALSO_READ_FALLBACK_WORDS)) # Line 80
        if article_id is not None and article_id != own_id:
            linked.append(article_id)                                            # Line 82
        cut = tokens[min(n_words, len(tokens)) - 1][1] if tokens else 0          # Line 83
        cursor = match.end() + cut                                               # Line 84
    pieces.append(body[cursor:])                                                 # Line 85
    return re.sub(r"\s+", " ", " ".join(pieces)).strip(), linked                # Line 86
```

### 🧠 The Walrus Operator `:=` (Line 78)
```python
tokens = [(w, m.end()) for m in re.finditer(r"\S+", segment) if (w := _word(m.group()))]
```
The **walrus operator `:=`** (assignment expression, Python 3.8+) allows you to assign a value to a variable inside an expression:
- It calls `_word(m.group())`, assigns the result to `w`, and tests if `w` is truthy (non-empty) in one step!
- Without `:=`, you'd need a separate loop or redundant function calls.

### How the Cursor-Cutting Works Visually:
```
Body: [ Good text before... ] [ ALSO READ: Headline of other article ] [ Good text after... ]
      ▲                     ▲                                        ▲
    cursor             match.start()                               cursor advances here!
      └───── appended ──────┘                                        └───── next chunk ────
```

---

## Section D: Corpus Boilerplate Excision (Lines 89-114)

```python
def _find_boilerplate(bodies) -> set:                                            # Line 89
    """Sentences that appear in many different articles are not about any specific topic."""
    counts = Counter()                                                           # Line 91
    for body in bodies:
        counts.update({s.strip() for s in SENTENCE_SPLIT.split(body) if len(s.strip()) > 25}) # Line 93
    return {s for s, c in counts.items() if c >= config.BOILERPLATE_MIN_DOCS}    # Line 94


def _strip_boilerplate(body: str, boilerplate: set) -> str:                      # Line 97
    return " ".join(s for s in SENTENCE_SPLIT.split(body) if s.strip() not in boilerplate).strip()
```

### Why Do We Need This?
News sites frequently repeat identical sentences:
- *"For more news, download the Indian Express App."*
- *"Click here to join our WhatsApp channel."*
- *"Subscribe to our newsletter for morning briefings."*

If 100 articles have *"Click here to join our WhatsApp channel"*, a search for "WhatsApp" would falsely recommend political, sports, and tech articles together just because of that boilerplate sentence!

- `counts = Counter()` counts sentence occurrences across documents (using a `set` per document so multiple mentions in one article only count once).
- If a sentence >25 characters appears in $\ge$ `BOILERPLATE_MIN_DOCS` (e.g. 4) distinct articles, it's marked as boilerplate and deleted from all articles!

---

## Section E: Tokenizer Management & Singletons (Lines 116-170)

```python
_EMBED_TOKENIZER = None                                                          # Line 120
_RERANK_TOKENIZER = None                                                         # Line 121


def get_embedder_tokenizer(model_name: Optional[str] = None):                   # Line 124
    """Load and cache the HuggingFace tokenizer for the embedding model (BERT family)."""
    global _EMBED_TOKENIZER
    if _EMBED_TOKENIZER is None:
        model_name = model_name or getattr(config, "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        revision = getattr(config, "EMBEDDING_REVISION", None)
        try:
            from transformers import AutoTokenizer                               # Line 131 (Lazy import!)
            _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
            _EMBED_TOKENIZER.model_max_length = 100000                          # Line 133
        except Exception as e:
            print(f"[DataLoader] Notice: Could not load embedder tokenizer ({e}).")
            _EMBED_TOKENIZER = None
    return _EMBED_TOKENIZER
```

### 🧠 Concepts to Master

#### 1. The Singleton Pattern with Module-Level Caching
Loading a HuggingFace tokenizer takes 200–500ms and loads a vocabulary file into RAM.
You never want to call `AutoTokenizer.from_pretrained(...)` every time an article is processed.
By holding `_EMBED_TOKENIZER` as a global variable and checking `if _EMBED_TOKENIZER is None:`, it is loaded **only once** for the entire lifespan of the process.

#### 2. Lazy Import (`from transformers import AutoTokenizer` on Line 131)
Notice this import is **inside the function**, not at the top of the file!
- Why? In lightweight environments (like running unit tests or offline utility scripts without heavy ML packages installed), top-level imports of `transformers` or `torch` would crash the program with `ModuleNotFoundError`.
- By importing only inside `get_embedder_tokenizer()`, if transformers isn't needed or available, the rest of `data_loader.py` still runs cleanly.

#### 3. Silencing Warning Noise (Line 133)
```python
_EMBED_TOKENIZER.model_max_length = 100000
```
By default, HuggingFace tokenizers print a loud red warning whenever you pass text longer than 512 tokens:
`"Token indices sequence length is longer than the specified maximum sequence length for this model..."`
Since our whole purpose in the next section is to calculate token counts of full articles to budget them down, setting `model_max_length` high silences this spam.

---

## Section F: Token-Budgeted Text Composition (Lines 172-308)

This is one of the most sophisticated algorithms in the system.

### The Problem: Why not just `text[:512]`?
1. Character slicing (`text[:500]`) is unpredictable — 500 characters might be 80 tokens or 150 tokens depending on vocabulary and punctuation.
2. If you let the neural network truncate at 256 tokens automatically, it will chop the text **in the middle of a sentence or word**:
   `"...the government decided to post-"`
   Transformers perform significantly worse when context ends with incomplete syntax.
3. **The Goal**: Keep the **title 100% whole**, keep the **description whole**, and pack as many **complete body sentences** as will fit under the token cap (e.g., 254 tokens). Stop at the last whole sentence!

```python
def compose_sentence_budgeted_text(
    title: str,
    description: str,
    body: str,
    budget: int,
    tokenizer=None,
    query_prefix: str = "",
) -> Tuple[str, int, bool]:
```

Let's trace its logic step-by-step:

### Case 1: What if Title + Description ALREADY exceeds the budget?
Look at Lines 231-248:
```python
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
```

### 🧠 Algorithmic Elegance: Binary Search on Token Budget
If a video description is 500 words long, how many words fit under our token budget?
- A naive linear scan would try 1 word, 2 words, 3 words... calling `tokenizer.encode()` 500 times. That's way too slow!
- **Binary Search** (`while low <= high: mid = (low + high) // 2`):
  - Checks 250 words. Too big? Check 125 words. Too small? Check 187 words.
  - Finds the exact maximum word count in just **8 steps** ($\log_2 500 \approx 9$) instead of 500 steps!

### Case 2: Title + Description fits, now greedily pack body sentences
Look at Lines 250-264:
```python
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
            break                                                # STOP! Do not break the sentence!

    return cur, cur_len, is_trunc
```

Notice: The moment a sentence overflows `effective_budget`, the loop immediately **breaks**. The partial sentence is discarded, ensuring the final output ends on a clean sentence period!

---

## Section G: Flexible Column Resolution & Robust Coercion (Lines 310-344)

In production data engineering, CSV sources change without warning:
- One batch calls the ID column `articleId`.
- Another calls it `id`.
- Another calls it `article_id`.
- Another calls it `ID`.

```python
_ID_CANDIDATES = ["articleId", "id", "article_id", "ID"]
_DATE_CANDIDATES = ["postPublishDate", "date", "published_at", "publishDate"]
_TITLE_CANDIDATES = ["articleTitle", "title", "headline", "articleName"]


def _resolve_column(raw: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first column name from *candidates* that exists in *raw*."""
    for c in candidates:
        if c in raw.columns:
            return c
    return None


def _col(raw: pd.DataFrame, candidates: list, default="") -> pd.Series:
    """Return the first column from *candidates* that exists in *raw*, else a constant Series."""
    for c in candidates:
        if c in raw.columns:
            return raw[c]
    return pd.Series([default] * len(raw), index=raw.index)
```

### 💡 Why this is defensive engineering at its best:
Instead of writing fragile code like `raw['articleTitle']` (which throws a `KeyError` if the CSV header is named `headline`), `_col(raw, _TITLE_CANDIDATES)` checks every candidate in order. If none exist, it gracefully returns a default empty series of identical length. The pipeline never crashes on schema variations!

Look at `_as_bool` (Line 328):
```python
def _as_bool(val) -> bool:
    """Robust boolean parsing for diverse CSV formats."""
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "y", "t"}
```
CSVs don't have boolean types — they have strings. Some CMS exports write `"1"`, some `"True"`, some `"YES"`, some `"y"`. `_as_bool` handles all of them safely.

---

## Section H: Deduplication — ONE Policy for the Whole Project (Lines 377-444)

```python
def _deduplicate(
    raw: pd.DataFrame,
    id_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
```

### The Business Rule
When the same news article is updated (e.g. developing breaking news):
1. **Latest `postPublishDate` wins.**
2. If dates are tied or missing, the **last row in the file** wins.
3. Also track **content conflicts**: Did two rows share the same ID but have different headlines? (Indicates upstream data corruption).

Let's study the Pandas sorting trick used on Lines 421-439:

```python
    raw = raw.copy()
    raw["_dedup_order"] = range(len(raw))                                        # 1. Record original line number

    if date_col:
        raw["_dedup_date"] = pd.to_datetime(raw[date_col], utc=True, errors="coerce") # 2. Parse timestamps to UTC
        raw = raw.sort_values(
            ["_dedup_date", "_dedup_order"],                                    # 3. Sort by: newest date, then highest line #
            ascending=[False, False],
            na_position="last",
        )
        raw = raw.drop(columns=["_dedup_date"])
    else:
        raw = raw.sort_values("_dedup_order", ascending=False)

    raw = raw.drop(columns=["_dedup_order"])
    raw = raw.drop_duplicates(subset=id_col, keep="first")                       # 4. First one is the winner!
    raw = raw.sort_index()                                                       # 5. Restore original file order!
    raw = raw.reset_index(drop=True)
```

### 🧠 Why this sequence is brilliant:
1. It adds a temporary helper column `_dedup_order = range(len(raw))`.
2. It sorts with `ascending=[False, False]`, putting the most recent date at the top. For date ties, the highest `_dedup_order` (the one later in the file) floats to the top.
3. It calls `drop_duplicates(keep="first")` — which keeps that top winner!
4. It calls `raw.sort_index()` to restore the original reading order of articles!

---

## Section I: `clean_dataframe()` — The Master Orchestrator (Lines 450-618)

This function coordinates the entire pipeline. Let's see the transformation:

```
Raw CSV DataFrame (Messy columns, duplicate IDs, curly quotes, ads, unbounded texts)
                                   │
                                   ▼
         1. Resolve ID column & Deduplicate (Latest date wins)
                                   │
                                   ▼
         2. Clean titles, descriptions, and normalize Unicode
                                   │
                                   ▼
         3. Strip "ALSO READ" blocks & extract editor links as ground truth
                                   │
                                   ▼
         4. Identify & remove repeated boilerplate sentences
                                   │
                                   ▼
         5. Standardize metadata (section, date, pageViews, is_live)
                                   │
                                   ▼
         6. Generate 3 ML Text Representations:
            - text_embed   (BGE embedding model, cap 254 tokens)
            - text_bm25    (title x3 for entity boost + desc + body)
            - text_rerank  (RoBERTa cross-encoder, cap 254 tokens)
                                   │
                                   ▼
Clean DataFrame (Standardized snake_case columns ready for PostgreSQL)
```

### 🧠 The 3 Text Representations: Why 3 Different Texts for 1 Article?

Look at Lines 593-600:
```python
clean_df["text_embed"] = embed_texts
clean_df["text_bm25"] = [
    f"{t}. {t}. {t}. {d}. {b}"
    for t, d, b in zip(clean_df.title, clean_df.description, clean_df.body)
]
clean_df["text_rerank"] = rerank_texts
```

| Field | Tokenizer / Algorithm | Recipe | Why this design? |
|---|---|---|---|
| **`text_embed`** | BAAI/bge-base-en-v1.5 (BERT) | Title + Desc + Body sentences up to 254 tokens | Dense neural embeddings have fixed 768-d vectors. Keeping strictly within budget prevents truncation errors. |
| **`text_bm25`** | Okapi BM25 (Bag-of-Words / Inverted Index) | `Title. Title. Title. Desc. Body` | BM25 counts word frequencies. Repeating the title **3 times** gives 3x higher term frequency to entities in the headline (e.g. "Rohit Sharma", "Election Commission")! |
| **`text_rerank`** | BAAI/bge-reranker-base (RoBERTa) | Title + Desc + Body sentences up to 254 tokens | Cross-encoders evaluate pairs: `(Query Article, Candidate Article)`. The model's total max length is 512 tokens. So each side gets **exactly 254 tokens** ($254 + 254 + 4 \text{ special tokens} = 512$). |

---

## Section J: `load_articles()` vs `clean_dataframe()` (Lines 625-650)

```python
def load_articles(paths=None) -> pd.DataFrame:
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
```

### 💡 Architectural Insight: DB Schema vs API Schema
Notice the distinction:
- **`clean_dataframe()`** emits `snake_case` columns (`page_views`, `unique_users`) and serializes `linked_ids` to a JSON string `"[101, 102]"`. This matches PostgreSQL column definitions in [`db/models.py`](file:///d:/indian%20express/Recommendation%20system/news%20recommender%20base/db/models.py).
- **`load_articles()`** adapts it for in-memory Python consumers (`evaluate.py`, `Recommender`), converting columns to `camelCase` (`pageViews`) and parsing `linked_ids` into real Python `list` objects.

---

## 🗺️ Complete Line Map of `src/data_loader.py`

| Lines | Function / Component | Core Purpose |
|---|---|---|
| **1-38** | Imports & Regex Constants | Compiled regexes for sentence splitting, emails, URLs, quotes table |
| **40-58** | `normalise_text`, `remove_links`, `_word` | NFKC normalization, quote translation, stripping URLs/emails |
| **60-68** | `_headline_index` | Tuple-indexed prefix lookup of article titles |
| **70-87** | `strip_also_read` | Excision of "ALSO READ" ads & ground-truth editor link extraction |
| **89-99** | `_find_boilerplate`, `_strip_boilerplate` | Cross-document frequency counting to strip shared site footers |
| **101-114**| `_subsection`, `_short_description` | JSON category parsing and clean word-boundary truncation |
| **120-159**| `get_embedder_tokenizer`, `get_reranker_tokenizer` | Singleton caching of HuggingFace tokenizers with lazy import |
| **161-265**| `compose_sentence_budgeted_text` | Binary search + sentence-by-sentence packing to exact token budget |
| **267-308**| `build_sentence_budgeted_texts` | Batch vectorized processing of budgeted texts |
| **315-343**| `_col`, `_resolve_column`, `_as_bool` | Flexible schema resolution across varying CSV headers |
| **349-374**| `load_raw_csv` | Multi-file disk loading with concatenated DataFrame output |
| **380-444**| `_deduplicate` | Unified deduplication engine (latest date wins, ties to file order) |
| **450-619**| `clean_dataframe` | Single authoritative cleaning pipeline for the entire project |
| **625-650**| `load_articles` | High-level facade with camelCase adaptation for Python evaluation |

---

## 🧠 Python Concepts Mastered in This Chapter

1. **Lookaround Assertions**: `(?<=[.!?])\s+` splits strings without consuming the delimiter punctuation.
2. **Unicode Normalization (NFKC)**: Unifies ligatures (`ﬁ` → `fi`), full-width characters, and composite accents.
3. **`str.translate()` Tables**: O(1) character-to-character replacement using integer code-point dictionaries (`ord(c)`).
4. **Tuple Keys in Dictionaries**: Using immutable tuples as keys to enable multi-word index lookups in $O(1)$ time.
5. **The Walrus Operator (`:=`)**: Assigning values inside comprehensions (`if (w := _word(m.group()))`).
6. **Binary Search over Functions**: Finding the maximum word boundary that fits under a token budget in $O(\log N)$ time.
7. **Singleton Pattern via Module Globals**: Lazy loading and caching heavy HuggingFace models.
8. **Defensive Schema Resolution**: Checking a candidate list of column names instead of assuming hardcoded headers.
9. **Two-Key Sorting in Pandas**: Sorting by date descending then index descending to resolve duplicate ties cleanly.

---

## ✏️ Chapter 2 Self-Test Quiz

Try answering these questions before checking the answers.

### Question 1
Why does `SENTENCE_SPLIT` use `(?<=[.!?])\s+` instead of `[.!?]\s+`?
<details>
<summary>👉 Click to see the answer</summary>
Because `[.!?]\s+` would match and delete the punctuation mark when splitting. The positive lookbehind `(?<=[.!?])` asserts that punctuation comes right before the space without including the punctuation mark in the split delimiter, preserving sentence-ending punctuation.
</details>

### Question 2
Why can't we use a `list` as a dictionary key in `_headline_index()`, and what does the code use instead?
<details>
<summary>👉 Click to see the answer</summary>
Lists are mutable in Python and therefore unhashable. Dictionaries require hashable keys. The code converts the list of words into an immutable <code>tuple</code>: <code>tuple(words[:4])</code>.
</details>

### Question 3
In `normalise_text()`, why is `text.translate(QUOTES)` preferred over multiple `.replace()` calls?
<details>
<summary>👉 Click to see the answer</summary>
<code>str.translate()</code> is implemented in C and performs a single pass over the string using a lookup table of Unicode code points. Chaining multiple <code>.replace()</code> calls scans the entire string repeatedly for every quote character.
</details>

### Question 4
What happens to an "ALSO READ <headline>" block when `strip_also_read()` finds it?
<details>
<summary>👉 Click to see the answer</summary>
The text of the "ALSO READ" block is cut out of the article body to avoid topic contamination, and if the headline matches an article in the corpus, that article's ID is saved in <code>linked_ids</code> as an editorial ground-truth recommendation.
</details>

### Question 5
Why does `compose_sentence_budgeted_text()` use binary search when trimming descriptions?
<details>
<summary>👉 Click to see the answer</summary>
Tokenizing text with HuggingFace models is computationally expensive. If a description has 500 words, linear search would call <code>tokenizer.encode()</code> hundreds of times. Binary search finds the exact word boundary in ~8 calls ($\log_2 500$).
</details>

### Question 6
Why does `text_bm25` repeat the article title three times: `f"{t}. {t}. {t}. {d}. {b}"`?
<details>
<summary>👉 Click to see the answer</summary>
BM25 ranks articles based on Term Frequency (TF). Repeating the title 3 times triples the term frequency of words in the headline, ensuring that articles sharing exact headline entities (names, places) get much higher keyword matching scores.
</details>

### Question 7
In `_deduplicate()`, what is the tie-breaker if two duplicate rows have the exact same `postPublishDate`?
<details>
<summary>👉 Click to see the answer</summary>
The row that appears later in the CSV file order wins (preserving the classic "keep last" behavior).
</details>

### Question 8
Why is `from transformers import AutoTokenizer` imported inside `get_embedder_tokenizer()` instead of at line 20?
<details>
<summary>👉 Click to see the answer</summary>
It is a lazy import. It prevents crashes in environments where heavy ML packages like <code>transformers</code> or <code>torch</code> are not installed, allowing lightweight data utilities to still use the rest of the file.
</details>

### Question 9
Why are both `text_embed` and `text_rerank` budgeted to 254 tokens instead of 256?
<details>
<summary>👉 Click to see the answer</summary>
Transformer models (like BERT and RoBERTa) require 2 special tokens: <code>[CLS]</code> at the start and <code>[SEP]</code> at the end. Budgeting text to 254 tokens leaves exact room for those 2 special tokens under the 256 token limit.
</details>

### Question 10
What is the difference between the return values of `clean_dataframe()` and `load_articles()`?
<details>
<summary>👉 Click to see the answer</summary>
<code>clean_dataframe()</code> returns database-ready snake_case columns (<code>page_views</code>, <code>unique_users</code>) and stringified JSON for <code>linked_ids</code>. <code>load_articles()</code> returns camelCase columns (<code>pageViews</code>, <code>uniqueUsers</code>) and parses <code>linked_ids</code> into Python lists for in-memory evaluators.
</details>

---

## 🚀 Next Step

Whenever you have finished reading Chapter 2 and testing yourself with the quiz, say **"Start Chapter 3"**!
We will dive into [`src/retrieval.py`](file:///d:/indian%20express/Recommendation%20system/news%20recommender%20base/src/retrieval.py) — where you will learn how **Okapi BM25**, **Fuzzy Indian Entity Matching**, **Dense BGE Embeddings**, and **Reciprocal Rank Fusion (RRF)** work under the hood!
