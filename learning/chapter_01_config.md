# 📖 Chapter 1: Configuration as a Module — Deep Dive

> **File**: [`config.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/config.py) (314 lines)
> **Difficulty**: ⭐ Easy → ⭐⭐ Medium (the functions at the end get mathy)
> **Time**: ~45-60 minutes for thorough study

---

## Section A: The Foundation (Lines 1-9)

```python
"""                                              # Line 1
Central settings. Every number you may want to tune lives here,
so you never have to hunt through the code.
"""                                              # Line 4
import os                                        # Line 5
from pathlib import Path                         # Line 6
                                                 # Line 7
ROOT = Path(__file__).parent                     # Line 8
DATA_DIR = ROOT / "data"                         # Line 9
```

### What's happening here?

**Line 1-4: Module docstring** — The triple-quoted string at the very top of a file is called a **docstring**. It describes what this file is for. Python stores it in `config.__doc__`.

**Line 5: `import os`** — The `os` module gives you access to operating system features: environment variables, file paths, CPU count, etc.

**Line 6: `from pathlib import Path`** — Instead of importing the whole `pathlib` module, this imports just the `Path` class directly. Now we can write `Path(...)` instead of `pathlib.Path(...)`.

**Line 8: `ROOT = Path(__file__).parent`** — This is a 3-step operation:

```python
__file__                    # "D:\...\news recommender base\config.py"
                            # __file__ is a magic variable Python auto-sets to
                            # the current file's path. Every .py file has this.

Path(__file__)              # Converts the string to a Path object
                            # Now we can call methods on it

Path(__file__).parent       # "D:\...\news recommender base"
                            # .parent = the directory containing this file
```

> 💡 **Why `Path` and not a plain string?** Because `Path` objects let you do this:

**Line 9: `DATA_DIR = ROOT / "data"`**

```python
# The / operator is OVERLOADED on Path objects!
ROOT / "data"    # = Path("D:\...\news recommender base\data")

# Without pathlib, you'd write:
os.path.join(str(ROOT), "data")   # messy, error-prone
```

> **Advanced Python Concept: Operator Overloading** 🧠
>
> When you write `a / b` in Python, Python actually calls `a.__truediv__(b)`. The `Path` class defines its own `__truediv__` method to join paths. This is called **operator overloading** — a class can redefine what `+`, `-`, `/`, `==`, etc. mean for its objects.
>
> ```python
> # Inside pathlib (simplified):
> class Path:
>     def __truediv__(self, other):
>         return Path(os.path.join(str(self), str(other)))
> ```

### 🔑 Key Takeaway
These 2 lines establish WHERE the project lives on disk. Every other file reference in the project is built from `ROOT` or `DATA_DIR`.

---

## Section B: Finding CSV Data Files (Lines 11-26)

```python
_env_csvs = os.getenv("DATA_CSV", "")                                           # Line 14
if _env_csvs:                                                                     # Line 15
    DATA_PATHS = [DATA_DIR / name.strip() for name in _env_csvs.split(",") if name.strip()]  # Line 16
else:                                                                             # Line 17
    _default_files = ["articles.csv", "articles_2.csv", "articles_clean_train_80.csv"]       # Line 18
    DATA_PATHS = [DATA_DIR / f for f in _default_files if (DATA_DIR / f).exists()]           # Line 19
    if not DATA_PATHS:                                                            # Line 20
        DATA_PATHS = [p for p in DATA_DIR.glob("*.csv") if "uploads" not in str(p)]          # Line 21
    if not DATA_PATHS:                                                            # Line 22
        DATA_PATHS = [DATA_DIR / "articles.csv"]                                  # Line 23

DATA_PATH = DATA_PATHS[0]                                                         # Line 25
CACHE_DIR = ROOT / "data" / "cache"                                               # Line 26
```

### Line-by-line:

**Line 14** — `os.getenv("DATA_CSV", "")`:
- Check if user set a `DATA_CSV` environment variable
- If not found, return `""` (empty string)
- The `_` prefix in `_env_csvs` is a Python convention: "this is a private/temporary variable, don't use it from outside this module"

**Line 15** — `if _env_csvs:`:
- An empty string `""` is **falsy** in Python. So `if "":` = `if False:`
- A non-empty string `"articles.csv"` is **truthy**. So `if "articles.csv":` = `if True:`

> **Python Truthiness** 🧠
> These are **falsy**: `None`, `0`, `""`, `[]`, `{}`, `False`, `0.0`
> Everything else is **truthy**: `"hello"`, `[1,2]`, `42`, `True`

**Line 16** — A list comprehension with TWO things happening:

```python
DATA_PATHS = [DATA_DIR / name.strip() for name in _env_csvs.split(",") if name.strip()]
```

Let's break it down with an example. If `DATA_CSV = "articles.csv, articles_2.csv, "`:

```python
_env_csvs.split(",")              # ["articles.csv", " articles_2.csv", " "]
                                   # Note the spaces!

# The list comprehension:
for name in ["articles.csv", " articles_2.csv", " "]:
    name.strip()                   # Remove whitespace: "articles.csv", "articles_2.csv", ""
    if name.strip():               # Filter: keep "articles.csv" ✓, "articles_2.csv" ✓, skip "" ✗
        DATA_DIR / name.strip()    # Build full path

# Result: [Path(".../data/articles.csv"), Path(".../data/articles_2.csv")]
```

**Lines 18-23** — A **fallback chain** (defensive programming):

```
Strategy 1: Check if the 3 known default files exist on disk
  ↓ (if none exist)
Strategy 2: Find ALL .csv files in the data/ folder (except uploads/)
  ↓ (if no .csv files at all)
Strategy 3: Just hardcode articles.csv (it might be created later)
```

> **Design Pattern: Graceful Degradation** 🧠
> Instead of crashing when data is missing, the code tries progressively simpler strategies. This is extremely common in production code.

**Line 25** — `DATA_PATH = DATA_PATHS[0]`: For backward compatibility, some old code might use `config.DATA_PATH` (singular) instead of `config.DATA_PATHS` (plural).

**Line 26** — `CACHE_DIR = ROOT / "data" / "cache"`: Chaining `/` operators! Each `/` joins one more path segment.

---

## Section C: Environment & Models (Lines 28-54)

```python
try:                                             # Line 29
    from dotenv import load_dotenv               # Line 30
    load_dotenv()                                # Line 31
except ImportError:                              # Line 32
    pass                                         # Line 33
```

### What's happening here?

**Lines 29-33** — The `python-dotenv` library reads `.env` files and puts their values into `os.environ`. But what if someone hasn't installed `python-dotenv`?

```python
try:
    from dotenv import load_dotenv    # Try to import it
    load_dotenv()                      # If it exists, use it
except ImportError:                    # If the package isn't installed...
    pass                               # ...just skip it silently
```

> **Pattern: Optional Dependencies** 🧠
> `try/except ImportError` is the standard way to make a dependency optional. The project will work without `python-dotenv` — you'd just need to set environment variables manually instead of using a `.env` file.

```python
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")    # Line 36
```

**Line 36** — `setdefault` sets a variable ONLY if it doesn't already exist. It's like saying "use `1` unless the user already chose something else." This suppresses a noisy HuggingFace warning.

```python
USE_NEURAL_MODELS = os.getenv("NEWSREC_NEURAL", "1") == "1"      # Line 39
```

**Line 39** — This is clever. Let's trace it:

```python
# If NEWSREC_NEURAL is not set:
os.getenv("NEWSREC_NEURAL", "1")   # returns "1"
"1" == "1"                          # True
USE_NEURAL_MODELS = True            # Use AI models

# If NEWSREC_NEURAL is set to "0":
os.getenv("NEWSREC_NEURAL", "1")   # returns "0"
"0" == "1"                          # False
USE_NEURAL_MODELS = False           # Skip AI models (use TF-IDF instead)
```

> ⚠️ Environment variables are ALWAYS **strings**. `os.getenv` returns `"0"` not `0`. That's why we compare with `== "1"` not `== 1`.

```python
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"                                         # Line 40
EMBEDDING_REVISION = os.getenv("NEWSREC_EMBED_REVISION", "a5beb1e3e68b9...")       # Line 41
```

**Lines 40-43** — Which AI models to use:
- `EMBEDDING_MODEL` = the model that turns text → 768-dimension vectors
- `RERANKER_MODEL` = the model that reads two texts side-by-side and judges similarity
- The long hex strings (revisions) are **Git commit SHAs** on HuggingFace — they pin the exact model version so results are reproducible

```python
EMBED_MAX_SEQ_LEN = 256    # Line 44
RERANK_MAX_LEN = 512       # Line 45
```

**Lines 44-46** — Token limits:
- The embedding model can process **256 tokens** max (roughly 200 words)
- The reranker can process **512 tokens** max (but that's for BOTH articles combined)
- In article-to-article mode, each side gets 256 tokens

```python
REMOTE_MODEL_URL = os.getenv("REMOTE_MODEL_URL", "").strip().rstrip("/")    # Line 51
```

**Line 51** — Method chaining:
```python
os.getenv(...)           # Get the URL string (or empty string)
    .strip()             # Remove whitespace from both ends
    .rstrip("/")         # Remove trailing slashes: "https://example.com/" → "https://example.com"
```

> **Method Chaining** 🧠: Each string method returns a NEW string, so you can chain `.strip().rstrip("/")`. This only works because each method returns the same type (string).

---

## Section D: The Tunable Numbers (Lines 56-138)

This section is where the "machine learning engineering" lives. Every number here was either **hand-picked** or **measured** through experiments.

### Pipeline Parameters (Lines 88-92)

```python
N_RETRIEVE = 50       # candidates taken from each search (BM25, semantic)
N_CANDIDATES = 20     # candidates kept after fusion, sent to the reranker
RRF_K = 60           # standard Reciprocal Rank Fusion constant
TOP_K = 10           # default number of recommendations
```

Think of it as a **funnel**:
```
   2446 articles in the database
          ↓ BM25 search
     50 best keyword matches          ← N_RETRIEVE
          ↓ Dense search
     50 best semantic matches         ← N_RETRIEVE
          ↓ RRF fusion
     20 best combined candidates      ← N_CANDIDATES
          ↓ Cross-encoder reranking
     10 final recommendations         ← TOP_K
```

### Blend Weights (Lines 94-100)

```python
WEIGHT_RERANK = 0.70    # cross-encoder (most accurate, 70% of final score)
WEIGHT_BM25 = 0.20      # keyword match (20% of final score)
WEIGHT_DENSE = 0.10     # semantic embedding (10% of final score)
```

Final score = `0.70 × reranker_score + 0.20 × bm25_score + 0.10 × dense_score`

> **Why not 100% reranker?** Because the reranker can be fooled by style similarity (two articles written the same way but about different topics). BM25 anchors the score to actual word overlap.

### Boost Multipliers (Lines 119-138)

```python
SECTION_BOOST_MULT = 1.072      # Same section? Multiply score by 1.072 (7.2% boost)
SUBSECTION_BOOST_MULT = 1.09    # Same subsection? Multiply by 1.09 (9% boost)
POPULARITY_BOOST_MULT = 1.044   # Popular article? Up to 4.4% boost
RECENCY_MAX_BOOST_MULT = 1.104  # Very recent? Up to 10.4% boost
```

These are **multiplicative**: if an article gets ALL boosts:
```
score × 1.072 × 1.09 × 1.044 × 1.104 = score × 1.346
```
A 34.6% boost in the best case. But notice how **conservative** these are — the biggest single boost is only 10.4%. The reranker score dominates.

### Half-Life Decay (Lines 127-135)

```python
SECTION_HALF_LIVES = {
    "live_blog": 0.25,      # 6 hours — live blogs go stale FAST
    "sports": 1.0,          # 1 day — yesterday's match is old news
    "entertainment": 2.0,   # 2 days — movies stay relevant longer
    "business": 3.0,        # 3 days — market analysis ages slowly
    "brand-story": 4.0,     # 4 days — brand content is evergreen-ish
    "default": 2.0
}
```

> **Half-life** is from radioactive decay. After one half-life, the boost drops by 50%. A sports article from 1 day ago gets 50% of the max recency boost. After 2 days, 25%. After 3 days, 12.5%. And so on.

---

## Section E: The Functions (Lines 141-183) — Here's Where Math Lives

### `_has_date()` — The NaN Self-Inequality Trick (Line 141-148)

```python
def _has_date(value) -> bool:
    if value is None:
        return False
    try:
        return not value != value    # ← THE CLEVER PART
    except Exception:
        return False
```

**Line 146 is tricky**: `value != value`

In math, everything equals itself: `5 == 5` is `True`. But **NaN** (Not a Number) and **NaT** (Not a Time) are special — they are NOT equal to themselves:

```python
float("nan") != float("nan")    # True!  (NaN ≠ NaN)
pd.NaT != pd.NaT                # True!  (NaT ≠ NaT)
5 != 5                           # False  (5 = 5, obviously)
```

So the logic is:
```python
value != value                   # True if NaN/NaT, False for normal values
not (value != value)             # False if NaN/NaT, True for normal values
return not value != value        # Returns True if value is a real date
```

> **Why not just `if pd.isna(value)`?** Because that requires importing pandas. This function lives in `config.py`, which is imported by EVERYTHING. Importing pandas in config would create a circular import issue and slow down startup.

### `recency_boost()` — Exponential Decay (Lines 151-170)

```python
def recency_boost(cand_date, ref_date, section: str, is_live: bool = False) -> float:
    if not _has_date(cand_date) or not _has_date(ref_date):
        return 1.0                    # No dates available → no boost (multiply by 1 = no change)
    
    days_old = max(0.0, (ref_date - cand_date).total_seconds() / 86400.0)
    #                    ↑ datetime subtraction gives timedelta
    #                                        ↑ convert seconds to days (86400 = 24×60×60)
    #          ↑ never negative (future articles get 0 days old)
    
    sec = "live_blog" if is_live else str(section or "").lower()
    half_life = SECTION_HALF_LIVES.get(sec, SECTION_HALF_LIVES.get("default", 2.0))
    #                              ↑ Look up half-life for this section
    #                                 If not found, try "default"
    #                                    If that's missing too, use 2.0
    
    return 1.0 + (RECENCY_MAX_BOOST_MULT - 1.0) * (0.5 ** (days_old / half_life))
    #      ↑ base (no boost)
    #            ↑ max possible boost amount (0.104)
    #                                          ↑ exponential decay factor
```

Let's trace with a **sports article that's 1 day old**:

```python
days_old = 1.0
half_life = 1.0  (sports section)

0.5 ** (1.0 / 1.0) = 0.5 ** 1 = 0.5     # 50% of max boost remains

1.0 + (1.104 - 1.0) * 0.5
= 1.0 + 0.104 * 0.5
= 1.0 + 0.052
= 1.052                                   # 5.2% boost
```

And a **sports article that's 3 days old**:
```python
0.5 ** (3.0 / 1.0) = 0.5 ** 3 = 0.125   # Only 12.5% remains

1.0 + 0.104 * 0.125 = 1.013              # Barely 1.3% boost (stale news)
```

> **Exponential Decay** 🧠: `0.5 ** (t / half_life)` is the classic half-life formula. It starts at 1.0 (fresh) and drops by 50% every `half_life` period. This is the same math used in radioactive decay, population modeling, and... news freshness!

---

## Section F: Relevance Admission — The Quality Gatekeeper (Lines 185-266)

This is the system that prevents BAD recommendations from appearing:

### The 4 Techniques:

```python
RELEVANCE_FLOOR = 0.40     # Technique 1: Hard floor — probability must be ≥ 40%
ELBOW_RATIO = 0.45          # Technique 2: Cliff drop — if score drops by 55%+ vs previous
ELBOW_MIN_KEEP = 4          # Safety: never apply cliff-drop below 4 survivors
LEXICAL_GATE = True         # Technique 3: Must share at least 1 keyword
ENTITY_GATE = True          # Technique 3b: Or share at least 1 named entity
RELATED_MIN_HEALTHY = 4     # Technique 4: Below 4 results, show trending instead
```

### `admission_cut()` — The Filtering Function (Lines 232-266)

```python
def admission_cut(prob_scores, entities=None) -> int:
    n = len(prob_scores)       # How many candidates we have
    if n == 0:
        return 0
    
    # Technique 1: Hard floor — scan until we find a score below 0.40
    keep = n                    # Start by keeping everything
    for i in range(n):
        if prob_scores[i] < RELEVANCE_FLOOR:    # Score < 0.40?
            keep = i                             # Cut everything from here down
            break
    
    # Entity gate: if we have enough survivors, cut at the first entity-less candidate
    if entities is not None and keep > ELBOW_MIN_KEEP:
        for i in range(1, keep):
            if not entities[i] and entities[i - 1]:   # Previous had entities, this doesn't
                keep = i
                break
    
    # Technique 2: Cliff drop — did the score suddenly collapse?
    if keep > ELBOW_MIN_KEEP:
        for i in range(1, keep):
            prev, cur = prob_scores[i - 1], prob_scores[i]
            if prev > 1e-9 and cur < prev * ELBOW_RATIO:   # Score dropped by 55%+?
                keep = i
                break
    
    return max(1, keep) if n else 0    # Always keep at least 1
```

**Example walkthrough**:
```
Scores: [0.85, 0.72, 0.68, 0.45, 0.18, 0.12, 0.09]
                                          ↑ BELOW 0.40! → keep = 4

After floor: keep 4 → [0.85, 0.72, 0.68, 0.45]

Cliff check: 0.85→0.72 (85% of previous, OK)
             0.72→0.68 (94% of previous, OK)
             0.68→0.45 (66% of previous, OK... 0.45 > 0.68×0.45=0.31)

Final: serve 4 recommendations (not 10!)
```

> **Why is this important?** Without admission_cut, an article about "Indian classical music" would ALWAYS show 10 "recommendations", even if only 2 articles in the database are actually about classical music. Positions 3-10 would be random noise. This function says "actually, only 2 are relevant — stop there."

---

## Section G: Pipeline Version Fingerprint (Lines 269-311)

```python
def get_pipeline_version() -> str:
    import hashlib                # Imported inside the function (lazy import)
    tokens = [
        f"embed_model={EMBEDDING_MODEL}",       # "embed_model=BAAI/bge-base-en-v1.5"
        f"mmr_lambda={MMR_LAMBDA}",              # "mmr_lambda=0.91"
        f"w_rerank={WEIGHT_RERANK}",             # "w_rerank=0.70"
        ...                                       # every single tunable parameter
    ]
    return hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()[:16]
```

**Step by step**:
```python
"|".join(tokens)
# "embed_model=BAAI/bge-base-en-v1.5|mmr_lambda=0.91|w_rerank=0.70|..."

.encode("utf-8")
# b"embed_model=BAAI/bge-base-en-v1.5|mmr_lambda=0.91|..."  (bytes)

hashlib.sha256(...).hexdigest()
# "1aa8dc568ca0f822a3b7c9d1e4f5..."  (64 hex characters)

[:16]
# "1aa8dc568ca0f822"  (first 16 characters only)
```

> **Why a hash?** If you change ANY parameter (even `MMR_LAMBDA` from 0.91 to 0.90), the hash changes completely. The ingestion pipeline (Chapter 10) compares this hash with the one stored in the database. If they differ → "Settings changed, I need to recompute ALL 24,000+ recommendations from scratch."

> **Lazy Import** 🧠: `import hashlib` is INSIDE the function, not at the top of the file. This is called a **lazy import** — the module is only loaded when this function is actually called. Since `get_pipeline_version()` is only called during ingestion (not on every request), this saves a tiny bit of startup time. Use this pattern for modules that are expensive to import or rarely needed.

---

## 🧪 Chapter 1 Quiz — Test Yourself

Answer these before moving to Chapter 2. Try to answer WITHOUT looking at the code.

### Easy
1. What does `Path(__file__).parent` return?
2. What is the difference between `os.getenv("KEY")` and `os.getenv("KEY", "default")`?
3. Why does line 39 compare with `== "1"` instead of `== 1`?

### Medium
4. If `DATA_CSV="a.csv, , b.csv"`, what does `DATA_PATHS` become? (Trace line 16)
5. What does `EMBED_TOKEN_BUDGET` evaluate to and why?
6. If a sports article is 2 days old, what is its recency boost multiplier?

### Hard
7. Why does `_has_date()` use `not value != value` instead of `pd.isna(value)`?
8. Given scores `[0.90, 0.85, 0.80, 0.35, 0.30]`, what does `admission_cut()` return?
9. If you change `WEIGHT_BM25` from 0.20 to 0.25, what happens to all stored recommendations?
10. Why is `hashlib` imported inside `get_pipeline_version()` instead of at the top of the file?

<details>
<summary>Click to reveal answers</summary>

1. The directory containing `config.py` → `D:\...\news recommender base`
2. Without default: returns `None` if not set. With default: returns `"default"` if not set.
3. Because `os.getenv()` always returns **strings**. The env var value `0` comes back as `"0"`, not the integer `0`.
4. `[DATA_DIR / "a.csv", DATA_DIR / "b.csv"]` — the empty string between commas is filtered out by `if name.strip()`.
5. `254` = 256 (max sequence length) − 2 (special tokens [CLS] and [SEP]). The model needs 2 slots for its own bookkeeping tokens.
6. `0.5 ** (2.0/1.0) = 0.25` → `1.0 + 0.104 × 0.25 = 1.026` → **2.6% boost** (getting stale)
7. Because `config.py` is imported by everything. If it imported pandas, it would add ~500ms to startup and risk circular imports. The self-inequality trick (`NaN != NaN`) achieves the same check with zero dependencies.
8. `3` — Floor check: `0.35 < 0.40` at index 3, so `keep = 3`. Cliff check: only runs if `keep > 4`, which it isn't. Result: keep 3 items.
9. `get_pipeline_version()` includes `w_bm25=...` in its hash. Changing it changes the hash. On next ingestion, `db/pipeline.py` sees the version mismatch and **forces a full recompute of all 24,000+ recommendations**.
10. **Lazy import** — `hashlib` is only needed when this function runs (during ingestion, not on every HTTP request). Importing it at module level would load it even when it's never used.

</details>

---

> [!TIP]
> **You've completed Chapter 1!** 🎉
> 
> You now understand:
> - How Python modules execute at import time
> - `pathlib.Path` and the `/` operator overloading
> - `os.getenv()` and environment variables
> - List comprehensions with filters
> - The NaN self-inequality trick
> - Exponential decay math for recency
> - Quality gatekeeper pattern (admission_cut)
> - Hash-based version fingerprinting
>
> Say **"Start Chapter 2"** when you're ready to dive into `src/data_loader.py` — where we'll learn about regex, Unicode normalization, and how raw messy CSVs get transformed into clean, AI-ready text.
