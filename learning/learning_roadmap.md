# 🎓 NewsRec v2.0 — Complete Code Learning Roadmap

> **Your goal**: Understand every line of this project's backend so you can write it yourself.
> **Your current skills**: Python basics, LLM/LangChain/LangGraph knowledge.
> **Gaps to fill**: Advanced Python, OOP, FastAPI, SQLAlchemy, NLP/IR algorithms, system design patterns.

---

## How This Roadmap Works

This roadmap has **12 Chapters**, organized in the exact order you should study the code. Each chapter:

1. 📂 **Tells you which file(s) to read**
2. 🧠 **Teaches you the Python/tech concepts you need BEFORE reading the code**
3. 🔍 **Walks through the code line-by-line explaining what it does and WHY**
4. 🔗 **Shows how this piece connects to the rest of the system**
5. ✏️ **Gives you a mini-exercise to test your understanding**

```mermaid
graph LR
    A["Ch 1: Config"] --> B["Ch 2: Data Loader"]
    B --> C["Ch 3: Retrieval"]
    C --> D["Ch 4: Reranker"]
    D --> E["Ch 5: Recommender"]
    E --> F["Ch 6: Calibration & Evidence"]
    F --> G["Ch 7: DB Connection"]
    G --> H["Ch 8: DB Models (ORM)"]
    H --> I["Ch 9: DB Repository"]
    I --> J["Ch 10: Ingestion Pipeline"]
    J --> K["Ch 11: FastAPI Server"]
    K --> L["Ch 12: Editor Routes"]
```

---

## The Big Picture First

Before any code, understand **what this system does** in plain English:

```
An editor at Indian Express uploads a CSV of 500 news articles.
The system:
  1. Cleans the text (removes junk, deduplicates)
  2. Creates a "fingerprint" (embedding) of each article using AI
  3. For EVERY article, finds the 10 most similar articles using 3 different methods
  4. Stores everything in PostgreSQL
  5. When a reader opens an article, the system instantly serves those 10 recommendations
```

The 3 methods of finding similar articles:
- **BM25** — old-school keyword matching ("does article B contain the same words as article A?")
- **Dense/Semantic** — AI embedding matching ("does article B mean the same thing as article A?")
- **Cross-Encoder Reranking** — the most accurate AI check ("reading both articles side-by-side, are they truly related?")

---

## 📋 Chapter Index

| # | Chapter | File(s) | Python Concepts You'll Learn | Difficulty |
|---|---------|---------|------------------------------|------------|
| 1 | Configuration as a Module | `config.py` | Module-level code, `os.getenv`, `pathlib.Path`, type hints, f-strings, `hashlib` | ⭐ Easy |
| 2 | Data Loading & Cleaning | `src/data_loader.py` | DataFrames, regex, Unicode normalization, closures, list comprehensions | ⭐⭐ Medium |
| 3 | Search Engines (Retrieval) | `src/retrieval.py` | OOP classes, `__init__`, numpy, BM25 algorithm, embeddings, caching | ⭐⭐⭐ Hard |
| 4 | Cross-Encoder Reranking | `src/reranker.py` | Tokenizers, sigmoid math, cross-encoder concept, token budgets | ⭐⭐⭐ Hard |
| 5 | The Recommender (Orchestrator) | `src/recommender.py` | `@dataclass`, composition pattern, the full pipeline, MMR algorithm | ⭐⭐⭐ Hard |
| 6 | Calibration & Evidence | `src/calibration.py`, `src/evidence.py` | Platt scaling, JSON serialization, deterministic explanations | ⭐⭐ Medium |
| 7 | Database Connection | `db/connection.py` | SQLAlchemy engine, connection pooling, context managers, generators | ⭐⭐ Medium |
| 8 | ORM Models | `db/models.py` | SQLAlchemy ORM, `relationship()`, `ForeignKey`, `__tablename__` | ⭐⭐ Medium |
| 9 | Repository Pattern | `db/repository.py` | Static methods, raw SQL, bulk operations, design patterns | ⭐⭐⭐ Hard |
| 10 | Ingestion Pipeline | `db/pipeline.py` | Threading, background jobs, incremental computation, state machines | ⭐⭐⭐⭐ Very Hard |
| 11 | FastAPI Server | `api/server.py` | FastAPI, Pydantic, async/lifespan, middleware, route decorators | ⭐⭐⭐ Hard |
| 12 | Editor Routes | `api/editor_routes.py` | APIRouter, file upload, background tasks, job polling | ⭐⭐ Medium |

---

## Chapter 1: Configuration as a Module

### 📂 File: [`config.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/config.py)

### 🧠 Concepts You Need First

#### 1.1 — Python Module Execution
When Python imports a file (`import config`), it **runs every line in that file from top to bottom**. This is different from a class or function — the code at module level executes immediately.

```python
# config.py runs these lines THE MOMENT someone does `import config`
ROOT = Path(__file__).parent    # this runs immediately
DATA_DIR = ROOT / "data"        # this too
```

This is why `config.py` has no classes or functions at the top — it's meant to be a **living dictionary of settings** that's already populated when any other file imports it.

#### 1.2 — `pathlib.Path` (Modern File Paths)
Forget `os.path.join()`. Python 3 has `pathlib`:

```python
from pathlib import Path

# __file__ = the current file's path
ROOT = Path(__file__).parent          # parent directory of config.py
DATA_DIR = ROOT / "data"              # the / operator joins paths!
DATA_DIR / "articles.csv"             # D:\...\news recommender base\data\articles.csv

# You can check if files exist
(DATA_DIR / "articles.csv").exists()  # True or False

# You can glob (search) for files
DATA_DIR.glob("*.csv")               # all CSV files in data/
```

> **Why not `os.path`?** `pathlib` is object-oriented — `Path` objects have methods like `.exists()`, `.glob()`, `.parent`. With `os.path` you'd write `os.path.exists(os.path.join(root, "data"))` — messy!

#### 1.3 — `os.getenv()` (Environment Variables)
Environment variables are settings stored OUTSIDE your code (in `.env` file or system):

```python
os.getenv("POSTGRES_HOST", "localhost")
#          ^key to look up    ^default if not found

# Your .env file:
# POSTGRES_HOST=localhost
# POSTGRES_PASSWORD=mypassword
```

> **Why use env vars?** So you never put passwords in your code. The `.env` file is in `.gitignore` — it never gets committed.

#### 1.4 — Type Hints
```python
def _has_date(value) -> bool:    # -> bool means "this function returns a boolean"
    ...

def recency_boost(cand_date, ref_date, section: str, is_live: bool = False) -> float:
    # section: str     = section must be a string
    # is_live: bool    = is_live must be a boolean
    # = False          = default value if not provided
```

Type hints don't enforce anything at runtime — they're **documentation for humans and tools**.

### 🔍 Code Walkthrough

```python
# Lines 8-9: Where is this project?
ROOT = Path(__file__).parent       # D:\...\news recommender base
DATA_DIR = ROOT / "data"           # D:\...\news recommender base\data
```

```python
# Lines 14-23: Which CSV files to load?
_env_csvs = os.getenv("DATA_CSV", "")     # Check if user specified custom CSVs
if _env_csvs:
    # User gave us specific files via environment variable
    DATA_PATHS = [DATA_DIR / name.strip() for name in _env_csvs.split(",")]
else:
    # Use defaults - check which ones actually exist on disk
    _default_files = ["articles.csv", "articles_2.csv", "articles_clean_train_80.csv"]
    DATA_PATHS = [DATA_DIR / f for f in _default_files if (DATA_DIR / f).exists()]
```

> ☝️ This is a **list comprehension** — `[expression for item in list if condition]`. It's just a compact for-loop that builds a list.

```python
# Lines 29-33: Load .env file (defensive import)
try:
    from dotenv import load_dotenv   # This package reads .env files
    load_dotenv()                    # Load variables into os.environ
except ImportError:
    pass                             # If python-dotenv isn't installed, just skip
```

> **Why try/except ImportError?** This way the code works even if someone hasn't installed `python-dotenv`. Defensive programming.

```python
# Lines 269-311: Pipeline version fingerprint
def get_pipeline_version() -> str:
    import hashlib
    tokens = [
        f"embed_model={EMBEDDING_MODEL}",     # What AI model we use
        f"mmr_lambda={MMR_LAMBDA}",            # Our diversity setting
        ...                                     # Every tunable parameter
    ]
    return hashlib.sha256("|".join(tokens).encode("utf-8")).hexdigest()[:16]
```

> **What is this?** It creates a unique "fingerprint" of ALL settings. If you change ANY setting (like `MMR_LAMBDA` from 0.91 to 0.85), the fingerprint changes. The system uses this to know: "settings changed → I need to recompute ALL recommendations".

### 🔗 How It Connects
Every single file in this project does `import config` and reads values like `config.TOP_K`, `config.MMR_LAMBDA`, etc. This is the **single source of truth** — no magic numbers scattered in code.

### ✏️ Exercise
Open `config.py` and answer:
1. What happens if you change `MMR_LAMBDA` from 0.91 to 0.80? (Hint: `get_pipeline_version()`)
2. What does `EMBED_TOKEN_BUDGET = EMBED_MAX_SEQ_LEN - EMBED_SPECIAL_TOKENS` evaluate to?
3. Why does `recency_boost()` use `0.5 ** (days_old / half_life)` — what shape does this create?

---

## Chapter 2: Data Loading & Cleaning

### 📂 File: [`src/data_loader.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/src/data_loader.py)

### 🧠 Concepts You Need First

#### 2.1 — Pandas DataFrames
A DataFrame is a table in Python — rows and columns, like an Excel sheet:

```python
import pandas as pd

# Read a CSV file into a DataFrame
df = pd.read_csv("articles.csv")

# Access columns
df["title"]                  # all titles (a Series)
df["title"][0]               # first article's title

# Filter rows
sports = df[df["section"] == "sports"]     # only sports articles

# Apply a function to every row
df["title"] = df["title"].apply(lambda x: x.strip())  # strip whitespace from every title
```

#### 2.2 — Regular Expressions (regex)
Regex is a mini-language for pattern matching in text:

```python
import re

# Find all words
re.findall(r"[a-z0-9]+", "Hello World 123")   # ['ello', 'orld', '123'] (lowercase!)

# Replace patterns
re.sub(r"\s+", " ", "too   many   spaces")     # "too many spaces"

# Named groups
pattern = r"(?P<name>\w+)\s+(?P<age>\d+)"
match = re.match(pattern, "Vaibhav 25")
match.group("name")   # "Vaibhav"
```

#### 2.3 — Unicode Normalization
Different keyboard/OS combos produce different byte sequences for the same character:

```python
import unicodedata

# "café" can be stored as 4 chars or 5 chars depending on encoding!
unicodedata.normalize("NFKC", text)   # standardize to one canonical form
```

### 🔍 What This File Does (The Flow)

```
Raw CSV → clean_dataframe() → Cleaned DataFrame
                ↓
      1. Rename columns (articleTitle → title)
      2. NFKC normalize all text
      3. Strip "ALSO READ..." boilerplate from bodies
      4. Remove sentences that appear in 4+ articles (boilerplate)
      5. Deduplicate (keep latest version of each article)
      6. Compose token-budgeted text representations:
         - text_embed: for the embedding model (254 tokens max)
         - text_bm25: for keyword search (title repeated 3x for weight)
         - text_rerank: for the cross-encoder (254 tokens max)
```

### 🔗 How It Connects
- **Called by**: `db/pipeline.py` (during CSV ingestion)
- **Produces**: A clean DataFrame with `text_embed`, `text_bm25`, `text_rerank` columns
- **These text columns** are what the AI models in chapters 3-4 actually process

### ✏️ Exercise
1. Why is the title repeated 3 times in `text_bm25`? (Think about keyword search)
2. What does "token-budgeted" mean? Why can't we just send the full article body to the AI?

---

## Chapter 3: Search Engines (Retrieval)

### 📂 File: [`src/retrieval.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/src/retrieval.py)

### 🧠 Concepts You Need First

#### 3.1 — Python Classes and OOP (The Big One You Need)

A class is a **blueprint for creating objects**. Think of it like a factory:

```python
class Dog:
    # __init__ runs when you CREATE a new dog
    def __init__(self, name, breed):
        self.name = name      # self = "this specific dog"
        self.breed = breed    # Stores data ON the object
    
    def bark(self):           # self is always the first parameter
        return f"{self.name} says Woof!"

# Creating objects (instances)
tommy = Dog("Tommy", "Labrador")    # __init__ runs with name="Tommy"
tommy.bark()                         # "Tommy says Woof!"
tommy.name                           # "Tommy" (stored on the object)
```

> **Key insight**: `self` is just "me, this specific object". When you write `self.postings = {}` inside a class, you're creating a dictionary that belongs to THIS specific instance.

#### 3.2 — `defaultdict` (Auto-Creating Dictionaries)
```python
from collections import defaultdict

# Regular dict crashes if key doesn't exist:
d = {}
d["cats"].append("Tom")   # KeyError! "cats" doesn't exist yet

# defaultdict auto-creates the value:
d = defaultdict(list)      # Missing keys auto-create empty lists
d["cats"].append("Tom")    # Works! d = {"cats": ["Tom"]}
```

#### 3.3 — NumPy Arrays (Fast Math)
```python
import numpy as np

# Regular Python list math is SLOW for 2000+ articles
scores = [0.1, 0.5, 0.9, 0.3]
# To multiply each by 2: [x * 2 for x in scores]  ← loops through each one

# NumPy does it in ONE operation (vectorized):
scores = np.array([0.1, 0.5, 0.9, 0.3])
scores * 2        # array([0.2, 1.0, 1.8, 0.6])  ← ALL AT ONCE (10-100x faster)
scores.max()      # 0.9
scores.argsort()  # array([0, 3, 1, 2]) ← indices that would sort it
```

#### 3.4 — BM25 Algorithm (Plain English)
BM25 answers: "Given a search query, how relevant is each document?"

It considers:
1. **Term Frequency (TF)**: How many times does the query word appear in the document? More = better, but with diminishing returns.
2. **Inverse Document Frequency (IDF)**: If a word appears in EVERY document (like "the"), it's useless. Rare words matter more.
3. **Document Length**: A 10-word doc with 5 query matches is more relevant than a 1000-word doc with 5 matches.

Formula: `score = Σ IDF(word) × (TF × (k1+1)) / (TF + k1 × (1 - b + b × docLen/avgLen))`

#### 3.5 — Embeddings (Plain English)
An embedding turns text into a list of 768 numbers (a "vector"). Similar texts get similar numbers:

```
"India won the cricket match"     → [0.21, -0.45, 0.87, ...]  (768 numbers)
"India defeated Australia in T20" → [0.22, -0.43, 0.85, ...]  (very similar!)
"Stock market crashed today"      → [-0.61, 0.12, -0.33, ...]  (very different!)
```

To measure similarity: **cosine similarity** = dot product of normalized vectors.

### 🔍 Three Classes in This File

```python
class BM25Index:
    """Keyword search — finds articles containing the same words."""
    def __init__(self, texts):
        # Tokenize every article, build an inverted index
        self.docs = [tokenize(t) for t in texts]          # [["india", "cricket"], ["stock", "market"], ...]
        self.postings = defaultdict(list)                   # {"india": [(0, 3), (5, 1)], ...}
        #                                                     word     article_0 appears 3 times, article_5 appears 1 time

class DenseIndex:
    """AI-powered semantic search — finds articles with similar MEANING."""
    def __init__(self, texts, vectors=None):
        # Either use pre-computed embeddings or compute them now
        if vectors is not None:
            self.vectors = vectors      # Use what's already computed
        else:
            model = get_embedder()      # Load the AI model
            self.vectors = model.encode(texts)  # Turn text → 768-dim vectors

class TfidfIndex:
    """Fallback: like DenseIndex but without the AI model (faster, less accurate)."""
```

All three classes have the **same methods** — this is called a **duck typing interface**:
```python
.scores_for_text(query)    # "How similar is each article to this search query?"
.scores_for_doc(position)  # "How similar is each article to article #N?"
.similarity_matrix(positions)  # "How similar are these articles to EACH OTHER?" (for diversity)
```

> **Why 3 classes with the same methods?** So the `Recommender` (Ch 5) doesn't care which search engine it's using. It just calls `.scores_for_text()` and it works whether the engine is BM25, TF-IDF, or full neural. This is the **Strategy Pattern**.

### 🔗 Key Function: Reciprocal Rank Fusion (RRF)
```python
def reciprocal_rank_fusion(rankings, k=60):
    """Merge multiple ranked lists into one. Each article gets:
       score = sum(1 / (k + rank_in_each_list))
    """
```
If article X is rank 1 in BM25 and rank 5 in semantic → `1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.032`
If article Y is rank 50 in BM25 and rank 2 in semantic → `1/(60+50) + 1/(60+2) = 0.009 + 0.016 = 0.025`
→ Article X wins because it's good in BOTH methods.

### ✏️ Exercise
1. What is an "inverted index"? Why is it called "inverted"?
2. If the BM25 model says article A is rank 3 and the semantic model says it's rank 40, what is its RRF score (with k=60)?
3. Why does `DenseIndex` normalize vectors to unit length?

---

## Chapter 4: Cross-Encoder Reranking

### 📂 File: [`src/reranker.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/src/reranker.py)

### 🧠 Concepts You Need First

#### 4.1 — Bi-Encoder vs Cross-Encoder

**Bi-Encoder** (what Ch 3's DenseIndex does):
```
Article A → [vector A]  ┐
                         ├→ cosine_similarity(A, B) = 0.85
Article B → [vector B]  ┘
```
Each article is encoded INDEPENDENTLY. Fast (encode once, compare any pair), but less accurate.

**Cross-Encoder** (what this Reranker does):
```
[Article A TEXT] + [Article B TEXT] → fed TOGETHER into one model → score 0.92
```
The model reads BOTH texts side-by-side and makes a direct judgment. Much more accurate, but SLOW (must re-run for every pair).

> **Real-world analogy**: Bi-encoder is like comparing two résumés by their keyword tags. Cross-encoder is like actually reading both résumés together and deciding who's a better match.

#### 4.2 — Sigmoid Function
The cross-encoder outputs a raw number (logit) like -2.3 or 4.7. We need to convert it to a probability (0 to 1):

```python
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

sigmoid(-5)   # 0.007  (very unlikely related)
sigmoid(0)    # 0.5    (maybe)
sigmoid(5)    # 0.993  (very likely related)
```

#### 4.3 — Token Budgets
AI models have a maximum input size (512 tokens for the reranker). When comparing two articles:
- Article A gets 254 tokens
- Article B gets 254 tokens  
- 4 tokens for separators
- Total = 512 ✓

If an article is 1000 tokens, we must **truncate** it to 254. The code carefully picks the most important sentences to keep.

### 🔗 How It Connects
- **Called by**: `src/recommender.py` (live path) and `db/pipeline.py` (ingestion)
- **Input**: Pairs of texts `[(article_A_text, article_B_text), ...]`
- **Output**: Probability scores `[0.92, 0.15, 0.67, ...]` — how related each pair is

### ✏️ Exercise
1. Why can't we use only the cross-encoder for everything (skip BM25 and embeddings)?
2. What would happen if we sent 800 tokens to a model that only accepts 512?

---

## Chapter 5: The Recommender (Orchestrator)

### 📂 File: [`src/recommender.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/src/recommender.py)

### 🧠 Concepts You Need First

#### 5.1 — `@dataclass` (Auto-Generated Classes)
Writing `__init__`, `__repr__`, etc. for simple data containers is tedious:

```python
# WITHOUT dataclass — lots of boilerplate
class Recommendation:
    def __init__(self, article, score, reason):
        self.article = article
        self.score = score
        self.reason = reason

# WITH dataclass — Python writes __init__ for you!
from dataclasses import dataclass

@dataclass
class Recommendation:
    article: dict
    score: float
    reason: str
    confidence_band: str = "MEDIUM"   # default value
```

Both do the same thing. `@dataclass` is just syntactic sugar.

#### 5.2 — Composition Pattern
The `Recommender` class doesn't inherit from anything. Instead, it **contains** other objects:

```python
class Recommender:
    def __init__(self, articles):
        self.bm25 = BM25Index(articles["text_bm25"])         # HAS-A BM25 engine
        self.semantic = DenseIndex(articles["text_embed"])     # HAS-A semantic engine
        self.reranker = Reranker()                             # HAS-A reranker
```

> **Composition vs Inheritance**: Inheritance = "A Dog IS-A Animal". Composition = "A Car HAS-A Engine". This project uses composition everywhere — the Recommender doesn't IS-A anything, it HAS several tools.

#### 5.3 — MMR (Maximal Marginal Relevance)
Problem: The top 10 most relevant articles might all be about the same sub-topic.
MMR balances **relevance** vs **diversity**:

```
Score = λ × relevance(article, query) − (1−λ) × max_similarity(article, already_selected)
```

- λ = 0.91 means 91% relevance, 9% diversity
- If an article is very similar to one we already picked, its score drops

### 🔍 The Full Pipeline (What `recommend()` Does)

```python
def recommend(self, query, article_id, top_k, method, use_boosts, diversify, section):
    # Step 1: Build a search query
    #   If article_id given → extract title + top IDF body terms
    #   If text query given → use it directly

    # Step 2: Two parallel searches (N_RETRIEVE=50 each)
    bm25_scores = self.bm25.scores_for_text(query)          # keyword scores
    dense_scores = self.semantic.scores_for_text(query)      # semantic scores

    # Step 3: Reciprocal Rank Fusion → merge into ~N_CANDIDATES=20
    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking])

    # Step 4: Cross-encoder reranking (the expensive, accurate step)
    rerank_scores = self.reranker.score_pairs(query_text, candidate_texts)

    # Step 5: Blend scores
    #   final = 0.70 * rerank + 0.20 * bm25_normalized + 0.10 * dense_normalized

    # Step 6: Apply boosts (if use_boosts=True)
    #   Same section? × 1.072
    #   Same subsection? × 1.09
    #   Popular article? × up to 1.044
    #   Recent article? × up to 1.104

    # Step 7: MMR diversity (if diversify=True)
    #   Remove near-duplicates, balance relevance vs variety

    # Step 8: Calibrate scores to 0-1 probability
    # Step 9: Build evidence/explanation JSON
    # Step 10: Return Result with recommendations
```

### 🔗 How It Connects
- **Used by**: `api/server.py` (the live path when precomputed results aren't available)
- **Contains**: BM25Index, DenseIndex, Reranker (from chapters 3-4)
- **Reads**: `config.py` for all thresholds and weights

### ✏️ Exercise
1. Trace a request: User clicks article "12345". What exactly happens step by step?
2. Why does the blend weight the reranker at 0.70 but BM25 at only 0.20?
3. If `MMR_LAMBDA = 1.0`, what would happen to diversity?

---

## Chapter 6: Calibration & Evidence

### 📂 Files: [`src/calibration.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/src/calibration.py) + [`src/evidence.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/src/evidence.py)

### 🧠 Concepts You Need First

#### 6.1 — Platt Scaling (Why Raw Scores Are Misleading)
The reranker might output scores like 0.45, 0.42, 0.41, 0.38. Are these "good"? You can't tell without context.

Platt scaling fits a sigmoid: `P = 1/(1 + e^-(a*score + b))` where `a=4.85, b=-2.60` were learned from labeled data. Now 0.45 → maybe 0.72 probability of being a good recommendation.

#### 6.2 — JSON Evidence
Each recommendation carries a structured explanation:

```json
{
  "shared_entities": ["Narendra Modi", "BJP"],
  "shared_terms": ["election", "mandate", "coalition"],
  "matching_sentence": "Modi secured a third consecutive term...",
  "agreement_count": 3,    // 3 out of 3 methods agree this is relevant
  "confidence_band": "HIGH"
}
```

This is **deterministic** (no LLM involved) — it's pure pattern matching.

### ✏️ Exercise
1. What are the 3 confidence bands and their thresholds?
2. Why is this project NOT using an LLM for explanations?

---

## Chapter 7: Database Connection

### 📂 File: [`db/connection.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/db/connection.py)

### 🧠 Concepts You Need First

#### 7.1 — SQLAlchemy (Python's Database Toolkit)
SQLAlchemy has two layers:
1. **Core** (low-level): Write SQL strings, execute them
2. **ORM** (high-level): Define Python classes that map to database tables

This project uses **both**:
```python
# ORM style (Chapter 8):
article = db.query(Article).filter(Article.id == "123").first()
article.title  # "Modi wins election"

# Core style (Chapter 9):
conn.execute(text("SELECT * FROM articles WHERE section = 'sports'"))
```

#### 7.2 — Connection Pooling
Creating a new database connection is EXPENSIVE (network handshake, authentication...). A **pool** keeps 10 connections alive and reuses them:

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,       # Keep 10 connections ready
    max_overflow=20,    # If all 10 busy, create up to 20 more temporarily
    pool_pre_ping=True, # Check "is this connection still alive?" before using it
)
```

#### 7.3 — Generators and `yield` (Important!)
```python
def get_db():
    db = SessionLocal()    # Create a database session
    try:
        yield db           # PAUSE HERE — give the session to the caller
    finally:
        db.close()         # After the caller is done, close the session

# Usage:
for session in get_db():
    session.query(Article).all()
# After the loop, db.close() runs automatically!
```

`yield` makes this a **generator**. It's like a function that can pause and resume. FastAPI uses this for **dependency injection** (Ch 11).

#### 7.4 — Context Manager (`with` statement)
```python
with SessionLocal() as db:    # creates a session
    db.query(Article).all()
# session automatically closes here, even if an error occurred
```

`with` guarantees cleanup. The `SessionLocal()` class has `__enter__` and `__exit__` methods that handle this.

### 🔍 Code Walkthrough

```python
# Lines 17-26: Build the database URL from environment variables
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
...
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# Result: "postgresql+psycopg2://postgres:postgres@localhost:5432/news_recommender"
```

```python
# Lines 29: The "Base" — all ORM models inherit from this
Base = declarative_base()
# Every class that inherits from Base becomes a database table
```

```python
# Lines 122-181: Schema migration without a migration tool
def ensure_schema():
    """Add new columns to existing tables. Safe to run repeatedly."""
    conn.execute(text(
        "ALTER TABLE article_recommendations "
        "ADD COLUMN IF NOT EXISTS rerank_score DOUBLE PRECISION"
    ))
    # IF NOT EXISTS = won't crash if the column already exists
```

> **Why not use Alembic (proper migrations)?** This project does lightweight migrations inline. For a production system you'd use Alembic, but for a research project, `ADD COLUMN IF NOT EXISTS` is simpler.

### ✏️ Exercise
1. What does `pool_pre_ping=True` do and why is it important for a long-running server?
2. Rewrite `get_db()` without using `yield` (use try/finally manually). Why is the `yield` version better?

---

## Chapter 8: ORM Models

### 📂 File: [`db/models.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/db/models.py)

### 🧠 Concepts You Need First

#### 8.1 — ORM = Object-Relational Mapping
Each Python class = one database table. Each attribute = one column:

```python
class Article(Base):                              # → CREATE TABLE articles (...)
    __tablename__ = "articles"                     # table name in PostgreSQL
    
    id = Column(String(100), primary_key=True)     # → id VARCHAR(100) PRIMARY KEY
    title = Column(Text, nullable=False)           # → title TEXT NOT NULL
    section = Column(String(100), index=True)      # → section VARCHAR(100) + CREATE INDEX
    embedding = Column(Vector(768))                # → embedding vector(768) (pgvector!)
```

#### 8.2 — Foreign Keys and Relationships
```python
class ArticleRecommendation(Base):
    article_id = Column(String(100), ForeignKey("articles.id", ondelete="CASCADE"))
    # ForeignKey = "this column MUST contain an ID that exists in articles.id"
    # CASCADE = if the article is deleted, delete all its recommendations too
    
    source_article = relationship("Article", foreign_keys=[article_id], back_populates="recommendations")
    # relationship() = creates a Python-level link. Now you can do:
    #   rec.source_article.title  → "Modi wins election"
    # And from the other side:
    #   article.recommendations   → [rec1, rec2, ..., rec10]
```

#### 8.3 — The `to_dict()` Pattern
ORMs give you Python objects, but HTTP APIs need JSON (dictionaries):
```python
def to_dict(self):
    return {
        "id": self.id,
        "title": self.title,
        ...
    }
```

### 🔍 The Three Tables

```mermaid
erDiagram
    articles ||--o{ article_recommendations : "has many"
    articles {
        string id PK
        text title
        text description
        text body
        string section
        datetime date
        vector768 embedding
        text text_embed
        text text_bm25
    }
    article_recommendations {
        int id PK
        string article_id FK
        string recommended_article_id FK
        int rank
        float score
        float rerank_score
        text reason
    }
    ingestion_jobs {
        string id PK
        string filename
        string status
        int total_articles
        int processed_articles
    }
```

### ✏️ Exercise
1. Why is `embedding` a `Vector(768)` and not a regular `Text` or `Float` column?
2. What does `ondelete="CASCADE"` mean in practice?
3. The `UniqueConstraint("article_id", "rank")` — what bug does this prevent?

---

## Chapter 9: Repository Pattern

### 📂 File: [`db/repository.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/db/repository.py)

### 🧠 Concepts You Need First

#### 9.1 — Static Methods
```python
class ArticleRepository:
    @staticmethod
    def count_articles(db):
        return db.query(Article).count()
    
    # Called as:
    ArticleRepository.count_articles(session)
    # NOT: repo = ArticleRepository(); repo.count_articles(session)
```

`@staticmethod` means "this method doesn't need `self` — it doesn't access any instance data". The class is just used as a **namespace** to group related functions.

> **Why not just free functions?** Organization. `ArticleRepository.save_recommendations()` is clearer than `save_recommendations()` — you know it's database-related.

#### 9.2 — The Repository Pattern
Instead of writing SQL queries everywhere in your code:
```python
# BAD: SQL scattered in server.py, pipeline.py, editor_routes.py...
db.execute("SELECT * FROM articles WHERE section = 'sports'")

# GOOD: All SQL lives in ONE place
class ArticleRepository:
    @staticmethod
    def list_articles(db, section=None): ...
    @staticmethod
    def save_recommendations(db, article_id, recs): ...
```

Now if you change the database schema, you only update `repository.py`.

#### 9.3 — Bulk Operations
Inserting 500 articles one by one = 500 database round trips (SLOW).
Bulk insert = 1 round trip with all 500 (FAST):

```python
@staticmethod
def bulk_upsert_articles_fast(db, article_dicts, chunk_size=500):
    """INSERT ... ON CONFLICT (id) DO UPDATE SET ...
    This inserts new articles AND updates existing ones in ONE query."""
```

### ✏️ Exercise
1. Why does `save_recommendations` delete old recommendations before inserting new ones?
2. What is the difference between `INSERT` and `INSERT ... ON CONFLICT DO UPDATE`?

---

## Chapter 10: Ingestion Pipeline

### 📂 File: [`db/pipeline.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/db/pipeline.py)

### 🧠 Concepts You Need First

#### 10.1 — Threading (Background Work)
```python
import threading

def heavy_work():
    # This runs in a SEPARATE thread (doesn't block the main program)
    for i in range(1000000):
        process(i)

# Start the work in the background
thread = threading.Thread(target=heavy_work, daemon=True)
thread.start()
# Main program continues immediately! heavy_work runs in parallel.
```

`daemon=True` means: "If the main program exits, kill this thread too."

#### 10.2 — State Machine (Job Status)
The ingestion job goes through states:
```
pending → processing → completed
                    ↘ failed
```

The `message` field uses a protocol: `STAGE:EMBEDDING|Processing article 150/500`
The frontend polls this message to show progress.

### 🔍 The Pipeline Steps (The Most Complex File)

```python
def run_ingestion_pipeline(csv_source, job_id, limit=None, recompute_all=False):
    # 1. PARSE: Read CSV file
    # 2. CLEAN: clean_dataframe() from Chapter 2
    # 3. STORAGE: bulk_upsert into PostgreSQL
    # 4. EMBEDDING: compute 768-dim vectors for new/changed articles
    # 5. RERANK: for each new article, find its top-10 recommendations
    #    - BM25 search → top 50
    #    - Dense search → top 50
    #    - RRF merge → top 20
    #    - Cross-encoder rerank → score all 20
    #    - Blend → boost → MMR → save top 10
    # 6. REVERSE PUSH: for existing articles, check if new articles should
    #    replace any of their current top-10
    # 7. RELOAD: refresh the in-memory search engine
```

> **Reverse Push is the clever part**: When you add 500 new articles, you don't just compute recommendations FOR them. You also check: "Should any of these new articles appear in the recommendations of the 2000 EXISTING articles?" This keeps the entire system fresh.

### ✏️ Exercise
1. Why does the pipeline use `threading.Thread(daemon=True)` instead of `asyncio`?
2. What does "O-centric scoring" mean in the reverse push context?
3. If you change `MMR_LAMBDA` in config, what happens on the next CSV upload?

---

## Chapter 11: FastAPI Server

### 📂 File: [`api/server.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/api/server.py)

### 🧠 Concepts You Need First

#### 11.1 — FastAPI Basics
FastAPI is a web framework — it turns your Python functions into HTTP endpoints:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/articles")          # When someone visits GET /api/articles
def get_articles():                 # Run this function
    return {"articles": [...]}     # Return this as JSON

@app.post("/api/recommend")        # When someone sends POST /api/recommend
def recommend(body: RecommendRequest):   # Parse the JSON body
    return {"recommendations": [...]}
```

#### 11.2 — Decorators (`@app.get`, `@app.post`)
A decorator wraps a function with extra behavior:

```python
@app.get("/api/status")    # This is a DECORATOR
def status():              # It registers status() as the handler for GET /api/status
    return {"ok": True}

# Without the decorator, you'd write:
def status():
    return {"ok": True}
app.get("/api/status")(status)    # Same thing, just uglier
```

#### 11.3 — Pydantic Models (Request Validation)
```python
from pydantic import BaseModel

class RecommendRequest(BaseModel):
    query: Optional[str] = None        # Optional string, defaults to None
    article_id: Optional[str] = None
    top_k: int = config.TOP_K          # Required integer, defaults to 10
    method: str = "hybrid+rerank"

# When someone sends: POST /api/recommend {"article_id": "123", "top_k": 5}
# FastAPI automatically:
#   1. Parses the JSON
#   2. Creates a RecommendRequest object
#   3. Validates types (top_k MUST be an int, or → 422 error)
#   4. Fills in defaults (method → "hybrid+rerank")
```

#### 11.4 — Lifespan (Startup/Shutdown)
```python
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # STARTUP CODE (runs once when server starts)
    ensure_schema()
    await run_in_threadpool(reload_engine_from_postgres)
    
    yield  # ← Server is running here, handling requests
    
    # SHUTDOWN CODE (runs when server stops)
    # (nothing in this project)
```

#### 11.5 — CORS Middleware
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow requests from ANY website
)
```
Without this, a browser at `http://localhost:5000` would block JavaScript from calling `http://localhost:5000/api/recommend` (browser security).

### 🔍 The Dual Serving Path (The Key Design Decision)

```python
@app.post("/api/recommend")
def recommend(body: RecommendRequest):
    # FAST PATH: Can we serve from pre-computed database? (sub-5ms)
    use_precomputed = (
        body.article_id           # Must be article-to-article
        and not body.query        # Not a text search
        and not body.section      # No section filter
        and body.method == "hybrid+rerank"
        and body.use_boosts
        and body.top_k <= config.TOP_K
    )
    
    if use_precomputed:
        # → Read from PostgreSQL article_recommendations table
        # → Return instantly (sub-5ms!)
    
    else:
        # SLOW PATH: Compute live (2-5 seconds)
        result = rec.recommend(query=body.query, ...)
```

> **Why two paths?** Pre-computing 10 recommendations for each of 2446 articles takes 30+ minutes during ingestion, but then every reader request is instant. The live path exists for search queries and edge cases.

### ✏️ Exercise
1. List all the conditions that must be true for the "fast path" to activate.
2. What is the difference between `@app.get` and `@app.post`?
3. Why does `lifespan` use `await run_in_threadpool(reload_engine_from_postgres)` instead of calling the function directly?

---

## Chapter 12: Editor Routes

### 📂 File: [`api/editor_routes.py`](file:///D:/indian%20express/Recommendation%20system/news%20recommender%20base/api/editor_routes.py)

### 🧠 Concepts You Need First

#### 12.1 — APIRouter (Modular Routes)
```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/db")   # All routes start with /api/db

@router.get("/health")                 # → GET /api/db/health
def health(): ...

@router.post("/upload-csv")            # → POST /api/db/upload-csv
def upload_csv(): ...

# In server.py:
app.include_router(router)             # Plug these routes into the main app
```

This keeps the server.py file smaller — editor routes live in their own file.

#### 12.2 — File Upload
```python
from fastapi import UploadFile

@router.post("/upload-csv")
async def upload_csv(file: UploadFile):
    # file.filename = "articles_batch_3.csv"
    # file.content_type = "text/csv"
    # file.read() = the actual bytes of the file
    
    # Save to disk, then start background processing
    content = await file.read()
    save_path = DATA_DIR / "uploads" / file.filename
    save_path.write_bytes(content)
    
    # Start ingestion in background thread
    thread = threading.Thread(target=run_ingestion_pipeline, args=(save_path, job_id))
    thread.start()
    
    return {"job_id": job_id}   # Frontend polls this job_id for progress
```

### ✏️ Exercise
1. Why does the upload endpoint return a `job_id` instead of waiting for processing to finish?
2. What happens if an editor tries to upload a second CSV while the first is still processing?

---

## 🗺️ The Complete Data Flow

Now that you know every file, here's how a full request flows:

### Flow 1: Editor Uploads CSV
```
Editor clicks "Upload" → POST /api/db/upload-csv
  → api/editor_routes.py saves file, creates job_id
  → threading.Thread starts run_ingestion_pipeline()
    → db/pipeline.py:
      → src/data_loader.py: clean_dataframe()
      → db/repository.py: bulk_upsert_articles_fast()
      → src/model_registry.py: get_embedder() → encode 768-d vectors
      → db/repository.py: bulk_set_embeddings()
      → For each article:
        → src/retrieval.py: BM25 + Dense search
        → src/reranker.py: Cross-encoder scoring
        → src/recommender.py: blend + boost + MMR
        → src/calibration.py: Platt scaling
        → src/evidence.py: build explanation
        → db/repository.py: save_recommendations()
      → Reverse push for existing articles
      → api/server.py: reload_engine_from_postgres()
```

### Flow 2: Reader Opens an Article
```
Reader clicks article "12345" → POST /api/recommend {"article_id": "12345"}
  → api/server.py: recommend()
    → Check: all fast-path conditions met? YES
    → db/repository.py: get_precomputed_recommendations("12345")
    → PostgreSQL returns 10 rows (sub-5ms!)
    → src/evidence.py: enrich_recommendation_evidence()
    → Return JSON with 10 recommendations
```

### Flow 3: Reader Searches "Modi election"
```
Reader types "Modi election" → POST /api/recommend {"query": "Modi election"}
  → api/server.py: recommend()
    → Check: fast-path? NO (has query, not article_id)
    → src/recommender.py: recommend(query="Modi election")
      → BM25 search: keyword match → top 50
      → Dense search: semantic match → top 50
      → RRF fusion → top 20
      → Reranker: score all 20 pairs
      → Blend: 0.70×rerank + 0.20×BM25 + 0.10×dense
      → MMR: diversify
      → Calibrate + build evidence
    → Return JSON with recommendations
```

---

## 📚 Suggested Study Order & Timeline

| Week | Chapters | Focus |
|------|----------|-------|
| 1 | 1-2 | Config + Data Loading (get comfortable with the codebase) |
| 2 | 3-4 | Search engines + Reranking (the IR/NLP core) |
| 3 | 5-6 | Recommender + Evidence (understand the full pipeline) |
| 4 | 7-9 | Database layer (SQLAlchemy + PostgreSQL) |
| 5 | 10 | Ingestion pipeline (the most complex piece) |
| 6 | 11-12 | FastAPI server (bringing it all together) |

> [!TIP]
> **How to study each chapter**: Open the file in your editor, read it alongside the chapter above, and try to predict what each line does before reading my explanation. When you can explain a file to someone else, you've mastered it.

---

## 🔑 Quick Reference: Python Concepts Index

| Concept | Where It Appears | Chapter |
|---------|-------------------|---------|
| `pathlib.Path` | `config.py` L8 | 1 |
| `os.getenv()` | `config.py` L14, `connection.py` L17 | 1, 7 |
| Type hints (`-> bool`) | `config.py` L141 | 1 |
| List comprehensions | `config.py` L16, everywhere | 1 |
| `try/except ImportError` | `config.py` L29 | 1 |
| `hashlib` | `config.py` L275 | 1 |
| Pandas DataFrames | `data_loader.py`, `server.py` | 2, 11 |
| Regex (`re`) | `data_loader.py`, `retrieval.py` | 2, 3 |
| Unicode (`unicodedata`) | `data_loader.py` | 2 |
| **Classes & `__init__`** | `retrieval.py`, `recommender.py` | 3, 5 |
| `defaultdict` | `retrieval.py` L77 | 3 |
| NumPy arrays | `retrieval.py`, `reranker.py` | 3, 4 |
| **`@dataclass`** | `recommender.py` L29 | 5 |
| **Composition pattern** | `recommender.py` L52 | 5 |
| **`@staticmethod`** | `repository.py` | 9 |
| **Generators & `yield`** | `connection.py` L52 | 7 |
| **Context managers (`with`)** | `connection.py`, `server.py` | 7, 11 |
| **SQLAlchemy ORM** | `models.py`, `connection.py` | 7, 8 |
| **`relationship()` & FK** | `models.py` L60, L137 | 8 |
| **Threading** | `pipeline.py`, `editor_routes.py` | 10, 12 |
| **FastAPI decorators** | `server.py` L239 | 11 |
| **Pydantic models** | `server.py` L141 | 11 |
| **Async context manager** | `server.py` L39 | 11 |
| **APIRouter** | `editor_routes.py` | 12 |

---

> [!IMPORTANT]
> **Ready to start?** Reply with **"Start Chapter 1"** and I'll do a deep, line-by-line walkthrough of `config.py` with you — we'll go through every single line, I'll quiz you, and we'll make sure you fully understand before moving to Chapter 2.
>
> Or if you want to jump to a specific chapter (e.g., "Start Chapter 7" for database stuff), just say so!
