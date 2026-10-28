# PROJECT_HANDOVER_FACTS.md

**Indian Express AI News Recommendation & Ingestion Platform — verified technical handover**

Prepared for: mentor / senior management review
Prepared on: 2026-10-28
Repository: `D:\indian express\Recommendation system\news recommender base` (branch `add_db`)
Method: every statement below was read from the code or produced by running the code. Where a
statement could not be verified it is marked **NOT FOUND**; where it is a reasoned inference it is
marked **INFERRED**.

---

## How to read this document

Each technical element is presented twice:

* **Facts** — what the code literally does, with a citation in the form
  `path/file.py:function_or_line`.
* **Plain-language explanation** — what the thing is, why a newsroom needs it, how it works step by
  step, the exact formula and constants, and a short real example.

Nothing in this file was taken from `README.md`, `PROJECT_AI_SPEC.md`, code comments or prior
handover notes unless the statement was then confirmed in the code. Comments that disagree with the
code are called out explicitly as **comment drift** and listed together in §17.7.

**Contents**

| | | |
|---|---|---|
| 1. Overview and executive summary | 8. Ingestion operations | 15. API reference |
| 2. Running the system | 9. Personalisation | 16. Tests |
| 3. Configuration and environment | 10. Evidence JSON contract | 17. Code-quality scan |
| 4. Data assets on disk and in the DB | 11. Security and operational safety | 18. Verified defects and risks |
| 5. Database schema and state | 12. Evaluation harness and results | 19. Capabilities that do not exist |
| 6. The pipeline, stage by stage | 13. Measured performance | 20. Handover recommendations |
| 7. Request routing and serving modes | 14. Models, dependencies, remote GPU | Verification log |


**Frontend / user-interface code (`frontend/`) is out of scope for this report.** No CSS, JS or
HTML was analysed. Where the interface receives a value from the backend, only the backend side is
documented.

---

# 1. System overview and executive summary

## 1.1 What the system is

**Facts**

* A FastAPI application, entry point `api/server.py:529-531` (`uvicorn.run("api.server:app", host="0.0.0.0", port=5000)`), version string `2.0.0` (`api/server.py:58-63`).
* Storage is PostgreSQL with the `vector` extension (`db/connection.py`, verified live: PostgreSQL 18.6, pgvector 0.8.7).
* Two neural models are used: `BAAI/bge-base-en-v1.5` for embeddings and `BAAI/bge-reranker-base` for cross-encoder re-scoring (`config.py:40-43`), each pinned to a commit SHA (`config.py:41`, `config.py:43`).
* Recommendations are produced by one scoring cascade implemented once and shared by the batch pipeline and the live request path: `src/scoring.py:1-10` (module docstring) with `blend` (`src/scoring.py:91-105`), `boost_parts` (`src/scoring.py:249-305`) and `mmr` (`src/scoring.py:308-325`).
* Results are served from two places: a precomputed table `article_recommendations` (20,000 rows, 10 per article, verified by SQL) or a live in-memory engine built at startup (`api/server.py:81`, `api/server.py:361-462`).

**Plain-language explanation**

The platform reads a CSV of published news articles, cleans the text, turns each article into a
mathematical vector that captures its meaning, and then, for every article, finds the ten other
articles that a reader would most plausibly want next. Those ten are stored in the database so the
website can serve them instantly. If an editor asks a question the precomputed table cannot answer
(a free-text search, a category filter, or a request for "no boosts"), the same scoring logic runs
live against the whole corpus in memory. The reason this matters for a newsroom: "related stories"
is the single highest-value on-page real estate, and it must be both relevant and fast.

## 1.2 Scope of this report

**Facts**

| Area | Covered | Evidence |
|---|---|---|
| Ingestion pipeline (`db/pipeline.py`) | Yes | `db/pipeline.py:321` `run_ingestion_pipeline` |
| Scoring / ranking (`src/scoring.py`, `src/recommender.py`) | Yes | `src/scoring.py`, `src/recommender.py:95` |
| Retrieval (`src/retrieval.py`) | Yes | `BM25Index` `src/retrieval.py:54`, `DenseIndex` `src/retrieval.py:202` |
| Reranking (`src/reranker.py`) | Yes | `src/reranker.py:315` |
| Calibration / confidence (`src/calibration.py`) | Yes | `src/calibration.py:40` |
| Evidence / explanations (`src/evidence.py`) | Yes | `src/evidence.py:475` |
| API surface (`api/server.py`, `api/editor_routes.py`) | Yes | — |
| Database (`db/connection.py`, `db/models.py`, `db/repository.py`) | Yes | — |
| Evaluation harness (`evaluate.py`) | Yes | `evaluate.py:1001-1032` |
| Frontend (`frontend/`, 1 HTML + 14 CSS + JS files) | **No — out of scope** | not analysed |
| Remote GPU worker notebook (`kaggle_gpu_worker.ipynb`) | Partially (contract only) | read, not executed |

**Plain-language explanation**

The report covers everything that decides *which* articles are recommended and *how confident* the
system is. It deliberately excludes the browser user interface, because the interface only displays
numbers computed by the backend; reviewing it would not change any conclusion about recommendation
quality.

## 1.3 How the facts were verified (read-only protocol)

**Facts**

* The only file written inside the repository is this report. Helper scripts were written outside the repository, in `C:\Users\Dell\AppData\Local\Temp\opencode\`.
* No row was inserted, updated or deleted; no ingestion was run against the live database; `POST /api/db/clear`, `/api/db/init`, `/api/db/reload`, `/api/db/upload-csv`, `/api/db/ingest-default` were never called.
* All in-memory engine builds pass `use_disk_cache=False` (`src/recommender.py:55`, `src/retrieval.py:207`, `src/retrieval.py:222-229`), so no `.npz` embedding cache file was created.
* Database reads used a read-only session (`conn.set_session(readonly=True)`).
* One side effect to disclose: starting the API through FastAPI's test client executes the `lifespan` hook (`api/server.py:38-55`), which calls `ensure_schema()` (`db/connection.py`) and `fail_orphaned_jobs()` (`db/repository.py`). `ensure_schema()` issues only idempotent `CREATE … IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS` statements, all of which are no-ops against the current schema (verified: all 23 article columns and 12 recommendation columns already exist). `fail_orphaned_jobs` updates rows only for jobs left in `pending`/`processing`; all three jobs are `completed`, so it changed nothing (job table verified before and after: 3 rows, all `completed`).

**Measured environment**

```
python 3.13.3
platform Windows-11-10.0.26300-SP0
cpu Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
cores 4 threads 8
ram_gb 17.05
torch 2.14.0+cpu cuda False
```

Installed versions (measured with `importlib.metadata`): pandas 2.3.3, numpy 2.4.2, scikit-learn
1.8.0, sentence-transformers 6.0.1, transformers 5.17.0, torch 2.14.0 (CPU build), fastapi 0.142.2,
uvicorn 0.49.0, starlette 1.7.0, pydantic 2.13.4, sqlalchemy 2.0.50, psycopg2-binary 2.9.13,
pgvector 0.5.1, python-dotenv 1.2.2, python-multipart 0.0.32, httpx 0.28.1, psutil 7.0.0.

**Plain-language explanation**

"Read-only" means the evidence was gathered by looking and measuring, not by changing anything. This
is why some experiments (for example "what would happen if we re-ingested everything with different
settings") are described but not executed: they would have rewritten the production tables.

## 1.4 Executive summary — the ten numbers that matter

| # | Fact | Value | Where it comes from |
|---|---|---|---|
| 1 | Articles in the live database | **2,000** at snapshot A (12:50); **2,446** at 13:18 — the DB grew during the audit (§5.6) | `select count(*) from articles` |
| 2 | Precomputed recommendation rows | **20,000** at snapshot A — exactly 10 per article, 0 sources with a different count (23,100 / 2,285 sources at 13:18) | `select … group by article_id having count(*) <> 10` → 0 |
| 3 | Articles with a stored embedding | **2,000 / 2,000 (100 %)** | `where embedding is not null` |
| 4 | Distinct sections in the corpus | **37** | `count(distinct section)` |
| 5 | Publication date range | **2026-09-26 → 2026-10-08** (UTC+5:30) | `min(date), max(date)` |
| 6 | Articles carrying editorial "Also Read" ground truth | **197** | `linked_ids not in ('[]','')` |
| 7 | Stored pipeline fingerprint vs current code fingerprint | **identical** (`6ae306216d00e08c`) | compared to `config.get_pipeline_version()` |
| 8 | Precomputed answer latency | **45.5 ms** from the running server with the GPU worker live; 23.9 ms in-process on CPU | §13.2, §13.1 |
| 9 | Live recompute latency, same machine | **≈1.0 s** with the GPU worker; **13.4–14.7 s** on CPU only | §13.2, §13.1 |
| 10 | Stored vs live top-10 agreement | **8.10 of 10** on average; identical list for only **2 of 10** sources; identical rank-1 for **10 of 10** | §18 H1 |

**Plain-language explanation**

The system is complete and internally consistent in its arithmetic (fingerprints match, every
article is embedded, every article has exactly ten stored recommendations). Its weak point is not
correctness of the maths but *latency of the live path*: roughly twenty times slower than the
precomputed path even with the GPU worker (a thousand times slower on CPU), which is why the fast
path exists. Two structural caveats are documented later: the precomputed table and the live engine
disagree about two of every ten articles, and a stored list is a snapshot of the corpus as it was
when that article was ingested (§18 H1). Several capabilities the documentation implies do not
exist in the code at all (§19).

## 1.5 Three score fields that are easy to confuse

**Facts**

| Field | Meaning in code | Written at |
|---|---|---|
| `score` | Calibrated confidence, `0.01–0.99`, produced by Platt scaling **plus** a signal-agreement bonus. This is what the API returns as `score`. | `src/recommender.py:307`, `src/calibration.py:47-55` |
| `boosted_score` | Relevance **after** metadata boosts, **before** calibration. This is the number the ordering actually used. | `src/recommender.py:313`, `db/pipeline.py:133` |
| `rerank_score` | Raw cross-encoder probability for the (source, candidate) pair, `0–1`. | `src/recommender.py:277`, `db/pipeline.py:134` |
| `unboosted_confidence` (inside evidence JSON only) | Platt probability of the *unboosted* relevance **without** the agreement bonus — therefore always ≤ `score`. | `db/pipeline.py:144`, `src/evidence.py:677` |

**Plain-language explanation**

Think of three separate questions. `rerank_score` = "how well does the neural model think these two
articles match?" `boosted_score` = "after the newsroom's editorial preferences (same section, same
sub-section, popularity, freshness, signal agreement) are applied, how high did this article climb?"
`score` = "converted into a human-readable confidence percentage, how sure are we?" The ordering on
the page is decided by `boosted_score`; the badge shown to the editor is `score`. They can move in
opposite directions — visible in the stored rows of the traced article in §6.12(h), where rank 4
has `score = 0.3175` but rank 5 has `score = 0.3405`.

---

# 2. Running the system

## 2.1 Commands that exist

**Facts**

| Action | Command | Source |
|---|---|---|
| Start the API + UI | `python api/server.py` → listens on `0.0.0.0:5000` | `api/server.py:529-531` |
| Ingest a CSV through the UI/API | `POST /api/db/upload-csv` (multipart, ≤ 50 MB, `.csv` only) | `api/editor_routes.py:81-120` |
| Ingest the bundled default dataset | `POST /api/db/ingest-default` | `api/editor_routes.py:121-146` |
| Force a full re-scoring of stored rows | `POST /api/db/reload` | `api/editor_routes.py:211` |
| Run the evaluation suite | `python evaluate.py --source postgres --samples 20 --skip-methods` | `evaluate.py:1001-1032` |
| Light mode (no model downloads) | `set NEWSREC_NEURAL=0` before starting | `config.py:39` |

**Plain-language explanation**

There is exactly one way to run the product: start `api/server.py`, which also serves the web pages.
Ingestion is triggered from the browser (or the same HTTP endpoint by script) and runs in a
background thread so the page stays responsive. Evaluation is a separate command-line tool used by
the team, not by readers.

## 2.2 What happens in the first 55 seconds after launch

**Facts**

Order of operations in `lifespan` (`api/server.py:38-55`):

1. `ensure_schema()` — creates the `vector` extension and adds any missing columns/indexes. Wrapped in `try/except`, so a database problem does not stop the app (`api/server.py:40-43`).
2. `ArticleRepository.fail_orphaned_jobs(db)` — marks ingestion jobs left in `pending`/`processing` as `failed` (`api/server.py:45-51`).
3. `reload_engine_from_postgres()` — loads **all** article rows plus their 768-dimension vectors into memory and builds the live engine (`api/server.py:54`, `api/server.py:81-121`).

Measured with the FastAPI test client on this machine (real output, trimmed):

```
[ModelRegistry] Checking remote cloud GPU worker at https://<redacted>.trycloudflare.com...
[ModelRegistry] [!] Remote worker at https://<redacted>.trycloudflare.com not responding. Falling back to local model.
[READY] Engine initialized from PostgreSQL: 2000 articles.
import api.server      :  52915.9 ms
lifespan (startup)     :  54694.0 ms   (ensure_schema + fail_orphaned_jobs + reload_engine_from_postgres)
RSS after startup      :   2621.2 MB
articles loaded        : 2000   engine ready: True
```

**Plain-language explanation**

The service cold-starts slowly — about 55 seconds and 2.6 GB of RAM — because it must load two
neural models from disk (419 MB embedder + 1,402 MB reranker, measured in the Hugging Face cache)
and copy every article vector into memory. Anyone restarting the server during a busy period should
expect roughly a minute of unavailability.

## 2.3 Ingestion endpoints — one of them cannot work

**Facts**

* `POST /api/db/ingest-default` reads `config.DATA_PATHS` and returns HTTP 404 when the file is missing (`api/editor_routes.py:121-146`).
* `config.DATA_PATHS` resolves to `data/articles.csv` when no candidate exists (`config.py:18-25`); measured: the `data/` directory contains **no** `articles.csv`, no `articles_2.csv` and no `articles_clean_train_80.csv`.
* Therefore `/api/db/ingest-default` currently returns 404 and `evaluate.py --source csv` raises `FileNotFoundError` (`src/data_loader.py:625` default path).

**Plain-language explanation**

The "load the demo dataset" button is broken because the demo file is not in the repository. This is
not a logic bug — it is a missing data file. Uploading a CSV works.

---

# 3. Configuration and environment

## 3.1 Environment variables (names only — no secret values are reproduced)

**Facts**

| Variable | Purpose | Read at | Required | Default if unset | Present in `.env` |
|---|---|---|---|---|---|
| `POSTGRES_USER` | database user | `db/connection.py` | yes (for DB features) | `postgres` | **yes** |
| `POSTGRES_PASSWORD` | database password | `db/connection.py` | yes | empty | **yes** |
| `POSTGRES_HOST` | database host | `db/connection.py` | yes | `localhost` | **yes** |
| `POSTGRES_PORT` | database port | `db/connection.py` | yes | `5432` | **yes** |
| `POSTGRES_DB` | database name | `db/connection.py` | yes | `news_recommender` | **yes** |
| `REMOTE_MODEL_URL` | optional cloud-GPU worker URL | `config.py:51` | no | `""` (run locally) | **yes** (61 characters) |
| `REMOTE_MODEL_TOKEN` | optional shared secret for that worker | `config.py:54` | no | `""` | **no** |
| `HF_TOKEN` | Hugging Face token (rate limits / private models) | not read by code; listed in `.env.example` | no | `""` | **no** |
| `NEWSREC_NEURAL` | `0` = light TF-IDF mode, no model downloads | `config.py:39` | no | `1` | **no** |
| `NEWSREC_EMBED_REVISION` | override pinned embedder commit | `config.py:41` | no | `a5beb1e3…` | **no** |
| `NEWSREC_RERANKER` | override reranker model id | `config.py:42` | no | `BAAI/bge-reranker-base` | **no** |
| `NEWSREC_RERANK_REVISION` | override pinned reranker commit | `config.py:43` | no | `2cfc18c9…` | **no** |
| `DATA_CSV` | comma-separated CSV paths to load | `config.py:14-16` | no | auto-detected | **no** |

**Plain-language explanation**

Six settings are needed to talk to the database and they are all present. The one optional setting
that *is* present — `REMOTE_MODEL_URL` — points at a cloud GPU worker that no longer exists.
Verified live:

```
REMOTE_MODEL_URL set: True length 61
URLError <urlopen error [Errno 11001] getaddrinfo failed>
elapsed 0.32s
```

Because `src/model_registry.py:150-158` only performs a 4-second `/health` ping and then silently
falls back to the local model (`src/model_registry.py:168-186`, `src/model_registry.py:195-209`),
the dead URL is harmless but invisible in the UI — except for two console lines at startup. This
matters for capacity planning: every measurement in this report is CPU-only.

## 3.2 Tunable parameters (single source of truth: `config.py`)

**Facts — retrieval and ranking**

| Constant | Value | Line | Effect of changing it |
|---|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | `config.py:40` | changes vector geometry; all stored embeddings become invalid |
| `EMBEDDING_REVISION` | pinned SHA | `config.py:41` | same as above; also changes the pipeline fingerprint |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | `config.py:42` | changes every `rerank_score` |
| `RERANKER_REVISION` | pinned SHA | `config.py:43` | same |
| `EMBED_MAX_SEQ_LEN` | `256` | `config.py:44` | how much article text the embedder reads |
| `RERANK_MAX_LEN` | `512` | `config.py:45` | total tokens per (query, passage) pair |
| `RERANK_PER_ARTICLE_TOKENS` | `256` | `config.py:46` | intended per-side share |
| `EMBED_TOKEN_BUDGET` | `254` | `config.py:76` | derived: 256 − 2 special tokens |
| `RERANK_TOKEN_BUDGET` | `254` | `config.py:80` | derived: (512 // 2) − 2 |
| `N_RETRIEVE` | `50` | `config.py:89` | candidates taken from each of BM25 and dense |
| `N_CANDIDATES` | `20` | `config.py:90` | candidates that reach the cross-encoder |
| `RRF_K` | `60` | `config.py:91` | fusion smoothing; larger = flatter blend of the two lists |
| `TOP_K` | `10` | `config.py:92` | default and maximum fast-path length |
| `WEIGHT_RERANK` | `0.55` | `config.py:101` | weight of the cross-encoder term |
| `WEIGHT_BM25` | `0.15` | `config.py:102` | weight of the keyword term |
| `WEIGHT_DENSE` | `0.30` | `config.py:103` | weight of the semantic term |
| `MMR_LAMBDA` | `0.91` | `config.py:109` | 1.0 = pure relevance, lower = more variety |
| `MMR_THETA` | `0.76` | `config.py:110` | similarity above which a candidate is penalised as a near-duplicate |
| `MMR_LAMBDA_SEARCH` | `0.88` | `config.py:115` | same, for free-text search |
| `MMR_THETA_SEARCH` | `0.85` | `config.py:116` | same, for free-text search |
| `MIN_RAW_RERANK_THRESHOLD` | `0.12` | `config.py:258` | below this top raw score the answer is flagged `low_confidence` |
| `FALLBACK_MIN_RELATED` | `3` | `config.py:269` | fewer related items than this triggers the separate "Trending in …" list |
| `FALLBACK_LIMIT` | `5` | `config.py:270` | cap on that fallback list |

**Facts — metadata boosts**

| Constant | Value | Line | Effect |
|---|---|---|---|
| `SECTION_BOOST_MULT` | `1.100` | `config.py:123` | +10 % when candidate section == source section |
| `SUBSECTION_BOOST_MULT` | `1.050` | `config.py:124` | +5 % when sub-section matches |
| `POPULARITY_BOOST_MULT` | `1.015` | `config.py:125` | up to +1.5 %, log-scaled on page views |
| `RECENCY_MAX_BOOST_MULT` | `1.040` | `config.py:126` | up to +4 %, exponential decay by section half-life |
| `BOOST_MODE` | `source_date_recency` | `config.py:199` | freshness measured against the **source article's** date, not the newest row |
| `SECTION_HALF_LIVES` | 40+ entries, e.g. `live_blog 0.25`, `sports 1.5`, `cities 2.5`, `entertainment 3.0`, `explained 7.0`, `books-and-literature 10.0`, `default 3.0` | `config.py:139-197` | decay speed of the recency boost per section |
| editorial boost | `1.40` **hard-coded**, not configurable | `src/scoring.py:269` | +40 % for a candidate that is in the source's editorial "Also Read" set |

**Facts — ingestion**

| Constant | Value | Line |
|---|---|---|
| `PRECOMPUTE_CANDIDATES` | `20` | `config.py:57` |
| `PRECOMPUTE_COMMIT_EVERY` | `5` | `config.py:58` |
| `INCREMENTAL_RECOMMENDATIONS` | `True` | `config.py:67` |
| `BOILERPLATE_MIN_DOCS` | `4` | `config.py:86` |
| `MAX_USER_HISTORY` | `10` | `config.py:119` |
| `USER_PROFILE_WEIGHT` | `0.35` | `config.py:120` |
| `QUERY_PREFIX` | `"Represent this sentence for searching relevant passages: "` | `config.py:70` |
| `SHORT_QUERY_WORDS` | `12` | `config.py:71` |
| `TEXT_RECIPE_VERSION` | `v2_sentence_token_budget` | `config.py:74` |

**Plain-language explanation**

Every number a manager or engineer might want to discuss lives in one file, `config.py`, instead of
being scattered through the code. That is genuinely true here: the weights (0.55 / 0.15 / 0.30), the
diversity settings (0.91 / 0.76), the four boost multipliers and 40+ section half-lives are all in
that single file, and both the batch pipeline and the live path read them from there. One
exception: the +40 % editorial-link boost is hard-coded inside `src/scoring.py:269`, so changing it
requires a code edit, not a config edit.

## 3.3 The pipeline fingerprint — and what it does not protect

**Facts**

* `config.get_pipeline_version()` (`config.py:284-335`) builds a SHA-256 over 33 tokens (`config.py:291-334`: 10 model/parameter tokens, 3 blend weights, 5 boost tokens, 2 output/scale tokens, and 13 consensus/agreement tokens) and returns the first 16 hex characters.
* Current value: `6ae306216d00e08c`. Every one of the 20,000 stored recommendation rows carries the same value (verified by SQL).
* `db/pipeline.py:321-360` compares the incoming version with the last completed job and forces a full recompute when they differ.

**Plain-language explanation**

The fingerprint is the system's defence against silently mixing results computed under different
settings: if somebody changes the blend weights, the fingerprint changes, and the next ingestion
recomputes every stored row rather than leaving half the table scored under old rules. **However**,
the fingerprint hashes only `config.py` values — it does **not** hash the source code. If somebody
edits the ranking maths in `src/scoring.py` without touching a config value, the fingerprint stays
the same and the database will happily mix old and new scoring. §6.10 shows a real consequence.

---

# 4. Data assets actually on disk and in the database

## 4.1 CSV files

**Facts** (measured by reading each file with pandas)

| File | Rows | Columns | Size | Publication date range |
|---|---|---|---|---|
| `data/articles_batches/batch_0_1.csv` | 1,000 | 27 | 4.56 MB | 2026-10-03 → 2026-10-08 |
| `data/articles_batches/articles_batch_03.csv` | 500 | 27 | 2.38 MB | 2026-09-30 → 2026-10-03 |
| `data/articles_batches/articles_batch_04.csv` | 500 | 27 | 2.40 MB | 2026-09-26 → 2026-09-30 |
| `data/articles_batches/articles_batch_05.csv` | 446 | 27 | 2.06 MB | 2009-10-02 → 2026-09-26 |
| `data/articles_batches/article_6.csv` | 600 | 33 | 2.72 MB | 2026-10-19 → 2026-10-22 |
| `data/articles_batches/article_7.csv` | 600 | 33 | 2.80 MB | 2026-10-16 → 2026-10-19 |
| `data/articles_batches/article_8.csv` | 531 | 33 | 2.49 MB | 2004-07-29 → 2026-10-16 |
| **Total on disk** | **4,177** | — | **19.4 MB** | — |

The 27-column header (first-generation export) is:

```
_id,articleId,articleName,articleTitle,articleUrl,sectionName,isProcessed,section1Slug,section2Slug,
section2Name,sectionsName,articleFeaturedImage,articleFeaturedImageAlt,duration,episode,postPublishDate,
pageViews,uniqueUsers,postType,algorithm,description,body_raw,premium,liveBlog,createdAt,updatedAt,__v
```

The newer `article_6/7/8.csv` files carry 33 columns — six extra analytics columns
(`activityTime`, `deletedAt`, `lastActivityDate`, `pageReferrer`, `pageType`, `sessionId`).

Column names are matched flexibly rather than positionally: ID from
`articleId | id | article_id | ID` (`src/data_loader.py:315`), date from
`postPublishDate | date | published_at | publishDate` (`src/data_loader.py:316`), title from
`articleTitle | title | headline | articleName` (`src/data_loader.py:317`).

**Plain-language explanation**

There is more news on disk than in the system. The database holds 2,000 articles — exactly the
1,000 + 500 + 500 of the three oldest files. A further **1,731 articles dated up to 2026-10-22 sit
uningested**, and they are 15 days newer than the newest article in the live database
(2026-10-08). Any demonstration of "how the system handles this week's news" is therefore currently
impossible without an ingestion run.

## 4.2 Other data assets

**Facts**

| Asset | Status |
|---|---|
| `data/cache/` | **empty** — no embedding cache file exists |
| `data/uploads/` | 17 staged ingestion copies, 53.8 MB total (one per upload attempt) |
| Hugging Face model cache (`~/.cache/huggingface/hub`) | 2.37 GB total; `models--BAAI--bge-base-en-v1.5` 419 MB, `models--BAAI--bge-reranker-base` 1,402 MB |
| `baseline_evaluation.json`, `tuning_results.csv`, `ablation_results.csv`, `blind_labeling_sample.csv`, `tests/test_integrity.py`, `run_experiments.py`, `freeze_pool.py`, `parity_test.py`, `cloud_gpu_worker.ipynb` | **deleted from the working tree** (`git status` shows them as ` D`). Only `kaggle_gpu_worker.ipynb` remains, untracked. |
| `data/articles.csv`, `data/articles_clean_train_80.csv` | **NOT FOUND** |

**Plain-language explanation**

The evidence trail from earlier tuning work — the baseline metrics file, the 200-configuration
tuning results, the ablation table, the human-labelling sample and the unit-test module — has been
deleted from the working copy. The claims still exist in `AGENTS.md`, but the artefacts behind them
no longer do, so they cannot be re-checked without re-running the experiments from scratch. That is
the single biggest documentation risk in this handover.

---

# 5. Database: schema, indexes and verified state

## 5.1 Server and extensions

**Facts** (read-only SQL)

```
PostgreSQL 18.6 on x86_64-windows, compiled by msvc-19.44.35229, 64-bit
[('plpgsql', '1.0'), ('vector', '0.8.7')]
```

## 5.2 Tables

**Facts**

`articles` — 2,000 rows, 23 columns:

| Column | Type | Null? |
|---|---|---|
| `id` | varchar(100) | NO (PK) |
| `title` | text | NO |
| `description` | text | YES |
| `body` | text | YES |
| `section` | varchar(100) | YES |
| `subsection` | varchar(100) | YES |
| `date` | timestamptz | YES |
| `url` | text | YES |
| `image` | text | YES |
| `is_live` | boolean | YES |
| `page_views` | integer | YES |
| `unique_users` | integer | YES |
| `linked_ids` | text (JSON array as text) | YES |
| `text_embed` | text | YES |
| `embedding` | `vector` (768-d) | YES |
| `created_at` / `updated_at` | timestamp | YES |
| `text_bm25` | text | YES |
| `text_rerank` | text | YES |
| `embed_token_count` | integer | YES |
| `rerank_token_count` | integer | YES |
| `is_truncated` | boolean | YES |
| `recipe_version` | varchar(32) | YES |

`article_recommendations` — 20,000 rows, 12 columns: `id` (serial PK), `article_id` (FK →
`articles.id`, `ON DELETE CASCADE`), `recommended_article_id` (FK → `articles.id`, `ON DELETE
CASCADE`), `rank` (1–10), `score` (double), `method`, `created_at`, `reason` (JSON text), `rerank_score`
(double), `pipeline_version`, `rank_raw` (integer), `boosted_score` (double).

`ingestion_jobs` — 3 rows, 11 columns: `id`, `filename`, `status`, `total_articles`,
`processed_articles`, `message`, `error`, `created_at`, `updated_at`, `refreshed_articles`,
`pipeline_version`.

## 5.3 Indexes — and the one that is missing

**Facts**

```
articles            | articles_pkey                        | UNIQUE btree (id)
articles            | ix_articles_id                       | btree (id)
articles            | ix_articles_date                     | btree (date)
articles            | ix_articles_section                  | btree (section)
articles            | ix_articles_section_date             | btree (section, date)
article_recommendations | article_recommendations_pkey     | UNIQUE btree (id)
article_recommendations | uq_source_article_rank           | UNIQUE btree (article_id, rank)
article_recommendations | idx_rec_source_rank              | btree (article_id, rank)
article_recommendations | ix_…_article_id                  | btree (article_id)
article_recommendations | ix_…_recommended_article_id      | btree (recommended_article_id)
ingestion_jobs      | ingestion_jobs_pkey                  | UNIQUE btree (id)
```

There is **no** index on `articles.embedding` — no `ivfflat`, no `hnsw`, no vector operator class
anywhere in the schema. Every index is a plain B-tree.

**Plain-language explanation**

The `vector` extension is installed and every article has a 768-number embedding stored, so the
database *could* do nearest-neighbour search. It does not: similarity search is done in the
application's memory with a NumPy matrix multiply (`src/retrieval.py:323-324` — `self.vectors @
self.vectors[pos]`), which at 2,000 articles is faster than starting a database scan. The pgvector
column is effectively a storage container, not a search index. This is a deliberate and reasonable
choice at this corpus size; it becomes the limiting factor somewhere in the tens of thousands of
articles, when the whole matrix no longer fits comfortably in RAM alongside the models.

## 5.4 Integrity checks actually run

**Facts** (read-only SQL, output quoted verbatim; snapshot taken before 12:50, when the corpus
held 2,000 articles — see §5.6 for the change that happened during this audit)

```
embedded: 2000
distinct sections: 37
date range: (2026-09-26 07:28:43+05:30, 2026-10-08 13:42:56+05:30)
max embed/rerank tokens: (254, 254)
truncated: 1987
recipe versions: [('v2_sentence_token_budget', 2000)]
linked_ids non-empty: 197
sources without exactly 10 recs: 0
distinct recommended ids: 1986
rec pipeline versions: [('6ae306216d00e08c', 20000)]
score stats: (0.1192, 0.9496, 0.7598)
boosted_score stats: (0.1053, 1.5741, 0.8415)
rerank_score stats: (0.0000, 1.0000, 0.7673)
page_views min/max: (117, 60438)
rank distribution: [(1, 2000), (2, 2000), … (10, 2000)]
```

Jobs:

```
job_788b7dcb | batch_0_1.csv        | completed | 1000/1000 | refreshed 0
job_743cff70 | articles_batch_03.csv| completed |  500/500  | refreshed 981
job_f489a027 | articles_batch_04.csv| completed |  500/500  | refreshed 1403
```

**Plain-language explanation**

Three things are worth noticing. First, the text budgeting was actually applied: no article exceeds
254 tokens for either model, which is the configured per-side limit. Second, 1,987 of 2,000 articles
(99.4 %) were truncated to fit — the models only ever see the headline, the summary and the first
few paragraphs. Third, and most striking: the stored **`boosted_score` reaches 1.5741**, i.e. above
1.0, while the calibrated `score` never exceeds 0.99. The boosted score is a relevance number
multiplied by up to ~1.96 of boosts, so it is not a probability and must never be displayed as one.

## 5.5 How schema changes are applied

**Facts**

* There is no migration tool. The schema is created by SQLAlchemy `Base.metadata.create_all` and then patched by `ensure_schema()` in `db/connection.py`, which is driven by two declarative lists (`_ADDED_COLUMNS`, `_ADDED_INDEXES`) and runs idempotent `ADD COLUMN IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`.
* `ensure_schema()` is called at startup (`api/server.py:41`) and before evaluation.

**Plain-language explanation**

Adding a column means adding one tuple to a list, and the change applies itself on the next start —
convenient, and safe because every statement is a no-op when already applied. The trade-off is that
there is no versioned history of schema changes and no rollback path; a bad patch has to be undone
by hand.

## 5.6 The database changed while this report was being written

This is not a defect in the code; it is a fact about the environment that any reader of the
numbers in §4–§13 must know. During the audit a **fourth ingestion job was running**:

```
job_3c2585d8 | articles_batch_05.csv | processing | 348/446 | refreshed 0
  created 2026-10-28 13:12:37   updated 13:18:45
  message: STAGE:RERANK|Precomputed 310/446 (1.1s/article, ETA 44m 20s (incl. ~41m 52s reverse))
```

What that means for the numbers in this report:

| Snapshot | Time | `articles` | `article_recommendations` |
|---|---|---|---|
| A — the state used by §4, §5.2–5.4, §12 (evaluate), §13 | before 12:50 | 2,000 | 20,000 |
| B — observed at the end of the audit | 13:18 | **2,446** | **23,100** (and rising) |
| C — distinct source articles with recommendations at 13:17 | 13:17 | — | 2,285 sources |

Every quantitative statement in this report is therefore labelled by the snapshot it came
from; they are each internally consistent but they do not all describe the same instant.
Two further consequences worth stating plainly:

1. **The API was serving a stale corpus while the DB grew.** The in-process engine was loaded
   from PostgreSQL at startup (12:4x) and only reloads when ingestion finishes
   (`reload_engine_from_postgres`, `api/server.py:81`). So a request during this window reads
   the 2,000-article index while 446 new articles are being written. This is the single-process
   behaviour of M5 and the same mechanism behind H1, observed live rather than inferred.
2. **A benchmark delta mixes two causes.** Comparing a metric measured now with one measured
   before 13:12 changes both the corpus size and (potentially) the code. Any future benchmark
   must record `count(*) from articles`, the active job id and `pipeline_version` alongside the
   metric — otherwise the delta is not attributable.

---

# 6. The Recommendation Pipeline, Stage by Stage

This is the core of the system. Everything after this section is bookkeeping
(how it is served, how it is measured, what is broken); this section is *how a
recommendation is actually produced*.

## 6.0 Stage and signal inventory

Every article that enters the system passes through the stages below, in this
order. "Can be toggled" records whether a caller or operator can switch the stage
off at request time, because that is what makes the ON/OFF measurement in
sub-section (h) possible at all.

| # | Stage / signal | What it decides | Runs at | Code | Toggleable? |
|---|---|---|---|---|---|
| 6.1 | CSV intake | Which rows are accepted into the system | ingestion only | `api/editor_routes.py:81-117`, `db/pipeline.py:362-370` | no |
| 6.2 | Clean, dedup, Also-Read extraction | One canonical row per article + editorial ground truth | ingestion only | `src/data_loader.py:450-618` | no |
| 6.3 | Text recipes and token budgets | Exactly which characters reach each model | ingestion only | `src/data_loader.py:560-602` | no |
| 6.4 | Embedding (`bge-base-en-v1.5`, 768-d) | The article's position in semantic space | ingestion (+ live rebuild) | `src/retrieval.py:DenseIndex`, `src/model_registry.py` | `NEWSREC_NEURAL=0` → TF-IDF |
| 6.5 | BM25 lexical retrieval | "Which articles share my distinctive words" | both | `src/retrieval.py:BM25Index`, `db/pipeline.py:43-53` | `method=bm25` |
| 6.6 | Dense retrieval | "Which articles are semantically close" | both | `src/retrieval.py:DenseIndex.scores_for_doc` | `method=semantic` |
| 6.7 | Reciprocal Rank Fusion | Merge the two lists into 20 candidates | both | `src/retrieval.py:reciprocal_rank_fusion` | implicit in `method=hybrid*` |
| 6.8 | Cross-encoder rerank | Precise pairwise relevance probability | both | `src/reranker.py:Reranker.score_pairs` | `use_rerank` via `method=hybrid` |
| 6.9 | Relevance blend | One 0–1 relevance number from three signals | both | `src/scoring.py:91-105` | weights are constants |
| 6.10a | Editorial (`Also Read`) boost | ×1.40 for an editor-linked article | both | `src/scoring.py:268-269` | `use_boosts` |
| 6.10b | Section boost | ×1.100 same section | both | `src/scoring.py:270-271` | `use_boosts` |
| 6.10c | Subsection boost | ×1.050 same subsection | both | `src/scoring.py:272-273` | `use_boosts` |
| 6.10d | Popularity boost | ×1.000–1.015, log-scaled page views | both | `src/scoring.py:276-281` | `use_boosts` |
| 6.10e | Recency boost | ×1.000–1.040, section half-life decay | both | `src/scoring.py:285-290`, `config.py:223-241` | `use_boosts` |
| 6.10f | Signal-consensus multiplier | ×0.30–×1.15 from three-signal agreement | both | `src/scoring.py:133-246` | `use_boosts` |
| 6.11 | MMR diversity | Demote near-duplicates among the top 10 | both | `src/scoring.py:308-325` | `diversify` |
| 6.12 | Platt calibration | Turn relevance into a stated confidence | both | `src/calibration.py:40-97` | no (always on) |
| 6.13 | Evidence JSON | The "why" record stored with each row | both | `src/evidence.py:build_evidence_reason` | no (always on) |
| 6.14 | Persistence + reverse push | Store 10 rows; update old articles that a new one is close to | ingestion only | `db/pipeline.py:215-320` | `recompute_all_recommendations` |
| 6.15 | Serving (fast path vs live) | Which of the two paths answers a request | request time | `api/server.py:370-377` | request shape decides |

Two stages that a reader of the older documentation would expect are **NOT
FOUND** in the current code: there is no relevance *admission gate* (no
`admission_cut`, no `RELEVANCE_FLOOR`, no elbow rule — `src/recommender.py:214`
carries the comment `Lower Filter: relevance admission (techniques 1 + 2)` and
then performs no filtering), and there is no personalisation *model* beyond the
single centroid blend described in §9.

---

## 6.1 CSV intake (parse and validate)

**(a) Plain-language explanation.** An editor drags a CSV of stories into the tool.
This stage decides whether the file is allowed in at all, then reads it into a
table. It is the only stage where a human-supplied file touches the machine, so
it is also the only stage with a size limit.

**(b) Code location.** Upload and validation: `api/editor_routes.py:81-117`.
Reading: `db/pipeline.py:362-370`. Job creation and thread launch:
`api/editor_routes.py:39-49`.

**(c) Inputs and outputs.**
* In: one multipart file upload; the CSV must end in `.csv`.
* Out: `data/uploads/job_<8 hex>_<name>.csv` on disk, one `ingestion_jobs` row
  with `status='processing'`, and — inside the background thread — a pandas
  `DataFrame` (`raw_df`) plus a printed validation summary.

**(d) Algorithm, step by step.**
1. `_require_idle(db)` — if any job is `pending` or `processing`, return HTTP 409
   *before* a single byte is written (`api/editor_routes.py:29-36`, called at `:90`).
2. `Path(file.filename).name` strips any directory component from the client-supplied
   name (`:92`), then the `.csv` suffix is enforced (`:93-94`, else HTTP 400).
3. The body is streamed to disk in 1 MB chunks (`:102`); the running total is
   compared with `MAX_FILE_SIZE` and the partial file is deleted on breach
   (`:104-107`, HTTP 413).
4. `_launch()` creates the job row and starts a daemon thread running
   `run_ingestion_pipeline` (`:39-49`).
5. The pipeline calls `pd.read_csv(csv_source)` with no dtype, encoding or
   `usecols` arguments (`db/pipeline.py:364`), then optionally truncates to
   `limit` rows (`:365-366`).

**(e) Exact values and constants.**

| Constant | Value | Where | Effect of changing |
|---|---|---|---|
| `MAX_FILE_SIZE` | `50 * 1024 * 1024` = 52,428,800 bytes | `api/editor_routes.py:26` | Larger accepts bigger batches but increases peak memory, because the whole frame is read into RAM and then cleaned into copies |
| chunk size | `1024 * 1024` = 1 MB | `api/editor_routes.py:102` | Only affects upload smoothness, not results |
| allowed suffix | `".csv"` | `api/editor_routes.py:93` | The only content check performed |
| `limit` | `Optional[int]`, default `None` | `db/pipeline.py:324-325` | Truncates to the first N rows — used for demos, and it silently biases the corpus if left set |

**(f) Worked example (real).** `data/articles_batches/batch_0_1.csv` is
4.56 MB, 1000 rows × 27 columns. It was accepted and produced job
`job_788b7dcb`, whose database row records `total_articles = 1000` and
`processed_articles = 1000` (this row was read directly from `ingestion_jobs`, see
§5). The next two jobs, `job_743cff70` (`articles_batch_03.csv`) and
`job_f489a027` (`articles_batch_04.csv`), each record 500.

**(g) Edge cases.**
* A file whose name does not end in `.csv` is rejected before writing (`:93`).
* A file larger than 50 MB is rejected mid-write and the partial file is unlinked
  (`:105-107`).
* A second upload while a job runs is rejected with 409 (`:29-36`).
* A CSV with a completely different column set is **not** rejected here; it fails
  or degrades later in 6.2, because `_col(..., default="")` returns empty strings
  for absent columns (`src/data_loader.py:320`).
* `limit` is applied after `read_csv`, so memory use is unaffected by it.

**(h) Measured ON vs OFF.** This stage cannot be toggled — there is no ingestion
without a file. Its measurable effect is on the corpus: the four CSVs currently
on disk contain 4,177 rows, and the database holds 2,000, so 2,177 rows have never
passed through this stage (§4).

**(i) Limitations.**
* No schema validation: a CSV with no title column yields empty titles rather
  than an error (`src/data_loader.py:320`, `:506`).
* `pd.read_csv` uses pandas defaults — `utf-8`, comma separator, `,` decimal, no
  `encoding_errors` handling — so a Latin-1 or semicolon-delimited export will
  raise or mis-parse.
* No content-type or magic-byte check; the `.csv` suffix is the entire guard.
* There is no dry-run mode: the only way to learn whether a file is acceptable is
  to ingest it.

---

## 6.2 Cleaning, deduplication and Also-Read extraction

**(a) Plain-language explanation.** Real newsroom exports are messy: the same
story appears twice (once in the morning, once updated), bodies contain leftover
"ALSO READ: …" promo blocks, and many articles share identical legal or
subscription sentences. This stage produces exactly one clean row per article, and
in doing so it extracts the editor's cross-links, which are the only human
ground truth the system will ever have.

**(b) Code location.** `clean_dataframe` — `src/data_loader.py:450-618`.
Deduplication — `:380-443`. Also-Read stripping — `:70-86`. Boilerplate detection —
`:89-98`. Column aliases — `:315-317`. Normalisation — `:44-68`.

**(c) Inputs and outputs.**
* In: the raw frame from 6.1.
* Out: a frame with snake_case columns `id, title, description, section,
  subsection, date, url, image, is_live, body, linked_ids, page_views,
  unique_users, text_embed, text_bm25, text_rerank, embed_token_count,
  rerank_token_count, is_truncated, recipe_version` (`:464-468`), plus a printed
  dedup summary (`:493-503`).

**(d) Algorithm, step by step.**
1. Resolve the ID column from `["articleId", "id", "article_id", "ID"]`
   (`:315`); if none exists, mint synthetic IDs `art_<md5(time)[:6]>_<n>` (`:478-481`).
2. Deduplicate by ID, keeping the row with the **latest** `postPublishDate`
   (`:380-443`); count rows whose text conflicted.
3. `normalise_text` every field: NFKC, smart quotes to ASCII, en/em dash to `-`,
   whitespace collapse (`:44-68`).
4. Build a headline index (`_headline_index`, `:517`) and, for each body, cut out
   every "ALSO READ `<headline>`" block whose headline matches a known article;
   the matched IDs become `linked_ids` (`:70-86`, `:519-523`).
5. Find sentences longer than 25 characters that appear in ≥ `BOILERPLATE_MIN_DOCS`
   articles and delete them from every body (`:89-98`, `:526-527`).
6. Fill `description` from the body's first 200 characters when empty (`:511-514`).
7. Resolve remaining metadata columns flexibly and coerce types.

**(e) Exact values and constants.**

| Constant | Value | Where | Effect of changing |
|---|---|---|---|
| `BOILERPLATE_MIN_DOCS` | `4` | `config.py:86` | Lower = more aggressive stripping; at 2–3 ordinary recurring phrases start disappearing from legitimate stories |
| dedup key | article ID | `src/data_loader.py:380-443` | Changing to content-hash would merge syndicated copies that have different IDs |
| dedup tie-break | latest `postPublishDate`, then last in file | `:380-443` | Determines which body text survives an update |
| Also-Read key length | 4 words (`ALSO_READ_KEY_WORDS`) | `src/data_loader.py` (`strip_also_read`) | Shorter keys over-match common headlines and delete real sentences |
| boilerplate sentence floor | > 25 characters | `:89-98` | Very short repeated lines are left in place |
| description fallback | first 200 chars of body | `:513` | Changes what the reranker sees for description-less rows |

**(f) Worked example (real).** Source article **10905842** —
*"'So we don't end up in jail': Kapil Sharma on why he watches younger comics"* —
survived cleaning as one row, section `entertainment`, subsection `Television`,
published `2026-10-04 10:38:01+05:30`, `page_views = 494`. Its cleaned body is
what produced the 23-term BM25 query in §6.5. Across the live corpus, 197 of 2,000
articles carry at least one `linked_ids` entry (read from the database, §5); those
197 are the articles for which §12 can compute ground-truth precision.

**(g) Edge cases.**
* No ID column → synthetic IDs are *not stable across runs* (they hash `time.time()`,
  `:480`), so re-uploading the same file creates duplicates instead of updates.
* Empty body → description stays empty; the article still enters the corpus and is
  embedded from its title alone.
* NaT/`None` dates survive into the DB as `NULL` and are excluded from
  `max(date)` recency arithmetic later (`config.py:202-209`).
* A body whose Also-Read headline does not exactly match any article's headline
  index is left untouched, so some promo text remains (`:70-86`).

**(h) Measured ON vs OFF.** Not toggleable at request time. The observable effect
is in the stored corpus: all 2,000 rows carry `recipe_version =
v2_sentence_token_budget`, 1,987 of 2,000 are marked truncated (i.e. cleaning fed
more text than the budget allows), and 197 rows carry editor links.

**(i) Limitations.**
* Deduplication is by ID only. Two rows for the same story with different IDs both
  survive and will later compete as near-duplicates; MMR (§6.11) is the only
  defence, and it only demotes, never merges.
* Synthetic IDs make re-upload non-idempotent (see edge cases).
* Boilerplate detection is corpus-local: ingesting a small batch strips less than
  ingesting the whole archive, so the same article can end up with different body
  text depending on batch order.

---

## 6.3 Text recipes and token budgets

**(a) Plain-language explanation.** Each model has a hard limit on how many word
pieces it can read. Rather than chopping text mid-sentence (which produces garbage
fragments), this stage builds three separate strings per article — one for the
embedding model, one for keyword search, one for the reranker — each cut at a whole
sentence boundary inside its budget.

**(b) Code location.** `src/data_loader.py:560-602`; the composer itself is
`compose_sentence_budgeted_text` (`:172`) driven by `build_sentence_budgeted_texts`
(`:294`).

**(c) Inputs and outputs.**
* In: cleaned `title`, `description`, `body`.
* Out: three text columns plus `embed_token_count`, `rerank_token_count`,
  `is_truncated`, `recipe_version`.

**(d) Algorithm, step by step.**
1. Title is always taken whole (it is the strongest signal and is short).
2. Description is taken whole; if it alone exceeds the budget it is trimmed at a
   word boundary by binary search on token count.
3. Body is appended **one whole sentence at a time** until the next sentence would
   break the budget; the loop then stops. Nothing is cut mid-sentence.
4. The same procedure runs twice — once with the embedding tokenizer at budget 254,
   once with the reranker tokenizer at budget 254 (`:571-588`).
5. `text_bm25` is built separately as `f"{title}. {title}. {title}. {description}. {body}"`
   (`:595-598`) — the title is repeated three times to give entity names triple weight.
6. `is_truncated` is `True` when either the embedding or the rerank text had to stop
   early (`:591`).

**(e) Exact values and constants.**

| Constant | Value | Where | Effect of changing |
|---|---|---|---|
| `EMBED_MAX_SEQ_LEN` | `256` | `config.py:44` | Sets the embedding window; a change changes the vector, so it is inside the version fingerprint |
| `EMBED_SPECIAL_TOKENS` | `2` | `config.py:75` | `[CLS]`/`[SEP]`; `EMBED_TOKEN_BUDGET = 256 − 2 = 254` (`:76`) |
| `RERANK_MAX_LEN` | `512` | `config.py:45` | Total pair window (query + candidate) |
| `RERANK_TOKEN_BUDGET` | `(512 // 2) − 2 = 254` | `config.py:80` | Per-side content budget; 254+254+4 specials = 512 |
| title repetition for BM25 | 3× | `src/data_loader.py:595-598` | Fewer = entity names lose their lexical advantage |
| `TEXT_RECIPE_VERSION` | `"v2_sentence_token_budget"` | `config.py:74` | In the fingerprint; stored per row as `recipe_version` |

**(f) Worked example (real).** Article 10905842 has `embed_token_count = 254` and
`rerank_token_count = 254`, and `is_truncated = True` — its body was longer than
one budget and was cut at the last whole sentence that fit. Of the 2,000 stored
rows, 1,987 hit the cap (99.4%), which is expected: news bodies routinely run to
400–1,200 words. The BM25 text for the same article therefore begins
`"So we don't end up in jail... . So we don't end up in jail... . So we don't end
up in jail... . <description>. <body>"`.

**(g) Edge cases.**
* A title longer than 254 tokens is taken whole anyway (step 1) and is then clipped
  by the tokenizer at inference time — the composed string can exceed the budget by
  itself.
* A single body sentence longer than the remaining budget is skipped, not split, so
  a body made of one enormous sentence contributes nothing.
* `text_embed` and `text_rerank` are produced by the *same* function with the *same*
  budget, so they are the same string unless the two tokenizers disagree about token
  counts. Two 254-token copies per article are therefore stored (`articles.text_embed`
  and `articles.text_rerank`).
* `EMBED_TOKEN_BUDGET`/`RERANK_TOKEN_BUDGET` are derived in `config.py`, so editing
  `EMBED_MAX_SEQ_LEN` silently moves both the budget and the fingerprint.

**(h) Measured ON vs OFF.** Not toggleable. Measured consequence: `max(embed_token_count)
= max(rerank_token_count) = 254` over all 2,000 rows, i.e. the cap binds for almost
every article, and 1,987/2,000 rows are flagged truncated.

**(i) Limitations.**
* Sentence-splitting is regex-based, so abbreviations ("Dr.", "Rs.", "No.") can end
  a "sentence" early and waste budget.
* Because the cap binds for 99% of articles, the tail of every long story is
  invisible to both models; a fact stated in paragraph 20 cannot influence the
  recommendation at all.
* The stored duplicate (`text_embed` ≈ `text_rerank`) doubles the text footprint for
  no benefit.

---

## 6.4 Embedding (semantic vector)

**(a) Plain-language explanation.** Every article is converted into 768 numbers
that encode its meaning. Two articles about the same event land near each other
even when they share no words. This is what lets the system find a story about
"a comic afraid of jail" when the neighbouring story never uses the word "jail".

**(b) Code location.** `src/retrieval.py:202-339` (`DenseIndex`); model loading and
the remote-GPU switch `src/model_registry.py`; the vector column and dimension are
declared in `db/models.py` (`embedding vector(768)`).

**(c) Inputs and outputs.**
* In: `text_embed` (≤254 tokens, §6.3) for each article.
* Out: one L2-normalised 768-float vector per article, stored in
  `articles.embedding` and held in one in-memory matrix for scanning.

**(d) Algorithm, step by step.**
1. `get_embedder()` returns a local `SentenceTransformer` or a remote HTTP client if
   `REMOTE_MODEL_URL` is set (`src/model_registry.py`).
2. `DenseIndex._load_or_build` hashes every text (`sha256(model|max_len|text)`) and
   looks it up in `data/cache/text_embedding_cache_<model>_len256.npz`; only unseen
   texts are encoded (`src/retrieval.py:220-311`).
3. Vectors are unit-normalised once, at build time, so cosine similarity is a plain
   dot product (`:329-333`).
4. At query time the whole 2,000×768 matrix is scanned with one matrix-vector
   product (`scores_for_doc`, `:323-325`); there is no vector index.

**(e) Exact values and constants.**

| Constant | Value | Where | Effect of changing |
|---|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | `config.py:40` | Different vector space; invalidates every stored vector and score (in the fingerprint) |
| `EMBEDDING_REVISION` | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | `config.py:41` | Pins the weights so a silent upstream update cannot change vectors (in the fingerprint) |
| dimensions | `768` | `db/models.py` `embedding vector(768)` | Changing it needs a schema migration |
| vector index | **none** | `db/connection.py` `_ADDED_INDEXES` | Only btree indexes exist; similarity is a linear scan |
| cache key | `sha256(model\|max_len\|text)` | `src/retrieval.py:261-275` | Includes the token cap, so a cap change forces re-embedding |

**(f) Worked example (real).** Article 10905842 and candidate 10895859 have cosine
similarity **0.6395** (computed directly from the stored vectors). The empirical
percentile of 0.6395 against the random-pair background is **0.9878** — i.e. this pair
is more similar than 98.8% of random article pairs in the corpus. That single number
supplies 0.30 × 0.9878 = **0.2963** of the final relevance, more than any other term.

**(g) Edge cases.**
* A row with no stored vector contributes a **zero** vector, hence zero cosine, rather
  than forcing a corpus-wide re-embed (`api/server.py:92-103`, `db/pipeline.py:66`).
* `NEWSREC_NEURAL=0` swaps in `TfidfIndex`, whose "similarity" is TF-IDF cosine, not
  semantics (`src/retrieval.py:341-367`).
* If the remote worker is configured but unreachable, the registry silently falls back
  to CPU (`src/model_registry.py`) — results stay correct, latency multiplies (§13).
* Duplicate text (the same wire copy twice) maps to the same cache key, so the second
  article is embedded for free but is also an exact vector duplicate.

**(h) Measured ON vs OFF.** Comparing `method=hybrid+rerank` (dense on) against
`method=bm25` (dense off) on source 10905842: the two top-10 lists share **7 of 10**
articles; on 10910523, **6 of 10**; on 10895859, **7 of 10**. So roughly a third of the
final list is only reachable through the vector signal.

**(i) Limitations.**
* Linear scan over the full corpus: fine at 2,000 rows, linear cost at 2 million.
* The vectors are frozen at ingest. Re-embedding only happens when `text_embed`
  changes or the vector is NULL (`db/pipeline.py:398-400`).
* No vector index means no approximate-search tuning, no per-section sharding, and
  no way to cap the scan cost.

---

## 6.5 BM25 lexical retrieval and the focused query

**(a) Plain-language explanation.** Before asking the neural model for an opinion,
the system asks a much dumber but very reliable question: which other articles use
the same unusual words? Rarity matters — "jail" is worth more than "the". For an
article-to-article request the "query" is not the article's full text but a
hand-built summary: its title words plus its 15 most distinctive body words.

**(b) Code location.** Index and scoring: `src/retrieval.py:54-197`. Query
construction: `db/pipeline.py:43-53` (ingestion) and `src/recommender.py:106-119`
(live). They are the same algorithm written twice.

**(c) Inputs and outputs.**
* In: `text_bm25` (title×3 + description + body, §6.3); for a request, the focused
  query string.
* Out: one score per article (`np.ndarray`, length = corpus size), plus the index's
  `idf` table and `postings` map.

**(d) Algorithm, step by step.**
1. Tokenise: lowercase, `[a-z0-9]+`, drop sklearn's English stop-word list and
   tokens shorter than 2 characters (`:38-46`).
2. Build `postings[term] = [(pos, tf), …]` and `idf[term]` (`:76-86`).
3. Build a 2-gram index over the vocabulary for typo tolerance (`:88-94`).
4. `normalize_query` corrects each query token to the nearest vocabulary word using
   2-gram candidate filtering plus `difflib.SequenceMatcher`, accepting a match at
   ratio ≥ 0.70 and weighting it 0.85 or the ratio (`:96-177`).
5. `scores()` sums the BM25 contribution of every query term over its postings list
   (`:179-188`).
6. The caller keeps the top `N_RETRIEVE = 50` and records the source's own score as
   `self_bm25` (`db/pipeline.py:172-178`, `src/recommender.py:118-119`).

**(e) Exact values and equations.**

```
idf(t)      = ln(1 + (N − n_t + 0.5) / (n_t + 0.5))                       retrieval.py:84
score(q,d)  = Σ_t  w_t · idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avg_len))   retrieval.py:187
k1 = 1.5, b = 0.75                                                        retrieval.py:69
w_t = 1.0 exact match, 0.85 × ratio fuzzy match                            retrieval.py:66-67
```

| Constant | Value | Where | Effect of changing |
|---|---|---|---|
| `k1` | `1.5` | `src/retrieval.py:69` | Higher = term frequency saturates more slowly (long articles score higher) |
| `b` | `0.75` | `src/retrieval.py:69` | 1.0 = full length normalisation, 0 = none |
| `N_RETRIEVE` | `50` | `config.py:89` | How deep each of the two retrieval lists goes before fusion |
| `FUZZY_MIN_RATIO` | `0.70` | `src/retrieval.py:66` | Lower = more typo corrections, more false corrections |
| `FUZZY_WEIGHT` | `0.85` | `src/retrieval.py:67` | How much a corrected term is trusted |
| focused-query body terms | `15` | `db/pipeline.py:52`, `src/recommender.py:113` | More terms = broader, less specific neighbours |
| self-normaliser | source's own BM25 score | `src/scoring.py:75-83` | See the limitation below — this caps the lexical term |

**(f) Worked example (real).** Source 10905842 produced this 23-term focused query
(verbatim from the stored evidence):

```
don end jail kapil sharma watches younger comics flattered standup rainas allahbadias
roasters samays tamanaah roasting usp cheeky offended samay offending sidharths pals
```

Its self-score is **self_bm25 = 197.5205**. Candidate 10895859 scores
**15.3867**, so its normalised value is 15.3867 / 197.5205 = **0.0779**, and it is
BM25 rank **2**. The two terms that actually matched, with the index's own weights:

| term | tf in candidate | idf | weight | contribution |
|---|---|---|---|---|
| `kapil` | 14 | 4.251 | 1.0 | 9.7911 |
| `sharma` | 6 | 2.685 | 1.0 | 5.5956 |

9.7911 + 5.5956 = 15.3867, which is exactly the raw score above — the evidence card's
terms therefore sum to the real BM25 score, not an approximation.

**(g) Edge cases.**
* `b`-normalisation divides by the corpus average length, so one 5,000-word outlier
  distorts every other article's length norm.
* A query term absent from the vocabulary scores `idf = 0` and contributes nothing
  (`:184`) — no error, no warning.
* The fuzzy matcher can "correct" a legitimate rare name onto a common word; because
  it fires on the *query* only, ingestion and live search can correct differently.
* `min_score=1e-9` in the live path (`src/recommender.py:151`) means any article with
  at least one matching term is eligible; a query whose terms are all stop-words
  yields an empty BM25 list.

**(h) Measured ON vs OFF.** `method=bm25` (no dense, no rerank) versus the full
pipeline, top-10 overlap: **7/10** (10905842), **6/10** (10910523), **7/10**
(10895859). Its own ordering is unrelated to the final one — for 10905842 the BM25
rank-1 is 10910523, while the pipeline's rank-1 is 10895859.

**(i) Limitations — one measured, one structural.**
* **The lexical term is nearly inert.** Because `self_bm25` is normalised against the
  *source's own* score — which includes its title repeated three times and all 15
  query terms — a neighbour can realistically reach only ~0.08. Multiplied by
  `WEIGHT_BM25 = 0.15`, the lexical signal contributes at most ≈ **0.012** to a
  relevance score, against ≈ 0.30 for the dense term. Measured across the 10 stored
  rows for 10905842, `bm25_norm` ranges 0.0309–0.0816.
* The focused query is built from a `set` of body terms ordered by IDF, so **terms
  with equal IDF are ordered arbitrarily**. Rebuilding the same query in a new process
  produced the same term *set* in a different *order* (verified: the stored string and
  a fresh recomputation differ in sequence but not content). BM25 scores are
  unaffected; the stored `bm25_query` string is not reproducible.

---

## 6.6 Dense retrieval

**(a) Plain-language explanation.** The same question as 6.5, asked in meaning-space
instead of word-space: "which articles have a vector pointing the same way as mine?"

**(b) Code location.** `src/retrieval.py:317-339`; used at `db/pipeline.py:181-183`
and `src/recommender.py:117`.

**(c) Inputs and outputs.**
* In: the source article's position (or a query vector for free-text search).
* Out: one cosine score per article; the top `N_RETRIEVE = 50` become the dense list.

**(d) Algorithm.** `scores_for_doc(pos)` returns `emb @ emb[pos]`; because vectors are
unit-normalised this equals cosine similarity. The source is set to `‑inf` so it
cannot recommend itself (`db/pipeline.py:182`).

**(e) Exact values.** Similarity = cosine = dot product of 768-d unit vectors.
`N_RETRIEVE = 50` (`config.py:89`). There is no similarity floor in the current code
(`MIN_RELEVANCE["dense"] = 0.50` exists at `config.py:262` but is only read by the
`semantic`-only method, not by the fused path).

**(f) Worked example.** For 10905842 → 10895859, cosine = **0.6395**, dense rank
**8**, percentile **0.9878** (percentile > cosine because random news pairs already
average ≈ 0.42 — see the quantile table at `src/scoring.py:46-58`).

**(g) Edge cases.** Zero-vector rows score 0.0; a section filter zeroes the score of
every article outside the section rather than removing rows
(`src/recommender.py:147-148`); duplicate stories occupy the same point in space.

**(h) Measured ON vs OFF.** `method=semantic` only versus the pipeline: **4/10**
overlap (10905842), **6/10** (10910523), **8/10** (10895859). On 10895859 the
semantic-only list is almost identical to the `hybrid` list, i.e. the reranker is what
disagrees with it (§6.8).

**(i) Limitations.** Cosine is a blunt instrument at this embedding size: the
background distribution is narrow (0.2578 at the 0th percentile to 0.6461 at the 98th,
`src/scoring.py:46-58`), so a 0.01 cosine difference spans several percentile points.
The percentile table is a fixed constant measured once on 50,000 random pairs and is
**not** in the pipeline fingerprint — recomputing it on a different corpus would change
every stored `dense_norm` while leaving the version hash unchanged.

---

## 6.7 Reciprocal Rank Fusion

**(a) Plain-language explanation.** Two lists, two different notions of "relevant".
RRF is the standard way to merge them without pretending their scores are comparable:
it rewards articles that appear high on *either* list, and rewards them more if they
appear on *both*.

**(b) Code location.** `src/retrieval.py:370-380`; called at `db/pipeline.py:187`
and `src/recommender.py:159`.

**(c) Inputs and outputs.** In: two ordered lists of corpus positions (≤50 each).
Out: a dict `{position: fused_score}`, of which the top `N_CANDIDATES = 20` are kept.

**(d) Algorithm.** For each list, add `1/(k + rank)` to that article's fused score,
rank starting at 1; sort descending; keep 20.

**(e) Exact formula and constants.**

```
fused(d) = Σ_lists  1 / (RRF_K + rank_of_d_in_that_list)        retrieval.py:370-380
RRF_K = 60                                                     config.py:91
N_CANDIDATES = 20                                              config.py:90
```

| Constant | Value | Where | Effect of changing |
|---|---|---|---|
| `RRF_K` | `60` | `config.py:91` | Higher flattens the advantage of rank 1; 60 is the standard value that keeps ranks 1–10 close together |
| `N_CANDIDATES` | `20` | `config.py:90` | The pool size handed to the cross-encoder; larger = better recall, linearly more reranker work |

**(f) Worked example (real).** For 10905842 the two 50-item lists overlapped enough
to produce a **90-article** fused pool (50 + 50 − 10 shared), of which the top 20 went
to the reranker. An article ranked 1st by BM25 and 8th by dense scores
1/61 + 1/68 = 0.016393 + 0.014706 = **0.031099**, while the traced candidate 10895859
(2nd by BM25, 8th by dense) scores 1/62 + 1/68 = **0.030835**. An article on only one list at
rank 20 scores 1/80 = **0.0125**, so dual presence is worth roughly 2.5× a single
mid-list appearance.

**(g) Edge cases.** 1/(60+1) vs 1/(60+50) differ by only 22% — RRF deliberately
compresses rank differences, so the top-20 cut is coarse. A position blocked by a
section filter never appears in either list.

**(h) Measured ON vs OFF.** Fusion cannot be disabled through the API. Its effect is
visible in the method comparison: `hybrid` (fused, no rerank) shares only **5/10**
with the full pipeline on 10905842 but **8/10** on 10895859 — the fused pool is a
reasonable candidate set, and the reranker is what reorders it.

**(i) Limitations.** RRF ignores score magnitude entirely: a BM25 score of 15.4 and a
score of 9.0 both count as "rank 2". It also cannot express "neither list is
confident", which is why the confidence gate lives in 6.8/6.12 instead.

---

## 6.8 Cross-encoder reranking

**(a) Plain-language explanation.** The two retrieval stages are fast and rough. This
stage reads the source text and each candidate text **together** and outputs a
probability that a human would consider them related. It is the most accurate and by
far the most expensive step.

**(b) Code location.** `src/reranker.py` (`build_reranker_pairs`, `score_pairs`,
`_to_probability`); called at `db/pipeline.py:193-201` and `src/recommender.py:169-184`.

**(c) Inputs and outputs.** In: the source's `text_rerank` (or a query-centred
passage for free-text search) and 20 candidate texts. Out: 20 probabilities in [0,1].

**(d) Algorithm.**
1. Build (query, passage) pairs, each side clipped to 254 content tokens so the pair
   fits the 512-token window (`_SIDE_BUDGET`).
2. For free-text search, `extract_query_centered_passage` keeps title and description
   whole and grows a body window around the sentence containing the first query token.
3. Run the cross-encoder in batches of 32, locally or on the remote worker.
4. `_to_probability` clips values already in [0,1] and sigmoids only genuine logits —
   the remote client returns raw logits, the local model returns probabilities, and
   applying sigmoid to both would trap every score in [0.5, 0.731].

**(e) Exact values and constants.**

| Constant | Value | Where | Effect of changing |
|---|---|---|---|
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | `config.py:42` | In the fingerprint |
| `RERANKER_REVISION` | `2cfc18c9415c912f9d8155881c133215df768a70` | `config.py:43` | In the fingerprint |
| `RERANK_MAX_LEN` | `512` (both sides together) | `config.py:45` | Longer = more context per pair, quadratically slower |
| per-side budget | `254` | `config.py:80` | Guarantees 254+254+4 = 512 |
| batch size | `32` | `db/pipeline.py:198` | Throughput only |
| `MIN_RAW_RERANK_THRESHOLD` | `0.12` | `config.py:258` | Below this top probability the whole answer is flagged `low_confidence` |

**(f) Worked example (real).** For source 10905842 the 20 candidate probabilities
ranged **0.0002 → 0.7154**, mean **0.0965**. Candidate 10895859 (the stored rank 1)
scored **0.0917** — the *highest* of the 20, i.e. reranker rank 1 in a pool where
nothing is very convincing. The wider context: 0.0917 is far below the 0.12 gate, yet
the request was **not** flagged low-confidence, because the gate reads the *maximum*
raw probability and that maximum was 0.7154. In other words the flag describes the
best candidate in the pool, not the candidate actually served at rank 1.

**(g) Edge cases.**
* 0.0002 and 0.0001 appear in real stored rows (ranks 4 and 10 of 10905842): the
  cross-encoder can output effectively zero yet the article still gets served.
* A remote worker that dies mid-run silently falls back to the local model, so scores
  stay comparable but latency jumps.
* `hybrid` and `bm25` methods never call it, so `rerank_score` is `NULL` for those rows.

**(h) Measured ON vs OFF.** Comparing "no boosts, no MMR" (rerank on) with `hybrid`
(rerank off) on the same three sources, the top-10 changes by **8/10**, **6/10**,
**5/10** articles respectively. Reranking is the single most disruptive stage
measured. Its cost: ≈ 12.7 s per article on local CPU vs ≈ 1.2 s with the GPU worker
(§13).

**(i) Limitations.** The cross-encoder is O(pool) per request: 20 pairs, each a full
forward pass over 512 tokens. At 2,000 articles and 20 candidates that is fine; the
pipeline does it for *every* article on ingestion, which is why a full recompute is
measured in hours, not minutes.

---

## 6.9 Relevance blend (the three-signal mix)

**(a) Plain-language explanation.** Three numbers now describe the same pair — the
cross-encoder's opinion, the word-overlap score, and the semantic percentile. The blend
turns them into one relevance number between 0 and 1, with the cross-encoder carrying
most of the weight.

**(b) Code location.** `src/scoring.py:91-105`. Called by both paths
(`db/pipeline.py:83-84`, `src/recommender.py:193-194`).

**(c) Inputs and outputs.** In: `rerank_scores` (probabilities), `bm25_scores` (raw),
`dense_scores` (raw cosine), the candidate positions, and `self_bm25`. Out:
`(relevance, bm25_norm, dense_norm)` — all three kept so the evidence card can show the
decomposition.

**(d) Algorithm.** BM25 is divided by the source's self-score and clipped to [0,1];
the dense cosine is mapped to its empirical percentile; the three are combined with
fixed weights.

**(e) Exact formula and constants.**

```
bm25_norm  = clip(bm25 / self_bm25, 0, 1)                        scoring.py:75-83
dense_norm = interp(cosine, COSINE_QUANTILES, 0…1)               scoring.py:61-72
relevance  = 0.55·rerank + 0.15·bm25_norm + 0.30·dense_norm      config.py:101-103
```

| Constant | Value | Where | Effect of changing |
|---|---|---|---|
| `WEIGHT_RERANK` | `0.55` | `config.py:101` | Dominant term; in the fingerprint |
| `WEIGHT_BM25` | `0.15` | `config.py:102` | Effectively ≈0.012 in practice (see 6.5(i)); in the fingerprint |
| `WEIGHT_DENSE` | `0.30` | `config.py:103` | Second-largest term; in the fingerprint |
| `COSINE_QUANTILES` | 101 fixed values, 0.2578 … 0.9428 | `src/scoring.py:46-58` | **Not** in the fingerprint |

Free-text search uses a two-term variant with **no dense term**:
`0.55·rerank + 0.15·bm25_norm` (`src/recommender.py:190`).

**(f) Worked example (real, verified to 6 decimal places).** Candidate 10895859 for
source 10905842:

```
0.550 × 0.091696  = 0.050433   (rerank_term)
0.150 × 0.077900  = 0.011685   (bm25_term)
0.300 × 0.987800  = 0.296340   (dense_term)
                  = 0.358458   → stored blend 0.3584
```

The same arithmetic reproduces every one of the 10 stored rows, which is the strongest
available evidence that the live code and the stored rows use the same formula.

**(g) Edge cases.** If `self_bm25` is 0 or missing the function falls back to
max-normalisation (`src/scoring.py:82-83`) — this is what happens for free-text search.
Negative cosines map to percentile 0.0 (clipped), so a negative dense score can no
longer produce a negative relevance term.

**(h) Measured ON vs OFF.** Weights are constants, not a toggle. Their importance: the
reranker alone (weight 0.55) cannot lift a candidate above ≈0.55, and the dense term
contributes 0.296 of a 0.358 total in the example above — i.e. **83% of the winning
score comes from the dense term**, because the reranker probability is tiny (0.0917).

**(i) Limitations.** This is the most consequential oddity in the whole system: the
weight table says the cross-encoder dominates, but in the stored data the *dense
percentile* dominates, because percentile-mapping inflates ordinary cosines (0.64 →
0.9878) while raw reranker probabilities stay low (0.09). Measured across the 10
stored rows for 10905842, `dense_norm` never falls below **0.9029** and 6 of 10 exceed
0.98, so the blend behaves almost like "0.30 for everyone + reranker differences".

---

## 6.10 Metadata and agreement boosts

Six multipliers are applied to the relevance score, in a single step, as a
**product** (`src/scoring.py:301-304`):

```
boosted = relevance × Π (editorial, section, subsection, popularity, recency, agreement)
```

### Cross-cutting questions (answers apply to all six)

**Why a product and not a sum?** A sum would let one large bonus compensate for
irrelevance; a product means every factor scales the *same* relevance number, so an
irrelevant article (relevance 0.02) stays irrelevant (0.02 × 1.40 = 0.028) while a
strong one gets a proportional nudge. The code makes this explicit by starting at 1.0
and multiplying (`src/scoring.py:301-304`).

**What is the maximum combined effect?** `1.40 × 1.10 × 1.05 × 1.015 × 1.04 × 1.15 =
1.9629`, i.e. a candidate can score **+96.3%**. The floor is **0.30** (the cross-domain keyword-fluke
penalty, `src/scoring.py:232`) with every other factor at 1.0, i.e. **−70%**.

**Can it reorder results?** Yes, and measurably. For source 10905842, ordering by
`blend` alone versus the stored final order:

| Candidate | blend (relevance only) | relevance-only rank | final rank |
|---|---|---|---|
| 10895859 | 0.3584 | 1 | 1 |
| 10896871 | 0.3355 | **2** | **6** |
| 10898555 | 0.3276 | 3 | 5 |
| 10898908 | 0.3222 | 4 | 7 |
| 10910523 | 0.3086 | **5** | **2** |
| 10909396 | 0.3046 | 7 | 3 |

Candidate 10896871 has the second-best relevance but lands 6th; 10910523 is 5th on
relevance and finishes 2nd. Every one of those moves is caused by these six factors.

**What prevents a boost from overpowering relevance?** Numerically, **nothing**. With a
maximum of 1.963, a boosted candidate at relevance 0.30 reaches 0.589 and outranks an
unboosted candidate at 0.50. Three design choices reduce the risk without removing it:
(i) the section/subsection/popularity/recency multipliers are deliberately small
(1.10/1.05/1.015/1.04 — together only +22.6%), so the large factors are the two that
carry editorial meaning (1.40 for a human-linked article) or evidential meaning
(0.30–1.15 for signal consensus); (ii) popularity is log-scaled, so the most-read story
in the corpus gets +1.5%, not +50%; (iii) the consensus multiplier can *demote* as well
as promote, and does so on 4 of the 10 stored rows for 10905842 (`agreement = 0.80`).
There is no cap, no relevance floor and no "boost may not exceed X%" rule anywhere in
the code — verified by reading `src/scoring.py:249-305` in full.

### 6.10a Editorial ("Also Read") boost

**(a) Plain-language explanation.** When an editor attached an "Also Read" link to a
story, that pairing is a human judgement that these two belong together. The system
trusts that judgement above every statistical signal and lifts the article 40%.

**(b) Code location.** `src/scoring.py:268-269`; the linked set comes from
`src/scoring.py:34-41`.

**(c) Inputs/outputs.** In: `source_row["linked_ids"]` (JSON string in the DB, list in
memory) and the candidate id. Out: `1.40` or `1.0`.

**(d) Algorithm.** `str(cand_row["id"]) in linked_set(source_row)` → 1.40.

**(e) Exact values.** `1.40`, hard-coded at `src/scoring.py:269` — **not** a `config.py`
constant, and therefore **not** in the pipeline fingerprint. Effect of changing it: at
1.40 an editorial link outweighs a same-section+same-subsection advantage (1.155)
comfortably; raising it to 2.0 would let a human link beat almost any relevance gap.

**(f) Worked example (real).** For source 10905842 none of the 10 stored recommendations
carries an editorial link, so all ten show `editorial: 1.0` (visible in the stored
`boost_items` of rank 1). Across the corpus, 197 articles have links at all (§5), so this
boost is rare but decisive where it fires.

**(g) Edge cases.** An ID that was linked but does not exist in the corpus is silently
ignored (`parse_ids` keeps only strings; the membership test simply fails). A link to an
article in another section applies the boost *and* keeps section at 1.0 — the editor
wins over the taxonomy.

**(h) Measured ON vs OFF.** Boosts-on versus boosts-off (configs A vs B) on 10905842:
top-10 overlap **8/10**; on 10910523 **7/10**; on 10895859 **7/10**. That is the
combined effect of all six factors, of which editorial contributes nothing for these
three sources.

**(i) Limitations.** Editorial links are sparse (197/2,000), biased towards what editors
chose to promote, and the ground truth used for evaluation is the same data — so any
metric that rewards this boost is partly measuring itself (§12).

### 6.10b Section boost

**(a) Plain-language explanation.** Readers who open a cricket story usually want
another cricket story. Same-top-level-section is a cheap, strong prior, and the tuning
exercise found it to be the single most valuable boost.

**(b) Code location.** `src/scoring.py:270-271`, constant at `config.py:123`.

**(c) Inputs/outputs.** In: `source_row["section"]`, `cand_row["section"]`. Out: `1.100`
or `1.0`.

**(d) Algorithm.** Exact string equality on the section name, guarded so an empty source
section never matches (`:270`).

**(e) Exact values.** `SECTION_BOOST_MULT = 1.100` (`config.py:123`). In the fingerprint
(`config.py:307`). Effect: +10% — enough to break a tie between two candidates whose
relevance differs by less than 10%, and measured to be worth ≈ +0.02 nDCG@5 in the
project's own ablation.

**(f) Two contrasting real articles for source 10905842 (entertainment/Television):**

| | 10895859 | 10898555 |
|---|---|---|
| section | entertainment | **lifestyle** |
| subsection | Television | Fitness |
| page views | 7,799 | 293 |
| published | 2026-09-27 (6.9 d older) | 2026-10-01 (2.97 d older) |
| section factor | **1.100** | 1.000 |
| subsection factor | **1.050** | 1.000 |
| popularity | 1.0122 | 1.0077 |
| recency | 1.0081 | 1.0306 |
| agreement | 1.0400 | 1.0400 |
| **total** | **1.2257 (+22.6%)** | **1.0802 (+8.0%)** |
| blend | 0.3584 | 0.3276 |
| final | 0.4394 → rank 1 | 0.3538 → rank 5 |

The same-section candidate converts a *lower* base relevance (0.3584) into a much
higher boosted score than the cross-section one (0.3276), which is exactly what the +10%
is for.

**(g) Edge cases.** Section names arrive as free text (`sectionName`/`section1Slug`);
"cities" vs "Cities" would not match — the comparison is case-sensitive on the stored
value, and cleaning does not lowercase section names. Empty section never matches.

**(h) Measured ON vs OFF.** Section is one of the six factors lumped into A-vs-B
(8/10, 7/10, 7/10 overlap). Isolating it exactly would need a code change per factor,
which this audit did not perform (stated as a limitation of the measurement, not of the
system).

**(i) Limitations.** A section can be huge (cities = 464 articles in the older corpus),
so "same section" is sometimes a weak signal; conversely a genuinely related story filed
under a different section (entertainment vs lifestyle in the example above) is
systematically disadvantaged.

### 6.10c Subsection boost

**(a) Plain-language explanation.** A finer version of 6.10b: Television is not
Bollywood, even though both are entertainment.

**(b) Code location.** `src/scoring.py:272-273`, constant at `config.py:124`.

**(c/d/e)** Exact equality on `subsection`, applied only when the source has one;
`SUBSECTION_BOOST_MULT = 1.050` (`config.py:124`), in the fingerprint (`config.py:308`).
+5% — deliberately half the section weight, because subsection labels are noisier.

**(f) Worked example (real).** Of the 10 stored recommendations for 10905842, exactly
one (10895859, Television) receives the subsection boost; nine do not, including four
that are in the same section but a different subsection (Bollywood, Telugu, or empty).
The empty case is common: 10898555 has `sub = (empty)` and so can never match.

**(g) Edge cases.** An empty candidate subsection yields 1.0 even when the source is
also empty, because the guard requires a truthy source value *and* equality — an
all-empty pair does not get a bonus. Subsection values are not normalised for case.

**(h) Measured ON vs OFF.** Folded into A-vs-B as above. Its individual contribution in
this sample is visible but small: it is worth +5% on 1 of 10 rows.

**(i) Limitations.** Subsection coverage in the live data is poor — 4 of 10 candidates in
the example have no subsection at all, so the factor is inert for a large share of pairs.

### 6.10d Popularity boost

**(a) Plain-language explanation.** A story that thousands of people are reading right
now is more likely to be a useful "read next" than one nobody opened. The boost is
logarithmic precisely so that a viral hit does not swallow the list.

**(b) Code location.** `src/scoring.py:276-281`, constant at `config.py:125`.

**(c) Inputs/outputs.** In: `page_views` of the candidate (accepts DB `page_views` and CSV
`pageViews`), and the corpus maximum. Out: a factor in `[1.0, 1.015]`.

**(d/e) Exact formula and values.**

```
popularity = 1 + (POPULARITY_BOOST_MULT − 1) · ln(1 + views) / ln(1 + max_views)
POPULARITY_BOOST_MULT = 1.015                                   config.py:125
max_views = 60,438  (measured, live corpus)                     §5
```

Because the ratio is logarithmic, going from 100 to 1,000 views moves the factor from
1.0077 to 1.0103 — +0.26 percentage points for a 10× traffic increase.

**(f) Worked example (real, verified).**

| candidate | views | ln(1+views)/ln(1+60438) | factor |
|---|---|---|---|
| 10895859 | 7,799 | 0.8140 | **1.0122** |
| 10908882 | 29,601 | 0.9333 | 1.0140 |
| 10898555 | 293 | 0.5162 | 1.0077 |
| 10896967 | 119 | 0.4350 | 1.0065 |

The most-read candidate in this list (29,601 views, ~38× the least-read) gets only
+1.40% versus +0.65% — the whole range is 0.74 percentage points.

**(g) Edge cases.** `views = 0` or missing → factor 1.0 (`:279`); a corpus where every
article has the same views makes the ratio 1.0 for the maximum and <1 for the rest, so
the factor becomes a proxy for "is the most-read article". `pageViews` in the live
DataFrame and `page_views` in the DB are both accepted (`:276-278`).

**(h) Measured ON vs OFF.** Cannot be isolated without a code change; in the combined
A-vs-B it is one of five factors. Its measured range across 10 rows is 1.0065–1.0140,
i.e. at most +1.4% — by construction the weakest of the six.

**(i) Limitations.** Page views are cumulative-forever and never decayed: an old
evergreen explainer can out-popularity a genuinely trending story. There is no view
window, no per-section normalisation, and no bot filtering.

### 6.10e Recency boost

**(a) Plain-language explanation.** News goes stale. But it goes stale at different
speeds: a live blog is dead in six hours, a book review is fine for ten days. The boost
therefore decays with a **section-specific half-life**, measured against the *source*
article's date rather than "now".

**(b) Code location.** `src/scoring.py:285-290` (caller), `config.py:223-241`
(`recency_boost`), `config.py:212-220` (`section_half_life`), `config.py:244-254`
(`recency_ref_date`).

**(c) Inputs/outputs.** In: candidate date, reference date, candidate section, `is_live`
flag. Out: a factor in `[1.0, 1.04]`.

**(d) Algorithm.** Compute days between reference and candidate (clamped at ≥0), look up
the half-life for the candidate's section (or `live_blog` when `is_live`), and add the
decayed remainder of the maximum bonus.

**(e) Exact formula and constants.**

```
days_old  = max(0, (ref_date − cand_date) / 86400)
half_life = SECTION_HALF_LIVES[section]           # 0.25 … 10.0 days
recency   = 1 + (RECENCY_MAX_BOOST_MULT − 1) · 0.5 ^ (days_old / half_life)
RECENCY_MAX_BOOST_MULT = 1.040                    config.py:126
BOOST_MODE = "source_date_recency"                config.py:199
```

Half-lives (`config.py:139-197`): live_blog 0.25 · trending 1.0 · sports 1.5 ·
cities/delhi/mumbai/chandigarh/pune/india 2.5 · world/entertainment 3.0 ·
business/technology/political-pulse 4.0 · legal 5.0 ·
explained/opinion/columns/education/upsc/lifestyle 7.0 · books 10.0 · default 3.0.

Effect of changing `RECENCY_MAX_BOOST_MULT`: the factor's ceiling, so at 1.04 freshness
is worth at most +4% — deliberately smaller than section (+10%). Changing `BOOST_MODE`
to `"multiplicative"` would score against the newest row in the corpus instead of the
source, which the project's own tuning rejected.

**(f) Worked example (real, verified).** For 10895859, entertainment half-life 3.0 days,
6.9 days older than the source: `0.5^(6.9/3.0) = 0.2029` → `1 + 0.04 × 0.2029 =`
**1.0081** (stored 1.0081). For 10898555, lifestyle half-life 7.0, 2.97 days older:
`0.5^(2.97/7) = 0.7452` → **1.0298** (stored 1.0306). The *older* story in a slow
section beats the *fresher* story in a fast section — that is the half-life doing its job.

**(g) Edge cases.** Missing/NaT dates → 1.0 (`config.py:237-238`). A candidate dated
*after* the source (possible when the corpus contains later batches) is clamped to 0
days old and gets the full +4%. A missing source date makes `recency_ref_date` fall back
to the corpus maximum (`config.py:252-254`), silently switching modes for that row.

**(h) Measured ON vs OFF.** In the combined A-vs-B comparison, recency contributes
1.0081–1.0400 across the 10 rows (max +4.0%). At the corpus level, the project's own
earlier ablation reported recency as net-negative on nDCG@5 (that harness is no longer
in the repository — see §19).

**(i) Limitations.** Recency is measured against the **source's** publication date, not
against "now", so a request served today for an article from last month rewards
candidates from last month — correct for "related reading", wrong for "what's happening".
The `is_live` override uses the candidate's flag, so a live blog candidate always decays
at the 6-hour rate regardless of its section.

### 6.10f Signal-consensus multiplier (agreement)

**(a) Plain-language explanation.** Three independent signals each have an opinion about
a candidate. When all three agree strongly, the candidate is almost certainly a good
recommendation and gets a bonus; when only one signal fires — the classic case being a
keyword that appears in a completely unrelated story — the candidate is pushed down
hard. This is the only boost that can penalise, and it is the reason the list is not
dominated by keyword coincidences.

**(b) Code location.** `src/scoring.py:108-246` (`_signal_grade`,
`consensus_multiplier`), called from `boost_parts` (`:292-299`).

**(c) Inputs/outputs.** In: BM25 rank, dense rank, raw cross-encoder probability, the
candidate row, the source row. Out: one of
`{1.15, 1.10, 1.08, 1.04, 1.00, 0.95, 0.80, 0.75, 0.40, 0.35, 0.30}`.

**(d) Algorithm.** Grade each signal (strong / moderate / weak / none), count how many
are at least moderate, then walk a priority table from "all three strong" down to
"single signal", stopping at the first match.

**(e) Exact values.**

| Signal | Strong | Moderate | Weak | None |
|---|---|---|---|---|
| BM25 rank (of 50) | ≤ 5 | 6–15 | 16–25 | >25 / absent |
| Dense rank (of 50) | ≤ 5 | 6–15 | 16–25 | >25 / absent |
| Cross-encoder prob | ≥ 0.55 | ≥ 0.35 | — | < 0.35 |

| Rule | Multiplier |
|---|---|
| all three strong | **1.15** |
| all three qualify, ≥2 strong | 1.10 |
| all three qualify | 1.08 |
| BM25 and dense both ≥ moderate | 1.04 |
| dense strong + CE strong, BM25 absent | 1.00 |
| dense strong + CE moderate, BM25 absent | 0.95 |
| BM25 strong, dense absent/far, CE not strong | **0.40** (keyword fluke) |
| BM25 strong, dense absent/far, CE strong, **different section** | **0.30** (cross-domain fluke) |
| BM25 strong, dense absent/far, CE strong, same section | 0.75 (topic mismatch) |
| dense ≥ moderate, BM25 absent/far, CE none | **0.35** (embedding drift) |
| only one signal qualifies | 0.80 |
| otherwise | 1.00 |

Editorial links bypass the table entirely and return 1.0 (`:170-171`) — a human link is
never penalised. Changing any threshold changes the stored scores, so the thresholds are
all in the fingerprint (`config.py:321-333`).

**(f) Worked example (real, verified against the stored rows for 10905842).**

| candidate | BM25 rank | dense rank | CE prob | grades | rule | factor |
|---|---|---|---|---|---|---|
| 10895859 | 2 strong | 8 moderate | 0.0917 none | S,M,N | BM25+dense ≥ moderate | **1.04** |
| 10898555 | 5 strong | 7 moderate | 0.0366 none | S,M,N | same | 1.04 |
| 10896871 | 16 weak | 9 moderate | 0.0595 none | W,M,N | only 1 qualifies | **0.80** |
| 10898908 | none | 49 none | 0.0721 none | N,N,N | only 0 qualify | 0.80 |

10896871 is the informative one: it has the second-best relevance score (0.3355) and is
in the right section, but only *one* signal corroborates it, so its 0.80 multiplier drags
it from rank 2 to rank 6. 10898908 is the "nobody vouches for this" case — dense rank 49
of 50, no lexical overlap, cross-encoder 0.07 — yet it still appears at rank 7, because
nothing removes it (see 6.15 and §19: there is no admission floor).

**(g) Edge cases.** A missing rank (`None`) is treated as "none", not as rank 0. A rank
of 900+ is also treated as missing (`:122`). For `method=hybrid`/`bm25`, `rerank_score`
is `None` → CE grade "none", so text-search and no-rerank paths can never reach the top
tiers of this table.

**(h) Measured ON vs OFF.** The multiplier is part of the `use_boosts` bundle, so A-vs-B
covers it (8/10, 7/10, 7/10 overlap). Its per-row effect is directly visible: four of the
ten stored rows for 10905842 carry 0.80, one carries 1.00, and five carry 1.04 — a spread
from −20% to +4% inside a single recommendation list.

**(i) Limitations.** The table is hand-written from editorial examples (the comments name
"Kapil Sharma jail" and "Flipkart sale vs Google lawsuit"), and its thresholds are tuned
to a 20-candidate pool. Because four of six tiers are driven by *ranks* rather than
scores, a pool that happens to be uniformly strong still produces "moderate" grades and
therefore no bonus.

---

## 6.11 MMR diversity

**(a) Plain-language explanation.** Ten near-identical stories about the same press
conference is a bad recommendation list even if all ten are relevant. MMR picks results
one at a time, each time slightly preferring novelty over pure relevance — but only
penalising a candidate when it is *very* similar to something already chosen.

**(b) Code location.** `src/scoring.py:308-325`; wrappers at `src/recommender.py:365-369`
and inline use at `db/pipeline.py:101-103`.

**(c) Inputs/outputs.** In: `boosted` scores, the candidate×candidate cosine matrix, `k`.
Out: an ordered list of candidate indices of length ≤ k.

**(d) Algorithm.** Start with an empty selection. Repeatedly pick the candidate
maximising `λ·score − (1−λ)·max(similarity to selected items that exceed θ)`. Stop at k.

**(e) Exact formula and constants.**

```
value(i) = λ · score(i) − (1−λ) · max{ sim(i,j) : j ∈ selected, sim(i,j) > θ }
λ(article) = 0.91,  θ(article) = 0.76      config.py:109-110
λ(search)  = 0.88,  θ(search)  = 0.85      config.py:115-116
```

| Constant | Value | Where | Effect of changing |
|---|---|---|---|
| `MMR_LAMBDA` | `0.91` | `config.py:109` | 1.0 = no diversity at all; at 0.91 the redundancy term can subtract at most 0.09 × similarity |
| `MMR_THETA` | `0.76` | `config.py:110` | Only pairs more similar than 0.76 are treated as redundant — ordinary topical closeness is never punished |
| `MMR_LAMBDA_SEARCH` | `0.88` | `config.py:115` | Search results are allowed slightly more variety |
| `MMR_THETA_SEARCH` | `0.85` | `config.py:116` | Search penalises only near-duplicates |

Both λ and θ are in the fingerprint (`config.py:299-300`). The ingestion path skips MMR
entirely when the candidate pool is ≤ `TOP_K` (`db/pipeline.py:101-103`).

**(f) Worked example (real).** The live trace for 10905842 produced the MMR order
`[17, 16, 0, 1, 4, 3, 2, 5, 7, 8]` over the 20-candidate pool — i.e. the item at pool
position 17 was chosen first, then 16, then 0. For the stored rows the effect was
negligible: the stored `rank_raw` equals `rank` for all 10 rows, meaning MMR did not
move anything at ingest time for this article.

**(g) Edge cases.** If no similarity exceeds θ, the penalty is 0 and MMR reduces to plain
score ordering (`max(redundant, default=0.0)` at `:319`). With a single candidate, MMR
returns it unchanged. The similarity matrix is computed only over the candidate pool
(`src/recommender.py:367`), so an exact duplicate outside the pool is invisible.

**(h) Measured ON vs OFF (real, three sources).** Boosts on, MMR on versus off (configs A
vs C): top-10 overlap **10/10** (10905842), **10/10** (10910523), **9/10** (10895859).
So MMR changed at most one position in ten. Compare with the *boosts* toggle (A vs B):
8/10, 7/10, 7/10. MMR is the weakest lever in the current configuration — at λ = 0.91 the
penalty term is small and θ = 0.76 excludes almost every real pair.

**(i) Limitations.** With λ = 0.91 the diversity mechanism is close to inert; measured
above, it moved one row in 30. The threshold is fixed, not per-section, so a section with
genuinely repetitive coverage (crime briefs) will still return near-duplicates.

---

## 6.12 Platt calibration

**(a) Plain-language explanation.** "Relevance 0.358" means nothing to an editor. "We are
38% confident this belongs next to that story" is actionable. A sigmoid fitted on editorial
links maps every relevance score onto a 0–1 confidence, and a three-level band makes it
readable.

**(b) Code location.** `src/calibration.py:40-97`; called at `src/recommender.py:281` and
`db/pipeline.py:126`.

**(c) Inputs/outputs.** In: `unboosted` relevance (clipped to [0,1]), the next candidate's
relevance (for the margin), and `agreement_count` (1–3). Out: a dict with
`calibrated_prob`, `percentage`, `band`, `band_label`, `band_color`, `margin`.

**(d) Algorithm.** Apply the logistic function to `a·s + b + 0.35·(agreement−1)`, clip to
[0.01, 0.99], then apply band thresholds with agreement overrides.

**(e) Exact formula and constants.**

```
logit = 4.85 · s − 2.60 + 0.35 · (agreement − 1)      calibration.py:50-51
prob  = sigmoid(logit), clipped to [0.01, 0.99]        calibration.py:52-55
HIGH   if prob ≥ 0.65   OR (agreement ≥ 3 AND prob ≥ 0.40)      calibration.py:60
MEDIUM if prob ≥ 0.35   OR (agreement ≥ 2 AND prob ≥ 0.25)      calibration.py:62
else LOW
margin = max(0, s − s_next)                                      calibration.py:84-86
```

| Constant | Value | Where | Effect of changing |
|---|---|---|---|
| `DEFAULT_PLATT_A` | `4.85` | `src/calibration.py:32` | Slope: how fast confidence rises with relevance |
| `DEFAULT_PLATT_B` | `-2.60` | `src/calibration.py:33` | Offset: `s = 0.536` is the 50% point |
| agreement weight | `0.35` per extra signal | `src/calibration.py:50` | Each corroborating signal adds a constant to the logit (≈ +9 points of probability near the middle) |
| `HIGH_CONF_THRESHOLD` | `0.65` | `:36` | Band boundary |
| `MED_CONF_THRESHOLD` | `0.35` | `:37` | Band boundary |

None of these are in the pipeline fingerprint: changing `a` or `b` changes every stored
`score` without changing `pipeline_version`.

**(f) Worked example (real, verified).** Candidate 10895859, relevance 0.358458,
agreement 2:

```
logit = 4.85 × 0.358458 − 2.60 + 0.35 × 1 = −0.5115
prob  = 1 / (1 + e^0.5115)                = 0.374847  → stored 0.3748  (band MEDIUM)
without the agreement bonus:                0.2970    → stored "unboosted_confidence": 0.297
```

Candidate 10896871, relevance 0.3355, agreement 1: `logit = 4.85 × 0.3355 − 2.60 =
−0.9730` → **0.2743** (stored 0.2743), band LOW. Candidate 10906391 in the live re-run had
boosted 0.6851 and displayed 0.7154: `logit = 4.85 × 0.654 − 2.60 + 0.35 = 0.9219` →
0.7154 ✓ (0.654 is its unboosted relevance).

**(g) Edge cases.** `s` is clipped to [0,1] before the sigmoid, so a relevance above 1.0
after boosts cannot inflate the confidence — because the calibration input is the
*unboosted* relevance, never the boosted score. Missing `next_score` → margin 0.0.

**(h) Measured ON vs OFF.** Calibration is always on. Its practical consequence is that
the displayed confidence is **not monotonic with rank**: for the 10 stored rows the scores
run 0.3748, 0.3201, 0.3159, **0.3175**, **0.3405**, 0.2743, 0.2616, 0.2308, **0.2355**,
0.2283 — rank 5 shows a *higher* confidence than rank 2, and rank 9 higher than rank 8,
because ordering uses the boosted score while confidence uses the unboosted one.

**(i) Limitations.** The slope (4.85) and offset (−2.60) are declared to be "fitted on
editorial links", but the fitting function (`fit_calibrator_on_corpus`,
`src/calibration.py:100`) is **never called anywhere in the codebase** — the values are
literals. The calibration therefore describes no measured distribution, and the
`agreement` bonus is a hand-added constant on top of it.

---

## 6.13 Evidence JSON (the "why")

**(a) Plain-language explanation.** Every recommendation carries a machine-readable
explanation: the shared names, the words that matched, the sentence that best explains
the link, the full score arithmetic and the human context (same section, days apart,
which batch each article came from). It is generated deterministically — no language
model is involved.

**(b) Code location.** `src/evidence.py` — `build_evidence_reason` (`:475`),
`score_breakdown` (`:663-683`), `enrich_recommendation_evidence` (`:730`),
`bm25_term_hits`, `extract_shared_entities`, `extract_shared_key_terms`,
`extract_best_matching_sentence`, `evaluate_signal_agreement`.

**(c) Inputs/outputs.** In: source row, candidate row, BM25 index, BM25 rank, dense rank,
raw rerank probability, the score components, ranks and admission info. Out: a JSON string
stored in `article_recommendations.reason`.

**(d) Algorithm.** Pure formatting: extract shared terms (IDF-ranked, limit 4), shared
entities (capitalised-phrase overlap, limit 3), the best matching sentence (top-overlap of
the first 15, truncated at 180 characters), the BM25 term contributions, and copy the
caller's numbers verbatim.

**(e) Exact values.** Limits: 4 key terms, 3 entities, 15 sentences searched, 180-character
sentence cap, 6-decimal rounding for probabilities. `rounded()` and `weighted()` return
`None` for a missing input rather than raising, which is why `hybrid`/`bm25` rows show
"not recorded" instead of a fake number.

**(f) Worked example (real, stored rank 1 for 10905842).**

```json
"summary": "Shares key entities (Kapil Sharma, Sharma) with 2 of 3 signals corroborated (entertainment)."
"agreement_count": 2  "confidence_band": "MEDIUM"
"boosts": ["Section boost (+10.0%)", "Subsection boost (+5.0%)", "Popularity boost (+1.2%)",
           "Recency boost (+0.8%)", "Agreement boost (+4.0%)"]
"score_breakdown": { "reranker_prob": 0.0917, "bm25_rank": 2, "dense_rank": 8,
   "blend": 0.3584, "components": { "rerank_term": 0.0504, "bm25_norm": 0.0779,
   "bm25_term": 0.0117, "dense_norm": 0.9878, "dense_term": 0.2963 },
   "unboosted_confidence": 0.297, "total_boost_mult": 1.2257,
   "boosted_raw": 0.4394, "platt_calibrated": 0.3748 }
"bm25_terms": [{"term": "kapil", "tf": 14, "idf": 4.251, "weight": 1.0, "contribution": 9.7911}, ...]
"context_facts": { "same_section": true, "same_subsection": true, "date_gap_days": 6.9,
   "source_batch": "Batch batch_0_1 (batch_0_1.csv)", "cand_batch": "Batch 04 (articles_batch_04.csv)" }
```

Every number in that block reconciles with 6.9/6.10/6.12 — that is the point of the stage.

**(g) Edge cases.** `rerank_score = None` (no reranker) → `reranker_prob: null` and the
weighted terms fall back to `None`. An article with no shared entity → the summary falls
back to the term/section text. A candidate whose text is empty → no best sentence.

**(h) Measured ON vs OFF.** Always on; the payload size is measurable: `GET
/api/db/recommendations/10905842` (10 rows with evidence) took **24.18 ms** versus
**3.59 ms** for a single article row, i.e. evidence serialisation is most of that endpoint's
cost.

**(i) Limitations.** The JSON embeds a hard-coded comparison against retired weights:
`provenance.spec_weights = {0.70, 0.20, 0.10}` with `has_weight_mismatch: true` and the
note "Live weights (0.55/0.15/0.3) from empirical tuning vs 0.70/0.20/0.10 initial spec"
(`src/evidence.py:611-628`). Nothing in the running system uses 0.70/0.20/0.10, so the card
reports a permanent, meaningless "mismatch". The `source_batch` string is also malformed
for files that do not start with "articles_batch" (`"Batch batch_0_1 (batch_0_1.csv)"`).

---

## 6.14 Persistence and the reverse push

**(a) Plain-language explanation.** Computing recommendations is expensive, so they are
computed once at ingestion and stored. When a new batch arrives, two things happen: the new
articles get their lists, and old articles that the *new* articles are close to get their
lists refreshed — because a better match may have just arrived. Everything else is left
untouched.

**(b) Code location.** `db/pipeline.py:321-637` (`run_ingestion_pipeline`),
`_precompute_for_article` (`:157-214`), `_reverse_push_for_old` (`:215-320`),
`_rank_and_build` (`:71-154`); storage in `db/repository.py`
(`save_recommendations`, delete-then-insert per source).

**(c) Inputs/outputs.** In: a cleaned frame. Out: `articles` upserts, `articles.embedding`
updates, `article_recommendations` rows (10 per source), and one `ingestion_jobs` row with
progress messages.

**(d) Algorithm.** init schema → version check (force full recompute if the fingerprint
moved) → parse/clean → upsert articles → embed only new/changed/vector-less rows → load
the **whole** corpus → forward precompute for the target set → reverse push for old
neighbours of new articles → reload the in-process engine → mark the job completed.

**(e) Exact values.** `PRECOMPUTE_CANDIDATES = 20`, `PRECOMPUTE_COMMIT_EVERY = 5`
(`config.py:57-58`), `TOP_K = 10`, `INCREMENTAL_RECOMMENDATIONS = True` (`config.py:67`),
`N_RETRIEVE = 50`. Commit every 5 articles bounds how much work is lost on a crash.

**(f) Worked example (real).** Job `job_f489a027` (`articles_batch_04.csv`, 500 rows)
recorded `refreshed_articles = 1403` — meaning 1,403 *existing* articles had their
recommendation list rewritten by the reverse push, versus 500 forward lists. That is why
ingestion time grows as the archive grows: the reverse set is larger than the batch.

**(g) Edge cases.** Re-uploading an unchanged file is idempotent: nothing is new or
changed, and the pipeline exits early. A changed *text* (`text_embed` differs) counts as
changed and triggers re-embedding (`:394-397`). A crash leaves the job `processing`; on
next startup `fail_orphaned_jobs()` marks it failed.

**(h) Measured ON vs OFF.** The full-corpus reload is the dominant cost: 2,000 articles
take ≈ 2.5 s to load but the engine construction that follows took **54.7 s** end-to-end in
the API lifespan measurement (§13), including two model loads.

**(i) Limitations.** The reverse push only *adds* candidates to a stored list; it re-ranks
(stored top-10 ∪ new neighbours), so a candidate that was 11th at original ingest can never
re-enter that article's list. This is the mechanism behind the stored-vs-live divergence
measured in 6.15.

---

## 6.15 Serving: the two paths

**(a) Plain-language explanation.** There are two ways to answer "more like this": read the
answer off a table (fast, frozen), or compute it on the spot (slow, current). The API picks
the table whenever the request looks like the default one, and computes otherwise.

**(b) Code location.** `api/server.py:361-378` (the rule), `:378-441` (fast path),
`:443+` (live path).

**(c) The exact rule.** The fast path is used only when **all** of these hold
(`api/server.py:370-377`): `article_id` present, `query` absent, `section` absent,
`method == "hybrid+rerank"`, `use_boosts` true, `top_k <= 10`. Additionally the table must
return at least `top_k` rows (`:386`).

**(d) What the fast path does.** Reads the 10 stored rows, orders them by `rank` when
`diversify=true` and by `rank_raw` when false (`db/repository.py:268`), runs
`enrich_recommendation_evidence` (pure formatting), and returns. It does not load a model.

**(e) Exact values.** Measured: fast path **23.94 ms** median, live path **14,684.78 ms**
median on local CPU (§13). The fast-path note string claims "sub-5ms"
(`api/server.py:398`) — that is the SQL lookup time, not the endpoint time.

**(f) Worked example (real).** `POST /api/recommend {"article_id": "10905842"}` returned
`method: "postgresql_precomputed_rerank"`, `low_confidence: false`, `best_relevance:
0.3748`, and the 10 stored ids starting `10895859`. The same request with `section`
specified fell through to the live engine and returned a different set of 10 ids.

**(g) Stored vs live divergence (measured, 10 sources).** Comparing the stored top-10 with
a fresh in-memory run of the live engine for the same 10 sources:

| source | overlap@10 | rank-1 identical |
|---|---|---|
| 10469207 | 10 | yes |
| 10895848 | 8 | yes |
| 10897831 | 10 | yes |
| 10901406 | 9 | yes |
| 10903712 | 8 | yes |
| 10905157 | 7 | yes |
| 10906491 | 5 | yes |
| 10907772 | 9 | yes |
| 10908985 | 8 | yes |
| 10910300 | 7 | yes |
| **mean** | **8.10 / 10** | **10 / 10** |

Per-candidate scores agree exactly (verified for 10905842: blend 0.3584, boosted 0.4394,
calibrated 0.3748 reproduce from the live code), so the difference is in *selection*, not
scoring: the stored lists are the product of incremental reverse pushes (6.14), while the
live engine re-derives the pool from the whole corpus. Expected mean disagreement: **1.9 of
10 rows**, always below rank 1.

**(h) Measured ON vs OFF.** Fast path 23.94 ms vs live 14,684.78 ms — a **613×**
difference. Against the GPU worker the live path drops to ≈1.2 s (§13), still ≈50× the fast
path.

**(i) Limitations.** `low_confidence` is hard-coded `false` on the fast path
(`api/server.py:396`), so the flag can only ever fire on the path that serves a minority of
traffic. `best_relevance` means "raw maximum cross-encoder probability" on the live path
and "calibrated score of row 1" on the fast path — the same field, two meanings.

---

## 6.16 Full end-to-end trace: article 10905842 → recommended 10895859

One real article followed from CSV row to served JSON. Every number below was read from
the database or recomputed from the stored vectors and text; nothing is illustrative.

**Source:** `10905842` — *"'So we don't end up in jail': Kapil Sharma on why he watches
younger comics"*, entertainment / Television, published 2026-10-04 10:38:01+05:30,
page views 494, `linked_ids` empty.

| # | Stage | Concrete value for this pair | Where it comes from |
|---|---|---|---|
| 1 | CSV intake | row from `batch_0_1.csv` (1,000 rows, 4.56 MB), job `job_788b7dcb` | `ingestion_jobs` |
| 2 | Clean | title above; section `entertainment`; subsection `Television`; 6.9-day gap to candidate | `articles` row |
| 3 | Text recipe | `embed_token_count = 254`, `rerank_token_count = 254`, `is_truncated = true` | `articles` row |
| 4 | BM25 text | title repeated 3× + description + body | `articles.text_bm25` |
| 5 | Embedding | 768-d unit vector; **cosine(10905842, 10895859) = 0.6395** | dot product of stored vectors |
| 6 | Focused query | 23 terms, beginning `don end jail kapil sharma watches younger comics …` | stored `reason.bm25_query` |
| 7 | BM25 retrieval | self_bm25 = **197.5205**; candidate raw BM25 = **15.3867** (kapil 9.7911 + sharma 5.5956); **BM25 rank 2** | recomputed + `reason.bm25_terms` |
| 8 | Dense retrieval | **dense rank 8** of 50 | `reason.method_ranks.dense` |
| 9 | RRF | 1/(60+2) + 1/(60+8) = 0.016129 + 0.014706 = **0.030835**; pool of 90 → top 20 | formula + trace pool size |
| 10 | Cross-encoder | **0.091696** (rank 1 of 20; pool max 0.7154, mean 0.0965) | `rerank_score` |
| 11 | Normalise | bm25_norm = 15.3867/197.5205 = **0.0779**; dense percentile = **0.9878** | §6.9 |
| 12 | Blend | 0.55×0.091696 + 0.15×0.0779 + 0.30×0.9878 = **0.358458** | `reason.components` |
| 13 | Boosts | editorial 1.0 × section 1.100 × subsection 1.050 × popularity 1.0122 × recency 1.0081 × agreement 1.0400 = **1.225703** | `reason.boost_items` |
| 14 | Boosted score | 0.358458 × 1.225703 = **0.439363** → stored **0.4394** | `boosted_score` |
| 15 | MMR | no movement at ingest (`rank_raw` = `rank` = 1) | `article_recommendations` |
| 16 | Calibration | logit = 4.85×0.358458 − 2.60 + 0.35 = −0.5115 → **0.374847** → stored **0.3748** | `score` |
| 17 | Band | agreement 2 of 3 → probability ≥ 0.35 → **MEDIUM** | `reason.confidence` |
| 18 | Evidence | 2 shared entities (Kapil Sharma, Sharma), 2 BM25 terms, 6.9-day gap, source/candidate batch labels as actually stored (`Batch batch_0_1 (batch_0_1.csv)` — malformed, §18 M3) | `reason` JSON |
| 19 | Persisted | one row, `pipeline_version 6ae306216d00e08c`, written 2026-10-28 12:13:57 | `article_recommendations` |
| 20 | Served | fast path, **23.94 ms**, `method: postgresql_precomputed_rerank`, `low_confidence: false` | API run (§13) |

**Counterfactual (same article, live engine today).** Recomputing from the whole corpus
produces a different winner: **10906391** at boosted 0.6851 / confidence 0.7154, then
10906396 (0.5885 / 0.6152), with 10895859 third (0.4394 / 0.3748). The stored rank 1 is the
stored rank 3 of a fresh run. Nothing is wrong with either number — they were computed
against different candidate pools (§6.14, §6.15g) — but it means the served answer for an
article is a decision made at ingestion time, not a property of the article.

---

# 7. Request Routing and Serving Modes

## 7.1 The two answers to the same question

`POST /api/recommend` has two implementations behind one URL. Which one runs is decided
by a six-clause boolean at `api/server.py:370-377`:

| Clause | Fast path (stored rows) | Live path (computed now) |
|---|---|---|
| `article_id` present | required | required or `query` |
| `query` absent | required | — |
| `section` absent | required | any |
| `method == "hybrid+rerank"` | required | any of four methods |
| `use_boosts` true | required | any |
| `top_k <= TOP_K (10)` | required | any |
| stored rows ≥ `top_k` | required (`:386`) | — |

If any clause fails, the live engine is called (`:433-441`). If the table lookup throws or
returns fewer rows, the exception is swallowed with a printed message and the live path
takes over (`:423-425`) — a silent, self-healing degradation.

## 7.2 Method matrix (live path only)

| `method` | Retrieval | Rerank | Boost | `rerank_score` |
|---|---|---|---|---|
| `bm25` | BM25 only, top 20 | no | yes | `null` |
| `semantic` | dense only, top 20 | no | yes | `null` |
| `hybrid` | BM25 + dense → RRF | no | yes | `null` |
| `hybrid+rerank` | BM25 + dense → RRF | **yes** | yes | probability |

`hybrid+rerank` silently degrades to `hybrid` when the reranker failed to load
(`src/recommender.py:100-101`). The fast path ignores `method` entirely and always reports
`postgresql_precomputed_rerank` (`api/server.py:391`), because the stored rows were all
produced by the reranked pipeline.

## 7.3 What each path returns

Both paths return `method, engine, source, low_confidence, best_relevance, notes,
recommendations[]`, but the fields do not mean the same thing:

| Field | Fast path | Live path |
|---|---|---|
| `source` | `"postgresql"` | `"live"` |
| `best_relevance` | calibrated `score` of row 1 (`:397`) | raw maximum cross-encoder probability (`src/recommender.py:198`) |
| `low_confidence` | always `false` (`:396`) | computed against `MIN_RAW_RERANK_THRESHOLD` |
| `notes` | one line claiming "sub-5ms" (`:398`) | engine notes, e.g. auto-corrections |
| per-rec `reason` | stored JSON, enriched | freshly built JSON |
| per-rec `evidence` | present (enriched) | absent |
| per-rec `rerank_score` | present (`:411`) | absent (`_result_json`, `:236-249`) |
| per-rec `confidence_band`, `margin`, `agreement_count` | **absent** | present (`:243-245`) |

## 7.4 The fallback ("Trending in <section>") block

`_attach_fallback` (`api/server.py:209-222`) counts every article-to-article request and,
**only if fewer than `FALLBACK_MIN_RELATED = 3` recommendations came back**, attaches a
second, clearly labelled list built by `ArticleRepository.get_trending_in_section`
(same section, within `max(7, 3 × half-life)` days of the source, ordered by page views,
capped at `FALLBACK_LIMIT = 5`).

Because the current pipeline has **no admission filter** (§6.0, §19), an article request
returns `min(top_k, 20)` rows — 10 by default — so the trigger condition
(`< 3`) can only be met by asking for `top_k ≤ 2`, by filtering to a section with fewer
than three articles, or on a corpus smaller than three articles. Measured on the running
server: `top_k=2` produced a fallback block; every default request produced 10 rows (§13).

## 7.5 Other serving endpoints

* `GET /api/db/recommendations/{article_id}` reads the same stored rows without the
  enrich step and is the cheapest way to inspect them (24.18 ms measured).
* `GET /api/articles?sort=confidence` orders the *articles* by their average stored
  `score` — the only place where the calibration is aggregated across the corpus.

---

# 8. Ingestion Operations

## 8.1 The job lifecycle

```
POST /api/db/upload-csv ──► ingestion_jobs row (status=processing)
                              │
                              ├─ STAGE:PARSE   → CLEAN → STORAGE → EMBEDDING → INDEX → RERANK → REVERSE
                              │
                              └─ status=completed | failed  (+ error text)
GET /api/db/jobs/{id}  → polled every 1.2 s by the editor UI
```

Every stage transition is written to `ingestion_jobs.message` as `STAGE:<NAME>|<text>`
(`db/pipeline.py`, `upd(message=...)`). Progress fields: `total_articles`,
`processed_articles`, `refreshed_articles`, `pipeline_version`, `error`.

## 8.2 Concurrency and failure handling

* One job at a time: `_require_idle` returns HTTP 409 if any job is `pending` or
  `processing` (`api/editor_routes.py:29-36`). This is enforced before the 50 MB write.
* Jobs run on a daemon thread (`api/editor_routes.py:39-49`), so a process exit kills an
  in-flight ingestion; the row is then cleaned up at next startup by
  `fail_orphaned_jobs()` (`api/server.py:41`).
* Commits happen every `PRECOMPUTE_COMMIT_EVERY = 5` articles (`config.py:58`), so a crash
  loses at most 5 articles of work.
* Version-change detection forces a full recompute when the fingerprint moved since the
  last completed job (`db/pipeline.py:354-360`).

## 8.3 What is recomputed, and at what cost

| Uploaded | Forward lists | Reverse-pushed lists (measured) |
|---|---|---|
| `batch_0_1.csv` (1,000) | 1,000 | 0 (`job_788b7dcb`) |
| `articles_batch_03.csv` (500) | 500 | **981** (`job_743cff70`) |
| `articles_batch_04.csv` (500) | 500 | **1,403** (`job_f489a027`) |

(A fourth job, `job_3c2585d8` for `articles_batch_05.csv`, was still running while this report
was written — see §5.6 and §13.4.)

The reverse set grew from 0 to 981 to 1,403 as the archive filled — the archive effect
described in §5. Each reverse item re-scores only the missing pairs
(`_reverse_push_for_old`), which is why 1,403 refreshed lists cost far less than 1,403 full
precomputes.

## 8.4 Operational endpoints

| Endpoint | Effect | Reversible? |
|---|---|---|
| `POST /api/db/init` | create extension/tables, run `ensure_schema()` | yes (idempotent) |
| `POST /api/db/reload` | rebuild the in-process engine from the DB | yes |
| `POST /api/db/clear` | **deletes all articles and recommendations** | **no** |
| `POST /api/db/ingest-default` | ingests `data/articles_clean_train_80.csv` | n/a — file absent, returns 404 |

`ingest-default` is broken in this tree: the file it points at does not exist, so the
endpoint answers 404. The only working ingestion entry point is a CSV upload.

---

# 9. Personalisation

## 9.1 What exists

`Recommender.recommend` accepts `user_history: list[str]` (`src/recommender.py:95-97`) and
uses it in three ways:

1. **A profile vector** — the last `MAX_USER_HISTORY = 10` readable IDs, weighted `0.8^r`
   by recency (newest first), normalised, and summed into one 768-d centroid
   (`_build_user_profile_vector`, `:322-341`).
2. **A score blend** — `final = (1 − 0.35)·relevance + 0.35·cosine(profile, candidate)`
   (`:223-226`), applied *before* the boosts.
3. **A block list** — history IDs are removed from the candidate pool
   (`:143-146`), so a reader is never shown what they already read.

If history is supplied without a query or an article, the engine invents a dummy query
string and sets all BM25 scores to zero (`:131-135`), so retrieval is purely vector-based.

## 9.2 What does not exist

* **No user model, no user table, no session.** There is no `users` table in the schema
  (§5) and no identifier anywhere in the request payload.
* **No way in.** `RecommendRequest` (`api/server.py:112-119`) has no `user_history` field
  and the call site (`:433-441`) never passes one. Personalisation is therefore
  **unreachable over HTTP**: every request runs with `user_history = None`.
* **No click feedback loop.** Nothing records what was clicked, so `page_views` (a global,
  cumulative counter) is the only behavioural input in the whole system, and it feeds only
  the 1.5%-range popularity boost (§6.10d).

**Plain-language summary:** the recommender contains the mathematics of a personalised
feed — centroid profile, recency decay, already-read suppression — but the product never
supplies the input, so in production it is indistinguishable from a non-personalised
system. Any statement that the platform "personalises per reader" would be unsupported by
the running code.

---

# 10. The Evidence JSON Contract

Every stored recommendation row carries `reason`, a JSON string produced by
`src/evidence.py:build_evidence_reason`. The live path produces the same structure
(`src/recommender.py:283-301`), so a client cannot tell the two apart except by the
`source` field of the response.

## 10.1 Key set

| Key | Type | Meaning | Optionally `null`? |
|---|---|---|---|
| `summary` | string | One-sentence human reading of the link | no |
| `verdict` / `template_why` | string | Short label + templated justification | no |
| `terms` | list[string] | IDF-ranked shared key terms (≤4) | no (may be empty) |
| `entities` | list[string] | Shared capitalised phrases (≤3) | no (may be empty) |
| `source_sentence` / `best_sentence` | string | Best-overlap sentence from source / candidate (≤180 chars) | yes |
| `signals` | list/string | Which retrieval branches found the candidate | no |
| `agreement_count` | int 1–3 | Number of corroborating signals | no |
| `agreement_label` / `status` | string | "2 of 3" and its wording | no |
| `boosts` | list[string] | Display strings, e.g. "Section boost (+10.0%)" | no |
| `context_facts` | object | same section/subsection, date gap, batch names | no |
| `confidence` | object | band, probability, margin | no |
| `score_breakdown` | object | the full arithmetic (see below) | no |
| `rank_relevance` | object | `rank_raw`, `final_rank`, `shift`, wording | no |
| `method_ranks` | object | bm25 rank, dense rank, reranker value/rank | ranks may be `null` |
| `margin_cutoff` | object | distance to the next candidate | yes |
| `bm25_terms` | list[object] | `term, tf, idf, weight, contribution` | no (may be empty) |
| `bm25_query` | string | the exact query string used | no |
| `cosine` | float | raw cosine similarity | no |
| `provenance` | object | weights actually used, boost caps, pipeline version | no |

## 10.2 `score_breakdown` — the audit trail

```json
{
  "reranker_prob": 0.0917, "bm25_rank": 2, "dense_rank": 8, "blend": 0.3584,
  "components": {"rerank_term": 0.0504, "bm25_norm": 0.0779, "bm25_term": 0.0117,
                 "dense_norm": 0.9878, "dense_term": 0.2963,
                 "weights": {"rerank": 0.55, "bm25": 0.15, "dense": 0.3}},
  "unboosted_confidence": 0.297,
  "boost_items": [{"name": "section", "mult": 1.1, "amount": "+10.0%"}, ...],
  "total_boost_mult": 1.2257, "boosted_raw": 0.4394, "platt_calibrated": 0.3748
}
```

The guarantee that makes this contract useful: **the numbers reconcile**. §6.16 checks
every step of this block against the stored vectors, the raw BM25 score and the
calibration constants, and all of them match to the stored precision.

## 10.3 Guarantees and non-guarantees

* **Guaranteed:** deterministic (no model call, no randomness); every boost displayed with
  the multiplier that was actually applied (`enrich_recommendation_evidence` is a pure
  formatter, `src/evidence.py:730`); unrecorded values stay `null` rather than being
  invented.
* **Not guaranteed:** the key set is not versioned and there is no schema validation; the
  `provenance.spec_weights` block advertises weights (0.70/0.20/0.10) that the system does
  not use and flags a permanent `has_weight_mismatch: true` (§6.13i); `source_batch` is
  malformed for batch files not named `articles_batch*`.
* **Not present:** there is no per-row `pipeline_version` *inside* the JSON — it lives in
  the `provenance` block and in the column. A row migrated from an older version keeps
  older numbers with a current-looking reason string.

---

# 11. Security and Operational Safety

## 11.1 Authentication and authorisation

**There is none.** No endpoint in `api/server.py` or `api/editor_routes.py` reads a token,
cookie, header or API key. Consequently:

| Endpoint | Consequence if reachable |
|---|---|
| `POST /api/db/clear` | deletes every article and recommendation (`api/editor_routes.py:189-208`) |
| `POST /api/db/upload-csv` | ingests up to 50 MB of attacker-supplied text into the corpus |
| `POST /api/db/init` | creates schema objects |
| `POST /api/recommend` | costs CPU/GPU per call |

The server binds `0.0.0.0:5000` (`api/server.py:529-531`), so on a shared network every one
of those is available to anyone who can reach the host.

## 11.2 CORS

`allow_origins=["*"]` with credentials disabled — any web page can call the API from a
browser (`api/server.py` `CORSMiddleware`). Combined with 11.1, a malicious page can
trigger `clear` on a reachable instance.

## 11.3 Input handling that *is* guarded

* Upload filename is reduced to its basename (`Path(...).name`, `:92`), so `../../x.csv`
  cannot escape `data/uploads/`.
* Only `.csv` is accepted (`:93`).
* Size is enforced while streaming, and the partial file is deleted on breach
  (`:104-107`).
* All database access goes through SQLAlchemy Core/ORM expressions, so the pipeline is not
  exposed to SQL injection through article text.
* Article text is never evaluated; the evidence JSON is `json.dumps`-ed.

## 11.4 The remote GPU worker

`REMOTE_MODEL_URL` is currently **set and reachable** (measured: `GET /health` returned
`status: ok, device: cuda, gpu: Tesla T4` in 0.85 s). `REMOTE_MODEL_TOKEN` is **not set**,
which means, per the worker's own contract, only `/health` is open and `/embed` and
`/rerank` should reject unauthenticated calls — but the worker in this deployment accepted
unauthenticated work (the audit's own runs embedded and reranked through it with no token,
and `config.REMOTE_MODEL_TOKEN` is empty so the client sent none). The tunnel URL is a
capability: anyone holding it can spend the GPU owner's quota.

Three different tunnel hostnames were observed live during this audit, each replacing the
previous one; a quick tunnel dies with its Colab session, so the URL in `.env` is a
short-lived secret that changes without notice. When it is stale, the client silently falls
back to CPU (`src/model_registry.py`), which changes latency by an order of magnitude
without any error surfacing to the caller.

## 11.5 Secrets handling

`.env` is git-ignored and holds `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`,
`POSTGRES_PORT`, `POSTGRES_DB`, `REMOTE_MODEL_URL` (all set). `REMOTE_MODEL_TOKEN` and
`HF_TOKEN` are absent. No secret is logged by the application. `.env.example` documents the
names. This report does not reproduce any value (§3).

## 11.6 Operational risks to note in handover

1. **Single process only.** `reload_engine_from_postgres()` mutates module globals; running
   uvicorn with `--workers > 1` would give each worker its own engine and its own view of
   the corpus (`api/server.py:81-108`).
2. **No backup or migration story.** Schema changes are applied by `ensure_schema()`; there
   is no `alembic`, no dump script, and `clear` has no confirmation parameter.
3. **Logs go to stdout only.** Nothing is persisted, and a `print`-based `[Fallback]` or
   `[ModelRegistry]` message is the only trace of a degraded mode.
4. **The database password is the only thing protecting the corpus**, because every API
   route is anonymous.

---

# 12. The Evaluation Harness, and What It Actually Says

## 12.1 What `evaluate.py` is

A single 977-line script with four independent parts (`evaluate.py`):

| Part | What it measures | Data source |
|---|---|---|
| 1 | Quality of the **stored** recommendations | `article_recommendations` + `articles` |
| 2 | Four-method comparison matrix (bm25 / semantic / hybrid / hybrid+rerank) | live in-memory engine |
| 3 | Free-text search demo on fixed queries | live in-memory engine |
| 4 | Human blind-label scoring from a CSV | a labels file provided by the user |

CLI: `--source auto|postgres|csv|api`, `--samples N`, `--api-url`, `--labels FILE`,
`--skip-methods`, `--save-baseline`, `--compare BASELINE.json`.

Part 1 is the one that matters for a handover, because it grades **the rows the reader is
actually served** by the fast path (§7), not a fresh recompute.

## 12.2 The command actually run

```powershell
python evaluate.py --source postgres --samples 20 --skip-methods --labels __no_labels__.csv
```

Why those flags: `--source postgres` grades the stored rows; `--samples 20` keeps the
run bounded; `--skip-methods` was used because Part 2 needs the live engine and roughly
triples the runtime; `--labels __no_labels__.csv` is a deliberate non-existent filename
because PowerShell cannot pass an empty string (`--labels ""` fails argument binding) and
Part 4 is optional.

Engine load: **137.65 s** (two model loads from local disk, CPU). Wall clock for the
report: 12:50:49 → 12:56:57.

## 12.3 Real output — Part 1

```
  A. SEMANTIC COHERENCE (Source ↔ Recommendations)
    Avg Cosine Similarity (source → recs):     0.6712
    Rank-Weighted Avg Similarity:               0.6833
    Avg Worst-Match Similarity:                 0.5974
    Std Dev (consistency across articles):      0.0716

  B. EDITORIAL GROUND-TRUTH (Also Read Links)
    Articles with Editor Curated Links:          197
    Precision@5 (Gold):                          0.0914  (9.1% of top-5 match editor picks)
    Hit@5:                                       41.1%  (any editor pick in top 5)
    Hit@10:                                      42.6%  (any editor pick in top 10)
    nDCG@10 (ranking quality):                   0.3939
    MRR (first relevant rank):                   0.3857
    MAP@5 (precision-recall trade-off):          0.3813

  C. SECTION & SUBSECTION ALIGNMENT
    Section Match P@5:                           71.0%
    Subsection Match P@5:                        35.0%

  D. DIVERSITY & REDUNDANCY
    Intra-List Diversity (ILD):                  0.3893  (0=identical, 1=maximally diverse)
    ILD Std Dev:                                 0.0930

  E. CATALOG COVERAGE & POPULARITY BIAS
    Unique Articles Recommended:                 1986 / 2000
    Catalog Coverage:                            99.3%
    Gini Coefficient (0=equal, 1=monopoly):      0.3205
    Top 5% Articles' Recommendation Share:       13.3%

  F. SCORE DISTRIBUTION
    Mean Relevance Score:                        0.7384
    Median Score:                                0.8590
    Std Dev:                                     0.2500
    Min / Max:                                   0.2018 / 0.9356
    % Scores Below 0.3 (low quality):            14.5%

  G. LATENCY
    Avg SQL Lookup Latency:                      10.37 ms
    P95 Latency:                                 7.32 ms
```

## 12.4 How to read those numbers (and the trap in them)

* **The ground-truth block has a different denominator from the rest.** Precision@5,
  Hit@5/10, nDCG@10, MRR and MAP@5 are computed over exactly the **197** articles that carry
  at least one editor link (`evaluate.py:246-257`). Sections C, D and F are computed over a
  **20-article random sample** (`random.sample(..., sample_size)`, seed 42,
  `evaluate.py:261-273`). They are not comparable sample sizes, and a 20-article sample
  cannot support a claim like "71% of recommendations are same-section" beyond ±10 points.
* **Precision@5 of 9.1% is not a failure.** Each source has on average ~1–3 editor links,
  so a 5-slot list can at best score ~20–60% on this metric. Hit@5 of 41.1% (any editor
  pick inside the top 5) is the more meaningful headline.
* **`P95 < mean` for SQL latency is arithmetically fine here**: a handful of cold first
  calls (connection setup) inflate the mean while 95% of the 2,000 lookups stay near 7 ms.
* **Score stats describe the *stored* calibration**, and the minimum (0.2018) is not the
  corpus minimum (0.1192, §5) because only a 20-article sample is measured.
* **No confidence interval is printed anywhere.** With 197 positives and 20 sampled
  articles, none of these numbers carries an error bar, and the harness has no bootstrap.

## 12.5 Real output — Part 3 (free-text, excerpt)

```
Query: 'flight crash'  (Best Relevance: 0.85)
  [74%] Plunged 14,000 feet in 29 secs: How stabbed Indian pilot averted flydubai crash
        signals: BM25 (rank 4, strong), Dense (rank 1, strong), Cross-Encoder (rank 1, 85.3%)
        boosts : Popularity +0.8%, Recency +0.8%, Agreement +15.0%
        terms  : crash (tf 5, idf 3.641, contribution 6.8812), flight (tf 12, idf 2.972, 6.5485)
        cosine : 0.6331 (~99th percentile)
```

This is a clean demonstration that the three signals agree and that the consensus
multiplier pays out its maximum (+15%) when they do. It is also the only place in the
output where a search query is traced end-to-end.

## 12.6 Part 2 was skipped, and Part 4 has nothing to read

* Part 2 (the four-method matrix) was skipped to bound runtime; it is exercised instead by
  the isolation measurements in §6 and §13, which answer the same question per source.
* Part 4 printed: `[Notice] Blind labeling file '__no_labels__.csv' not found.` — i.e. no
  human labels exist in this tree, so **there is no human-judged quality number for this
  system at all**.

## 12.7 What is missing from the harness (verified by inspection)

* `--save-baseline` **crashes**: it reads `config.ARTICLE_DISCARD_FLOOR`
  (`evaluate.py:675`), an attribute that no longer exists in `config.py`. This audit
  deliberately did not run it (it writes a file, and the brief forbids writes), so the
  claim is code-derived, not observed.
* No `baseline_evaluation.json` exists in the repository, so `--compare` has nothing to
  compare against and no regression gate can be evaluated.
* `run_experiments.py`, `freeze_pool.py`, `parity_test.py` and the ablation harness
  referenced by the older documentation are **absent from this tree** (deleted), so no
  tuning result can be reproduced from the current checkout.
* No test asserts any quality threshold. Nothing fails if nDCG@10 halves.

## 12.8 The honest summary for management

The system retrieves topically coherent neighbours (average cosine 0.67, worst-match 0.60)
and covers the catalogue almost completely (99.3%), but agreement with editors' own links
is modest: **41.1% of articles have at least one editor pick in the top 5**, and the
ranking quality of that agreement is nDCG@10 = 0.39. There is no measured human judgement,
no confidence interval, and no regression baseline in the repository.

---

# 13. Measured Performance

Everything below was measured on the handover machine: Windows 11, Intel i5-8365U
(4 cores / 8 threads), 17.05 GB RAM, `torch 2.14.0+cpu`, `cuda_available = False`.
Two configurations appear: **CPU-only** (remote worker stale, local fallback in use) and
**GPU-assisted** (the remote Tesla T4 worker live in `.env`).

## 13.1 In-process (FastAPI `TestClient`, CPU-only, no GPU worker)

| Operation | Median | Notes |
|---|---|---|
| module import | **52,915.9 ms** | mostly torch + transformers import |
| lifespan (startup) | **54,694.0 ms** | `ensure_schema` + `fail_orphaned_jobs` + engine build + 2 model loads |
| RSS after startup | 2,621.2 MB | 2,000 articles, 768-d matrix, two models |
| RSS after workloads | 3,265.2 MB | +644 MB across the requests below |
| `GET /api/status` | 1.56 ms | |
| `GET /api/articles?page=1&per_page=20` | 11.95 ms | |
| `GET /api/articles?section=cities` | 12.63 ms | |
| `GET /api/articles?sort=confidence` | 41.93 ms | aggregate ordering |
| `GET /api/articles/trending?n=6` | 9.37 ms | indexed section+date query |
| `GET /api/articles/{id}` | 3.59 ms | |
| `GET /api/sections` | 3.38 ms | |
| `GET /api/db/recommendations/{id}` | 24.18 ms | 10 stored rows + evidence |
| `GET /api/db/jobs?limit=5` | 4.01 ms | |
| `POST /api/recommend` (fast path) | **23.94 ms** | stored rows |
| `POST /api/recommend` (live, CPU) | **14,684.78 ms** | 20 cross-encoder pairs on CPU |
| `POST /api/recommend` (live, no boosts/MMR) | 13,396.13 ms | boosts are free; the reranker is the cost |
| `POST /api/recommend` (search, CPU) | 18,388.75 ms | query-centred passages |
| `POST /api/recommend` (`method=hybrid`) | 153.64 ms | no cross-encoder |
| `POST /api/recommend` (`method=bm25`) | 126.00 ms | |

## 13.2 Real HTTP against the running server (GPU worker live)

Same endpoint, same article, `127.0.0.1:5000`, three repetitions each:

| Request | median | min | max |
|---|---|---|---|
| fast path (`article_id` only) | **45.5 ms** | 33.6 | 58.3 |
| fast path, `diversify=false` | 34.3 ms | 34.1 | 35.1 |
| live path (section forced) | **999.4 ms** | 989.4 | 2,110.6 |
| live path, no boosts / no MMR | 968.6 ms | 934.6 | 996.4 |
| search `hybrid+rerank` | **2,265.7 ms** | 1,620.4 | 2,717.1 |
| search `method=hybrid` | 438.5 ms | 411.1 | 687.0 |
| search `method=bm25` | 405.7 ms | 364.5 | 416.4 |
| search `method=semantic` | 420.1 ms | 404.7 | 440.1 |

**The GPU worker is worth ≈14.7× on article-to-article** (14,684.78 ms → 999.4 ms) and
≈8× on search. Removing boosts and MMR changes nothing measurable (999.4 → 968.6 ms),
confirming that the entire live cost is the cross-encoder.

## 13.3 Per-stage cost, isolated (GPU worker live, three sources)

| Configuration | median per request |
|---|---|
| full pipeline (boosts + MMR) | 1,232–1,296 ms (first call 75,865 ms = model warm-up) |
| no boosts, MMR on | 1,238–1,258 ms |
| boosts on, MMR off | 1,171–1,246 ms |
| no boosts, no MMR | 1,166–1,213 ms |
| `hybrid` (no rerank) | 80–112 ms |
| `bm25` only | 83–108 ms |
| `semantic` only | 77–122 ms |

Cross-encoder share of a live request: **≈90%** (1.2 s minus the ~100 ms retrieval floor).

## 13.4 Startup cost, and ingestion cost measured from the job log

| Operation | Measured |
|---|---|
| `evaluate.py` engine load (CPU, 2 models) | 137.65 s |
| API lifespan (CPU, 2 models) | 54.69 s |
| corpus load from PostgreSQL (2,000 rows + 768-d matrix) | ≈2.5 s |
| DB round-trip, warm pool | 0–1 ms |
| DB round-trip, cold connect | 58 ms |

Ingestion cost is recorded by the pipeline itself, so it can be read out of `ingestion_jobs`
rather than estimated (read-only SQL; timestamps are `created_at` → `updated_at`):

| Job | Batch | Uploaded | Reverse-refreshed | Elapsed | Per uploaded article |
|---|---|---|---|---|---|
| `job_788b7dcb` | `batch_0_1.csv` | 1,000 | 0 | 02:09:47 → 02:28:35 = **18 m 47 s** | **1.13 s** (forward only) |
| `job_743cff70` | `articles_batch_03.csv` | 500 | 981 | 02:30:06 → 02:50:00 = **19 m 54 s** | 2.39 s |
| `job_f489a027` | `articles_batch_04.csv` | 500 | 1,403 | 11:57:04 → 12:21:08 = **24 m 04 s** | 2.89 s |
| `job_3c2585d8` | `articles_batch_05.csv` | 446 | (in progress) | self-reported: `Precomputed 310/446 (1.1s/article, ETA 44m 20s (incl. ~41m 52s reverse))` | 1.1 s forward |

Two read-outs:

1. **Forward precompute is ≈1.1 s per article on the T4**, and that number is stable across
   jobs (1.13 s measured on job 1, 1.1 s self-reported by job 4).
2. **The reverse pass dominates as the archive grows.** The same 500-article batch took
   19 m 54 s with 1,500 old articles and 24 m 04 s with 2,000 (+21 %), while the number of
   old lists refreshed rose from 981 to 1,403 (+43 %). Job 4's own ETA attributes ~41 m 52 s
   of its 44 m 20 s to the reverse pass. In other words: re-ingesting a fixed batch gets
   slower every time you do it, even though the batch is unchanged.


## 13.5 A measured client-side trap (worth knowing before benchmarking)

While timing the deployed server from `http://localhost:5000`, **every** request took
≈2.04 s — including a static favicon. The same requests to `http://127.0.0.1:5000` took
16–61 ms. Cause: on this host `localhost` resolves to `::1` first, the server listens on
IPv4 `0.0.0.0` only, and each connection stalls ~2 s before falling back. This is a client
configuration issue, not a server defect — but any dashboard, health check or load test
using `localhost` will report a 2-second API that is actually a 45-millisecond API.

## 13.6 Scaling limits implied by these numbers

* Ingestion is O(articles × 20 cross-encoder pairs) forward, plus a reverse pass that grows with
  the archive. Measured: ≈1.1 s/article forward on GPU, 2.4–2.9 s/article once the reverse pass
  is included (§13.4), and ≈13 s/article on CPU (§13.1). A full 2,000-article recompute is
  therefore roughly **40 minutes on GPU** and **≈7 hours on CPU** — and a full recompute is what
  any change to a fingerprint token triggers (`config.py:284-335`).
* The live path cannot serve interactive traffic on CPU (14.7 s/request). Production
  therefore depends on the free-tier tunnel staying alive, which is the single largest
  operational risk in the system (§11.4, §18).
* Retrieval itself is cheap: BM25 + dense + RRF + boosts on 2,000 articles is ~100 ms,
  of which the 2,000×768 matrix product is a few milliseconds.

---

# 14. Models, Dependencies and the Remote GPU Worker

## 14.1 Models in use

| Role | Model | Revision (pinned) | Size of output |
|---|---|---|---|
| Embedding | `BAAI/bge-base-en-v1.5` | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | 768-d unit vector |
| Cross-encoder | `BAAI/bge-reranker-base` | `2cfc18c9415c912f9d8155881c133215df768a70` | 1 probability in [0,1] |
| Fallback lexical | TF-IDF (`sklearn`, 1–2 grams, sublinear) | — | cosine similarity |

Both revisions are pinned in `config.py:41-43`, are part of the pipeline fingerprint
(`config.py:293-295`), and were verified live on the remote worker: `/health` returned
`embedding_revision = a5beb1e3…` and `reranker_revision = 2cfc18c9…`, i.e. **identical to
the local constants**. That is the one property that makes CPU/GPU mixing safe.

## 14.2 Installed dependency versions (measured, not from any file)

| Package | Version | Used for |
|---|---|---|
| Python | 3.13.3 | runtime |
| torch | 2.14.0+cpu | model execution (`cuda_available = False`) |
| sentence-transformers | 6.0.1 | model loading |
| transformers | 5.17.0 | tokenizers, cross-encoder head |
| numpy | 2.4.2 | vectors, scoring |
| pandas | 2.3.3 | CSV handling, corpus frame |
| scikit-learn | 1.8.0 | stop-words, TF-IDF fallback, Platt fitting |
| fastapi | 0.142.2 | API |
| starlette | 1.7.0 | ASGI layer |
| uvicorn | 0.49.0 | server |
| pydantic | 2.13.4 | request model |
| sqlalchemy | 2.0.50 | ORM (2.0 style) |
| psycopg2-binary | 2.9.13 | PostgreSQL driver (actually used) |
| psycopg | **not installed** | listed in `requirements.txt` but absent |
| pgvector | 0.5.1 | `vector(768)` type |
| python-dotenv | 1.2.2 | `.env` loading |
| python-multipart | 0.0.32 | file upload |
| httpx | 0.28.1 | test client |
| psutil | 7.0.0 | not imported by the backend |

**`requirements.txt` contains no version pins at all** (13 bare package names, and it lists
both `psycopg2-binary` and `psycopg[binary]`). The versions above are whatever happened to
be installed; a fresh `pip install -r requirements.txt` today may produce a different stack.

## 14.3 The remote worker (Mode A, live inference)

Defined in `kaggle_gpu_worker.ipynb` (8 cells, nbformat 4). It exposes exactly three
endpoints, and `src/model_registry.py` implements the matching client:

```
GET  /health  -> {status, device, gpu, embedding_model, embedding_revision,
                  reranker_model, reranker_revision}
POST /embed   {"texts": [...]}       -> {"vectors": [[768 floats], ...]}   (L2-normalised)
POST /rerank  {"pairs": [[q, d], …]} -> {"logits": [...]}                  (raw logits)
```

Verified live during this audit: `status: ok`, `device: cuda`, `gpu: Tesla T4`, response in
**0.85 s** for `/health` (which includes the tunnel round-trip).

Client behaviour (`src/model_registry.py`): ping `/health` with a short timeout; on any
failure or timeout, **silently** use the local CPU model instead. Requests are chunked
(128 items per call) with a 120 s timeout. `REMOTE_MODEL_TOKEN` is optional; when set it is
sent as a bearer token and the worker rejects `/embed` and `/rerank` without it.

## 14.4 Consequences of the fallback design

* Results are identical either way (same pinned weights), so correctness is not at risk.
* Latency changes by an order of magnitude, and **nothing surfaces that fact to the caller**
  beyond a printed `[ModelRegistry] [!] Remote worker ... not responding` line.
* The tunnel URL is ephemeral. Measured in this audit: three different hostnames appeared
  in `.env` within a few hours; two of them were already dead by the time they were used.
* The free Colab/Kaggle session expires. When it does, production silently becomes a
  14-second-per-request service.

---

# 15. API Reference

## 15.1 Public recommendation API (`api/server.py`)

| Method | Path | Parameters | Response |
|---|---|---|---|
| `POST` | `/api/recommend` | JSON body (below) | recommendations payload |
| `GET` | `/api/status` | — | engine name, article count, sections, config, fallback stats |
| `GET` | `/api/articles` | `page ≥ 1` (default 1), `per_page 1–100` (default 20), `section`, `sort=date\|confidence` | `{total, page, per_page, articles[]}` |
| `GET` | `/api/articles/trending` | `n 1–20` (default 6) | most-read articles |
| `GET` | `/api/articles/{article_id}` | — | one article |
| `GET` | `/api/sections` | — | 37 section names |
| `GET` | `/` | — | frontend shell (out of scope, §0) |
| `GET` | `/favicon.svg`, `/favicon.ico` | — | icons |

### `POST /api/recommend` request body (`api/server.py:112-119`)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `query` | string? | `null` | free-text search |
| `article_id` | string? | `null` | "more like this" |
| `top_k` | int | `10` | rows returned |
| `method` | string | `"hybrid+rerank"` | `bm25`, `semantic`, `hybrid`, `hybrid+rerank` |
| `use_boosts` | bool | `true` | the six multipliers of §6.10 |
| `diversify` | bool | `true` | MMR / `rank` vs `rank_raw` ordering |
| `section` | string? | `null` | hard filter (disables the fast path) |

Validation: at least one of `query` / `article_id` is required, else HTTP 400
(`api/server.py:363-367`). An unknown `article_id` on the live path raises `KeyError` →
HTTP 404 (`:458-459`). An unhandled error returns 500 with the exception text
(`:460-462`) — exception strings are therefore exposed to clients.

**There is no `user_history` field**, so personalisation (§9) cannot be requested.

### Response shape — the two paths do not return the same keys

Measured against the running server with `POST {"article_id":"10905842","top_k":10}`
(fast path) and the same request with `"section":"india"` added (live path):

```
FAST path  method: postgresql_precomputed_rerank
  top-level keys : best_relevance, diversify, engine, low_confidence, method, notes,
                   recommendations, source, source_article
  rec[0] keys    : article, boosted_score, evidence, rank, rank_raw, reason,
                   rerank_score, score
  confidence_band / margin / agreement_count -> None (absent)

LIVE path  method: hybrid+rerank
  rec[0] keys    : agreement_count, article, boosted_score, confidence_band, margin,
                   rank, rank_raw, reason, score
  rerank_score / evidence -> absent
```

Fast-path payload (`api/server.py:390-416`, per-rec keys built at `:399-414`):

```json
{ "method": "postgresql_precomputed_rerank", "engine": "...", "source": "postgresql",
  "source_article": {...}, "diversify": true,
  "low_confidence": false,                 // hard-coded, see §18
  "best_relevance": 0.3748,                // = row-1 `score`, see §18
  "notes": ["⚡ Served directly from PostgreSQL precomputed table (sub-5ms, ...)"],
  "recommendations": [ { "article": {...}, "score": 0.3748, "boosted_score": 0.4394,
                         "rank": 1, "rank_raw": 1, "rerank_score": 0.091696,
                         "reason": "{...enriched JSON...}", "evidence": {...} } ],
  "fallback": {...}                        // only when the list is short (§7.4)
}
```

Live-path payload (`_result_json`, `api/server.py:225-252`):

```json
{ "method": "hybrid+rerank", "engine": "...", "source": "live",
  "low_confidence": false, "best_relevance": 0.7154,   // max raw CE prob
  "notes": [],
  "recommendations": [ { "article": {...}, "score": 0.3748, "boosted_score": 0.4394,
                         "confidence_band": "MEDIUM", "margin": 0.0498,
                         "agreement_count": 2, "rank": 1, "rank_raw": 1,
                         "reason": "{...}" } ],
  "fallback": {...}
}
```

**A client must handle both key sets.** `confidence_band`, `margin` and `agreement_count`
exist only on the live path; `rerank_score` and `evidence` exist only on the fast path.
Field semantics also differ — see §7.3 for `best_relevance`, `low_confidence` and
`rerank_score`.

## 15.2 Database and editor API (`api/editor_routes.py`, prefix `/api/db`)

| Method | Path | Parameters | Effect |
|---|---|---|---|
| `GET` | `/health` | — | connectivity + pgvector + table presence |
| `POST` | `/init` | — | create extension/tables, `ensure_schema()` |
| `POST` | `/upload-csv` | multipart `file` (≤50 MB, `.csv`) | start ingestion; 409 if busy, 413 if too big, 400 if not `.csv` |
| `POST` | `/ingest-default` | `limit` (int?) | **404 in this tree** (target CSV absent) |
| `GET` | `/jobs/{job_id}` | — | one job row |
| `GET` | `/jobs` | `limit` (default 10) | recent jobs + summary |
| `GET` | `/recommendations/{article_id}` | `limit` (10), `diversify` (true) | stored rows with evidence |
| `POST` | `/clear` | — | **deletes all articles and recommendations** |
| `POST` | `/reload` | — | rebuild the in-process engine |

## 15.3 Error codes in use

| Code | When |
|---|---|
| 400 | no `query` and no `article_id`; upload is not `.csv`; DB has no articles indexed |
| 404 | article not found in the index; `/ingest-default` with a missing file |
| 409 | an ingestion job is already active |
| 413 | upload exceeds 50 MB |
| 500 | any unhandled exception, with the exception text in `detail` |

# 16. Tests — What Exists, What Runs, What Passes

## 16.1 Inventory: there is no automated test suite

```
PS> Test-Path tests
False
PS> Test-Path .github
False
PS> Get-ChildItem -Force | Select-Object Name
.git  api  data  db  frontend  learning  scratch  scripts  src  venv  __pycache__
.env  .env.example  .gitignore  AGENTS.md  config.py  evaluate.py  explain.md
import_colab_export.py  kaggle_gpu_worker.ipynb  opencode.md
PROJECT_HANDOVER_FACTS.md  README.md  requirements.txt
```

No `pytest.ini`, `tox.ini`, `pyproject.toml`, `setup.cfg` or `conftest.py` exists at the
repository root, and there is no `.github/` directory, so no CI ever runs anything.
`git status --porcelain` additionally reports:

```
 D tests/test_integrity.py
```

i.e. the only unit-test file the project ever had was **deleted from the working tree**
(the deletion is not committed). `tests/` therefore does not exist on disk.

| Location | Contents | Runnable assertions? | In git? |
|---|---|---|---|
| `tests/` | **absent**; one deleted file `tests/test_integrity.py` | — | deletion uncommitted |
| `scratch/verify_all_fixes.py` | two printed check groups over `src/scoring.py` / `src/evidence.py` | yes — prints `PASSED`, exit 0 | no (untracked) |
| `scratch/test_evidence_restructure.py` | 35 assertions in 5 groups | yes — prints `[PASS]`/`[FAIL]`, exit 1 on failure | no (untracked) |
| `scratch/test_live_recommend.py` | builds the live engine from the DB and prints results | no assertions — exploratory | no |
| `scratch/v2_blend_compare.py`, `v2_blend_results.json`, `blend_comparison_results.json` | offline blend comparison artefacts | no | no |
| `scripts/recompute_recommendations.py`, `scripts/recompute_target_articles.py`, `scripts/refresh_evidence.py` | maintenance jobs that **write to the database** | no | no |
| `learning/*.md` (3 files, 102 KB) | study notes | no | yes |

Two consequences for handover: the only files that behave like tests live in an untracked
directory (`scratch/`), so a fresh clone contains **no tests at all**; and the three tracked
`scripts/*.py` are write-capable maintenance jobs, not tests.

## 16.2 Run 1 — `scratch/verify_all_fixes.py` → PASS

```powershell
python scratch/verify_all_fixes.py
```

Real output (trimmed to the tail):

```
Amazon Data Center 3-way consensus (expected 1.15): 1.15
Semantic match without keywords (expected 1.00): 1.0
Embedding drift penalty (expected 0.35): 0.35

=== 2. VERIFYING EVIDENCE & CONFIDENCE (NO 999, NO CONTRADICTION) ===
Confidence for score=0.35 with 3-signal agreement: HIGH (44.95%)
Signals for missing BM25 (no 999):
   BM25 Keyword (not in top-50 keyword results)
   Dense Vector (Rank #2 ? ? Strong)
   Cross-Encoder (Rank #1, 95.0% ? ? Strong)

All architectural verification checks PASSED!
EXIT=0
```

The `? ?` is the Windows console code page failing to render the `★` glyph that
`src/evidence.py:334` emits; the stored string is correct.

**What it covers:** the consensus multiplier tiers (`src/scoring.py:183-232`), the embedding
drift penalty, the "missing BM25 must not be encoded as rank 999" convention, and that the
confidence vocabulary stays HIGH/MEDIUM/LOW. It is a smoke test: each value is printed next
to the constant it expects, and the pass condition is a human reading it. Exit code 0.

## 16.3 Run 2 — `scratch/test_evidence_restructure.py` → 32 passed, 3 failed

```powershell
$env:PYTHONPATH='.'
python scratch/test_evidence_restructure.py
```

`PYTHONPATH` is required because this script imports `config` and `src.*` from the repository
root and does not bootstrap `sys.path` itself.

Real output (trimmed):

```
  [PASS] Provenance records live blend weights
  [PASS] Provenance flags spec mismatch (0.65/0.15/0.20 vs 0.70/0.20/0.10)
  [PASS] Provenance records boost caps

--- 5. Testing enrich_recommendation_evidence formatter ---
  [PASS] Enriched rec contains confidence_band 'HIGH'
  [PASS] Enriched rec contains cosine_display
  [PASS] Enriched rec contains provenance block
  [PASS] Enriched rec contains rank_relevance promotion description

==========================================
Summary: 32 passed, 3 failed
==========================================
EXIT=1
```

The three failures, verbatim:

```
  [FAIL] BM25 #1 and CE #4 (Dense #8 > 5) yields 2 of 3 agreement
  [FAIL] Only CE #3 in top-5 yields 1 of 3 agreement
  [FAIL] No signal in top-5 yields 0 of 3 agreement
```

| Assertion | Test expects | Code returns | Why |
|---|---|---|---|
| BM25 #1, CE #4, Dense #8 | 2 | **3** | dense rank 8 is within the MODERATE band 6–15 |
| BM25 #12, Dense #18, CE 0.88 | 1 | **2** | BM25 rank 12 is MODERATE; CE 0.88 is STRONG |
| BM25 #10, Dense #15, CE 0.20 | 0 | **2** | both ranks are MODERATE; CE 0.20 contributes nothing |

These are **stale expectations, not regressions**. `evaluate_signal_agreement` counts a signal
when it is STRONG *or* MODERATE (`src/evidence.py:364`), where MODERATE means candidate rank
6–15 (`src/scoring.py:114`) or cross-encoder probability 0.35–0.55 (`src/scoring.py:117`).
The tests encode an older rule in which only a top-5 hit counted. The current rule is
deliberate — it must match the boost table in `consensus_multiplier()` (`src/scoring.py:158-232`)
so the displayed count equals the multiplier actually applied — but nobody updated the tests,
and nobody has decided which behaviour is correct (§18).

Note also the third PASS line above: the test **enshrines** the hard-coded provenance flag
`0.70/0.20/0.10`, which is neither the live weights (0.55/0.15/0.30, `config.py:101-103`)
nor the value named in the test's own title (0.65/0.15/0.20). The mismatch flag is therefore
permanently `true` in every evidence blob (§10.3, §18).

## 16.4 Run 3 — the only end-to-end verification is `evaluate.py`

The `evaluate.py` run in §12 is the only check in this audit that exercises stored data,
joins, and metric computation together. Its value and its limits:

| It does prove | It does not prove |
|---|---|
| 20,000 stored rows (snapshot A) load and join to `articles` | that any individual recommendation is correct |
| 1,986 of 2,000 articles receive recommendations, so the fast path serves real data | that the score ordering is right (no labels: "Also Read" ground truth exists for 197/2,000 sources) |
| Evidence JSON parses for every sampled row | that the live and stored paths agree (§7, §18) |
| Semantic coherence is computable (0.6712 avg) | that the models behave as configured (GPU vs CPU not asserted) |

`evaluate.py` never fails on quality — it prints numbers. There is no threshold that turns a
regression into a non-zero exit code, so it cannot be used as a gate.

## 16.5 What is not tested at all

| Capability | Test in tree | Status |
|---|---|---|
| CSV intake → clean → dedup → Also-Read extraction | none | untested (the deleted `tests/test_integrity.py` used to cover this) |
| Ingestion pipeline end-to-end (`run_ingestion_pipeline`, `db/pipeline.py:321`) | none | untested — would require DB writes, so out of scope for this audit |
| Reverse push (`_reverse_push_for_old`, `db/pipeline.py:215`) | none | untested; measured indirectly in §7/§18 |
| `_focused_query` determinism (`db/pipeline.py:52`) | none | untested — and it is not deterministic (§18) |
| API contract / routing (`api/server.py`) | none | untested in repo; one ad-hoc `TestClient` run was made for §13 |
| Calibration fit (`fit_calibrator_on_corpus`, `src/calibration.py:100`) | none | the function is never called anywhere (§17) |
| Remote-GPU parity | `parity_test.py` **deleted** (`git status`) | untested in this tree |
| Fallback block (`_attach_fallback`, `api/server.py:209`) | none | untested; measured live for §7.4 |
| Evidence JSON key stability | partially — `test_evidence_restructure.py` | the key set is asserted, not its numeric truth |
| Personalisation | none | unreachable over HTTP anyway (§9, §19) |

## 16.6 Verdict

| Measure | Value |
|---|---|
| Automated test files that exist | 2 (`scratch/`, both untracked) |
| Assertions executed | 35 (plus printed checks with no assertion) |
| Assertions failing | **3** |
| Suite exit code | **1** |
| CI pipelines | 0 |
| Tests present in a fresh clone | **0** |

The honest statement for management: **the code base currently has no test suite.** What it
has is two untracked smoke scripts, one of which fails three assertions for a documented
reason, plus an evaluation script that reports quality without enforcing it. Any change to
scoring, ingestion or the API is unguarded by automation today.

---


# 17. Code-Quality Scan

## 17.1 Scope, tooling, and the limits of this scan

```
PS> $files = (Get-ChildItem -Recurse -Include *.py -File |
              Where-Object { $_.FullName -notmatch '\\venv\\|__pycache__' }).FullName
PS> "$($files.Count)"
27
```

27 Python files (≈6,100 lines of application code, plus the untracked `scratch/` and
`scripts/` helpers). No static analyser is installed in this environment:

```
ruff   : NOT INSTALLED
flake8 : NOT INSTALLED
pylint : NOT INSTALLED
mypy   : NOT INSTALLED
bandit : NOT INSTALLED
```

So every finding below comes from a regex or AST scan written for this audit, with the exact
command shown. **Limits, stated plainly:** there is no type checking, no security linter, and
no complexity analysis in this report. What follows covers markers, exception handling,
unused imports, zero-reference definitions, and comment-vs-code contradictions — not style.

All 27 files parse (`ast.parse`, no bytecode written):

```
parsed OK: 27 of 27
syntax errors: none
```

Largest modules (lines, `Measure-Object -Line`):

| Module | Lines | Role |
|---|---|---|
| `evaluate.py` | 881 | evaluation harness (§12) |
| `src/evidence.py` | 845 | evidence JSON + formatter (§10) |
| `db/pipeline.py` | 558 | ingestion (§8) |
| `src/data_loader.py` | 542 | cleaning, token budgets (§6.2–6.3) |
| `api/server.py` | 461 | HTTP + routing (§7, §15) |
| `db/repository.py` | 378 | SQL access (§5) |
| `src/reranker.py` | 349 | cross-encoder (§6.8) |
| `src/recommender.py` | 344 | live engine (§7) |
| `src/retrieval.py` | 316 | BM25 / dense (§6.5–6.6) |
| `config.py` | 286 | every tunable (§3) |
| `src/scoring.py` | 265 | blend / boosts / MMR (§6.9–6.11) |

## 17.2 TODO / FIXME / HACK / XXX / "not implemented"

```powershell
Select-String -Path $files -Pattern 'TODO|FIXME|HACK|\bXXX\b|not implemented|NotImplemented'
```

```
config.py:49: # When set to a valid URL (e.g. https://xxx.trycloudflare.com), embeddings and rerankings
```

**One match, and it is not a marker** — it is the literal `xxx.trycloudflare.com` inside a
comment explaining the tunnel URL format. There are **zero** `TODO`, `FIXME`, `HACK` or
`NotImplemented` markers in the code base.

This is genuinely good news and it also means something less good: the known defects in §18
are not flagged anywhere in the source. A new maintainer gets no in-code warning that the
admission filter is missing, that `PRECOMPUTE_CANDIDATES` does nothing, or that the evidence
provenance block is hard-coded.

## 17.3 Exception handling

Bare `except:` — none:

```
--- bare except ---
none
```

`except Exception` — 63 occurrences across 17 files:

| File | Count | File | Count |
|---|---|---|---|
| `api/server.py` | 11 | `src/evidence.py` | 7 |
| `evaluate.py` | 11 | `db/connection.py` | 6 |
| `api/editor_routes.py` | 4 | `db/pipeline.py` | 4 |
| `src/retrieval.py` | 4 | `src/model_registry.py` | 3 |
| `src/recommender.py` | 3 | `db/models.py` | 2 |
| `src/data_loader.py` | 2 | `config.py` | 1 |
| `src/scoring.py` | 1 | `src/calibration.py` | 1 |
| `src/reranker.py` | 1 | `scripts/refresh_evidence.py` | 1 |
| `scratch/v2_blend_compare.py` | 1 | | |

Catching broadly is defensible for a demo service, but four of these sites convert a failure
into a *silently different answer*, which matters for handover:

| Site | What the catch does | Consequence |
|---|---|---|
| `src/model_registry.py:87-89, 139-141` (plus the health check at `:150-161`) | any remote error → use the local model | GPU→CPU downgrade with no user-visible signal; latency ×10 (§13.3, §14.4) |
| `api/server.py:423-424` | precomputed lookup error → `print()` and fall through | the request is silently served by the live engine instead, ~20× slower and a different list (§7) |
| `api/server.py:338-343` | `/api/articles/trending` error → `{"articles": []}` with HTTP 200 | an empty widget, no error code |
| `src/evidence.py:419-420` | batch-file scan error → print and continue | `source_batch` falls back to a fabricated label (§18) |

## 17.4 Unused imports

```
evaluate.py:32: unused import 'defaultdict'  (from collections.defaultdict)
evaluate.py:34: unused import 'Tuple'  (from typing.Tuple)
db\pipeline.py:23: unused import 'extract_shared_entities'  (from src.evidence.extract_shared_entities)
api\server.py:13: unused import 'List'  (from typing.List)
scripts\recompute_target_articles.py:6: unused import 'np'  (from numpy)
scripts\recompute_target_articles.py:9: unused import 'Article'  (from db.models.Article)
scripts\refresh_evidence.py:14: unused import 'json'  (from json)
```

Seven in code that runs, all harmless. The scanner also flagged two categories that are **not**
defects and should not be "fixed" blind:

* `db/__init__.py:5-7` — package re-exports; they are the module's public surface.
* `db/connection.py:191` — `from . import models  # noqa`, an import-for-side-effect that
  registers the tables before `Base.metadata.create_all`. It is already marked `noqa` and the
  scanner ignores that comment.

## 17.5 Zero-reference definitions

AST scan for functions defined and never named anywhere else:

```
api\server.py:291 'get_articles'            → route handler (decorator-referenced)
api\server.py:482 'serve_index'             → route handler
api\editor_routes.py:57-212 (9 functions)   → route handlers
src\calibration.py:100  'fit_calibrator_on_corpus'   → genuinely dead
src\data_loader.py:40   'first_words'                → genuinely dead
src\data_loader.py:156  'get_tokenizer'              → genuinely dead
src\data_loader.py:294  'build_token_budgeted_texts' → genuinely dead
```

After removing the 14 decorator-referenced route handlers and the middleware
(`_revalidate_static`), **four** functions are truly unreferenced:

| Function | Significance |
|---|---|
| `src/calibration.py:100 fit_calibrator_on_corpus` | the routine that produced the Platt constants — so the reported calibration cannot be reproduced from this repository (§6.12) |
| `src/data_loader.py:40 first_words` | leftover from the retired word-count recipe |
| `src/data_loader.py:156 get_tokenizer` | the loader builds its own tokenizer path elsewhere |
| `src/data_loader.py:294 build_token_budgeted_texts` | the batch version of the text composer; the pipeline calls the per-row composer instead |

## 17.6 Config knobs that do nothing

| Knob | Declared | Read by | Effect |
|---|---|---|---|
| `PRECOMPUTE_CANDIDATES = 20` (`config.py:57`) | "fused candidates sent to the reranker during ingestion" | **only** `evaluate.py:661` (baseline record) | **none** — ingestion uses `N_CANDIDATES` (`db/pipeline.py:501`). Tuning it changes nothing, silently |
| `BODY_WORDS_EMBED` / `BODY_WORDS_RERANK = None` (`config.py:83-84`) | retired word counts | only `evaluate.py:677-678` | none (intentionally retired, but still written into every baseline) |
| `MIN_RELEVANCE["tfidf"]` (`config.py:263`) | light-mode threshold | never — `recommender.py:210` only asks for the semantic index | none in neural mode |

`BOOST_MODE` is *not* in this list: it is read by `config.recency_ref_date`
(`config.py:244-254`), which both paths call.

## 17.7 Comment-vs-code contradictions found by the scan

| Claim in the code | What the code does | Cite |
|---|---|---|
| "Lower Filter: relevance admission (techniques 1 + 2)" | nothing is filtered; the lines below only build `source_linked_set`. No `admission_cut`, `RELEVANCE_FLOOR` or `ELBOW_*` exists anywhere in the tree | `src/recommender.py:214-217` |
| "Served directly from PostgreSQL precomputed table (sub-5ms, …)" — shown to every API client | measured 33.6–45.5 ms per request (§13.2), ~10× the claim | `api/server.py:398` |
| `"engine": "PostgreSQL pgvector + Precomputed Cross-Encoder"` | no vector search happens at serve time — a btree join on `(article_id, rank)`; the vector column is only read to build the in-memory engine at startup | `api/server.py:392`, §5.3 |
| `spec_weights = {0.70, 0.20, 0.10}` + `has_weight_mismatch` | live weights are `0.55/0.15/0.30`; the flag is therefore permanently `true` and the "spec" exists nowhere in the repo | `src/evidence.py:613-619`, `:944-949` |
| Confidence band fallback at `0.60` / `0.35` | the calibrator that wrote the stored band uses `0.65` / `0.35`; the enricher can relabel the same row differently | `src/evidence.py:924` vs `src/calibration.py:36-37` |
| "…corroborated by N of 3 signals in top-5" | agreement now also counts candidate ranks 6–15, so "top-5" in the sentence a reader sees is false for those rows | `src/evidence.py:931`, `src/scoring.py:114` |
| "calibrated against 50,000 random document pairs from the news corpus" | **NOT FOUND** — no script or artefact in the repository reproduces `COSINE_QUANTILES` | `src/scoring.py:44-46` |
| "Relevance blend weights (must sum to 1.0)" | 0.55 + 0.15 + 0.30 = 1.00 ✓ (example of a comment that holds) | `config.py:94-103` |

## 17.8 Verdict

| Check | Result |
|---|---|
| Files parse | 27 / 27 |
| TODO/FIXME/HACK markers | **0** |
| `NotImplemented` stubs | 0 |
| Bare `except:` | 0 |
| Broad `except Exception` | 63 (4 of them hide a changed behaviour) |
| Unused imports in running code | 7 |
| Truly dead functions | 4 |
| Config knobs with no effect | 3 |
| Misleading comments/strings shown to users or auditors | 8 |

The code is readable and internally consistent where it counts — one blend implementation
(`src/scoring.py`), one evidence builder, no stubs, no bare excepts. The problems are not
sloppiness; they are **stale claims**: comments, UI strings and evidence fields that describe
a system the code no longer is. For a handover, those are more dangerous than dead code,
because they will be believed.

---


# 18. Verified Defects and Risks

Everything in this section was observed in this tree: measured, executed, or read directly
from the cited line. Nothing here is inherited from `PROJECT_AI_SPEC.md` or from the
`AGENTS.md` memory file; where those documents disagree with the code, §17.7 records the
disagreement and this section describes what the code does.

**Severity scale used here**

| Level | Meaning |
|---|---|
| Critical | data loss or unauthorised access reachable in normal operation |
| High | a reader or API consumer receives a wrong/misleading answer, or a documented workflow cannot run |
| Medium | a maintainer or tuner is misled, or a latent failure waits for a change |
| Low | cosmetic, or an edge case outside the shipping corpus |

**Register**

| # | Severity | Defect | One-line effect |
|---|---|---|---|
| C1 | **Critical** | no authentication; CORS `*`; `POST /api/db/clear` deletes everything | anyone who can reach the port can erase the corpus |
| H1 | High | stored recommendation lists are frozen snapshots stamped with the *current* algorithm version | two correct answers to "more like this" (§7.1) |
| H2 | High | `low_confidence` is hard-coded `false` on the fast path | the only "no good match" signal never fires where it matters |
| H3 | High | displayed `score` is not monotonic in `rank` — 1,996 of 2,020 sources | looks broken; sorting by score contradicts the order |
| H4 | High | the two serving paths return different key sets | a client cannot read one field list for both |
| H5 | High | `evaluate.py --save-baseline` crashes at the end of a full run | the regression baseline cannot be refreshed |
| H6 | High | dependencies are unpinned, and two PostgreSQL drivers are declared | a fresh install is not reproducible |
| H7 | High | the GPU worker is an unauthenticated public tunnel on a URL that dies with the session | silent 10× slowdown, free GPU for anyone with the link |
| H8 | High | the BM25 signal contributes ≤ 0.012 of a 1.0 blend despite a 0.15 weight | keyword matching is effectively switched off |
| M1 | Medium | `best_relevance` means two different things on two paths | dashboards compare unlike numbers |
| M2 | Medium | `provenance.has_weight_mismatch` is permanently `true` and cites weights that exist nowhere | every audit trail claims a contradiction |
| M3 | Medium | `context_facts.source_batch` is malformed and, for unknown IDs, fabricated | evidence states provenance that is wrong |
| M4 | Medium | `POST /api/db/ingest-default` returns 404 — its CSV is deleted | a documented entry point is dead |
| M5 | Medium | engine state lives in module globals behind one lock | `--workers > 1` serves stale models (single process only) |
| M6 | Medium | the dense term is ~83 % of a traced blend, driven by a quantile table that cannot be regenerated from the repo | one global curve decides relevance |
| M7 | Medium | Platt constants and the cosine table are outside the pipeline fingerprint | changing them silently invalidates every stored score |
| M8 | Medium | the focused BM25 query is built from a `set`, so term order is not reproducible | re-ingesting identical data can change stored lists |
| M9 | Medium | there is no relevance admission gate anywhere | 10 rows are always returned, 14.5 % below 0.30 confidence |
| M10 | Medium | `top_k > 10` or any `section` filter silently leaves the fast path | 45 ms → ~1 s, different answer, no warning |
| L1 | Low | synthetic IDs embed a timestamp | ID-less CSV re-upload duplicates the corpus |
| L2 | Low | `evidence.admission` is always `null` | a documented contract field carries no information |

---

## C1. No authentication, open CORS, and a destructive endpoint

**What.** `POST /api/db/clear` calls `ArticleRepository.clear_all_data(db)`
(`api/editor_routes.py:190-196`), which deletes every article and every recommendation.
No route in the application checks a credential: there is no API key, no session, no
`Depends(...)` guard anywhere. CORS allows every origin:

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)   # api/server.py:65-68
```

**Evidence.** `grep` for authentication primitives across `api/`, `db/`, `src/` returns
nothing (NOT FOUND): no `HTTPBasic`, `OAuth2`, `APIKeyHeader`, `Depends(security...)`.
The server binds `0.0.0.0:5000` (`api/server.py:529-531`), so it is reachable from the LAN.

**Impact.** Any machine on the same network — or any web page a staff member visits, because
`allow_origins=["*"]` and the browser will happily send the request — can call `/api/db/clear`
and destroy 20,000 recommendations and 2,000 articles. There is no backup job in the repo
(§19). This is the single highest-severity item in the report.

---

## H1. Stored lists are frozen snapshots, but stamped with the current algorithm version

**What.** The fast path serves rows from `article_recommendations` verbatim
(`api/server.py:378-422`). Those rows were written during ingestion. When later batches
arrive, only *new* articles and *old articles close to a new article* are recomputed
(`_reverse_push_for_old`, `db/pipeline.py:215`). An old article that no new article is close
to keeps its list forever.

**Evidence (measured, 10 sources, live GPU engine vs stored rows).**

| Measure | Result |
|---|---|
| Mean overlap of the two top-10 lists | **8.10 / 10** |
| Sources whose top-10 is identical | **2 / 10** |
| Sources whose rank-1 is identical | **10 / 10** |
| Per-pair scores for the same (source, candidate) pair | identical |

Worked example: for source `10905842`, the stored rank-1 is `10895859`
(`boosted_score` 0.4394, `score` 0.3748, `rerank_score` 0.0917). Recomputed live against
today's corpus, the top three are `10906391` (0.6851 / 0.7154), `10906396` (0.5885 / 0.6152)
and then `10895859`. Both lists are "correct" for their own corpus.

The crucial detail: `article_recommendations.pipeline_version` equals
`config.get_pipeline_version()` today (`6ae306216d00e08c`, §5.4). So the stamp says "current
algorithm" — and it is — but the fingerprint covers **code and parameters, not the corpus**.
A list computed when 1,000 articles existed is indistinguishable from one computed against
2,000.

**Why the two paths can disagree while every score matches.** `rank-1` is stable and per-pair
scores are identical, so the scoring code is consistent. The difference is *selection*: the
live path chooses from the whole corpus; the stored list was chosen from the corpus that
existed at write time (stored `created_at` for this source: 2026-10-28 02:20; newest article
`updated_at`: 11:58).

**Impact for management.** "Similar articles" is a function of *when an article was ingested*
as much as of its content. Two readers looking at different articles get lists computed
against different-sized corpora, and no field in the API says so.

---

## H2. `low_confidence` is hard-coded `false` on the fast path

**What.** The fast path builds its payload with a literal:

```python
"low_confidence": False,                                   # api/server.py:396
"best_relevance": precomputed[0].get("score", 0.0),        # api/server.py:397
```

The live path computes it properly from the raw cross-encoder maximum
(`src/recommender.py:198-199`).

**Evidence.** Real responses from the running server for the same source:

```
FAST  {"method":"postgresql_precomputed_rerank", "low_confidence": false, "best_relevance": 0.3748}
LIVE  {"method":"hybrid+rerank",                 "low_confidence": false, "best_relevance": 0.7154}
```

Both say `false` here, but for different reasons: the fast path says it unconditionally; the
live path says it because 0.7154 > 0.12. The stored rank-1 cross-encoder probability for that
pair is **0.0917** — below the 0.12 threshold — so a genuine low-confidence case exists and the
fast path still reports `false`.

**Impact.** `low_confidence` is the only "nothing really matches" signal the API exposes
(`config.py:258`). It is always `false` on the path that serves every article-to-article click
with default settings. A reader cannot be warned; a dashboard cannot count it.

---

## H3. The displayed `score` contradicts the order (99 % of sources affected)

**What.** Ranking uses `boosted_score` (relevance × boosts, `db/pipeline.py:95-103`). The
`score` field shown to users and returned by the API is the Platt-calibrated value including a
+0.35-logit bonus per extra agreeing signal (`src/calibration.py:47-55`). The two are not
monotonic in each other.

**Evidence (read-only query over every stored row at the time; 1,996 of 2,020 sources had at least one inversion).**

```
sources with non-monotonic stored score by rank: 1996 of 2020
```

Example, source `10560602`:

| rank | score | boosted_score |
|---|---|---|
| 3 | 0.8500 | 0.9481 |
| 4 | **0.8786** | 0.9101 |
| 9 | 0.6710 | 0.7401 |
| 10 | **0.7562** | 0.7111 |

Rank 4 displays a *higher* confidence than rank 3, and rank 10 higher than rank 9. The cause is
the agreement bonus: a candidate corroborated by all three signals gets a large calibration
lift even though it ranks lower.

**Impact.** Any consumer that sorts or compares by `score` — including a human reading the
JSON — sees an ordering violation. The design is intentional (§6.12) but nothing in the payload
distinguishes "ordering key" from "confidence". See also M1.

---

## H4. The two serving paths return different key sets

**What.** Measured key lists for one recommendation, same endpoint:

```
FAST : article, boosted_score, evidence, rank, rank_raw, reason, rerank_score, score
LIVE : agreement_count, article, boosted_score, confidence_band, margin, rank, rank_raw, reason, score
```

**Cause.** The fast path emits stored columns directly (`api/server.py:399-414`); the live path
emits a fixed dict (`_result_json`, `api/server.py:236-249`). Neither includes the other's
fields.

**Impact.** `confidence_band`, `margin` and `agreement_count` are `undefined` for every
article-to-article request with default parameters — the common case. `rerank_score` and
`evidence` are `undefined` on the live path. A client written against one path silently
renders blanks on the other; there is no versioning or schema in the response to detect it.

---

## H5. `evaluate.py --save-baseline` cannot run

**What.** `save_baseline()` records:

```python
"article_discard_floor": config.ARTICLE_DISCARD_FLOOR,     # evaluate.py:675
```

`config.ARTICLE_DISCARD_FLOOR` does not exist — `grep` over `config.py` finds no such name.
Accessing it raises `AttributeError`.

**Evidence.** Code-derived, **deliberately not executed**: `--save-baseline` rewrites
`baseline_evaluation.json`, which is a project file, and this audit is read-only. The failure
point is unambiguous: the attribute is referenced and never defined.

**Impact.** The documented "refresh the reference" workflow (§12.7) fails *after* the full
evaluation has run — six minutes of work lost — and no baseline is written. `--compare`
therefore has nothing current to compare against; the only baseline files in git history are
both deleted in this working tree.

---

## H6. Dependencies are unpinned, and two PostgreSQL drivers are declared

**What.** `requirements.txt` in full:

```
pandas
numpy
scikit-learn
sentence-transformers
fastapi
uvicorn[standard]
python-dotenv
sqlalchemy
psycopg2-binary
psycopg[binary]
pgvector
python-multipart
```

No version constraints anywhere; no lock file; `torch` and `transformers` are not listed at all
(they arrive transitively through `sentence-transformers`).

**Evidence of how far an install can drift.** Measured environment versus what the GPU
notebook asserts (§14.2): `sentence-transformers` **6.0.1 installed vs 6.1.0 asserted**,
`transformers` **5.17.0 installed vs 5.19.0 asserted**. Both drivers are present
(`psycopg2-binary` 2.9.13 and `pgvector` 0.5.1); the code imports `psycopg2`
(`db/connection.py`), so `psycopg[binary]` is installed and unused.

**Impact.** A fresh install today does not reproduce today's scores. The embedding model
*revision* is pinned (`config.py:41,43`), which protects the vectors, but the cross-encoder's
numerics depend on the installed `transformers`/`torch`, and no test would catch a change.
This is the classic handover failure: "it worked on the machine we built it on".

---

## H7. The GPU worker is an unauthenticated public tunnel

**What.** The Colab/Kaggle notebook (`kaggle_gpu_worker.ipynb`) can require an
`X-Worker-Token` header — `_check_token` raises 401 when `WORKER_TOKEN` is set — and
`/health` reports `auth_required`. The client sends the header only if
`REMOTE_MODEL_TOKEN` is set (`src/model_registry.py:20-35`).

**Evidence (live, this session).**

```
REMOTE_MODEL_URL set: True   len: 55   token set: False
status = ok        device = cuda        gpu = Tesla T4
auth_required = False
embedding_revision = a5beb1e3...   (matches config.py:41)
reranker_revision  = 2cfc18c9...   (matches config.py:43)
```

So: the worker is open to anyone with the URL, the client holds no token, and the tunnel URL
changed three times during a single working session (it dies with the Colab runtime).

**Impact.** Two separate problems.

1. *Security:* free use of the GPU by anyone who obtains the URL; no rate limit, no origin
   check (`api/server.py` has none either).
2. *Silent degradation:* when the tunnel dies, `model_registry` catches the error and uses the
   local CPU model (`src/model_registry.py:87-89, 139-141`). Nothing in the API response says
   which device answered. Measured effect: ~1.2 s per article request on the T4 versus ~13 s
   on CPU (§13.3) — a 10× change with no signal.

---

## H8. The BM25 signal contributes at most 0.012 of a 1.0 blend

**What.** `self_normalized_bm25` divides each candidate's BM25 score by the *source article's
own* BM25 self-score (`src/scoring.py:75-83`), capped at 1.0. For article-to-article, the
source article matches its own focused query almost perfectly, so the denominator is huge
relative to every neighbour.

**Evidence (article `10905842`, traced).** `self_bm25 = 197.5205`; the best neighbour BM25 =
16.1088, so `bm25_norm ≤ 0.082`. The BM25 term is `WEIGHT_BM25 × bm25_norm = 0.15 × 0.082 ≈`
**0.012** — about 1.2 % of the score, while the weight in `config.py:102` advertises 15 %.

**Impact.** Keyword overlap — names, places, case names, product names — is effectively
invisible to the blend. A story that shares only an uncommon entity with the source cannot
outrank a story that is merely in the same neighbourhood of embedding space. This is the
mechanism behind the qualitative complaint that recommendations "drift topic" (see M6).

---

## M1. `best_relevance` means two different numbers

| Path | Value | Scale |
|---|---|---|
| Fast (`api/server.py:397`) | row-1 `score` | Platt-calibrated 0.01–0.99, includes agreement bonus |
| Live, `hybrid+rerank` (`src/recommender.py:198`) | `max(raw rerank)` | raw cross-encoder probability |
| Live, `bm25` (`:204`) | max normalised BM25 | 0–1, scale-free |
| Live, `semantic` (`:209`) | max cosine | 0.25–0.95 for this corpus |

Measured for the same source: `0.3748` (fast) vs `0.7154` (live). Both are "the best relevance"
of the same request. Any threshold, alert or dashboard built on this field is comparing
different quantities depending on the query shape.

## M2. The provenance block is permanently in a mismatch state

`src/evidence.py:613-619` (and the same block re-created at `:944-949`) hard-codes
`spec_weights = {rerank: 0.70, bm25: 0.20, dense: 0.10}` and sets
`has_weight_mismatch = (live != those values)`. Live weights are `0.55 / 0.15 / 0.30`
(`config.py:101-103`), so the flag is `true` in **every** evidence blob, and the accompanying
sentence tells the reader that the live weights deviate from an "initial spec" of
0.70/0.20/0.10 that does not exist in the repository. The unit test that asserts this
behaviour is itself titled "0.65/0.15/0.20 vs 0.70/0.20/0.10" (§16.3) — three different weight
triples, none of them the live one. An auditor reading the JSON concludes the deployed weights
are wrong.

## M3. `source_batch` is malformed and sometimes invented

`get_article_batch` (`src/evidence.py:404-421`) derives the label from the filename:
`stem.replace("articles_batch_", "").replace("articles_", "")`. Executed against this tree:

```
get_article_batch('10905842') -> 'Batch batch_0_1 (batch_0_1.csv)'
get_article_batch('10910523') -> 'Batch batch_0_1 (batch_0_1.csv)'
get_article_batch('99999999') -> 'Batch 01 (articles_batch_01.csv)'
```

Three problems: the label reads "Batch batch_0_1"; unknown IDs are silently assigned
`Batch 01 (articles_batch_01.csv)`, a file that **was deleted from the tree**
(`git status`: `D data/articles_batches/articles_batch_01.csv`); and the first call reads
seven CSVs (≈20 MB) synchronously inside a request to populate a module-global map. The stored
evidence JSON of every article therefore states a batch provenance that is wrong.

## M4. `POST /api/db/ingest-default` cannot work

`api/editor_routes.py:131-133` resolves `data/articles_clean_train_80.csv` and raises 404 if it
is missing. `Test-Path data/articles_clean_train_80.csv` → `False` (and `config.py:18` still
lists it as a default dataset). The only working ingestion entry point is a CSV upload. This
matters beyond cosmetics: the repository's own README-style instructions and `config.py` still
name this dataset, so a new operator's first attempt fails.

## M5. Single-process engine assumption

The live engine and corpus are module-level globals (`api/server.py`, `rec`, `articles`)
swapped under `_reload_lock` (`:35`) by `reload_engine_from_postgres()` (`:81-84`). Ingestion
runs in a background thread of the *same* process (`api/editor_routes.py:39-49`) and reloads
only that process. With `uvicorn --workers 2`, a second worker keeps serving the old index;
with a separate ingestion process, no worker reloads. There is no shared cache, no pub/sub, no
version check between processes. The deployment must remain a single process, and nothing
enforces that.

## M6. One global curve decides most of the score

`cosine_percentile` (`src/scoring.py:61-72`) maps raw cosine to a percentile using a fixed
101-value table (`COSINE_QUANTILES`, `:46-58`) that the comment attributes to "50,000 random
document pairs". No script in the repository produces that table (NOT FOUND). In the traced
example a raw cosine of **0.6395** becomes **0.9878**, and the dense term is **82.7 %** of the
winning blend (§6.9). Consequences: (a) the blend is dominated by a curve rather than by the
embedding model's discrimination; (b) the curve cannot be re-derived or validated from the
repo; (c) it is not in the fingerprint (M7), so recalibrating it changes every stored score
without triggering a recompute.

## M7. The fingerprint does not cover everything that changes scores

`config.get_pipeline_version()` (`config.py:284-335`) hashes 33 tokens: models, revisions,
token limits, weights, boost multipliers and mode, MMR, recipe, consensus thresholds. It does
**not** include:

| Omitted constant | Where | Effect of changing it |
|---|---|---|
| `DEFAULT_PLATT_A = 4.85`, `DEFAULT_PLATT_B = -2.60` | `src/calibration.py:32-33` | every stored `score` and band becomes incomparable |
| `COSINE_QUANTILES` (101 values) | `src/scoring.py:46-58` | every stored `boosted_score` moves |
| `BOOST_CAP`-style editorial constant `1.40` | `src/scoring.py:269` | ordering changes; the version does not |

The first two are read at score time by the pipeline, so this is not theoretical: editing
either file silently produces rows that look current.

## M8. The focused query is not reproducible

```python
top_terms = sorted(set(body_toks), key=lambda t: -bm25.idf.get(t, 0.0))[:15]   # db/pipeline.py:52
```

The sort key is IDF only. Terms that occur the same number of times have identical IDF — which
is most terms — and ties are resolved by the iteration order of a Python `set`, which depends
on string hashing (randomised per process unless `PYTHONHASHSEED` is fixed). Two ingestions of
the same CSV can therefore build different 15-term queries, which changes BM25 ranks, which
changes the candidate pool under RRF, which changes the stored list.

**Impact.** "Re-ingest to check" is not a valid reproduction test for this pipeline, and two
environments cannot be compared row-for-row.

## M9. There is no relevance admission gate

`src/recommender.py:214` carries the comment `# --- Lower Filter: relevance admission
(techniques 1 + 2) -----------` but the lines that follow only compute
`source_linked_set`; no candidate is dropped. A repository-wide grep finds **no**
`admission_cut`, `RELEVANCE_FLOOR` or `ELBOW_*` symbol (NOT FOUND). The pipeline likewise
ranks every candidate it has (`db/pipeline.py:95-103`).

**Evidence of the consequence.** Every default request returns 10 rows. In the stored corpus,
**14.5 %** of recommendations have `score < 0.30` (§12.3, part F), and the minimum stored
`score` is 0.1192. The evidence JSON carries an `admission` field that is `None` for every row
because no caller passes one (`db/pipeline.py:137-152`, `src/evidence.py:490`).

## M10. Two ordinary parameters silently leave the fast path

The fast-path condition (`api/server.py:370-377`) requires `top_k <= config.TOP_K` (10) **and**
no `section`. Raising the slider above 10, or filtering to a category — both normal product
interactions — re-runs the whole live engine. Measured (§13.2):

| Request | Time |
|---|---|
| fast path, `top_k=10` | **45.5 ms** |
| live path (`section=india`) | **999.4 ms** |
| live, no boosts/no MMR | 968.6 ms |

Nothing in the response warns the client that it changed path; only `source` and `method`
differ. The `notes` string on the fast path claims "sub-5ms" (§17.7), which makes the contrast
worse, not better.

## L1. Synthetic IDs are not idempotent

For a CSV with no ID column, IDs are `art_<md5(time)[:6]>_<n>` (`src/data_loader.py:478-481`).
The prefix changes on every run, so re-uploading the same file creates a second copy of the
corpus instead of updating the first. All shipped batch files carry `articleId`, so the live
database is unaffected — this is a trap for future uploads only.

## L2. `admission` is always `null`

`build_evidence_reason(..., admission: Optional[dict] = None)` (`src/evidence.py:490`) is
called without that argument by both producers (`db/pipeline.py:137`, `src/recommender.py:283`),
so `evidence["admission"]` is `None` in every stored row (20,000 of 20,000 at snapshot A). The key is documented in
§10.1 as part of the contract; it carries no information.

---

## 18.9 Things that look like defects but are intentional

| Looks wrong | Actually |
|---|---|
| `boosted_score` exceeds 1.0 (max 1.5741) | it is relevance × up to 1.9629 of multipliers (§6.10); only `score` is bounded to 0.99 |
| stored list ≠ live list | by design (incremental maintenance); the defect is H1's missing staleness marker, not the incrementality |
| `rank_raw` differs from `rank` | MMR and boosts are supposed to reorder (§6.10–6.11); measured shift examples in §6.10 |
| P@5 of 0.0914 against "Also Read" | the target is editorial placement, available for only 197 of 2,000 sources; it is a weak proxy, not a quality verdict (§12.4) |
| many recommendations share the source's section (71 %) | the section boost is deliberate (§6.10) and is the only boost with a measured positive effect |

## 18.10 Which defects a given audience will actually notice

| Audience | Will notice | Will not notice |
|---|---|---|
| Reader | H1 (list changes without the article changing), H3 (score/order mismatch if shown), M9 (always 10 items, some irrelevant) | C1, H6, M7 |
| Engineer integrating the API | H4 (missing keys), H2 (`low_confidence` useless), M10 (latency cliff), M1 (two `best_relevance` scales) | M6, M8 |
| Data/ML owner | H8 (keyword signal dead), M6 (dense curve dominates), M8 (non-reproducible), M7 (fingerprint gaps) | C1 |
| IT / security | C1 (no auth, `*` CORS, destructive endpoint), H7 (open GPU tunnel), H6 (unpinned deps) | H3, M3 |
# 19. Capabilities That Do Not Exist

This section exists so that nobody inherits a false expectation. Each row was checked by
searching the code; "NOT FOUND" means the symbol, table, column or route does not exist
anywhere in this tree. Nothing in this list is a judgement about whether it *should* exist —
that is §20.

| # | Capability one might assume | What actually exists | Evidence |
|---|---|---|---|
| 1 | A relevance floor that drops weak candidates | **Nothing.** Every request returns `min(top_k, 20)` rows; 14.5 % of stored rows score below 0.30 | `src/recommender.py:214` (comment only); no `admission_cut`/`RELEVANCE_FLOOR`/`ELBOW_*` (NOT FOUND) — §18 M9 |
| 2 | Personalisation ("for you") | `Recommender.recommend(user_history=...)` exists, but **no HTTP field reaches it** | `api/server.py` `RecommendRequest` has no `user_history`; §9 |
| 3 | Any authentication or authorisation | none | no `HTTPBasic`/`OAuth2`/`APIKeyHeader`/`Depends` guard (NOT FOUND) — §11, §18 C1 |
| 4 | Versioned schema migrations or rollback | additive `ensure_schema()` only: `ADD COLUMN IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS` | `db/connection.py:123-161` — §5.5 |
| 5 | An approximate-nearest-neighbour index (pgvector HNSW/IVFFlat) | none — retrieval is an in-memory NumPy scan; the DB has btree indexes only | §5.3 index listing; `src/retrieval.py` |
| 6 | Response caching / CDN | none for API responses; only `Cache-Control: no-cache` on static files | `api/server.py:514` |
| 7 | A job queue, retries, or resumability | one daemon thread per upload; no retry; a killed job is relabelled `failed` at next startup | `api/editor_routes.py:39-49`, `api/server.py:41` |
| 8 | Multi-process or multi-instance serving | single process by construction — engine in module globals | `api/server.py:35, 81-84` — §18 M5 |
| 9 | More than 10 stored recommendations per article | `article_recommendations` holds exactly 10 per source (`rank` 1..10) | §5.4 rank distribution |
| 10 | Click / dwell / CTR feedback | no table, no column, no endpoint. `page_views` and `unique_users` are static CSV columns | `db/models.py`; event/click tables NOT FOUND |
| 11 | A/B testing or experiment assignment | none | NOT FOUND |
| 12 | Metrics, tracing, structured logs, alerts | one `/api/status` JSON plus `print()` statements | `api/server.py:257-290` |
| 13 | Rate limiting or quotas | none | NOT FOUND |
| 14 | Backup / restore / retention | no job, no script, no policy in the repository | NOT FOUND |
| 15 | Corpus versioning, so a stored list knows which corpus produced it | only `pipeline_version`, which fingerprints **code**, not data | `config.py:284-335` — §18 H1 |
| 16 | Editable or re-runnable tuning and parity harness | `run_experiments.py`, `freeze_pool.py`, `parity_test.py`, `tests/test_integrity.py`, `baseline_evaluation.json` and `blind_labeling_sample.csv` are all **deleted in this working tree** | `git status` — §16 |
| 17 | Deleting a single article | no delete endpoint; the only removal path is `/api/db/clear`, which removes everything | `api/editor_routes.py:190-196` |
| 18 | Non-English content, entity resolution, topic classification | none; the only entity handling is BM25 fuzzy matching of Indian names | `src/retrieval.py` fuzzy block |

**Plain-language summary for management.** The system does four things: it ingests news CSVs,
it precomputes 10 "more like this" links per article, it serves those links very fast, and it
can recompute live on demand. It does **not** know who the reader is, does not learn from what
readers click, does not decide that nothing matches, does not survive a second server process,
and has no operational safety net (no auth, no backup, no metrics, no rollback). Those are not
oversights in the code so much as scope that was never started.

---



---

# 20. Handover Recommendations

Ordered by risk, not by effort. Each item names the file to touch and how to tell it worked.
Nothing here requires a rewrite; every item is a local change to an existing function.

## 20.1 Before anyone else touches the system (safety — hours, not days)

| # | Action | Where | Why | Done when |
|---|---|---|---|---|
| 1 | Put the API behind a credential or bind it to `127.0.0.1`; require a token for `POST /api/db/clear` | `api/server.py:58-68`, `api/editor_routes.py:190` | C1: anyone on the LAN, or any web page a staff member visits, can delete the corpus | unauthenticated `/api/db/clear` returns 401/403 |
| 2 | Restrict CORS to the real front-end origins | `api/server.py:65-68` | C1 | `allow_origins` is an explicit list |
| 3 | Decide the GPU-tunnel question in writing: set `WORKER_TOKEN` on the worker **and** `REMOTE_MODEL_TOKEN` locally, or state that an open GPU is accepted | `.env` (not committed), notebook cell 3 | H7: currently `auth_required = false` on a public URL | `/health` reports `auth_required: true` and inference still works |
| 4 | Take and schedule a `pg_dump` | outside the repo | no backup/retention exists anywhere (§19 #14) | a restore into a scratch database has been done once |
| 5 | Pin the environment: `pip freeze` into a lock file; drop `psycopg[binary]` if unused; record the GPU worker's versions | `requirements.txt` | H6: no version constraints at all; two PostgreSQL drivers declared | a fresh venv reproduces today's `/health` and `evaluate.py` numbers |

## 20.2 First two weeks (correctness and reproducibility)

| # | Action | Where | Why |
|---|---|---|---|
| 6 | Compute `low_confidence` on the fast path from the stored `rerank_score` maximum, using the same `MIN_RAW_RERANK_THRESHOLD` as the live path — and decide `best_relevance` once (recommended: raw CE maximum on both paths, calibrated `score` reported separately) | `api/server.py:396-397` | H2, M1 |
| 7 | Emit one response schema from both paths: add `confidence_band`/`margin`/`agreement_count` to the fast path (they are inside the stored evidence) and `rerank_score`/`evidence` to the live path | `api/server.py:399-414`, `:225-252` | H4 |
| 8 | Fix baseline production: remove the dead `article_discard_floor` key or restore the constant, then capture one baseline on a **frozen** corpus (no job running) | `evaluate.py:675` | H5 |
| 9 | Remove the `spec_weights` block or replace it with the live values; fix or drop `source_batch` | `src/evidence.py:613-619, 944-949, 404-421` | M2, M3 |
| 10 | Recreate `tests/` (the file is deleted, not absent from history), add a contract test for both response shapes, and resolve the three stale agreement assertions one way or the other | `tests/`, `src/evidence.py:288-365` | §16.3, §16.6 — no automated guard exists today |
| 11 | Make the focused query deterministic: `sorted(set(body_toks), key=lambda t: (-bm25.idf.get(t, 0.0), t))` | `db/pipeline.py:52` | M8 |
| 12 | Make list staleness visible: store `corpus_size` / `computed_at_corpus_max_date` on the recommendation rows, or expose it in the API | `db/models.py`, `db/pipeline.py` | H1 — today a frozen list is indistinguishable from a fresh one |
| 13 | Delete `PRECOMPUTE_CANDIDATES` or use it in `db/pipeline.py:501` | `config.py:57` | §17.6 — a tunable with no effect |

## 20.3 First quarter (product decisions this code cannot make for you)

| # | Decision | Options | Consequence to accept |
|---|---|---|---|
| 14 | Is a short, honest list better than 10 items? | (a) implement a relevance floor + keep the existing "Trending in `<section>`" fallback; (b) delete the admission comment and the "techniques 1+2" claim | (a) makes `low_confidence` and the fallback block meaningful; (b) keeps recall at the cost of always filling the slot |
| 15 | Rescue the lexical signal | normalise BM25 by the candidate maximum or by a corpus percentile (as dense already is) instead of by the source's self-score | H8: today `0.15` of weight delivers ≤ 0.012 of score |
| 16 | Make the score reproducible | commit the script that produced `COSINE_QUANTILES`, regenerate it, and add it plus the Platt constants to the fingerprint | M6, M7 |
| 17 | Rebuild quality measurement | more ground-truth links, human labels (the sample CSV is deleted), a metric with a pass/fail threshold in CI | §12.4: P@5 is 0.0914 against a 197-source target; there is no gate |
| 18 | Personalisation: in or out? | if in, add `user_history` to `RecommendRequest` — the engine already supports it | §19 #2, privacy review needed |
| 19 | Replace the free Colab tunnel | a stable inference host, or CPU-only with an async queue and a smaller reranker | H7, §13.6: interactive traffic is impossible on CPU (14.7 s/request) |

## 20.4 What not to do

* **Do not tune weights or boost multipliers yet.** The frozen-pool harness (`freeze_pool.py`,
  `run_experiments.py`) is deleted and there is no current baseline. Any change to a
  fingerprint token triggers a full recompute (≈40 min on GPU, ≈7 h on CPU) with nothing to
  compare against.
* **Do not run `--save-baseline` while an ingestion job is active.** The corpus was growing
  during this audit (§5.6); the resulting baseline would mix two different datasets.
* **Do not run `uvicorn --workers > 1`.** The engine is process-local (§18 M5); extra workers
  serve whatever they loaded at boot.
* **Do not treat `pipeline_version` as proof of freshness.** It fingerprints code, not corpus
  (§18 H1) — the live example in this report proves it: a list computed against 1,000 articles
  carries the same stamp as one computed against 2,446.
* **Do not quote `boosted_score` as a probability.** It reaches 1.5741 by design (§6.10).
* **Do not trust the "sub-5ms" note** in the API's `notes` field; measured 33.6–45.5 ms (§13.2).

## 20.5 The handover snapshot to record on the day of transfer

Copy these values into the handover document, with a timestamp, before anyone changes anything:

| Item | How to obtain | Value at the time of this report |
|---|---|---|
| Algorithm fingerprint | `python -c "import config;print(config.get_pipeline_version())"` | `6ae306216d00e08c` |
| Corpus size | `select count(*) from articles` | 2,000 at snapshot A; 2,446 at snapshot B (§5.6) |
| Stored recommendations | `select count(*), count(distinct article_id) from article_recommendations` | 20,000 / 2,000 (A); 23,100 / 2,285 (B) |
| Distinct `pipeline_version` in stored rows | `select distinct pipeline_version from article_recommendations` | `6ae306216d00e08c` only |
| Token budgeting actually applied | `select max(embed_token_count), max(rerank_token_count) from articles` | 254 / 254 |
| Text recipe | `select distinct recipe_version from articles` | `v2_sentence_token_budget` |
| Last completed job | `select id, filename, updated_at from ingestion_jobs order by updated_at desc limit 1` | `job_f489a027`, `articles_batch_04.csv`, 12:21:08 |
| Worker health | `GET {REMOTE_MODEL_URL}/health` | `device=cuda`, `gpu=Tesla T4`, `auth_required=false`, revisions match `config.py:41,43` |
| Quality baseline | `python evaluate.py --source postgres --samples 20 --skip-methods --labels __none__.csv` | §12.3 (Part 1 numbers) |

## 20.6 If only three things get done

1. **Close the front door** (auth on `/api/db/*`, CORS list, token on the GPU worker). Nothing
   else in this report can hurt the organisation as quickly as C1.
2. **Restore a measurement loop** (fix `save_baseline`, recreate `tests/`, capture one baseline
   on a frozen corpus). Without it, every later change is an opinion.
3. **Fix `low_confidence` and the response schema** on the fast path. These are one-line
   changes that make the API honest about what it knows, and they remove the two differences
   between the fast and live paths that a client is most likely to trip over.

---

# VERIFICATION LOG

Every statement in this report came from one of the commands below, run in this working tree
on **2026-10-28 (13:18 local)**. Nothing was copied from `PROJECT_AI_SPEC.md`, `AGENTS.md`,
`README.md`, or any other document; where those documents disagree with the code, §17.7
records the disagreement and the report follows the code.

## V.1 Read-only protocol actually followed

| Rule | How it was honoured |
|---|---|
| One deliverable | `PROJECT_HANDOVER_FACTS.md` — the only project file created. No other file in the repository was created, modified, moved or deleted |
| No DB writes | every SQL statement executed was `SELECT` or a SQLAlchemy query; no `INSERT`/`UPDATE`/`DELETE`/`TRUNCATE`/`DROP`, no `ensure_schema()`, no ingestion invoked by this audit |
| No destructive endpoints | `/api/db/clear`, `/api/db/init`, `/api/db/reload`, `/api/db/upload-csv`, `/api/db/ingest-default` were **never called** |
| No disk writes by experiments | in-memory engine builds used `use_disk_cache=False`, so `data/cache/` was not written (it is empty) |
| Helper scripts | live only in `C:\Users\Dell\AppData\Local\Temp\opencode\` — outside the repository |
| Secrets | `.env` values were never printed. Only these facts are reported: each `POSTGRES_*` and `REMOTE_MODEL_URL` is set; `REMOTE_MODEL_TOKEN` and `HF_TOKEN` are not; the URL is 55 characters |
| Frontend | not inspected; out of scope (§1.2) |

**Incidental side effect, disclosed:** importing the project under Python creates/refreshes
`__pycache__` bytecode directories inside the repository. They already existed, are
gitignored, and contain no source. No `.py` file's contents changed.

**One environment change not caused by this audit:** a fourth ingestion job was started at
13:12:37 (see §5.6). It was observed, never touched.

## V.2 Environment

| Check | Command | Result |
|---|---|---|
| Python | `venv\Scripts\python.exe -V` | 3.13.3 |
| OS / CPU / RAM | `platform`, `systeminfo`-level probes | Windows 11, i5-8365U (4C/8T), 17.05 GB |
| torch | in-process probe | 2.14.0+cpu, `cuda_available=False` |
| installed libs | in-process probe | sentence-transformers 6.0.1, transformers 5.17.0, numpy 2.4.2, pandas 2.3.3, scikit-learn 1.8.0, fastapi 0.142.2, sqlalchemy 2.0.50, psycopg2-binary 2.9.13, pgvector 0.5.1 |
| `psycopg` | import probe | **not installed** (§14.2) |
| linters | `Get-Command ruff/flake8/pylint/mypy/bandit` | all NOT INSTALLED (§17.1) |
| git state | `git status --porcelain` | 34 entries: 18 deletions, 9 modifications, 7 untracked (includes this report) |
| tests dir | `Test-Path tests` | `False` |
| CI | `Test-Path .github` | `False` |

## V.3 Database (read-only)

| Probe | Result |
|---|---|
| server / extension | PostgreSQL 18.6, pgvector 0.8.7 |
| `count(*)` articles / recs / distinct sources | 2,000 / 20,000 / 2,000 at snapshot A; 2,446 / 23,100 / 2,285 at 13:18 (§5.6) |
| jobs | 3 completed + 1 processing (`job_3c2585d8`, 348/446 at 13:18:45) |
| versions | `article_recommendations.pipeline_version` = `6ae306216d00e08c` only, equal to `config.get_pipeline_version()` |
| token budgets | `max(embed_token_count) = 254`, `max(rerank_token_count) = 254` |
| text recipe | `v2_sentence_token_budget` on 100 % of rows |
| truncation | 1,987 of 2,000 articles flagged `is_truncated` |
| editorial links | 197 articles carry ≥ 1 `linked_ids` entry |
| score ranges | `score` 0.1192–0.9496 (avg 0.7598); `boosted_score` 0.1053–1.5741; `rerank_score` 0.0000–1.0000 |
| indexes | `\di` — btree only, no vector index (§5.3) |
| non-monotonic `score` by `rank` | 1,996 of 2,020 sources (§18 H3) |

## V.4 Live serving path

| Probe | Result |
|---|---|
| running server | `venv\Scripts\python.exe api/server.py`, PIDs 21900 / 17456, port 5000, 911 MB RSS |
| worker `/health` | `status=ok, device=cuda, gpu=Tesla T4, auth_required=False`, both revisions equal to `config.py:41,43` |
| fast path | `POST /api/recommend {"article_id":"10905842"}` → `postgresql_precomputed_rerank`, keys as quoted in §15.1 |
| live path | same body + `"section":"india"` → `hybrid+rerank`, keys as quoted in §15.1 |
| fallback block | `top_k=2` attached a 5-item block; `/api/status` reported 26 requests / 2 triggered (rate 0.0769) |
| stored vs live (`live_vs_stored.py`) | 10 sources: mean overlap@10 = 8.10/10, identical top-10 = 2/10, identical rank-1 = 10/10 |
| full trace (`trace_table.py`) | source 10905842 → 10 rows with every blend component, boost and calibration step reproduced to 4 dp (§6.16) |
| ON/OFF isolation (`isolation.py`) | 3 sources × 7 configurations; timings and overlaps quoted in §13.3 and §6.10 |
| HTTP latency (`http_timing.py`) | fast 45.5 ms (min 33.6), live 999.4 ms, live no-boost/no-MMR 968.6 ms, search `hybrid+rerank` 2,265.7 ms, `hybrid` 438.5 ms, `bm25` 405.7 ms, `semantic` 420.1 ms — all against `127.0.0.1` |
| `localhost` trap | same requests via `localhost` took ≈2.04 s each (IPv6-first stall) — §13.5 |

## V.5 Evaluation harness

```powershell
python evaluate.py --source postgres --samples 20 --skip-methods --labels __no_labels__.csv
```

* Started 12:50:49, finished 12:56:57; engine load **137.65 s**; full output saved to a shell
  log (91,711 bytes) and quoted in §12.3.
* Part 1: 197 ground-truth sources, P@5 0.0914, Hit@5 41.1 %, nDCG@10 0.3939, MRR 0.3857,
  MAP@5 0.3813, coverage 1,986/2,000 (99.3 %), Gini 0.3205, latency avg 10.37 ms.
* Part 3: free-text demo `flight crash` returned a 74 % match with three agreeing signals.
* Part 4: `Blind labeling file '__no_labels__.csv' not found.` — expected (deliberate name;
  PowerShell cannot pass an empty string to `--labels`).
* Part 2 was skipped by design (`--skip-methods`), because it triples runtime.

**Why `--save-baseline` was not run:** it rewrites `baseline_evaluation.json`, a project file,
and the audit is read-only. It is also provably broken (`evaluate.py:675` reads a config
attribute that does not exist — §18 H5), and a fourth ingestion job was writing to the DB at
the time (§5.6), so any saved baseline would have described a moving corpus.

## V.6 Tests

| Command | Outcome |
|---|---|
| `python scratch/verify_all_fixes.py` | `All architectural verification checks PASSED!`, exit 0 |
| `$env:PYTHONPATH='.'; python scratch/test_evidence_restructure.py` | `Summary: 32 passed, 3 failed`, exit **1** — the three stale agreement assertions quoted verbatim in §16.3 |

`scratch/test_live_recommend.py` was read but not run: it builds the live engine from the DB
and prints results without assertions, and the same behaviour was measured directly by
`isolation.py` / `live_vs_stored.py` (§13.3, §18 H1).

## V.7 Code-quality scans

| Command | Output |
|---|---|
| `Get-ChildItem -Recurse -Include *.py` (excluding `venv`, `__pycache__`) | 27 files |
| `ast.parse` over all 27 | `parsed OK: 27 of 27`, `syntax errors: none` |
| `Select-String -Pattern 'TODO\|FIXME\|HACK\|\bXXX\b\|not implemented\|NotImplemented'` | 1 hit — `config.py:49`, the literal `xxx.trycloudflare.com` in a comment |
| `Select-String -Pattern '^\s*except\s*:'` | `none` |
| `Select-String -Pattern 'except Exception' \| Group-Object Path` | 63 across 17 files (table in §17.3) |
| AST unused-import scan | 21 hits: 7 real (quoted in §17.4), 12 `db/__init__.py` re-exports, 1 `# noqa` side-effect import, plus scratch files |
| AST zero-reference scan | 18 hits: 14 decorator-referenced route handlers, 4 genuinely dead functions (§17.5) |
| `Select-String` for `admission_cut\|RELEVANCE_FLOOR\|ELBOW` | no definition anywhere; only the misleading comment at `src/recommender.py:214` |
| `Select-String` for `PRECOMPUTE_CANDIDATES\|BODY_WORDS_*` | each read only by `evaluate.py` (baseline record) — §17.6 |
| `Select-String` for auth primitives (`HTTPBasic\|OAuth2\|APIKeyHeader`) | none |

## V.8 Notebook and data inventory

| Probe | Result |
|---|---|
| notebook cells | 8 (`kaggle_gpu_worker.ipynb`, untracked) — run order, endpoint contract, auth support and checkpoint behaviour recorded in §14.3 |
| CSV files on disk | 4,177 rows in `data/articles_batches/` vs 2,000 in the DB at snapshot A → 1,731 rows not yet ingested at that time (the gap is what `job_3c2585d8` was closing) |
| `data/cache/` | empty — no embedding cache present |
| batch file names | `batch_0_1.csv`, `articles_batch_03..05.csv`, `article_6..8.csv`; `articles_batch_01.csv` and `_02.csv` are deleted in the working tree — which is what makes `source_batch` label unknown IDs with a non-existent file (§18 M3) |
| `data/articles_clean_train_80.csv` | `False` — hence `/api/db/ingest-default` → 404 (§18 M4) |
| `data/uploads/`, `data/cache/` | gitignored, not present |

## V.9 Commands that failed and were re-run

Recorded so the log is complete:

| Command | Failure | Resolution |
|---|---|---|
| `Get-ChildItem -Filter "pytest.ini","tox.ini",...` | PowerShell `Filter` accepts one string only | re-ran as a plain directory listing; confirmed no such file exists |
| `python -c "... text('select ...') ..."` (inline SQL) | quoting broke the string literal (`SyntaxError`) | moved the query into a temp `.py` file |
| `python "$env:TEMP\...\job_probe.py"` | `ModuleNotFoundError: db` — `sys.path[0]` is the script's directory | added `sys.path.insert(0, <repo root>)` to the helper; re-ran successfully |
| `python evaluate.py ... --labels ""` | PowerShell cannot bind an empty argument | used a deliberately non-existent filename `__no_labels__.csv` |

## V.10 Not run, and why

| Not run | Reason |
|---|---|
| `--save-baseline` | rewrites a project file; also broken (§18 H5); corpus was moving (§5.6) |
| any ingestion | would write to the live database (read-only protocol) |
| `/api/db/clear`, `/init`, `/reload`, `/upload-csv`, `/ingest-default` | destructive or state-changing |
| `scripts/recompute_recommendations.py`, `recompute_target_articles.py`, `refresh_evidence.py` | all three write to the database |
| `import_colab_export.py` | writes embeddings into the database |
| `run_experiments.py`, `freeze_pool.py`, `parity_test.py`, `tests/test_integrity.py` | **deleted in this working tree** (`git status`) |
| frontend JS/CSS | explicitly out of scope (§1.2) |
| Part 2 of `evaluate.py` | skippable by design; triples runtime and needs the live engine |

## V.11 Measurements, in the order they were taken

| # | Measurement | Snapshot |
|---|---|---|
| 1 | code read; environment; library versions | — |
| 2 | DB inventory, indexes, jobs (snapshot A) | before 12:20 |
| 3 | `evaluate.py` Part 1 (2,000 articles) | 12:50:49–12:56:57 |
| 4 | live trace of 10905842, stored-row arithmetic check | ~12:55 |
| 5 | stored-vs-live overlap, ON/OFF isolation | ~12:00–12:40 |
| 6 | HTTP latency table (server + GPU live) | after 12:56 |
| 7 | test runs, code-quality scans | after 13:00 |
| 8 | DB re-check → discovered `job_3c2585d8` in progress and 2,446 articles | 13:17–13:18 |
| 9 | non-monotonic score audit | 13:17 |

Because of step 8, every quantitative claim in this report names the snapshot it belongs to;
§5.6 lists the two snapshots explicitly.

---

*End of report.*

