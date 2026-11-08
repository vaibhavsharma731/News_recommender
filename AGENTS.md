# AGENTS.md — NewsRec v2.0 Project Memory

> Persistent architecture memory for AI agents working in this repo.
> Last verified against the working tree on **2026-10-26** (branch `add_db`).
> **Authoritative source = live code, NOT `PROJECT_AI_SPEC.md`.** The spec doc is
> several revisions stale (see "Spec Drift" below). Always re-read the code.

---

## 0. Project Identity

| | |
|---|---|
| Name | Indian Express AI News Recommendation & Ingestion Platform ("NewsRec v2.0") |
| Stack | FastAPI + PostgreSQL/pgvector + sentence-transformers + vanilla JS frontend |
| Root | `D:\indian express\Recommendation system\news recommender base` |
| Run | `python api/server.py` → http://localhost:5000 |
| Light mode | `NEWSREC_NEURAL=0` → TF-IDF, no model downloads |
| Git | `origin` → `vaibhavsharma731/News_recommender`, `repo2` → `..._2` |
| Branch | `add_db` (5 commits ahead of `origin/add_db`, plus uncommitted work) |

### Live database state (verified 2026-10-22)
- PostgreSQL `news_recommender` @ localhost:5432, user `postgres`, pgvector **0.8.7**
- `articles`: **2446** rows, **2446 embedded** (100% coverage)
- `article_recommendations`: **24458** rows, all `pipeline_version = 1aa8dc568ca0f822`
- `ingestion_jobs`: **5**, all `completed` (batches 01–05, 500/500/500/500/446 articles)
- Reverse-push refresh counts per batch: 0 → 498 → 982 → 1401 → 1733 (growing archive effect)

---

## 1. Architecture at a Glance

```
CSV upload ──► api/editor_routes.py ──► db/pipeline.py (background thread)
                                              │
              ┌───────────────────────────────┼───────────────────────────┐
              ▼                               ▼                           ▼
      src/data_loader.py            src/retrieval.py            src/reranker.py
      clean_dataframe()             BM25Index / DenseIndex     Reranker.score_pairs()
      (token-budgeted texts)        TfidfIndex (fallback)       (cross-encoder)
              │                               │                           │
              └──────────────┬────────────────┴───────────────────────────┘
                             ▼
                   src/recommender.py  Recommender.recommend()
                   RRF → blend → boosts → MMR → calibrate → evidence
                             │
              ┌──────────────┴──────────────────────┐
              ▼                                     ▼
    PostgreSQL (precomputed)              In-memory live path
    article_recommendations               api/server.py → /api/recommend
```

### Dual serving path (`api/server.py:341`)
Fast-path fires when ALL hold: `article_id` set, `query` null, `section` null,
`method == "hybrid+rerank"`, `use_boosts == True`, `top_k <= config.TOP_K`.
→ serves `article_recommendations` rows directly (sub-5ms).
Otherwise → live in-memory `Recommender`.

---

## 2. File-by-File Map

### `config.py` — single source of tunables (160 lines)
| Key | Value | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 384-d, L2-normalised (Pipeline C) |
| `EMBEDDING_DIM` | `384` | vector dimension |
| `EMBEDDING_REVISION` | pinned SHA `5c38ec7c…` | env `NEWSREC_EMBED_REVISION` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | cross-encoder |
| `RERANKER_REVISION` | pinned SHA `2cfc18c9…` | env `NEWSREC_RERANK_REVISION` |
| `EMBED_MAX_SEQ_LEN` | `256` | embedding token cap |
| `RERANK_MAX_LEN` | `512` | **total pair** cap (query+passage) |
| `RERANK_PER_ARTICLE_TOKENS` | `256` | intended per-side split |
| `EMBED_TOKEN_BUDGET` | `254` | 256 − 2 special tokens |
| `RERANK_TOKEN_BUDGET` | `254` | `(512//2) − 2` |
| `TEXT_RECIPE_VERSION` | `v2_sentence_token_budget` | |
| `N_RETRIEVE` | `50` | per branch |
| `N_CANDIDATES` | `20` | to reranker |
| `RRF_K` | `60` | |
| `TOP_K` | `10` | |
| `MMR_LAMBDA` | `0.91` | MEASURED (was 0.75) |
| `MMR_THETA` | `0.76` | MEASURED (was 0.80) |
| `SECTION_BOOST_MULT` | `1.072` | MEASURED (was 1.08) |
| `SUBSECTION_BOOST_MULT` | `1.09` | MEASURED (was 1.05) |
| `POPULARITY_BOOST_MULT` | `1.044` | MEASURED (was 1.06) |
| `RECENCY_MAX_BOOST_MULT` | `1.104` | MEASURED (was 1.10) |
| `BOOST_MODE` | `source_date_recency` | recency vs source article, not corpus max |
| `MIN_RAW_RERANK_THRESHOLD` | `0.12` | raw CE prob gate → `low_confidence` |
| ~~`ARTICLE_DISCARD_FLOOR`~~ | **deleted** | was inert while the double-sigmoid bug pinned scores to 0.5–0.731 |
| `RELEVANCE_FLOOR` / `ELBOW_RATIO` / `ELBOW_MIN_KEEP` | `0.40` / `0.45` / `4` | `config.admission_cut()` — the post-rerank relevance gate |
| `LEXICAL_GATE` / `ENTITY_GATE` | `True` / `True` | pre-rerank gate: no shared BM25 term AND no shared entity → blocked |
| `WEIGHT_RERANK` / `WEIGHT_BM25` / `WEIGHT_DENSE` | `0.70` / `0.20` / `0.10` | user-set; the measured search chose `0.5/0.3/0.2` |
| `SECTION_HALF_LIVES` | Tiered IE decay: live_blog 0.25d, trending 1d, sports 1.5d, cities/india 2.5d, world/ent 3d, business/tech 4d, legal 5d, explained/opinion/upsc 7d, books 10d, default 3d | read via `section_half_life()` (hyphen/underscore/case tolerant) |
| `FALLBACK_MIN_RELATED` / `FALLBACK_LIMIT` | `3` / `5` | technique 4: below 3 related items the API adds a separate "Trending in `<section>`" list; window = `fallback_window_days(section)` = `max(7, 3 × half-life)` |
| `PRECOMPUTE_COMMIT_EVERY` | `5` | |
| `REMOTE_MODEL_URL` | env | Colab/Cloudflare GPU worker |

`get_pipeline_version()` = sha256 of all model+param+recipe tokens, first 16 hex chars.
Stored on `article_recommendations.pipeline_version` and `ingestion_jobs.pipeline_version`.
**Changing any listed param changes the version → next ingestion auto-forces a full recompute.**

### `src/data_loader.py` (650 lines) — THE authoritative loader
Explicitly documented as the single authoritative data loader; do not duplicate cleaning logic.
- `clean_dataframe(raw_df)` — the one cleaning function.
- Dedup policy: **latest `postPublishDate` wins**, tie → last in file order. (`_deduplicate`, L380)
  Spec doc still says `keep="last"` — outdated.
- Column aliases: ID ∈ `articleId|id|article_id|ID`; date ∈ `postPublishDate|date|published_at|publishDate`;
  title ∈ `articleTitle|title|headline|articleName`.
- No ID column → synthetic IDs `art_<md5prefix>_<n>` (collision-safe, L480).
- Normalisation: NFKC, smart quotes → ASCII, en/em dash → `-`, whitespace collapse.
- `strip_also_read()` (L70) — removes inline "ALSO READ <headline>" blocks, extracts
  editor link IDs via 4-word headline dictionary (`ALSO_READ_KEY_WORDS=4`);
  fallback excision `ALSO_READ_FALLBACK_WORDS=14`. These links are the **ground truth** for evaluation.
- `_find_boilerplate()` — sentences >25 chars appearing in ≥ `BOILERPLATE_MIN_DOCS` (4) articles are stripped.
- Token-budgeted composition (v2 recipe): `compose_sentence_budgeted_text()` (L172) — title whole,
  description whole (trimmed at word boundary via binary search if needed), then body **sentence by
  sentence** until budget full; stops at last whole sentence. Emits `is_truncated`.
- `text_bm25` = `title. title. title. desc. body` (title ×3 for entity weight).
- `load_articles()` renames to camelCase (`pageViews`, `uniqueUsers`) and parses `linked_ids` to lists.

### `src/retrieval.py` (381 lines)
- `tokenize()` — `[a-z0-9]+`, drops sklearn ENGLISH_STOP_WORDS, min length 2. No naive "s" stripping.
- `BM25Index` — Okapi BM25, k1=1.5, b=0.75, IDF = `ln(1+(N-n+0.5)/(n+0.5))`.
  - Fuzzy Indian-entity correction: 2-gram candidate index → `difflib.SequenceMatcher`,
    `FUZZY_MIN_RATIO=0.70`, `FUZZY_WEIGHT=0.85`, adjacent-token merge at ratio ≥0.80,
    repeat-char normalisation (`akkhhlesh`→`akhlesh`).
  - `normalize_query()` returns `(clean_query, corrections, expanded_terms)`.
- `DenseIndex` — bge vectors, unit-normalised, cosine = dot product.
  - **Dual cache**: master `data/cache/text_embedding_cache_<model>_len<256>.npz`
    keyed by `sha256(model|len|text)`; legacy `text_embedding_cache.npz` keyed by `sha256(text)`
    is auto-migrated. Incremental — only unseen texts get encoded.
  - Linear scan only (no HNSW/ANN).
- `TfidfIndex` — light fallback, `(1,2)` ngrams, `sublinear_tf`.
- `reciprocal_rank_fusion(rankings, k=RRF_K)` — `sum 1/(k+rank)`.

### `src/reranker.py` (403 lines)
- `build_reranker_pairs()` — enforces ≤ `RERANK_MAX_LEN` (512) tokens per pair.
  - **Article-to-article**: both sides trimmed to `_SIDE_BUDGET` = 254.
  - **Live search**: query trimmed to 254, passage gets `_CONTENT_BUDGET - q_tokens`.
  - Asserts + hard-clamps on violation, logs `VIOLATION`.
- `extract_query_centered_passage()` — title whole + description whole, then body sentence
  containing first query token, expanding left-then-right to fill budget.
- `Reranker._to_probability(raw)` — the model is `num_labels=1`, so `CrossEncoder.predict` ALREADY
  applies sigmoid. This clips values already in `[0,1]` and sigmoids only genuine logits (which is
  what `RemoteCrossEncoderClient` returns). Skipping it double-sigmoids every score into
  `[0.5, 0.731]`.
- `Reranker.score()` — `activation_fn=Identity` then `_to_probability`.
- `Reranker.score_pairs()` — budget-safe public entry point, returns true probabilities in `[0,1]`.

### `src/scoring.py` (125 lines) — the one implementation of the numeric cascade
Pure functions, no DB, no model. Added 2026-10-26 (Change 13) because `db/pipeline.py` and
`src/recommender.py` each carried their own copy of these steps and the copies had drifted.
- `parse_ids(raw)` / `json_ids(raw)` / `linked_set(row)` — linked-ID handling that accepts the DB form
  (JSON string), the CSV form (list) and both dicts and pandas Series.
- `norm(values)` — max-normalise and **clip to `[0,1]`** (the live path used to skip the clip, so a
  negative cosine produced a negative relevance term).
- `blend(rerank, bm25, dense, positions)` — `config.WEIGHT_*` weights; returns
  `(relevance, bm25_norm, dense_norm)`. The reranker term is evaluated in float64.
- `boost_parts(source_row, cand_row, max_views, max_ref_date)` — returns
  `{editorial, section, subsection, popularity, recency, total}`; `total` is the product.
  Values are **unrounded**; rounding happens once, in `evidence.build_evidence_reason()`.
- `mmr(scores, similarity, top_k, lam, theta)` — penalty applies only when similarity > θ.

### `src/recommender.py` (412 lines)
`Recommender.recommend(query, article_id, user_history, top_k, method, use_boosts, diversify, section)`
- Article-to-article builds a **focused query**: title tokens + top-15 IDF body terms.
- Relevance blend comes from `src.scoring.blend()`:
  - Article-to-article: `WEIGHT_RERANK*rerank + WEIGHT_BM25*bm25_norm + WEIGHT_DENSE*dense_norm`
  - Text search: `0.70*rerank + 0.20*bm25_norm` (no dense term)
- `low_confidence` gate uses **raw** CE probability vs `MIN_RAW_RERANK_THRESHOLD` (0.12).
- Relevance gate: `config.admission_cut()` (hard floor + elbow + entity veto; editorial links bypass).
  Replaced the deleted `ARTICLE_DISCARD_FLOOR`, which could never fire.
- `_boost_parts()` — thin wrapper over `src.scoring.boost_parts()`; `total` is the product, so
  `_boost_mult` is gone (one formula, not two).
- `_mmr()` — thin wrapper over `src.scoring.mmr()`; λ `MMR_LAMBDA` (article) / `MMR_LAMBDA_SEARCH`
  (search); θ `MMR_THETA` (article) / `MMR_THETA_SEARCH` (search).
- Personalization: profile centroid with `0.8^r` decay, blended at weight 0.35.
- Returns `Result(recommendations, low_confidence, best_relevance, method, notes)`.
- Each `Recommendation` carries: `article, score (calibrated), boosted_score (pre-calibration,
  post-boost — the value the ordering used), calibrated_score, confidence_band,
  margin, agreement_count, rank, rank_raw, reason (JSON evidence string)`.
- `_explain()` is **deleted** (Change 13); the cascade is built once, by
  `evidence.build_evidence_reason()`, and called identically from here and from the pipeline.

### `src/calibration.py` (169 lines)
- Platt scaling: `P = σ(a*score + b)`, defaults `a=4.85`, `b=-2.60`, clipped to [0.01, 0.99].
- Bands: HIGH ≥ 0.65, MEDIUM ≥ 0.35, else LOW.
- `fit_calibrator_on_corpus()` — positives = editorial Also-Read pairs, negatives = 3× random pairs;
  80/20 split, reports Brier score. Ground truth is editorially biased (documented).

### `src/evidence.py` — deterministic, zero-LLM explanations
- `extract_shared_key_terms()` — IDF-ranked overlap (limit 4).
- `extract_shared_entities()` — capitalized-phrase regex overlap (limit 3), multi-word first.
- `extract_best_matching_sentence()` — top-overlap sentence from first 15, truncated at 180 chars.
- `evaluate_signal_agreement()` — BM25 rank ≤20, Dense rank ≤20, CE ≥0.35 → agreement count 1–3.
- `bm25_term_hits()` — the actual query terms that scored a candidate, with `tf`, `idf`, `weight`,
  `contribution`. Uses the identical formula as `BM25Index.scores()`, so the contributions sum
  exactly to the real BM25 score (verified to 4 dp).
- `build_evidence_reason()` — **formats the whole score cascade itself** (since Change 13) and
  serialises the JSON blob stored in `recommendations.reason`. Signature:
  `(source_row, cand_row, bm25_index, bm25_rank, dense_rank, rerank_score, …, confidence_info,
  scores=, rank_raw=, final_rank=, cand_pos, bm25_query, cosine, admission)`. Emits
  `summary, terms, entities, source_sentence, best_sentence, signals, agreement_count, boosts,
  context_facts, confidence, score_breakdown, rank_relevance, method_ranks, margin_cutoff`,
  plus `bm25_terms`, `bm25_query`, `cosine`, `admission`. `rounded()` / `weighted()` return
  `None` for a missing input (the `hybrid` method has no cross-encoder score) instead of raising.
  The boost `amount` string is derived from the **same rounded multiplier that is printed**, so
  `1.0414` can never sit next to `+4.2%`.
- `enrich_recommendation_evidence()` — **pure formatter, no arithmetic**. It reads the stored
  `score_breakdown` (`components`, `unboosted_confidence`, `boost_items`, `total_boost_mult`,
  `boosted_raw`, `platt_calibrated`) and copies it. It never invents a number:
  unrecorded values stay `None` and the UI prints "not recorded". `agreement_count` is read from
  the scorer, not re-derived. The only things it computes are text facts from the article rows
  (batch file, section/subsection match, publish-time gap). Verified to read rows written by the
  previous revision unchanged.
- `get_article_batch()` — maps article ID → "Batch NN (file.csv)" by scanning `data/articles_batches/*.csv`.

### `src/model_registry.py` (211 lines)
- Thread-safe singleton cache (`threading.Lock` + dict) — one model instance per process.
- `get_embedder()` / `get_cross_encoder()` — if `config.REMOTE_MODEL_URL` set, ping `/health`
  (4s timeout) and use `RemoteEmbedderClient` / `RemoteCrossEncoderClient`; on any failure
  falls back to the local model automatically.
- Remote contract: `POST /embed` `{"texts":[...]}` → `{"vectors":[[...]]}`;
  `POST /rerank` `{"pairs":[[q,d],...]}` → `{"logits":[...]}`; `GET /health` → `{"status":"ok"}`.
  Chunked at 128 per request, 120s timeout.

### `db/connection.py` (212 lines)
- Env-driven URL (`POSTGRES_*` or `DATABASE_URL`), `pool_size=10, max_overflow=20, pool_pre_ping=True`.
- `check_db_health()` — connectivity, pgvector extension, `articles` table presence.
- `ensure_schema()` — idempotent migrations driven by two small declarative tables instead of 12
  inline statements: `_ADDED_COLUMNS` = `(table, column, SQL type, why)` for
  `rerank_score`, `rank_raw`, `boosted_score`, `pipeline_version` (recs);
  `refreshed_articles`, `pipeline_version` (jobs);
  `embed_token_count`, `rerank_token_count`, `is_truncated`, `recipe_version` (articles);
  `_ADDED_INDEXES` = `CREATE INDEX IF NOT EXISTS ix_articles_section_date ON articles (section, date)`,
  which serves technique 4's "popular in this section, within N days" lookup.
  Adding a column now = adding one tuple.
- `init_db()` — creates extension + `Base.metadata.create_all` + `ensure_schema()`.
- `evaluate.py` calls `ensure_schema()` before querying, so a standalone run against a DB written
  by an older revision (missing `boosted_score`) does not fail with `UndefinedColumn`.

### `db/models.py` (197 lines)
**`articles`** — `id` (PK, VARCHAR 100), `title` (NOT NULL), `description`, `body`, `section`,
`subsection`, `date` (TZ), `url`, `image`, `is_live`, `page_views`, `unique_users`,
`linked_ids` (JSON string), `text_embed`, `text_bm25`, `text_rerank`,
`embed_token_count`, `rerank_token_count`, `is_truncated`, `recipe_version`,
`embedding` **vector(384)**, `created_at`, `updated_at`.
Indexes: `articles_pkey(id)`, `ix_articles_id`, `ix_articles_section`, `ix_articles_date`.

**`article_recommendations`** — `id` SERIAL PK, `article_id` FK→articles CASCADE,
`recommended_article_id` FK→articles CASCADE, `rank` (1..10), `rank_raw`,
`score` (Platt confidence on the unboosted blend, 0..1),
`boosted_score` (relevance AFTER boosts, BEFORE calibration — the value the ordering used),
`rerank_score` (raw CE prob, O-centric), `pipeline_version`,
`method` (`neural_hybrid_rerank`), `reason` (JSON evidence), `created_at`.
Unique `(article_id, rank)`; index `(article_id, rank)`.

**`ingestion_jobs`** — `id` PK (`job_<hex8>`), `filename`, `pipeline_version`, `status`
(`pending|processing|completed|failed`), `total_articles`, `processed_articles`,
`refreshed_articles`, `message` (`STAGE:X|msg` protocol), `error`, `created_at`, `updated_at`.

### `db/repository.py` (412 lines)
`ArticleRepository` static methods: `corpus_rows` (ORM articles → plain rows for **both** corpus
loaders; `camel_keys=False` gives the snake_case the pipeline uses, `camel_keys=True` the
`pageViews`/`created_at` the live `Recommender` DataFrame uses — a field can no longer be mapped one
way in one loader and another way in the other), `bulk_upsert_articles_fast` (chunked
`INSERT … ON CONFLICT`), `bulk_set_embeddings`,
`save_recommendations` (delete-then-insert per source article, atomic),
`get_precomputed_recommendations` (join, orders by `rank` or `coalesce(rank_raw, rank)`),
`get_trending_in_section` (technique 4: `section = ?` + source-relative date window,
`ORDER BY page_views DESC`, `LIMIT 5`, model-free, served by `ix_articles_section_date`),
`list_articles_ranked_by_confidence` (avg stored score desc, then rec count, then date),
`clear_all_data`, job CRUD, `get_ingestion_summary` (**one** query with `FILTER` aggregates instead
of five round-trips), `get_active_job`, `fail_orphaned_jobs`.
Every scan is `ArticleRepository.corpus_rows`-shaped; `update_job` only touches the fields passed.
There is **no** vector-search method here: the exact in-memory NumPy scan is 20–30× faster than a
pgvector sequential scan at this corpus size (Change 3 deleted the Postgres retriever).

### `db/pipeline.py` (670 lines) — the ingestion brain
`run_ingestion_pipeline(csv_source, job_id, limit=None, recompute_all_recommendations=False)`
1. `init_db()`, record `pipeline_version`.
2. **Version-change detection** — if last completed job's `pipeline_version` ≠ current →
   force `recompute_all_recommendations = True` (score-scale integrity).
3. Parse CSV → `clean_dataframe()` → compute `new_ids` / `changed_ids` (changed = `text_embed` differs).
4. `bulk_upsert_articles_fast` (chunk 500). `linked_ids` goes in as `json_ids(...)`, so a list can
   never be stored as a Python repr.
5. Embed only `needs_emb_ids` = new ∪ changed ∪ embedding-is-NULL.
6. `_load_full_corpus()` — **whole DB** (old + new) → records + 768-d matrix, via `corpus_rows`.
7. Target set = `new_ids ∪ changed_ids ∪ (uploaded ids lacking current-version recs)`.
   Early-exit if nothing to do (idempotent re-upload).
8. `_precompute_for_article()` — BM25 (focused query) + dense → RRF → top-20 → CE rerank →
   `_rank_and_build()` → calibrate → evidence JSON.
   Stores `score` (calibrated) and `rerank_score` (raw CE, 6 dp) + `rank_raw`.
9. **Reverse push** (`_reverse_push_for_old`) — inverts the candidate map to find old articles
   close to new ones; for each: merges stored top-10 (reusing stored `rerank_score` **only if
   `pipeline_version` matches**, else re-scores) + new neighbours, re-scores only the missing pairs
   O-centrically, then calls the **same** `_rank_and_build()`.
10. `reload_engine_from_postgres()` in-process, then job marked `completed`.

`_rank_and_build(source_row, cand_positions, rerank_vec, records, emb, bm25, bm25_all, dense_all,
bm25_query, bm25_pos_of, dense_pos_of, max_views, max_date, current_ver)` is the **single** ranking
path for forward and reverse rows: blend (`src.scoring.blend`) → boosts (`boost_parts`) → MMR
(`scoring.mmr`, or plain relevance order when the pool is ≤ `TOP_K`) → `config.admission_cut()` →
calibration → `build_evidence_reason()`. Change 13 deleted `_calculate_boosts`, `_apply_mmr` and
`_generate_reason` (167 lines) so the reverse pass can no longer record less evidence than the
forward pass for identical work. `_titles_summary()` replaced two copies of the
`"“A”, “B” (+3 more)"` string builder.

Job message protocol: `STAGE:PARSE|CLEAN|STORAGE|EMBEDDING|INDEX|RERANK|REVERSE|COMPLETE`.
Progress bar reserved: 0–40% embedding, 40–95% rerank, 95–100% reverse.
ETA includes a forward-extrapolated estimate of the reverse pass.

### `api/server.py` (532 lines)
- `lifespan()` → `ensure_schema()`, `fail_orphaned_jobs()`, `reload_engine_from_postgres()` in threadpool.
- Globals `rec` (Recommender) and `articles` (DataFrame); `_reload_lock` guards swaps.
- `reload_engine_from_postgres()` builds the live engine from `ArticleRepository.corpus_rows(arts,
  camel_keys=True)` + one vector matrix; a row without a stored vector contributes a zero vector
  (zero cosine) exactly like the pipeline, instead of forcing a whole-corpus re-embed. The old
  hand-written row mapping (`_row_to_dict` and the inline loader) is deleted — Change 13.
- Routes: `GET /api/status`, `GET /api/articles` (page, per_page≤100, section, sort=date|confidence),
  `GET /api/articles/trending`, `GET /api/articles/{id}`, `POST /api/recommend`,
  `GET /api/sections`, `GET /`, `/favicon.svg`, `/favicon.ico`; mounts `/css`, `/js`.
- `POST /api/recommend` response: `{method, engine, source, low_confidence, best_relevance, notes,
  recommendations:[{article, score, boosted_score, confidence_band, margin, agreement_count,
  rank, rank_raw, reason, evidence?}]}`. The old `unboosted_score` was removed — it actually held
  the *calibrated* score, which was misleading next to `score` (the same number).
- Fast-path response adds `source_article`, `diversify`, per-rec `rerank_score` + `boosted_score`.
- `fallback` — **optional** second list (technique 4), present only on an article-to-article request
  whose related list came back with fewer than `config.FALLBACK_MIN_RELATED` (3) items:
  `{heading, note, type: "fallback", count, window_days, items: [{...article, type: "fallback",
  pageViews}]}`. Items are same-section, within `[source_date ± fallback_window_days]`, ordered by
  `page_views`, capped at `FALLBACK_LIMIT` (5), and exclude the source plus everything already in
  `recommendations`. They carry **no** `score` / `boosted_score` / `rerank_score` /
  `confidence_band` / `margin` / `agreement_count` / `reason` / `evidence` — they are popularity,
  not relevance. `recommendations` itself is never padded or reordered. Search requests never get
  a `fallback` block (no source section to scope it to).
- `_fallback_block()` / `_attach_fallback()` in `api/server.py`; `/api/status` reports
  `fallback.{min_related, limit, requests, triggered, items_served, rate}` since process start.
- CORS wide open (`allow_origins=["*"]`).
- `/css` and `/js` are `StaticFiles` mounts, wrapped by a `_revalidate_static` middleware that adds
  `Cache-Control: no-cache`. Without it browsers reuse a stale stylesheet/script after an edit
  (Starlette sends `ETag`/`Last-Modified` but no `Cache-Control`); unchanged files still answer 304.
- Entry: `uvicorn.run("api.server:app", host="0.0.0.0", port=5000, reload=False)`.

### `api/editor_routes.py` (219 lines) — prefix `/api/db`
`GET /health`, `POST /init`, `POST /upload-csv` (50 MB cap, 1 MB chunks, .csv only,
409 if a job is active, background `threading.Thread(daemon=True)`),
`POST /ingest-default` (`data/articles_clean_train_80.csv`), `GET /jobs/{id}`,
`GET /jobs` (+ summary), `GET /recommendations/{id}`, `POST /clear`, `POST /reload`.
Three shared helpers (Change 13): `_require_idle(db)` = the 409 guard (upload checks it **before**
writing 50 MB), `_launch(db, job_id, filename, csv_source, **kwargs)` = create job + daemon thread,
`_reload_live_engine()` = the late `reload_engine_from_postgres` import, used by `/clear` and `/reload`.

---

## 3. Frontend (`frontend/`, 1 HTML + 14 CSS + 8 JS)

Load order in `index.html` is **critical**:
CSS `variables → base → layout → sidebar → header → hero → cards → tabs → rightpanel → search →
toast → animations → editor → evaluator`; JS `api → state → ui → evaluator → hero → tabs → editor → app`.

### Two independent "tab" systems — don't confuse them
1. **Role switch** (Editor ↔ Reader) — `app.js:358-397`, `setAppRole()`; persists to `localStorage('app_role')`.
2. **Reader tabs** (Similar Articles ↔ Semantic Search) — `tabs.js:29-50`, `switchTab()`.

### `js/state.js` (30 lines)
Closure-based store: `{topK:10, method:'hybrid+rerank', useBoosts:true, diversify:true,
activeSection:'', activeTab:'similar', engineInfo:null, allSections:[], theme}`.
`get()` returns a shallow copy → always destructure. `subscribe()` exists but is **never called**
(pub/sub is inert). `State`/`Api` are **not** exposed on `window`; `Tabs`/`Hero`/`Editor`/`Evaluator` are.

### `js/api.js` (149 lines)
`API_BASE` = `location.origin + '/api'` on http(s), else `http://localhost:5000/api`.
Endpoints: `status, trending, articles, sections, getArticle(dead), recommend, dbHealth, dbInit,
uploadCsv, ingestDefault, jobStatus, recentJobs, precomputedRecs`.
`dbInit`/`uploadCsv`/`ingestDefault` bypass `_fetch` and only read `err.detail` (inconsistent errors).

### `js/tabs.js` (854 lines) — biggest file
- **"Halo" feed**: each article becomes a `.feed-widget` (centre story + top/left/right/bottom slots).
  `planHaloSlots(n)` → n≤6 all top/bottom; n≥7 one per side; n≥9 two per side. Default topK=10 → `{top:3, bottom:3, left:2, right:2}`.
- Infinite scroll: `FEED_PAGE=100`, `FEED_CHUNK=15` widgets, two `IntersectionObserver`s.
- `loadWidgetRecommendations()` deliberately **omits `section`** so `hybrid+rerank` stays on the
  precomputed path; the category filter is applied client-side instead.
- `buildOrbitTile()` renders rank chip, score chip, boost badges, agreement, and an "🔬 Eval" badge
  that opens `window.Evaluator.showModal(...)`.
- `renderFallback()` (technique 4) renders `data.fallback` **below** the halo as its own dashed
  block with its own heading + "not directly related" note. It is called from `renderWidgetRecs()`
  on every render, and removes any previous block first, so MMR toggles cannot duplicate it.
  Fallback titles go through `escHtml()` (new; DB strings never touch `innerHTML` raw here).
- Search tab: `performSearch()` calls `Api.recommend({query, topK, method, useBoosts, diversify, section})`.
- Quick Story Finder: client-side fuzzy title/section match, max 15 results, ↑/↓/Enter/Esc.
- `Tabs` exposes `buildFeedWidget` and `renderWidgetRecs` so a halo can be rendered directly
  (used by the render checks; no feed boot required).

### `js/editor.js` (594 lines)
Upload → `Api.uploadCsv()` → `job_id` → `startPipelineMonitor()` polling `Api.jobStatus()` every 1200 ms.
Parses `STAGE:X|msg`, de-dupes via a `logged` Set, drives the 6-node pipeline diagram
(`parse|clean|storage|embed|rerank|reverse`), renders the log window, completion banner,
DB-health badges, article inspector (`Api.precomputedRecs`), and job history tables.

### `js/evaluator.js` (505 lines)
Renders the **7 grounded evaluation items** (no charts anywhere — pure HTML/CSS):
1 shared entities + BM25 key terms · 2 side-by-side passage pair ·
3 score cascade (CE → BM25 rank → Dense rank → blend → boosts → Platt delta) ·
4 3-stage rank progression (Raw → Boosted → Final MMR) · 5 method agreement ·
6 cutoff margin · 7 editorial context facts.
`renderCard(rec, sourceArt, index, defaultOpen)` and `showModal(rec, sourceArt, candArt)`.

### `js/ui.js` (286 lines)
`showToast`, `buildArticleCard`, `renderCards`, `renderNotes`, `renderResultMeta`,
`buildCategoryPills` (**never called**), `showSpinner`. Has `escapeHtml`-free raw innerHTML.

### `js/app.js` (423 lines)
Theme toggle, sidebar, control bindings (topK slider, section filter, method, boosts, diversity),
randomizer widget, bootstrap sequence (`Api.status` → `Api.sections` → `Hero.init` → `Tabs.init`
→ `Editor.init` → role wiring → reveal app).

### CSS
`variables.css` is the only token file (`--cat-*` category colours, layout sizes, radii, shadows).
It is also the whole glass system: every surface in the app already reads `--bg-surface` /
`--bg-card` / `--border-*`, so those tokens are **translucent plates**, and `--glass-blur`,
`--glass-sat`, `--glass-sheen`, `--glass-shadow`, `--ambient` complete the recipe. Retune glass
there, never per-component.
`base.css` paints the single ambient colour field (`body::before`, fixed, transform-only
`ambient-drift`) that the glass refracts, then applies the blur to six chrome selectors
(`.navbar`, `.sidebar`, `.right-panel`, `.toast`, `.story-finder-dropdown`, `.eval-modal-card`) and
the plate shadow/specular edge to a card list. `.glass` is the portable plate for new components.
**Cards deliberately skip `backdrop-filter`** — invisible against an already-soft field, but one GPU
layer per card. `.main-content` must stay `transparent` or the field is hidden. Light theme
`[data-theme="light"]` overrides the tokens only (it is the only such block now); `evaluator.css`
and parts of `editor.css` still hard-code light-on-dark colours and do **not** follow the light theme.

---

## 4. Known Bugs / Drift (verified 2026-10-22)

### P0 — functional
1. ~~`app.js:335` — assigns to `const statusData`~~ **fixed (Change 6d)**.
2. ~~`app.js:72` — calls `syncCategoryPills(sec)` which does not exist~~ **fixed (Change 6d)**.
3. ~~`renderCards()` wipes the notes banner `renderNotes()` just prepended~~ **fixed (Change 6d)**.
4. ~~`rightpanel.css` redefines the global `@keyframes shimmer`~~ **fixed (Change 6d)**.
5. ~~`.btn-secondary` used 4× in `index.html` but never defined~~ **fixed (Change 6d)** — aliased to
   `.btn-ghost`.
6. `editor.js:150` uses raw `fetch('/api/db/clear')` with a hard-coded path instead of `Api`.
6a. ~~`app.js` randomizer requested `perPage: 200`~~ **fixed (Change 12)** — `/api/articles` caps
    `per_page` at 100, so the request 422'd and the randomizer widget stayed empty.
6b. ~~Editing CSS/JS had no visible effect until a hard refresh~~ **fixed (Change 12)** —
    `StaticFiles` sent no `Cache-Control`; `/css` and `/js` now revalidate.

### P1 — correctness
7. XSS risk: `ui.js`, `tabs.js`, `app.js`, `hero.js` inject API strings via `innerHTML` without
   escaping. Only `evaluator.js` has `escapeHtml`.
8. `evaluator.js:110` prints "(High)" regardless of actual score.
9. ~~`src/evidence.py:enrich_recommendation_evidence()` fabricates fallbacks (0.65 / 0.82 / 0.064
    margin) and hard-codes boost multipliers (1.08/1.05/1.10).~~ **fixed (Change 10/13)** — it is now
    a pure formatter over the stored `score_breakdown`: unrecorded values stay `None` and the UI
    prints "not recorded", and the boost labels come from the real multipliers the scorer applied.
10. `evaluator.js:237` labels a signed score delta as "Platt Probability Calibration".
11. `evaluator.js` accepts `rec.evidence`; `tabs.js:185` only reads `rec.reason` → inconsistent.
12. `tabs.js:430` sets `dataset.loaded = 'true'` *before* the await → failed requests never retry.
13. `tabs.js:535` `buildFeed()` has no request sequencing → concurrent calls can interleave.
14. ~~`tabs.js:254` `ev.signals.join(...)` assumes an array → throws on a string.~~
    **fixed** — `Array.isArray` guard.
15. ~~`db/pipeline.py:390` reverse-push passes `rel_pos + 1` as both bm25_rank and dense_rank →
    **fabricated agreement ranks** in reverse-pushed evidence.~~
    **fixed** — real corpus ranks via `argsort` of the full BM25/dense score vectors, and since
    Change 13 the reverse pass runs the same `_rank_and_build()` as the forward pass.
16. ~~`db/pipeline.py:497` `_apply_mmr` hard-codes the similarity threshold `0.80`, ignoring
    `config.MMR_THETA` (0.76).~~ **fixed** — `_apply_mmr` is deleted (Change 13); both paths call
    `src.scoring.mmr(..., theta=config.MMR_THETA)`.
17. ~~`_calculate_boosts` / `_boost_mult` recency uses corpus-max `date`, ignoring `BOOST_MODE`.~~
    **fixed** — both call `config.recency_boost()` / `config.recency_ref_date()`; measured max
    abs diff between the two paths is `0.0`. `_calculate_boosts` is deleted (Change 13); the recency
    reference is passed into `src.scoring.boost_parts()`.
18. ~~`db/pipeline.py:660` persists `linked_ids` as `str(...)` → not JSON.~~
    **was not a live bug, latent risk now closed.** `clean_dataframe()` already returns `linked_ids`
    as JSON *strings*, so `str()` was a no-op and the stored value was valid JSON (live rows confirm
    `["10911764"]`). Change 13 replaced it with `scoring.json_ids(...)`, which normalises a list to
    JSON instead of a Python repr — so feeding a list would now also be correct.
18a. **Double sigmoid** — `Reranker.score()` applied sigmoid on top of `CrossEncoder.predict`'s
    already-sigmoided output, trapping every score in `[0.5, 0.731]`. Fixed via
    `_to_probability()`. This is why the old discard floor could never fire.
18b. ~~Technique 4 (graceful degradation) not implemented — a source with no related article now
    legitimately returns 1 item, and the UI has no "Trending in `<section>`" fallback list.~~
    **fixed (Change 11)** — optional `fallback` block in `/api/recommend`, rendered as its own
    labelled section below the halo. Measured trigger rate on 50 real sources: **34.0%**. The block
    is excluded from evaluation by construction (never stored, never read by `evaluate.py`).

### P1b — experiment harness: 19–22 verified fixed, 23 still open
19. ~~`run_experiments.py` passed the scalar `a0_val_ndcg` as the paired baseline array →
    `IndexError: too many indices`.~~ **fixed** — line 496 passes the per-article `a0_ndcg` list;
    the scalar is only used for the `delta_vs_a0` field. (The shipped CSVs still came from an older
    revision, so re-run the script if the numbers matter.)
20. ~~`import_colab_export.py` ran `db.query(Article.id).all()` (SQLAlchemy 2.0 autobegins) and then
    `with db.begin():` → `InvalidRequestError: A transaction is already begun`, so the Colab export
    could never be imported.~~ **fixed** — one `with SessionLocal.begin() as db:` wraps the reads and
    the writes. The unit test that "passed only because it failed earlier" was itself broken and is
    fixed too (it now hands the fake session to `__enter__`).
21. ~~`freeze_pool.py:75` iterated a `set` to build the focused BM25 query → not bit-reproducible.~~
    **fixed** — `sorted(..., key=lambda pair: (-idf, term))` gives a deterministic tie-break.
22. ~~`freeze_pool.py:165` ignored `config.RERANKER_REVISION` and defaulted `--max-len 192`.~~
    **fixed** — `max_length = max_length or config.RERANK_MAX_LEN` and
    `revision=config.RERANKER_REVISION`.
23. `parity_test.py` can still report **"FULL PARITY VERIFIED" with a dead tunnel** — the client falls
    back to the local model on HTTP error, so it compares local vs local (cosine 1.0, same Top-5).
    **Still open.** Fix = make the parity check assert the remote answer came from the remote.

### P1c — found and fixed in Change 13 (backend audit)

24. ~~`tests/test_integrity.py` could not run at all — `tests/` is not a package, so
    `python tests/test_integrity.py` died on `import config` and `-m unittest tests.test_integrity`
    died on `ModuleNotFoundError`. The "15 invariant tests" had never been executed.~~
    **fixed** — `sys.path` bootstrap in the test module; **16 tests run and pass**.
25. ~~`build_evidence_reason()` raised `TypeError` when `rerank_score` was `None` (the `hybrid`
    method never calls the cross-encoder).~~ **fixed** — `rounded()` / `weighted()` return `None`.
26. ~~The live path divided the dense score by its maximum **without clipping** to `[0,1]`, so a
    negative cosine produced a negative relevance term, while the ingestion path clipped. The same
    pair scored differently depending on which path served it.~~ **fixed** — `src.scoring.norm()`
    clips in both.
27. ~~Reverse-push rows carried less evidence than forward rows for identical work (no
    `rank_relevance`, `method_ranks`, `margin_cutoff`) — the evaluation card showed "not recorded".~~
    **fixed** — one `_rank_and_build()` for both.
28. ~~`api/server.py`'s `reload_engine_from_postgres()` rebuilt article rows by hand and answered
    "one embedding missing" by re-embedding the whole corpus.~~ **fixed** — `corpus_rows()`; a row
    without a vector contributes a zero vector (zero cosine), exactly like the pipeline.
29. ~~`db/pipeline.py` booted with an unused `from sqlalchemy.orm import Session` and `api/server.py`
    imported `load_articles` it never called.~~ **fixed** — 13 unused imports removed across 9 files.

### P2 — dead code
`State.subscribe`, `State.theme/engineInfo/allSections` (write-only), `State.addToHistory`/`saveHistory`
(called from `hero.js:44` but don't exist), `Api.getArticle`, `window.UI`, `buildCategoryPills`,
`tabs.js` `'editor'` tab branch, `editor.js` `activeJobId` + legacy `.rec-table-wrap` table,
duplicate `escapeHtml` in `editor.js` and `evaluator.js`, duplicate section-colour maps
(`variables.css` `--cat-*` unused vs `ui.js` `SECTION_COLORS`).

### Spec drift — `PROJECT_AI_SPEC.md` is stale
| Spec says | Code actually |
|---|---|
| `RERANK_MAX_LEN 160` | `512` (pair cap) |
| `BODY_WORDS_EMBED 350` / `BODY_WORDS_RERANK 150` | retired → `None`, token budgets 254 |
| `MMR_LAMBDA 0.75`, θ 0.80 | `0.91`, θ `0.76` |
| Section/subsection/popularity boosts 1.08/1.05/1.06 | `1.072/1.09/1.044` |
| dedup `keep="last"` | latest-date-wins |
| `MIN_RELEVANCE['reranker'] 0.05` | `MIN_RAW_RERANK_THRESHOLD 0.12` |
| `test_postgres_setup.py` exists | **deleted** (staged for removal) |
| No mention of calibration, evidence, boost_mode, remote GPU | all present |

`README.md` is also partially stale (says "6 JS modules", "30 candidates", "12 modular CSS files").

---

## 5. Evaluation & Experiment Harness

### 5.1 The measured configuration (source of truth for `config.py`'s MEASURED values)
Winner row of `tuning_results.csv`, `config_hash = 82e2663f`:
```
w_rerank=0.5  w_bm25=0.3  w_dense=0.2
sec_mult=1.072  subsec_mult=1.09  pop_mult=1.044  rec_mult=1.104
boost_cap=None  boost_mode=source_date_recency
mmr_lambda=0.91  mmr_theta=0.76
```
`82e2663f` = `sha256("sec=1.072|sub=1.09|pop=1.044|rec=1.104|cap=None|mode=source_date_recency|lam=0.91|theta=0.76|w=0.5,0.3,0.2")[:8]`
— verified reproducible. Computed at `run_experiments.py:484-485`.

**Objective:** `composite_rank_score = catalog_coverage − 0.005 × median_age_days`
(maximize), on a **20-article validation split**, subject to constraints
`ΔnDCG@5 ≥ −0.01` AND `paired_ci_low ≥ −0.01` AND `near_dup_rate ≤ A0 + 0.005`.
Bootstrap 1000 samples, seed 42. Tie-break on `ndcg5`.
**nDCG was NOT maximized** — the winner is only rank **14 of 200** by nDCG@5.

**Winner's metrics (validation, n=20):** nDCG@5 `0.214574` (Δ vs A0 `+0.019343`,
95% CI `[0.0000, +0.058028]`), recall@5 `0.325`, hit@5 `0.35`, near_dup `0.055`,
coverage `0.038103`, median_age_days `−0.276476`, mean_unboosted_rel `0.662476`.
Best nDCG@5 seen anywhere in the search = `0.243957` (`a080c3a4`) — **rejected** by the objective.
Only **9 of 200** configs pass the constraint set.

### 5.2 Ablation ladder (`ablation_results.csv`, validation n=20)
| step | nDCG@5 | paired Δ vs A0 CI | recall5 | hit5 | near_dup | median_age_days |
|---|---|---|---|---|---|---|
| A0 relevance only | 0.195232 | [0, 0] | 0.275 | 0.30 | 0.095 | +0.08913 |
| A1 + section 1.08 | **0.215876** | [0, +0.0598] | 0.275 | 0.30 | 0.105 | +0.04042 |
| A2 + subsection 1.05 | 0.197423 | [0, +0.0066] | 0.275 | 0.30 | 0.095 | +0.06743 |
| A3 + popularity 1.06 | 0.195232 | [0, 0] | 0.275 | 0.30 | 0.100 | +0.08913 |
| A4 + recency 1.10 | 0.175889 | [−0.0580, 0] | 0.225 | 0.25 | 0.105 | −0.01788 |
| A5 all boosts | 0.175889 | [−0.0580, 0] | 0.225 | 0.25 | 0.130 | −0.27648 |
| A6 all + MMR(0.75,0.80) | 0.175889 | [−0.0580, 0] | 0.225 | 0.25 | **0.000** | −0.15072 |

Read-outs: **section boost is the only clearly positive lever** (+0.0206 nDCG@5);
**recency is net-negative** (−0.0193); popularity is exactly neutral;
MMR changed the selected sets (near-dup 0.13→0.00) but **not** nDCG/recall/hit.

### 5.3 `baseline_evaluation.json` (saved 2026-10-18, 1709 articles — STALE)
```
gt_p5 0.0939 | gt_hit5 0.4697 | gt_hit10 0.5455
gt_ndcg10 0.38506 | gt_mrr 0.34173 | gt_map5 0.32374
section_p5 0.55 | subsection_p5 0.20
semantic_coherence_avg 0.65016 | rank_weighted 0.66313 | ild_avg 0.40189
catalog_coverage 0.99707 | gini 0.30659 | top_5pct_rec_share 0.12147
score_mean 0.41076 | score_median 0.37945 | score_below_0.3 0.272
avg_latency_ms 5.15 | p95_latency_ms 6.96
```
Its embedded `config` block records the **old** `mmr_lambda 0.75`, `rerank_max_len 160`,
`body_words_embed 350` → any `--compare` diff misreports.

### 5.4 File purposes
| File | Purpose |
|---|---|
| `evaluate.py` (977 L) | 4 parts: (1) PostgreSQL precomputed quality, (2) 4-method comparison matrix, (3) free-text demo, (4) human blind-label summary. CLI: `--source auto\|postgres\|csv\|api`, `--samples 100`, `--api-url`, `--labels`, `--save-baseline`, `--compare`. |
| `run_experiments.py` (673 L) | Frozen-pool story-clustered 60/20/20 split (cosine ≥0.82 adjacency → connected components) → ablation A0–A6 → 200-config random search (`RandomState(42)`) → selection → test-set eval → blind-label CSV. |
| `freeze_pool.py` (221 L) | Builds `data/cache/frozen_pool.pkl`: top-20 candidates per GT source with all raw signals cached → tuning needs **zero model calls**. Stores `corpus=2467`, `max_page_views=69218`, `max_corpus_date=2026-10-08`. |
| `parity_test.py` (166 L) | Local CPU vs Colab T4 parity: embeddings pass if min cosine ≥ 0.9999; reranker passes if Top-5 ordering 100% identical. |
| `import_colab_export.py` (147 L) | Validated atomic import: hash match, IDs exist, 768-dim, no NaN/Inf, L2 norm in [0.990, 1.010], single transaction. |
| `tests/test_integrity.py` (304 L) | 15 invariant tests (see below). |
| `eda.py` | Exploratory analysis via `src.data_loader`. |
| `cloud_gpu_worker.ipynb` | Colab T4 GPU worker (contract below). |
| `blind_labeling_sample.csv` | 136 pairs, 20 source articles, **0 labels filled** → Part 4 currently contributes nothing. |

**`tests/test_integrity.py` covers:** dedup latest-date-wins · dedup conflict counting ·
synthetic-ID collision guard · flexible column alias mapping · pipeline-version fingerprint
changes with `EMBED_MAX_SEQ_LEN` · `BODY_WORDS_*` retired → `None` · whole-sentence token
budgeting · description word-boundary trim · embedding cache key = `sha256(model|len|text)` ·
reranker 50/50 split ≤254/side and total pair ≤512 · model revisions pinned to 40-char SHAs ·
pipeline version includes revisions · import rejects NaN/Inf · calibration monotonic + bands ·
deterministic evidence, zero LLM calls.
No `__init__.py`, no conftest, needs network for HF tokenizers → not offline-safe.

### 5.5 `cloud_gpu_worker.ipynb` — the remote GPU worker (THE notebook)
**File:** `cloud_gpu_worker.ipynb`, `nbformat 4.2`, 8 cells, `accelerator: GPU`, `colab.gpuType: T4`.
Last commit touching it: `39a46f7` (2026-10-18); working-tree mtime 2026-10-22 13:29 (**uncommitted edits**).

**Cell map (run in order):**
| # | Type | Purpose |
|---|---|---|
| 0 | md | Title + guarantees (pinned revisions, seeds, fp32, Colab Secrets) |
| 1 | code | `pip install fastapi uvicorn "sentence-transformers==6.1.0" "transformers==5.19.0"` + download/install `cloudflared-linux-amd64.deb`. **Does not pin `torch`** and never reads `colab_requirements.txt` (which does pin `torch==2.14.0`). |
| 2 | code | Seeds 42 (`random`/`np`/`torch`/`cuda`/`cudnn.deterministic`), loads `HF_TOKEN` from `google.colab.userdata`, asserts `sentence_transformers==6.1.0`, `transformers==5.19.0`, `numpy` startswith `2.`, prints device+GPU. |
| 3 | code | Loads `SentenceTransformer(BAAI/bge-base-en-v1.5, revision=a5beb1e3…)` `max_seq_length=256`; `CrossEncoder(BAAI/bge-reranker-base, revision=2cfc18c9…, max_length=512)`. |
| 4 | code | **Mode A**: FastAPI on `127.0.0.1:8000` in a daemon thread + `cloudflared tunnel --url http://127.0.0.1:8000`; scrapes the `https://*.trycloudflare.com` URL from `stderr` (40 × `readline()`); prints the `REMOTE_MODEL_URL=…` line. |
| 4b | code | **Keep-alive**: `while True` loop every **240 s** — pings own `/health`, runs a 64×64 CUDA add to keep the context warm, prints heartbeat. |
| 5 (a) | code | **Connection test**: full chain public-tunnel → Cloudflare → FastAPI → model; asserts `/embed` shape `(2,768)` with norms ≈1 and `/rerank` scores relevant > irrelevant. |
| 5 (b) | code | **Mode B**: resumable batch precompute to `CHECKPOINT_DIR=/content/drive/MyDrive/newsrec_colab/checkpoint.json`, checkpoint every 50 articles, final `newsrec_export.json`. |

**Endpoint contract (verified cell ≡ client in `src/model_registry.py`):**
```
GET  /health  -> {status:"ok", device, gpu, embedding_model, embedding_revision,
                  reranker_model, reranker_revision}
POST /embed   {"texts":[str]}        -> {"vectors":[[768 floats], ...]}   (L2-normalised, float32)
POST /rerank  {"pairs":[[q,d], ...]} -> {"logits":[float, ...]}          (raw logits; sigmoid client-side)
```
Empty input → `{"vectors": []}` / `{"logits": []}`. Both use `batch_size=32`, `np.float32`.
Client chunks at 128, 120 s timeout, and **silently falls back to the local model on any error**
(so a dead tunnel degrades to CPU with no user-visible warning).
Mode B export shape: `{"config_hash": str, "articles": {id: {"embedding":[768], "title": str}}}`.
⚠️ Live `.env` currently points at `https://cents-citizen-sullivan-mate.trycloudflare.com` —
**verified DEAD** (`getaddrinfo failed`). Quick tunnels die with the Colab session; the URL must
be re-pasted into `.env` after every restart. Note `.env` has a **leading space** after
`REMOTE_MODEL_URL=` (`REMOTE_MODEL_URL= https://…`) — python-dotenv strips it, but it is fragile.

**Mode B bugs (confirmed by reading the cell):**
- `run_batch_precompute` reads `row['id']`, `row['title']`, `row.get('text_embed')` — but the raw
  batch CSVs (`data/articles_batches/*.csv`) have **camelCase** columns (`articleId`,
  `articleTitle`, …). Pointing it at a raw batch file raises **`KeyError: 'id'`**. It only works if
  fed a **pre-cleaned** CSV (output of `clean_dataframe`/`load_articles`).
- `config_hash` is a **manually typed string** argument. Nothing computes it, and
  `import_colab_export.py:40-50` *requires* it to equal `config.get_pipeline_version()` → silent
  `--force-hash` pressure.
- The export contains **embeddings only, no `recommendations`** — so the importer's
  recommendation-import branch (`import_colab_export.py:113-126`) is never exercised by this notebook.
- Encodes **one article at a time** (`embedder.encode(text_embed, …)` on a bare string) — no batching,
  ~2446 sequential calls. Mode A batches at 32.
- Uses `np` and `embedder` without importing/binding them in-cell → `NameError` if run standalone
  after a runtime restart.
- If the checkpoint's `config_hash` differs, it **silently discards the old checkpoint** (line 443-447)
  and then overwrites it — prior work lost with no warning.
- Two different cells are both labelled **"Step 5"**.

**Other notebook issues:** `/embed` and `/rerank` have **no auth / rate limit / origin check** on a
public tunnel (free-tier T4 exposed to anyone with the URL). `fix_numpy_pin.py` is now a **no-op**
(it rewrites a `numpy==2.5.3` string that no longer exists, and reformats the whole JSON with
`indent=1`). The `cloudflared` stderr scrape uses blocking `readline()` and can hang.

### 5.6 Measured-vs-production gap (most important finding)
| Measured setting | Where it should apply | Reality |
|---|---|---|
| `MMR_THETA = 0.76` | MMR dedup threshold | `db/pipeline.py:497` **hard-codes `> 0.80`** → DB precompute (what Part 1 measures) ignores it. Live path (`recommender.py:397`) uses it. |
| `BOOST_MODE = "source_date_recency"` | recency reference | **`BOOST_MODE` is referenced NOWHERE in the codebase.** Both `pipeline.py:466-471` and `recommender.py:384-390` recency-boost against the **corpus max date** — i.e. the `multiplicative` mode the search rejected. |
| blend `0.5 / 0.3 / 0.2` | relevance blend | `recommender.py:190`, `pipeline.py:150`, `pipeline.py:345` all hard-code `0.55 / 0.30 / 0.15`. The measured blend is nowhere. |
| `MMR_LAMBDA = 0.91`, boost mults | | ✅ correctly wired in both paths. |

### 5.7 Baseline protocol — how to measure an upgrade against the current code
`baseline_evaluation.json` is the **regression reference**. Only `--save-baseline` writes it
(`evaluate.py:save_baseline`), and it saves **Part 1 only** — i.e. the quality of the rows already
sitting in `article_recommendations`, not a live recompute. So it grades the **ingestion pipeline**,
which is what the reader fast-path actually serves.

```powershell
# refresh the reference (only when you intend to move the goalposts)
python evaluate.py --source postgres --samples 100 --save-baseline

# after changing code, DO NOT overwrite — compare
python evaluate.py --source postgres --samples 100 --compare baseline_evaluation.json
```

**Preconditions that must hold before a baseline is meaningful** (verified 2026-10-22):
- `config.get_pipeline_version()` **==** `article_recommendations.pipeline_version`
  (currently both `1aa8dc568ca0f822`). If they differ, the DB was scored under an older config —
  recompute before benchmarking, or the numbers describe code you no longer have.
- `articles.recipe_version` == `config.TEXT_RECIPE_VERSION` for every row
  (currently all 2446 are `v2_sentence_token_budget`).
- `max(articles.rerank_token_count)` == `config.RERANK_TOKEN_BUDGET` (254) and
  `max(articles.embed_token_count)` == `config.EMBED_TOKEN_BUDGET` (254) — proves the 512-pair /
  256-embed budgeting was actually applied at ingest.
- `REMOTE_MODEL_URL` must be empty (or the tunnel alive) — otherwise `model_registry` silently
  falls back to CPU and the `avg_latency_ms` in the baseline is not comparable.
  The writer records `compute_device` so a CPU/GPU mix-up is visible.

**Caveats when reading the deltas:**
- Ground truth is only the **209** articles carrying ≥1 editorial "Also Read" link, so `gt_p5`,
  `gt_ndcg10`, `gt_mrr`, `gt_map5` are computed over 209 sources, not 2446.
- `avg_latency_ms` / `p95_latency_ms` measure the SQL lookup only — not model inference.
- `section_p5` / `subsection_p5` are diluted by articles with an empty section (denominator counts
  every rec, numerator only matches) — see bug list H.
- `compare_baselines` treats a missing key as `0.0`, so a key removed from the writer shows up as a
  fake regression. Do not delete keys from `save_baseline`'s config dict.
- `baseline_evaluation_legacy_2026-10-18.json` is the pre-refresh reference (1709 articles,
  `mmr_lambda 0.75`, `rerank_max_len 160`). Diffs against it are meaningless — keep it only as history.

---

## 6. Data Layout

```
data/
├── articles_batches/          articles_batch_01..05.csv   (~2400 KB each) → 2446 articles
├── cache/                     text_embedding_cache.npz (7.2 MB legacy),
│                              text_embedding_cache_BAAI_bge-base-en-v1.5_len256.npz (6 KB),
│                              articles_2467_vectors.npy (7.4 MB),
│                              cleaned_articles.pkl (26 MB), frozen_pool.pkl (376 KB)
└── uploads/                   job_<id>_<name>.csv (ingestion staging, gitignored)
```
CSV columns seen: `articleId`, `articleTitle`, `description`, `body`, `sectionName`,
`section1Slug`, `section2Name`, `section2Slug`, `postPublishDate`, `articleUrl`,
`articleFeaturedImage`, `liveBlog`, `pageViews`, `uniqueUsers`.
43 distinct sections; largest: cities 464, entertainment 419, india 296, sports 229, lifestyle 154.

`.env` (gitignored) holds `POSTGRES_*` + `REMOTE_MODEL_URL`.
`.env.example` is the template. `.gitignore` excludes `.env`, `venv/`, `data/cache/`, `data/uploads/`.

---

## 7. Working Agreements

1. **Never trust `PROJECT_AI_SPEC.md` for numbers** — read `config.py`.
2. **`src/data_loader.clean_dataframe()` is the only cleaning path.** Do not write parallel cleaners.
3. **Any change to a `get_pipeline_version()` token changes the version hash** → the next
   ingestion forces a full recompute (slow but correct). Update the doc string if you change it.
4. **Ingested vs live scoring must stay in sync — and it now is, by construction.** The blend, boost
   table, MMR and linked-ID handling live in **`src/scoring.py`**; both `db/pipeline.py` and
   `src/recommender.py` call it. Do not re-derive those formulas in a caller: edit `src/scoring.py`
   once and both paths change together. (Before Change 13 the two copies had drifted — bugs #15–#17,
   #26.) The only intentionally different piece is the text-search blend, which has no dense term.
   Verify any change here with a golden run, not by eyeballing: same fixtures → same rows → diff.
5. **`reason` is a JSON string**, not plain text. The frontend parses it (`ui.js`, `tabs.js`,
   `evaluator.js`). Keep the key set compatible: `summary, terms, entities, source_sentence,
   best_sentence, signals, agreement_count, boosts, context_facts, confidence,
   score_breakdown, rank_relevance, method_ranks, margin_cutoff`.
6. **Only one ingestion job at a time** — `get_active_job()` gates uploads with HTTP 409.
7. **Single-process assumption** — `reload_engine_from_postgres()` mutates in-process globals.
   Running uvicorn with `--workers > 1` desynchronises the engines.
8. Frontend has **no build step and no framework** — plain classic scripts, global scope.
   Any new module must be appended to `index.html` in the right order.
