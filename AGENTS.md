# AGENTS.md — NewsRec v2.0 Project Memory

> Persistent architecture memory for AI agents working in this repo.
> Last verified against the working tree on **2026-10-22** (branch `add_db`).
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
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | 768-d, L2-normalised |
| `EMBEDDING_REVISION` | pinned SHA `a5beb1e3…` | env `NEWSREC_EMBED_REVISION` |
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
| `ARTICLE_DISCARD_FLOOR` | `0.05` | drops unrelated candidates |
| `MAX_USER_HISTORY` / `USER_PROFILE_WEIGHT` | `10` / `0.35` | personalization |
| `SECTION_HALF_LIVES` | live_blog 0.25d, sports 1d, entertainment 2d, business 3d, brand-story 4d, default 2d | |
| `INCREMENTAL_RECOMMENDATIONS` | `True` | reverse-push on |
| `PRECOMPUTE_COMMIT_EVERY` | `5` | |
| `REMOTE_MODEL_URL` | env | Colab/Cloudflare GPU worker |

`get_pipeline_version()` = sha256 of all model+param+recipe tokens, first 16 hex chars.
Stored on `article_recommendations.pipeline_version` and `ingestion_jobs.pipeline_version`.
**Changing any listed param changes the version → next ingestion auto-forces a full recompute.**

### `src/data_loader.py` (649 lines) — THE authoritative loader
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

### `src/retrieval.py` (380 lines)
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

### `src/reranker.py` (382 lines)
- `build_reranker_pairs()` — enforces ≤ `RERANK_MAX_LEN` (512) tokens per pair.
  - **Article-to-article**: both sides trimmed to `_SIDE_BUDGET` = 254.
  - **Live search**: query trimmed to 254, passage gets `_CONTENT_BUDGET - q_tokens`.
  - Asserts + hard-clamps on violation, logs `VIOLATION`.
- `extract_query_centered_passage()` — title whole + description whole, then body sentence
  containing first query token, expanding left-then-right to fill budget.
- `Reranker.score()` — `activation_fn=Identity` then stable sigmoid `1/(1+e^-clip(logit,±30))`.
- `Reranker.score_pairs()` — budget-safe public entry point.

### `src/recommender.py` (462 lines)
`Recommender.recommend(query, article_id, user_history, top_k, method, use_boosts, diversify, section)`
- Article-to-article builds a **focused query**: title tokens + top-15 IDF body terms.
- Relevance blend:
  - Article-to-article: `0.55*rerank + 0.30*bm25_norm + 0.15*dense_norm`
  - Text search: `0.65*rerank + 0.35*bm25_norm`
- `low_confidence` gate uses **raw** CE probability vs `MIN_RAW_RERANK_THRESHOLD` (0.12).
- Discard floor filter: `rerank_score >= 0.05` OR candidate is an editorial link.
- `_boost_mult()` — editorial 1.40 × section × subsection × log-popularity × recency half-life.
- `_mmr()` — λ `0.91` (article) / `0.88` (search); θ `config.MMR_THETA` (article) / `0.85` (search).
  **Penalty applies only when similarity > θ.**
- Personalization: profile centroid with `0.8^r` decay, blended at weight 0.35.
- Returns `Result(recommendations, low_confidence, best_relevance, method, notes)`.
- Each `Recommendation` carries: `article, score (calibrated), calibrated_score, confidence_band,
  margin, agreement_count, rank, rank_raw, reason (JSON evidence string)`.

### `src/calibration.py` (170 lines)
- Platt scaling: `P = σ(a*score + b)`, defaults `a=4.85`, `b=-2.60`, clipped to [0.01, 0.99].
- Bands: HIGH ≥ 0.65, MEDIUM ≥ 0.35, else LOW.
- `fit_calibrator_on_corpus()` — positives = editorial Also-Read pairs, negatives = 3× random pairs;
  80/20 split, reports Brier score. Ground truth is editorially biased (documented).

### `src/evidence.py` (570 lines) — deterministic, zero-LLM explanations
- `extract_shared_key_terms()` — IDF-ranked overlap (limit 4).
- `extract_shared_entities()` — capitalized-phrase regex overlap (limit 3), multi-word first.
- `extract_best_matching_sentence()` — top-overlap sentence from first 15, truncated at 180 chars.
- `evaluate_signal_agreement()` — BM25 rank ≤20, Dense rank ≤20, CE ≥0.35 → agreement count 1–3.
- `build_evidence_reason()` — serialises a JSON blob into `recommendations.reason`.
- `enrich_recommendation_evidence()` — standardises the **7 grounded evaluation items**
  across Editor Inspector + Reader tiles.
  ⚠️ **Contains hard-coded fallbacks** that fabricate values when the API omits them
  (e.g. `rerank_val = 0.65`, `final_score = 0.82`, boost percentages hard-coded to
  1.08/1.05/1.10 instead of reading `config`). Treat its numbers as unreliable when
  the upstream fields are absent.
- `get_article_batch()` — maps article ID → "Batch NN (file.csv)" by scanning `data/articles_batches/*.csv`.

### `src/model_registry.py` (194 lines)
- Thread-safe singleton cache (`threading.Lock` + dict) — one model instance per process.
- `get_embedder()` / `get_cross_encoder()` — if `config.REMOTE_MODEL_URL` set, ping `/health`
  (4s timeout) and use `RemoteEmbedderClient` / `RemoteCrossEncoderClient`; on any failure
  falls back to the local model automatically.
- Remote contract: `POST /embed` `{"texts":[...]}` → `{"vectors":[[...]]}`;
  `POST /rerank` `{"pairs":[[q,d],...]}` → `{"logits":[...]}`; `GET /health` → `{"status":"ok"}`.
  Chunked at 128 per request, 120s timeout.

### `db/connection.py` (222 lines)
- Env-driven URL (`POSTGRES_*` or `DATABASE_URL`), `pool_size=10, max_overflow=20, pool_pre_ping=True`.
- `check_db_health()` — connectivity, pgvector extension, `articles` table presence.
- `ensure_schema()` — idempotent `ALTER TABLE … ADD COLUMN IF NOT EXISTS` for:
  `rerank_score`, `rank_raw`, `pipeline_version` (recs); `refreshed_articles`, `pipeline_version` (jobs);
  `embed_token_count`, `rerank_token_count`, `is_truncated`, `recipe_version` (articles).
- `init_db()` — creates extension + `Base.metadata.create_all` + `ensure_schema()`.

### `db/models.py` (192 lines)
**`articles`** — `id` (PK, VARCHAR 100), `title` (NOT NULL), `description`, `body`, `section`,
`subsection`, `date` (TZ), `url`, `image`, `is_live`, `page_views`, `unique_users`,
`linked_ids` (JSON string), `text_embed`, `text_bm25`, `text_rerank`,
`embed_token_count`, `rerank_token_count`, `is_truncated`, `recipe_version`,
`embedding` **vector(768)**, `created_at`, `updated_at`.
Indexes: `articles_pkey(id)`, `ix_articles_id`, `ix_articles_section`, `ix_articles_date`.

**`article_recommendations`** — `id` SERIAL PK, `article_id` FK→articles CASCADE,
`recommended_article_id` FK→articles CASCADE, `rank` (1..10), `rank_raw`,
`score` (display, 0..1), `rerank_score` (raw CE prob, O-centric), `pipeline_version`,
`method` (`neural_hybrid_rerank`), `reason` (JSON evidence), `created_at`.
Unique `(article_id, rank)`; index `(article_id, rank)`.

**`ingestion_jobs`** — `id` PK (`job_<hex8>`), `filename`, `pipeline_version`, `status`
(`pending|processing|completed|failed`), `total_articles`, `processed_articles`,
`refreshed_articles`, `message` (`STAGE:X|msg` protocol), `error`, `created_at`, `updated_at`.

### `db/repository.py` (407 lines)
`ArticleRepository` static methods: `bulk_upsert_articles_fast` (chunked `INSERT … ON CONFLICT`),
`bulk_set_embeddings`, `search_similar_by_vector` (pgvector `<=>` cosine, **no ANN index**),
`save_recommendations` (delete-then-insert per source article, atomic),
`get_precomputed_recommendations` (join, orders by `rank` or `coalesce(rank_raw, rank)`),
`list_articles_ranked_by_confidence` (avg stored score desc, then rec count, then date),
`clear_all_data`, job CRUD, `get_ingestion_summary`, `get_active_job`, `fail_orphaned_jobs`.

### `db/pipeline.py` (884 lines) — the ingestion brain
`run_ingestion_pipeline(csv_source, job_id, limit=None, recompute_all_recommendations=False)`
1. `init_db()`, record `pipeline_version`.
2. **Version-change detection** — if last completed job's `pipeline_version` ≠ current →
   force `recompute_all_recommendations = True` (score-scale integrity).
3. Parse CSV → `clean_dataframe()` → compute `new_ids` / `changed_ids` (changed = `text_embed` differs).
4. `bulk_upsert_articles_fast` (chunk 500).
5. Embed only `needs_emb_ids` = new ∪ changed ∪ embedding-is-NULL.
6. `_load_full_corpus()` — **whole DB** (old + new) → records + 768-d matrix.
7. Target set = `new_ids ∪ changed_ids ∪ (uploaded ids lacking current-version recs)`.
   Early-exit if nothing to do (idempotent re-upload).
8. `_precompute_for_article()` — BM25 (focused query) + dense → RRF → top-20 → CE rerank
   → blend `0.55/0.30/0.15` → boosts → MMR → discard-floor filter → calibrate → evidence JSON.
   Stores `score` (calibrated) and `rerank_score` (raw CE, 6 dp) + `rank_raw`.
9. **Reverse push** (`_reverse_push_for_old`, L255) — inverts candidate map to find old articles
   close to new ones; for each: merges stored top-10 (reusing stored `rerank_score` **only if
   `pipeline_version` matches**, else re-scores) + new neighbours, re-scores only the missing pairs
   O-centrically, recomputes BM25/dense/boosts/MMR, overwrites top-10.
10. `reload_engine_from_postgres()` in-process, then job marked `completed`.

Job message protocol: `STAGE:PARSE|CLEAN|STORAGE|EMBEDDING|INDEX|RERANK|REVERSE|COMPLETE`.
Progress bar reserved: 0–40% embedding, 40–95% rerank, 95–100% reverse.
ETA includes a forward-extrapolated estimate of the reverse pass.

### `api/server.py` (467 lines)
- `lifespan()` → `ensure_schema()`, `fail_orphaned_jobs()`, `reload_engine_from_postgres()` in threadpool.
- Globals `rec` (Recommender) and `articles` (DataFrame); `_reload_lock` guards swaps.
- Routes: `GET /api/status`, `GET /api/articles` (page, per_page≤100, section, sort=date|confidence),
  `GET /api/articles/trending`, `GET /api/articles/{id}`, `POST /api/recommend`,
  `GET /api/sections`, `GET /`, `/favicon.svg`, `/favicon.ico`; mounts `/css`, `/js`.
- `POST /api/recommend` response: `{method, engine, source, low_confidence, best_relevance, notes,
  recommendations:[{article, score, unboosted_score, confidence_band, margin, agreement_count,
  rank, rank_raw, reason, evidence?}]}`.
- Fast-path response adds `source_article`, `diversify`, per-rec `rerank_score`.
- CORS wide open (`allow_origins=["*"]`).
- Entry: `uvicorn.run("api.server:app", host="0.0.0.0", port=5000, reload=False)`.

### `api/editor_routes.py` (221 lines) — prefix `/api/db`
`GET /health`, `POST /init`, `POST /upload-csv` (50 MB cap, 1 MB chunks, .csv only,
409 if a job is active, background `threading.Thread(daemon=True)`),
`POST /ingest-default` (`data/articles_clean_train_80.csv`), `GET /jobs/{id}`,
`GET /jobs` (+ summary), `GET /recommendations/{id}`, `POST /clear`, `POST /reload`.

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
- Search tab: `performSearch()` calls `Api.recommend({query, topK, method, useBoosts, diversify, section})`.
- Quick Story Finder: client-side fuzzy title/section match, max 15 results, ↑/↓/Enter/Esc.

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
Light theme `[data-theme="light"]` overrides only 13 tokens — `evaluator.css` and parts of
`editor.css` hard-code light-on-dark colours and do **not** follow the light theme.

---

## 4. Known Bugs / Drift (verified 2026-10-22)

### P0 — functional
1. `app.js:335` — assigns to `const statusData` → TypeError swallowed by try/catch →
   `checkReaderDatabaseState()` never refreshes engine status.
2. `app.js:72` — calls `syncCategoryPills(sec)` which **does not exist** → ReferenceError thrown
   *before* `triggerActiveTabReload()` → changing the category filter never rebuilds the feed.
3. `tabs.js:628` + `ui.js:193` — `renderCards()` does `container.innerHTML = ''` and wipes the
   notes banner that `renderNotes()` just prepended → search notes always disappear.
4. `rightpanel.css:282` redefines the global `@keyframes shimmer` (opacity pulse) → breaks
   `.skeleton` sweep animation app-wide.
5. `.btn-secondary` is used 4× in `index.html` but **never defined** in any CSS file.
6. `editor.js:150` uses raw `fetch('/api/db/clear')` with a hard-coded path instead of `Api`.

### P1 — correctness
7. XSS risk: `ui.js`, `tabs.js`, `app.js`, `hero.js` inject API strings via `innerHTML` without
   escaping. Only `evaluator.js` has `escapeHtml`.
8. `evaluator.js:110` prints "(High)" regardless of actual score.
9. `src/evidence.py:enrich_recommendation_evidence()` fabricates fallbacks (0.65 / 0.82 / 0.064 margin)
   and hard-codes boost multipliers (1.08/1.05/1.10) that no longer match `config`
   (1.072/1.09/1.104).
10. `evaluator.js:237` labels a signed score delta as "Platt Probability Calibration".
11. `evaluator.js` accepts `rec.evidence`; `tabs.js:185` only reads `rec.reason` → inconsistent.
12. `tabs.js:430` sets `dataset.loaded = 'true'` *before* the await → failed requests never retry.
13. `tabs.js:535` `buildFeed()` has no request sequencing → concurrent calls can interleave.
14. `tabs.js:254` `ev.signals.join(...)` assumes an array → throws on a string.
15. `db/pipeline.py:390` reverse-push passes `rel_pos + 1` as both bm25_rank and dense_rank →
    **fabricated agreement ranks** in reverse-pushed evidence (the forward path computes them properly).
16. `db/pipeline.py:497` `_apply_mmr` hard-codes the similarity threshold `0.80`, ignoring
    `config.MMR_THETA` (0.76) — the live path (`recommender._mmr`) uses the config value.
    → **Ingested and live MMR differ.**
17. `db/pipeline.py:421-474` `_calculate_boosts` uses `max_date` (corpus max) for recency,
    while `config.BOOST_MODE = "source_date_recency"` implies source-relative recency.
    `src/recommender.py:_boost_mult` also uses `self.articles["date"].max()` — both ignore the
    declared `BOOST_MODE`.
18. ~~`db/pipeline.py:660` persists `linked_ids` as `str(...)` → not JSON.~~
    **VERIFIED NOT A BUG.** `clean_dataframe()` already returns `linked_ids` as JSON *strings*
    (`src/data_loader.py:554` → `json.dumps(l)`), and `pipeline.py:613` uses that DataFrame
    directly (`df.to_dict("records")`). So `str()` is a **no-op** and the stored value is valid JSON.
    Live rows confirm (`["10911764"]`). The latent risk is only if someone feeds a *list* into
    `bulk_upsert_articles_fast` — that would silently store a Python repr. Keep it in mind when
    refactoring, but the current path is correct.

### P1b — experiment harness is not reproducible as shipped
19. `run_experiments.py:467` sets `a0_val_ndcg` to a **scalar** but passes it as the paired
    baseline array at `:496` → `compute_bootstrap_ci` indexes a 0-d array →
    **`IndexError: too many indices for array`**. Re-running the script today overwrites
    `ablation_results.csv` then dies before writing `tuning_results.csv`. The shipped CSVs came
    from an earlier revision. Fix = pass the per-article `a0_ndcg` list (as STEP 5 does at `:432`).
20. `import_colab_export.py:60` runs `db.query(Article.id).all()` (SQLAlchemy 2.0 autobegins),
    then `:106` does `with db.begin():` → **`InvalidRequestError: A transaction is already begun`**
    → rollback → re-raise. **The Colab export can never be imported.** The unit test passes only
    because it fails earlier at the NaN check.
21. `freeze_pool.py:75` iterates a **`set`** to build the focused BM25 query
    (`set(body_toks)`), so candidate selection depends on `PYTHONHASHSEED` → the frozen pool is
    not bit-reproducible across processes.
22. `freeze_pool.py:165` ignores `config.RERANKER_REVISION` and defaults `--max-len 192`
    (production = 512). Every `tuning_results.csv` number comes from an **unpinned, truncated**
    cross-encoder.
23. `parity_test.py` can report **"FULL PARITY VERIFIED" with a dead tunnel** — the client falls
    back to the local model on HTTP error, so it compares local vs local (cosine 1.0, same Top-5).

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
4. **Ingested vs live scoring must stay in sync.** `db/pipeline.py` reimplements BM25/blend/
   boosts/MMR separately from `src/recommender.py` — they have already drifted (bug #15, #16, #17).
   Prefer routing the pipeline through `src.recommender.py` helpers when touching this.
5. **`reason` is a JSON string**, not plain text. The frontend parses it (`ui.js`, `tabs.js`,
   `evaluator.js`). Keep the key set compatible: `summary, terms, entities, source_sentence,
   best_sentence, signals, agreement_count, boosts, context_facts, confidence,
   score_breakdown, rank_relevance, method_ranks, margin_cutoff`.
6. **Only one ingestion job at a time** — `get_active_job()` gates uploads with HTTP 409.
7. **Single-process assumption** — `reload_engine_from_postgres()` mutates in-process globals.
   Running uvicorn with `--workers > 1` desynchronises the engines.
8. Frontend has **no build step and no framework** — plain classic scripts, global scope.
   Any new module must be appended to `index.html` in the right order.
