# 📰 AI News Recommender

Give it an article (or a few words like "flight crash") and it returns the most related
articles from the dataset, with a match score and a "why recommended" line.

## Quick start

```bash
cd news_recommender
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python api/server.py
```

Open **http://localhost:5000** in your browser. The frontend is served automatically by FastAPI.

The **first run downloads two models (~1.5 GB, needs internet once)** and embeds all articles
(about a minute on CPU). Embeddings are cached in `data/cache/`, so later starts are instant.

> **Datasets are not distributed with this repository.** News article corpora are not published
> here, so a fresh clone has code but no articles. Supply your own CSV and ingest it through the
> editor upload screen or `POST /api/db/upload-csv` before the reader view has anything to show.

No internet or short on disk? Run in light mode (TF-IDF, no downloads, lower quality):

```bash
# Linux / Mac
NEWSREC_NEURAL=0 python api/server.py

# Windows PowerShell
$env:NEWSREC_NEURAL="0"; python api/server.py
```

The app switches to light mode automatically if the models cannot be loaded.

## Project layout

```
news_recommender/
├── api/
│   ├── __init__.py
│   ├── server.py            FastAPI backend (REST endpoints + serves frontend)
│   └── editor_routes.py     Ingestion routes (/api/db: upload, jobs, health)
├── db/
│   ├── connection.py        Engine, env config, idempotent schema migrations
│   ├── models.py            SQLAlchemy tables (articles, recommendations, jobs)
│   ├── repository.py        All SQL access
│   └── pipeline.py          Ingestion + precompute pipeline
├── frontend/
│   ├── index.html           Main HTML shell
│   ├── css/                 14 modular CSS files (variables, base, layout, etc.)
│   └── js/                  8 JS modules (api, state, ui, evaluator, hero, tabs, editor, app)
├── config.py                ALL tunable numbers in one place
├── evaluate.py              Compare methods + demo queries
├── requirements.txt
├── .env.example             Template for DB + model worker settings
├── PROJECT_HANDOVER_FACTS.md Verified technical handover (architecture, numbers, defects)
├── data/                    Not shipped. Your CSVs and caches live here (git-ignored)
└── src/
    ├── data_loader.py       CSV -> clean table (dedupe, remove boilerplate, build texts)
    ├── retrieval.py         BM25 keyword search, embedding search, TF-IDF fallback, RRF fusion
    ├── reranker.py          Cross-encoder that re-scores the best candidates
    ├── scoring.py           The one scoring/boost/MMR implementation (shared by both paths)
    ├── evidence.py          Deterministic "why recommended" evidence
    ├── calibration.py       Platt scaling of scores into confidence bands
    ├── model_registry.py    Model loading, remote GPU worker client
    └── recommender.py       The full pipeline, boosts, diversity, explanations
```

Read the code in this order: `config.py` -> `data_loader.py` -> `retrieval.py` -> `reranker.py` -> `scoring.py` -> `recommender.py` -> `api/server.py`.

For a detailed, measured description of every stage, see `PROJECT_HANDOVER_FACTS.md`.

## How it works

1. **Clean:** duplicate articles are collapsed (latest publish date wins) and repeated boilerplate sentences are removed.
2. **Two searches:** BM25 (exact words, names) and embeddings (meaning).
3. **Fuse:** Reciprocal Rank Fusion merges both lists into 20 candidates (`config.N_CANDIDATES`).
4. **Rerank:** a cross-encoder reads input + candidate together and scores the match.
5. **Boost (article input only):** small bonus for same section and similar publish time.
6. **Diversify:** MMR avoids near-identical results.
7. **Explain:** shared keywords, same section, same time.

If the best score is below the threshold in `config.MIN_RELEVANCE`, the UI says
"no close match found" instead of showing weak results confidently
(this is what happens for "flight crash", because the dataset has no such article).

## Incremental ingestion (adding articles later)

Each upload stores the new articles and precomputes a top-10 for them. But what about the
articles that were already there? Their precomputed lists would go stale (they could never
recommend the new arrivals). The pipeline keeps them fresh with a **reverse push**:

1. **New articles** are searched against the **whole database** (BM25 + semantic → RRF →
   top-20 → cross-encoder rerank), not just the uploaded batch. So new articles can
   recommend old ones.
2. **Reverse push:** for every new article `N`, we look at the existing articles `O` that
   `N` was close to. For each such `O` we score `rerank(O, N)` **with `O` as the query**
   (O-centric), merge it into `O`'s current top-10 (reusing the stored `rerank_score` for
   `O`'s existing entries), re-apply boosts + MMR and keep the top-10. If `N` is not good
   enough it is simply ignored, and `O` keeps its old list.

Only old articles that are genuinely close to a new article are touched — the rest are
left completely alone, so the cost scales with the new batch, not the whole archive.

This is controlled by `config.INCREMENTAL_RECOMMENDATIONS` (default `True`). Passing
`recompute_all_recommendations=True` to `run_ingestion_pipeline` forces a full rebuild of
every article's list instead (correct but slower).

Every precomputed row now stores two scores:

| Column | Meaning |
|---|---|
| `score` | final blended relevance shown in the UI (0–1) |
| `rerank_score` | raw cross-encoder probability for this exact (source → recommended) pair, computed O-centrically — the comparable value the reverse push uses |

Existing databases are migrated automatically: `db.connection.ensure_schema()` runs
`ALTER TABLE article_recommendations ADD COLUMN IF NOT EXISTS rerank_score` on startup and
before every ingestion.

### Editor visibility into uploads

The Editor dashboard has an **Upload History & Status** panel:

* a summary tile row — total uploads, total articles ingested, total old
  recommendations refreshed, and failed jobs;
* one row per upload — when it ran, the file/job id, how many articles it contained,
  how many **existing** articles the reverse push refreshed, and its status.

During an upload the live pipeline log streams the reverse-push step as its own node
(`6. Reverse Update`): how many existing articles are being re-scored and a sample of the
affected article titles. The per-job count is stored in `ingestion_jobs.refreshed_articles`
(auto-migrated with `ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS refreshed_articles
INTEGER DEFAULT 0`).

Both the rerank and reverse-update steps show a **time-left that covers the whole job**: the
rerank ETA adds an estimate of the upcoming reverse pass (`ETA … (incl. ~… reverse)`), and the
reverse step itself streams a live ETA while it walks the progress bar's final 5% (95% → 100%).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/status` | Engine health-check + metadata |
| `GET`  | `/api/articles` | Paginated article list (query: page, per_page, section, sort=`date`&#124;`confidence`) |
| `GET`  | `/api/articles/trending` | Top N most recent articles |
| `GET`  | `/api/articles/{id}` | Single article by ID |
| `POST` | `/api/recommend` | Unified recommendation (query, article_id, or user_history) |
| `GET`  | `/api/sections` | All distinct section names |

Interactive API docs available at **http://localhost:5000/docs** (Swagger UI).

## Evaluate

```bash
python evaluate.py
```

- **Part 1** prints top results for demo queries, including ones with no match in the data.
- **Part 2** compares BM25 / semantic / hybrid / hybrid+rerank using a rough automatic proxy (same-section precision@5).
- **Part 3** (optional): create `data/eval_labels.json` such as `{"10897396": [10898185, 10896576]}`
  (article id -> ids you judge relevant) to get real Precision@5 and nDCG@10. Label 20-30 articles; it takes ~15 minutes.

## Tuning tips

- Wrong "no match" warnings (too many or too few)? Change `MIN_RELEVANCE` in `config.py` after looking at the scores in `evaluate.py` Part 1.
- Results too similar to each other? Lower `MMR_LAMBDA` (e.g. 0.6).
- Want same-story articles ranked higher? Raise `RECENCY_BOOST` / `SECTION_BOOST`.
