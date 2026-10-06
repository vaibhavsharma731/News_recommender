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
│   └── server.py           FastAPI backend (REST endpoints + serves frontend)
├── frontend/
│   ├── index.html           Main HTML shell
│   ├── css/                 12 modular CSS files (variables, base, layout, etc.)
│   └── js/                  6 JS modules (api, state, ui, hero, tabs, app)
├── config.py                ALL tunable numbers in one place
├── evaluate.py              Compare methods + demo queries
├── requirements.txt
├── .env                     HuggingFace token (not tracked by git)
├── data/
│   └── articles.csv         Your dataset
└── src/
    ├── data_loader.py       CSV -> clean table (dedupe, remove boilerplate, build texts)
    ├── retrieval.py         BM25 keyword search, embedding search, TF-IDF fallback, RRF fusion
    ├── reranker.py          Cross-encoder that re-scores the best candidates
    └── recommender.py       The full pipeline, boosts, diversity, explanations
```

Read the code in this order: `config.py` -> `data_loader.py` -> `retrieval.py` -> `reranker.py` -> `recommender.py` -> `api/server.py`.

## How it works

1. **Clean:** 738 rows -> 699 unique articles; repeated boilerplate sentences are removed.
2. **Two searches:** BM25 (exact words, names) and embeddings (meaning).
3. **Fuse:** Reciprocal Rank Fusion merges both lists into 30 candidates.
4. **Rerank:** a cross-encoder reads input + candidate together and scores the match.
5. **Boost (article input only):** small bonus for same section and similar publish time.
6. **Diversify:** MMR avoids near-identical results.
7. **Explain:** shared keywords, same section, same time.

If the best score is below the threshold in `config.MIN_RELEVANCE`, the UI says
"no close match found" instead of showing weak results confidently
(this is what happens for "flight crash", because the dataset has no such article).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/status` | Engine health-check + metadata |
| `GET`  | `/api/articles` | Paginated article list (query: page, per_page, section, sort) |
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
