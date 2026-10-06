# 📰 News Article Recommender

Give it an article (or a few words like "flight crash") and it returns the most related
articles from the dataset, with a match score and a "why recommended" line.

## Quick start

```bash
cd news_recommender
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The **first run downloads two models (~1.5 GB, needs internet once)** and embeds all articles
(about a minute on CPU). Embeddings are cached in `data/cache/`, so later starts are instant.

No internet or short on disk? Run in light mode (TF-IDF, no downloads, lower quality):
`NEWSREC_NEURAL=0 streamlit run app.py`  (Windows PowerShell: `$env:NEWSREC_NEURAL=0`).
The app also switches to light mode automatically if the models cannot be loaded, and says so in the sidebar.

## Project layout

```
news_recommender/
├── app.py              Streamlit frontend (UI only, no search logic)
├── evaluate.py         Compare methods + demo queries
├── config.py           ALL tunable numbers in one place
├── requirements.txt
├── data/
│   └── articles.csv    your dataset
└── src/
    ├── data_loader.py  CSV -> clean table (dedupe, remove boilerplate, build texts)
    ├── retrieval.py    BM25 keyword search, embedding search, TF-IDF fallback, RRF fusion
    ├── reranker.py     cross-encoder that re-scores the best candidates
    └── recommender.py  the full pipeline, boosts, diversity, explanations
```

Read the code in this order: `config.py` -> `data_loader.py` -> `retrieval.py` -> `reranker.py` -> `recommender.py` -> `app.py`.

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
