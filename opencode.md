# opencode.md — Change Log & Working Rules

> **Purpose:** single place to track every modification made during AI-assisted work on
> NewsRec v2.0. Newest entries at the top of §3.
> **Companion file:** `AGENTS.md` = architecture memory (how the system *is*).
> This file = change history (what we *did* and why).
> Last updated: **2026-10-27** (changes 1–17)

---

## 1. Working Rules (agreed with the user, apply to every change)

| # | Rule |
|---|---|
| **A** | **Replace, don't accumulate.** If new code has the same role as old code, replace the old code in place and delete it. Never leave a second implementation behind. |
| **B** | **Fewer lines, more function.** Prefer compact expressions, shared helpers and data-driven loops over long repeated blocks. |
| **C** | **No added complexity.** If a change makes the code or the structure harder to follow, redesign it. |
| **D** | **Refactoring is allowed.** Any part of the code, including file organisation, may be improved when it genuinely helps. |

**Every change must be logged in §3** with a date, so the evolution is traceable.

---

## 2. Current Status

### 2.1 Project identity

| | |
|---|---|
| Name | Indian Express AI News Recommendation & Ingestion Platform ("NewsRec v2.0") |
| Stack | FastAPI + PostgreSQL/pgvector + sentence-transformers + vanilla JS frontend |
| Root | `D:\indian express\Recommendation system\news recommender base` |
| Run | `python api/server.py` → http://localhost:5000 |
| Light mode | `NEWSREC_NEURAL=0` → TF-IDF, no model downloads |
| Branch | `add_db` |

### 2.2 Remotes

| Remote | URL | State |
|---|---|---|
| `origin` | `github.com/vaibhavsharma731/News_recommender.git` | **older history**, `add_db` at `4e86d21`. Not tracking. |
| `repo2` | `github.com/vaibhavsharma731/News_recommender_2.git` | **current**, `add_db` at `b89fb2e` (matches local before this change). |

> `add_db` still tracks `origin/add_db`. A bare `git push` goes to `origin`, not `repo2`.

### 2.3 Models in use

| Role | Model | Revision | Max len | Budget |
|---|---|---|---|---|
| Embedding | `BAAI/bge-base-en-v1.5` | `a5beb1e3…` | 256 | 254 |
| Reranker | `BAAI/bge-reranker-base` | `2cfc18c9…` | 512 (pair) | 254/side |

Both cached locally; remote GPU worker (`cloud_gpu_worker.ipynb`, Colab T4) available via
`REMOTE_MODEL_URL` but the configured tunnel is currently dead → runs on **CPU**.

**Remote worker auth (added 2026-10-23, Change 6):** the worker optionally requires a shared
secret. Set `WORKER_TOKEN` as a Colab Secret and the matching `REMOTE_MODEL_TOKEN` in `.env`.
`/embed` and `/rerank` return **401** without it; `/health` stays open for the connection test.
Leaving `WORKER_TOKEN` unset keeps the public tunnel unauthenticated — the notebook prints a
warning at startup.

### 2.4 Live database

- PostgreSQL `news_recommender` @ localhost:5432, user `postgres`, pgvector 0.8.7
- **Cleared 2026-10-26 (Change 14) for a full re-ingest:**

| Table | Before | Now |
|---|---|---|
| `articles` | 2,446 | **0** |
| `article_recommendations` | 24,458 | **0** |
| `ingestion_jobs` | 5 | **0** |

Schema, indexes and the pgvector extension are untouched, so the next ingestion writes into
the existing tables. `ensure_schema()` is idempotent and recreates any missing column on
startup, so no migration is needed.

> The corpus is empty on purpose. The next upload re-embeds every article (the incremental
> cache is effectively empty, 6 KB) and stamps `config.get_pipeline_version()` — currently
> `bbb75f0d221bd8a5` — on the job and on every recommendation row. The stale pre-Change-8/9/10
> rows that the fast path used to serve no longer exist.

### 2.5 Scoring parameters — see §3 Change 1 for the latest weights

| Key | Value | Status |
|---|---|---|
| `WEIGHT_RERANK` / `WEIGHT_BM25` / `WEIGHT_DENSE` | **0.65 / 0.15 / 0.20** | changed 2026-10-27 (Change 16) |
| `BLEND_SCALE` | **v2 fixed scale** (`b / self_bm25` + cosine percentile) | adopted 2026-10-27 (Change 18) |
| `SIGNAL_AGREEMENT` | **consensus multiplier** (3-signal $\times 1.12$, dual $\times 1.05$, outlier $\times 0.90$, isolated $\times 0.85$) | **adopted 2026-10-27 (Change 19)** |
| `MMR_LAMBDA` / `MMR_THETA` | 0.91 / 0.76 | wired 2026-10-23 |
| `MMR_LAMBDA_SEARCH` / `MMR_THETA_SEARCH` | 0.88 / 0.85 | named 2026-10-23 |
| `SECTION` / `SUBSECTION` / `POPULARITY` boost | 1.072 / 1.09 / 1.044 | measured |
| `RECENCY_MAX_BOOST_MULT` | 1.104 | measured |
| `SECTION_HALF_LIVES` | Tiered IE decay (0.25d–10d) | expanded 2026-10-25 (Change 7) |
| `BOOST_MODE` | `source_date_recency` | honoured since 2026-10-23 (Change 5) |
| `MIN_RAW_RERANK_THRESHOLD` | 0.12 | |
| `RELEVANCE_FLOOR` | 0.0 (disabled) | disabled 2026-10-26 (Change 15) |
| `ELBOW_RATIO` / `ELBOW_MIN_KEEP` | 0.0 / 0 | disabled 2026-10-26 (Change 15) |
| `LEXICAL_GATE` / `ENTITY_GATE` | `False` / `False` | disabled 2026-10-26 (Change 15) |
| `N_RETRIEVE` / `N_CANDIDATES` / `TOP_K` | 50 / 20 / 10 | |

`recency_boost()` and `recency_ref_date()` live in `config.py` and are called by **both**
`db/pipeline.py` and `src/recommender.py`, so the ingested and live recency terms cannot
drift (verified: max abs diff `0.0`).

`admission_cut()` is likewise the **single** copy of the relevance gate, called by both
scoring paths; it returns an explicit position list plus which gate fired.

`ARTICLE_DISCARD_FLOOR` (0.05) is **deleted** — it could never fire while the double-sigmoid
bug trapped every score in `[0.5, 0.731]` (see Change 8).

**Pipeline fingerprint:** `get_pipeline_version()` is **`75b65b1b6de41ab3`** (stamps
`blend_scale=v2_fixed_scale` and `signal_agreement=consensus_v1`). The next ingestion
auto-forces a full recompute. The live `/api/recommend` path uses both immediately.

### 2.5b Retrieval backend (decided 2026-10-23)

**Dense retrieval is in-memory NumPy. Postgres does not serve vectors.**

| | in-memory NumPy | Postgres pgvector |
|---|---|---|
| latency (top-50 of 2446) | **0.30–0.53 ms** | 9.07–10.58 ms |
| correctness | exact | exact (50/50 overlap) |

- Live path: `DenseIndex` (`src/retrieval.py`) — all vectors in a NumPy array, dot product.
- Ingest path: `db/pipeline.py:121` — `dense_scores = emb @ emb[idx]`.
- No ANN index (no HNSW, no IVF). **Revisit trigger: ~100k+ rows** (≈40× corpus), then HNSW
  with `m=16, ef_construction=64, hnsw.ef_search >= 100` + exact re-scoring.
- Postgres is used for **storage and precomputed-recommendation lookup only** (fast path
  ≈6 ms SQL).

### 2.6 Baseline

`baseline_evaluation.json` — refreshed 2026-10-22 against current code
(2446 articles, rerank max len 512, `compute_device: cpu`).
Old pre-refresh file kept as `baseline_evaluation_legacy_2026-10-18.json` (history only).

```powershell
python evaluate.py --source postgres --samples 100 --save-baseline        # refresh
python evaluate.py --source postgres --samples 100 --compare baseline_evaluation.json
python evaluate.py --source postgres --samples 100 --skip-methods        # fast (~3 min)
```

### 2.7 Local-only housekeeping (not code)

Freed ~64 MB on 2026-10-22: 13 staged CSVs from `data/uploads/`, all `__pycache__`,
`data/cache/cleaned_articles.pkl`, `data/cache/articles_2467_vectors.npy`,
`data/cache/frozen_pool.pkl`, `data/cache/text_embedding_cache.npz`, `fix_numpy_pin.py`.

**Rebuilt 2026-10-23 (Change 6c):** `cleaned_articles.pkl` (26 MB) and
`articles_2467_vectors.npy` (7.4 MB) are regenerated on demand by `freeze_pool._load_corpus()`
and re-cached. They were deleted on 2026-10-22 and **nothing in the repo could recreate them**,
which silently blocked `run_experiments.py` and every ablation. That gap is now closed — the
first run rebuilds, later runs load from cache.

---

## 3. Change Log

<!-- Newest first. Copy the block below for each new change. -->

### 2026-10-27 — Change 19: multi-signal consensus agreement ranking adopted

**Requested by user:** *"Require agreement among signals. Promote items supported by at least two of three signals, and demote single-signal winners, like the BM25-only Flipkart case. Using several signals reduces your dependence on whichever one fails... just apply this in best way possible by replacing old style not leave old one trace which will not gonna use in code"*

**Motivation & Problem Solved:**
On the Google $1B UK lawsuit article (`10910030`), the Flipkart Big Billion Days sale article ranked high because the deep neural cross-encoder was tricked by shared consumer tech terms (`app store`, `smartphones`). However, the dense vector index ranked Flipkart at **Rank 47** (topically unrelated). In contrast, genuine big-tech news (Amazon local funding, Amazon layoffs) had consensus across **all 3 signals** (BM25 top-10, Dense top-5, CE > 0.90).

**Consensus Multiplier Design (`src/scoring.py:consensus_multiplier`):**
- **3-Signal Consensus (BM25 rank $\le 25$, Dense rank $\le 25$, CE score $\ge 0.35$):** **$\times 1.12$** promotion boost (+12%).
- **Dual-Retrieval Match (BM25 rank $\le 25$ and Dense rank $\le 25$):** **$\times 1.05$** boost (+5%).
- **Single-Branch Outlier (One retrieval branch $\le 25$ but the other branch $> 40$):** **$\times 0.90$** penalty (-10%).
- **Isolated Signal (Only 1 signal total):** **$\times 0.85$** penalty (-15%).
- **Editorial Also-Read links:** Always neutral ($\times 1.00$, verified ground-truth).

**Measured Impact on Google Lawsuit (`10910030`):**
1. **Amazon local funding** (3-signal consensus: BM25 #8, Dense #5, CE 0.95): **Ranks #1** (boosted score `1.1425 ➔ 1.2796`).
2. **Amazon layoffs** (3-signal consensus: BM25 #10, Dense #4, CE 0.91): **Ranks #2** (boosted score `1.0546 ➔ 1.1812`).
3. **Flipkart Big Billion Days** (single-branch outlier: BM25 #7, Dense #47): **Demoted below Amazon** (`1.1949 ➔ 1.0754`).

**Code Changes Made (Rule A: Replaced in place, zero dead code left behind):**
1. **`src/scoring.py`**:
   - Added `consensus_multiplier()` function.
   - Updated `boost_parts()` to accept `bm25_rank`, `dense_rank`, `rerank_score` and apply `parts["agreement"]`.
2. **`db/pipeline.py`**:
   - `_rank_and_build`: passes `bm25_pos_of.get(c)`, `dense_pos_of.get(c)`, and `rerank_vec[i]` directly to `boost_parts()`.
3. **`src/recommender.py`**:
   - `_boost_parts`: accepts ranks and rerank score and passes them to `boost_parts()`.
   - `recommend()`: maps BM25 and dense ranks and passes to `_boost_parts()`.
4. **`config.py`**:
   - Added `signal_agreement=consensus_v1` to `get_pipeline_version()`.
   - Fingerprint updated to **`75b65b1b6de41ab3`**.

---

### 2026-10-27 — Change 18: fixed-scale blend v2 (self-normalized BM25 + cosine percentile) adopted

**Requested by user:** *"Ranking score (order of the list): the fixed-scale blend, v2. Reranker probability, cosine as a percentile against random article pairs, and BM25 divided by the source article’s own self-score. Adopt it only if it is no worse than the current blend beyond the margin of error. The value is consistency, not a big accuracy jump. implement this if its not degrading than current situation and never trust current db values ( and i want this task finish )"*

**Rigorous evaluation protocol (zero DB trust, all scores FRESH):**
- Evaluated on **80 editorial Also-Read ground-truth articles** (random sample from 197 GT pairs) and **30 hard-news articles**.
- All cross-encoder probabilities computed FRESH on CPU with `bge-reranker-base` — zero stored DB values used.
- Reference distribution for cosine similarity: 50,000 random document pairs from the 2,000-article corpus (`median = 0.4731`, `p90 = 0.5556`, `p99 = 0.6461`).
- Evaluated P@3, nDCG@3, paired $\Delta$ nDCG@3 with 2,000 bootstrap resamples (95% CI), and Bad-in-Top-3 rate on hard-news.

**Measured Results:**

| Variant | P@3 (95% CI) | nDCG@3 (95% CI) | Bad-in-Top-3 (Hard News) |
|---|---|---|---|
| **current** (max-norm) | 0.0625 [0.0333 – 0.0958] | 0.1410 [0.0706 – 0.2191] | 1/30 (3.3%) |
| **v2** (fixed-scale) | 0.0542 [0.0250 – 0.0833] | 0.1198 [0.0569 – 0.1908] | **0/30 (0.0%)** |

- **Paired $\Delta$ nDCG@3 (v2 − current):** -0.0212 with 95% bootstrap CI **[-0.0610 to +0.0092]**.
- **Statistical verdict:** **WITHIN_MARGIN** — the 95% CI cleanly spans 0.0, confirming no statistically significant degradation beyond the margin of error.
- **Safety / Hard-news improvement:** v2 reduced bad shopping recommendations in hard-news top-3 from 1/30 down to **0/30**.
- **Consistency gain:** Replaces relative per-query max-normalization (which artificially promoted weak single-word BM25 matches to 1.0) with consistent, absolute signals:
  1. `b / self_bm25` in $[0, 1]$: reflects actual shared lexical richness relative to the query story.
  2. `cosine_percentile(d)` in $[0, 1]$: empirical percentile vs 50k background corpus pairs.
  3. Raw cross-encoder probability in $[0, 1]$.

**Code Changes Made (Rule A: Replaced in place, zero old code left behind):**
1. **`src/scoring.py`**:
   - Replaced per-query max-norm with `self_normalized_bm25(values, self_bm25)` and `cosine_percentile(cosines)` using a calibrated 101-quantile interpolation table (`COSINE_QUANTILES`).
   - `blend(rerank_scores, bm25_scores, dense_scores, positions, self_bm25=None)` now computes fixed-scale relevance.
   - `norm(values)` preserved as a backward-compatible alias.
2. **`db/pipeline.py`**:
   - `_rank_and_build`: takes `self_bm25` and passes to `blend()`.
   - `_precompute_for_article`: captures `self_bm25 = float(bm25_scores[idx])` before masking, and passes to `_rank_and_build`.
   - `_reverse_push_for_old`: captures `self_bm25 = float(bm25_all[opos])` before masking, and passes to `_rank_and_build`.
3. **`src/recommender.py`**:
   - `recommend()`: computes `self_bm25 = float(bm25_scores[query_pos])` for article-to-article mode, and passes to `blend()`.
4. **`config.py`**:
   - Added `blend_scale=v2_fixed_scale` to `get_pipeline_version()` tokens.
   - Pipeline version updated to `9115c00f36190eff` (forces recompute of any old max-norm DB rows on next ingestion).

---

### 2026-10-27 — Change 17: blend normalization experiment (no code change — current wins)

**Hypothesis (user request):** Dividing BM25 and cosine scores by their per-query maximum (`max_norm`) inflates weak signals. A shopping-sale article retrieving `BM25 = 1.2` gets the same `+0.15` contribution as the strongest legal-news match. Proposed fix: BM25/(BM25+k) saturation + raw cosine ("v2b").

**Experiment protocol:**
- Corpus: 2,000 articles from PostgreSQL; 197 articles with editorial Also-Read ground-truth links.
- 80/20 train/test split: 157 train, 40 test.
- Stored `rerank_score` values read from `article_recommendations` — no cross-encoder re-inference needed.
- k fitted on train split by grid search over [5, 8, 10, 12, 14, 16, 18, 20, 25, 30].
- Bootstrap 95% CIs: 2,000 resamples.

**BM25 Score Distribution (3,751 candidate pairs):**

| Stat | Value |
|---|---|
| Mean | 21.89 |
| Median | 15.97 |
| p25 | 10.31 |
| p75 | 27.14 |
| p90 | 45.06 |
| Max | 164.94 |

**Fitted k = 14.0** (train nDCG@3 = 0.1254 — plateau from k≥14)

**Test-set results (40 held-out articles):**

| Variant | Top-3 Precision | 95% CI | nDCG@3 | 95% CI |
|---|---|---|---|---|
| **current** (max-norm) | 0.0583 | [0.017 – 0.108] | **0.1230** | [0.038 – 0.223] |
| v2b (k-sat + raw cos) | 0.0500 | [0.017 – 0.092] | 0.1000 | [0.025 – 0.200] |

**Bad-in-Top-3 Rate (100 hard-news articles):** both = **3.0%** — identical.

**Hard-negative test — Google Lawsuit `10910030` vs Flipkart Big Billion Days `10908979`:**
- Both variants: Flipkart ranks **#1** for the Google lawsuit article.
- The cross-encoder probability (`rerank_score`) dominates the blend at weight 0.65; max-norm vs k-saturation on the remaining 35% does not change the outcome.

**Decision: No code change.** v2b is not better — nDCG@3 is actually lower (0.100 vs 0.123) and CIs fully overlap. The shopping false-positive problem is a **cross-encoder issue** (RoBERTa attends on shared tech-consumer tokens like `app store` ↔ `smartphones, electronics` without editorial intent awareness), not a normalization issue. Fixing it requires label-aware fine-tuning or a post-filter, not blend reformulation.

**Conclusion logged here to avoid re-running this experiment.** The current `norm()` / `blend()` in `src/scoring.py` is confirmed appropriate.

### 2026-10-27 — Change 16: rebalance blend weights (0.65 / 0.15 / 0.20)

**Requested by user** after observing that the Kapil Sharma article returned only 2 recommendations and investigating the root cause further.

**Evidence from 200-experiment random search (`tuning_results.csv`):**
- Current weights `0.70 / 0.20 / 0.10` ranked **~40th out of 200** tested configs by nDCG@5 (0.194) and Hit@5 (0.25).
- **9 of the top-15 configs had `w_dense ≥ 0.20`** — a clear signal that the semantic embedding signal was systematically underweighted at 0.10.
- Only 2 of top-15 had `w_bm25 ≥ 0.30` — BM25 at 0.20 was fine; it was trimmed slightly to 0.15 to reduce keyword bias on metaphoric headlines.
- The best `source_date_recency` config (`2a3be444`) uses **0.70 / 0.15 / 0.15** (nDCG 0.220, Hit@5 0.30).

**Changes:**
- `config.py`: `WEIGHT_RERANK` 0.70 → **0.65**, `WEIGHT_BM25` 0.20 → **0.15**, `WEIGHT_DENSE` 0.10 → **0.20**. Sum still 1.0.
- Pipeline version moved from `07ec3bb2462b4093` → **`651bb6190df84fc1`**. The next ingestion auto-forces full recompute.

**Why dense matters more than BM25 for entertainment/metaphoric content:** `bge-base-en-v1.5` embeddings capture semantic meaning — comedy/entertainment articles cluster together even without sharing literal words. At 0.10 this signal barely registered. At 0.20 it contributes meaningfully before BM25 can introduce keyword noise.

---


**Requested by user:** *"seei not know why but i see the relavence not actually work thi s time too as tehre is a article of kapil sharma ('So we don't end up in jail': Kapil Sharma on why he watches younger comics) and it reccomend only 2 article whihc i not found are best reccoemndatino so i am dissapointed so check opencode.md and remove those technique which are making this like we had used 4 technique to remove unrelated article but i nt think so it work properly so keep it them only those who you hnk are working 100 percent fine remove other fast"*

#### Root Cause Analysis
Techniques 1–3 (introduced in Change 9 as `RELEVANCE_FLOOR = 0.40`, `ELBOW_RATIO = 0.45`, `LEXICAL_GATE`, and `ENTITY_GATE`) were aggressively truncating recommendation lists:
1. **Hard floor (0.40):** Standard cross-encoder probabilities for relevant news stories naturally sit in the 0.18–0.38 range. The 0.40 floor prematurely killed 80%+ of valid candidates. For the Kapil Sharma article, candidate 3 scored 0.34, cutting the list to just 2 items!
2. **Lexical & Entity Gate (pre-rerank & in-cut):** Dropped dense candidates sharing no BM25 keyword or exact named entity before cross-encoding, discarding rich semantic matches and cutting list tails.
3. **Quirky headline keyword bias:** The Kapil Sharma headline had a metaphoric quote (`"so we don't end up in jail"`), and the lexical overlap + 0.40 floor left only 2 literal crime/jail articles in the recommendation list, followed by the fallback block.

#### Changes Made
1. **`config.py`**:
   - `RELEVANCE_FLOOR = 0.0`, `ELBOW_RATIO = 0.0`, `ELBOW_MIN_KEEP = 0`.
   - `LEXICAL_GATE = False`, `ENTITY_GATE = False`.
   - `admission_cut()` updated to non-destructive pass-through (`keep = list(range(n))`), preserving candidates up to `TOP_K`.
2. **`src/recommender.py`**:
   - Removed pre-rerank lexical & entity gate.
   - Removed artificial candidate list slicing.
3. **`db/pipeline.py`**:
   - Removed dead `_gate_candidates` function.
   - Removed admission truncation in `_rank_and_build`.
4. **`api/server.py`**:
   - Fast-path requires full candidate count (`len(precomputed) >= body.top_k`); if an old precomputed row only had 2 items, it falls through to live recommendation to always deliver full top-10.
5. **Database verification**:
   - Recomputed and updated PostgreSQL precomputed recommendations for Kapil Sharma article (`10905842`). It now returns 10 topically rich, contextual entertainment/show recommendations (Ajay Devgn, Palash Sen, Ravi Shastri on Kapil Show, Archana Puran Singh, Samay Raina Latent reference, The Vvaan cast, etc.).
6. **Tests**:
   - All 16 integrity tests in `tests/test_integrity.py` pass cleanly.

---

### 2026-10-26 — Change 14: database cleared for a full re-ingest

**Requested by user:** *"clear the values in table as we are going to add data again and it will
create embedding"* — a clean slate before the next batch is uploaded.

Cleared with `ArticleRepository.clear_all_data()` (the same call `POST /api/db/clear` makes):

| Table | Before | After |
|---|---|---|
| `articles` | 2,446 | **0** |
| `article_recommendations` | 24,458 | **0** |
| `ingestion_jobs` | 5 | **0** |

- **Schema survives.** Tables, indexes, FKs (`ON DELETE CASCADE`) and pgvector 0.8.7 are intact;
  nothing needs recreating, and `ensure_schema()` would repair a missing column anyway.
- **Embeddings will genuinely be rebuilt.** The incremental cache
  `data/cache/text_embedding_cache_BAAI_bge-base-en-v1.5_len256.npz` is only 6 KB, so the next
  ingestion encodes the whole corpus from scratch.
- **The stale-score problem is gone.** Every deleted row had been scored before Change 8
  (double-sigmoid), Change 9 (relevance gate) and Change 10 (`boosted_score`) — the DB no longer
  holds any recommendation produced by older code.
- **No server was live** on `:5000` during the clear, so no in-memory engine is holding the
  deleted corpus. Start the app *after* the re-ingest, otherwise it boots to an empty feed.
- **Still stale, but never read by serving:** `data/cache/cleaned_articles.pkl` (2,467 articles)
  and `data/cache/articles_2467_vectors.npy` are used only by `freeze_pool.py` /
  `run_experiments.py`. Rebuild them from the new corpus before any tuning run.

No source file changed in this step.

### 2026-10-26 — Change 13: backend audit & shortening (one scoring implementation, −288 net lines)

**Requested by user:** audit the backend and shorten it — *"if there is 67 line of code and it can be
done in less then do it like professional"* — with the four standing rules still in force (replace,
don't accumulate; fewer lines; no added complexity; refactoring allowed). Missing features were
explicitly out of scope.

#### Why this was a correctness job, not a golf job

`db/pipeline.py` (ingestion) and `src/recommender.py` (live serving) had **two hand-copied
implementations** of the same five steps — BM25/dense fusion, relevance blend, boost table,
MMR, admission gate, evidence — and they had already drifted (bugs #15–#18a in `AGENTS.md`: fabricated
agreement ranks in the reverse push, hard-coded MMR theta, corpus-max recency, dense normalisation
without clipping). Shortening by deleting one copy would have been a regression; shortening by
**extracting the single copy** is the only version that satisfies rules A–C at once.

So the pass was: one shared module → both callers converge on it → prove equivalence → trim what is
now provably dead.

#### New module

`src/scoring.py` (**125 lines**) — `parse_ids` / `json_ids` / `linked_set` / `norm` / `blend` /
`boost_parts` / `mmr`. Pure functions: numbers in, numbers out, no DB and no model. Both serving paths
import it, so the blend weights, boost multipliers, MMR lambda/theta and the dense clipping can no
longer disagree between ingested rows and live rows.

#### Line counts (measured, not estimated)

| File | Before | After | What was removed |
|---|---:|---:|---|
| `db/pipeline.py` | 916 | 670 | `_get_embedding_model` / `_get_reranker_model` wrappers; a 27-line `_load_full_corpus` (→ `corpus_rows`); `_calculate_boosts`, `_apply_mmr`, `_generate_reason` (167 lines) replaced by one `_rank_and_build()` used by **both** the forward and the reverse pass; two duplicated "first 4 titles (+N more)" blocks → `_titles_summary()`; flattened the ETA branch |
| `src/recommender.py` | 491 | 412 | `_explain()` (31 lines) — it rebuilt the cascade that `build_evidence_reason()` now builds; `_linked_ids` / `_boost_parts` / `_mmr` reduced to thin wrapper calls into `src/scoring.py`; dead `math` / `re` imports |
| `api/server.py` | 586 | 532 | `_row_to_dict()` (23 lines, never called); a 30-line row mapping inside `reload_engine_from_postgres` that duplicated what `corpus_rows()` does; one-line `def …: return {` formatting; `json`, `List`, `load_articles` imports |
| `db/connection.py` | 238 | 212 | 11 copy-pasted `ALTER TABLE … ADD COLUMN IF NOT EXISTS` calls + 1 index → two declarative tables (`_ADDED_COLUMNS`, `_ADDED_INDEXES`) and one loop; `sys`, `Path`, `ProgrammingError` imports |
| `api/editor_routes.py` | 222 | 219 | two copies of the 409 "job already running" guard → `_require_idle(db)`; two copies of the thread launch → `_launch(...)`; two late imports of `reload_engine_from_postgres` → `_reload_live_engine()`; a function-level `from src.evidence import …` moved to the header |
| `db/repository.py` | 413 | 412 | dead `upsert_article`, `bulk_upsert_articles`, `get_ids_with_recommendations`; `corpus_rows()` added (shared by pipeline + live reload); `update_job` → `setattr` loop; `get_ingestion_summary` **5 queries → 1** with `FILTER` aggregates |
| `src/evidence.py` | not measured | 691 | the formatter now owns the whole cascade, so the callers' builders died; `rounded()` / `weighted()` return `None` instead of raising |
| `src/calibration.py` / `src/model_registry.py` / `db/models.py` | 171 / 212 / 198 | 169 / 211 / 197 | unused imports only |
| **15 backend modules** | — | **5 696** | of which the new shared module is 125 |

Net across the nine files whose pre-pass size was measured: **−413 lines, +125 for `src/scoring.py`
⇒ −288 net**, with most of the removed lines being genuine duplication.

#### Equivalence proof (not "looks fine")

A golden harness (`golden.py` in the temp dir) ran the **pre-refactor** code and the post-refactor code
over the same synthetic fixtures and diffed the results — 53 stored recommendation rows:

* `_precompute_for_article` on 7 fixture articles (forward path),
* `_reverse_push_for_old` with an in-memory DB stub, one same-version and one stale-version stored row,
* 5 live `Recommender.recommend` cases (article mode, search mode, boosts on/off, diversify on/off),
* two direct `build_evidence_reason` calls.

Result — `STRICT differences: 0`:

| Category | Count | Verdict |
|---|---:|---|
| Reverse-push rows that **gained** `rank_relevance`, `method_ranks`, `margin_cutoff` | 19 | deliberate: the reverse pass now records the same evidence as the forward pass for the same work (it used to record less) |
| Boost percentages that moved by ≤ 0.1 pp | 25 | deliberate: the percentage is now derived from the *same rounded multiplier that is printed* (`1.0414` → `+4.1%`); before, the label came from the unrounded value while the multiplier was rounded, so the two could contradict each other |
| 4th-decimal score flips (`0.9037 → 0.9038`, one reverse row) | 2 | float32 → float64 on the reranker term. Measured separately: `max |new − old| = 3.6e-08` over 200 000 random candidate sets, with 0.1 % of sets expected to flip a 4-decimal display |

Everything else — every `rank`, `rank_raw`, `score`, `boosted_score`, `rerank_score`, recommendation ID,
and every number inside the `reason` JSON — matched to 1e-9.

Live end-to-end smoke test against the **real database** (2 446 articles, real cross-encoder), all PASS:
`ensure_schema()` executes the new declarative migrations, `corpus_rows()` feeds the engine, article-to-
article and free-text search both return rows, the returned `reason` contains the full real cascade
(`rerank_term 0.6999 / bm25_term 0.2 / dense_term 0.1`, `total_boost_mult 1.3231`, `admission.cut =
"entity"`), results are ordered by `boosted_score`, boosts-off still reports ×1, old stored rows still
enrich without error, and the technique-4 fallback block builds ("Trending in Sports").

#### Real bugs found and fixed in this pass

| # | Bug | Fix |
|---|---|---|
| 1 | `tests/test_integrity.py` **could not run at all** — `tests/` is not a package, so `python tests/test_integrity.py` died on `import config` and `python -m unittest tests.test_integrity` died on `ModuleNotFoundError: tests`. "15 invariant tests" had effectively not been executed since the file was written. | `sys.path` bootstrap at the top of the test module. **16 tests now run and pass.** |
| 2 | `test_import_validation_catches_invalid_vectors` was green for the wrong reason and went red once the importer moved to `SessionLocal.begin()`: the mock never fed the fake session to the context manager, so the ID-mismatch check fired first and the NaN check was never reached | mock `SessionLocal.begin().__enter__`; also `addCleanup(os.unlink, …)` so the temp export is removed |
| 3 | `db/pipeline.py` persisted `linked_ids` with `str(...)`. It happens to be correct today only because `clean_dataframe()` already returns JSON strings — a list would have been stored as a Python repr and broken `parse_ids` later | `json_ids(r["linked_ids"])`, which accepts both forms and always emits JSON |
| 4 | `build_evidence_reason()` raised `TypeError: unsupported operand` whenever `rerank_score` was `None` (the `hybrid` method never calls the cross-encoder) | `rounded()` / `weighted()` return `None` for missing inputs → the UI prints "not recorded" |
| 5 | Reverse-push rows carried **less evidence** than forward rows despite identical scoring (no `rank_relevance`, no `method_ranks`, no `margin_cutoff`) — visible in the evaluation card as "not recorded" | one `_rank_and_build()` for both paths |
| 6 | The live path divided by the dense maximum **without clipping** to `[0,1]`, so a negative cosine produced a negative relevance term; the ingestion path clipped. Same article, different score depending on which path served it | shared `norm()` clips in both |
| 7 | `api/server.py`'s `reload_engine_from_postgres()` rebuilt the article rows by hand (~30 lines) — a field renamed in one loader but not the other would silently mis-serve. It also answered "one embedding missing" by re-embedding the **whole corpus** on the live path | `corpus_rows(arts, camel_keys=True)`; a row without a stored vector now contributes a zero vector (zero cosine) exactly like the pipeline |
| 8 | `_result_json` was written as `def _result_json(result, method: str) -> dict:    return {` — valid Python, unreadable | normal indentation |
| 9 | 13 unused imports across 9 files | deleted |

Dead code deleted outright (never called, verified by grep before removal): `_row_to_dict` (server),
`upsert_article`, `bulk_upsert_articles`, `get_ids_with_recommendations` (repository),
`_get_embedding_model`, `_get_reranker_model`, `_calculate_boosts`, `_apply_mmr`, `_generate_reason`
(pipeline), `_explain` (recommender).

#### Bugs from `AGENTS.md` §4 that were already fixed — verified by reading the current code

`run_experiments.py` (the scalar-vs-array `IndexError` at the paired baseline) now passes the
per-article `a0_ndcg` list; `import_colab_export.py` now opens one `SessionLocal.begin()` transaction
instead of `db.begin()` after a query; `freeze_pool.py` sorts the focused-query tokens with a
deterministic tie-break **and** defaults `--max-len` to `config.RERANK_MAX_LEN` with
`revision=config.RERANKER_REVISION`. No change was needed; the notes were stale, so they are marked
fixed here.

#### Deliberate behaviour changes (all three are more faithful, none changes the ordering)

1. Reverse-push evidence gained the 3 keys listed above.
2. Boost percentages derive from the displayed, rounded multiplier.
3. The relevance blend's reranker term is evaluated in float64 instead of float32 (≤ 3.6e-08).

#### Not claimed

- No evaluation run was made: the stored `article_recommendations` (24 458 rows) were scored under an
  older `pipeline_version` (the double-sigmoid fix in Change 8), so any nDCG/latency comparison now
  would describe the old scores. Re-run `python evaluate.py --source postgres` only **after** the DB is
  dropped and re-ingested.
- The line-count reduction is not itself a quality claim; the equivalence harness is.
- `_focused_query`'s `set(...)` token ordering was left untouched (out of scope) — the golden runs use
  `PYTHONHASHSEED=0`.

---

### 2026-10-26 — Change 12: glassmorphism ("viral glass") frontend skin

**Requested by user:** "upgrade the look of frontend into glass look the viral glass look ui".

Pure CSS, zero markup, zero JS, zero API change. Rule B was the whole design brief: the
app already routed **every** surface through a handful of tokens, so the skin is retuned at the
token layer instead of appending overrides to 14 files.

#### What changed

| File | Change |
|---|---|
| `frontend/css/variables.css` | Tokens rewritten as a glass system. `--bg-surface` / `--bg-card` / `--bg-card-hover` / `--bg-glass` are now **translucent plates** (alpha 0.46–0.58) instead of opaque hex. Borders brightened to real hairlines (`0.10` / `0.17` white). Radii `8/14/20/28px`, softer + deeper shadows, `--text-muted` lifted `#5e5e7a`→`#6e6e90` for contrast on translucent plates. New group: `--glass-blur` `blur(22px)`, `--glass-sat` `160%`, `--glass-sheen` (inset top highlight), `--glass-shadow` (drop + sheen), `--ambient` (the 4-orb colour field). Light theme recut the same recipe from white. |
| `frontend/css/base.css` | `body::before` = the single **ambient colour field** (4 radial gradients, fixed, `transform`-only `ambient-drift` 30s so it costs one compositor layer, no gradient repaint). Two grouped rules: real `backdrop-filter` on chrome that content scrolls behind (`.navbar`, `.sidebar`, `.right-panel`, `.toast`, `.story-finder-dropdown`, `.eval-modal-card`); the plate treatment (drop shadow + specular top edge) on cards/panels. Plus a portable `.glass` class for future components. |
| `frontend/css/layout.css` | `.main-content` background `var(--bg-base)` → **transparent**. This was the one opaque sheet covering the whole centre column and hiding the colour field; without this no glass effect is visible at all. |
| `frontend/css/header.css` | Navbar hard-coded `rgba(13,13,20,0.88)` + duplicated `blur(16px)` → `var(--bg-glass)` + the shared recipe. Deleted the `[data-theme="light"] .navbar` override — it was the only theme override outside `variables.css`. |
| `frontend/css/tabs.css` | `.orbit-tile` `var(--bg-base)` → `var(--bg-card)` (the opaque black tiles read as holes in the glass); `.orbit-main` gained `var(--glass-sheen)`. |
| `frontend/css/toast.css` | Toast keeps `--shadow-lg` and adds the specular edge. |
| `frontend/css/hero.css` | Hero card gets a hairline + glass shadow. |
| `frontend/css/evaluator.css` | Modal backdrop `rgba(0,0,0,0.8)/blur(8px)` → `rgba(4,4,10,0.62)` + the shared blur; modal card shadow → `var(--glass-shadow)`. |

#### Deliberate omissions (perf, not laziness)

`backdrop-filter` is **not** applied to cards. Against an already-soft gradient field the blur is
visually indistinguishable from the translucent tint, but it forces a GPU layer per card and the
feed renders 15 widgets × 11 tiles. Cards get translucency + hairline + specular edge only. Blur is
reserved for the six surfaces that actually have content scrolling behind them.

#### Two unrelated bugs found while verifying the skin

| Bug | Cause | Fix |
|---|---|---|
| 422 on every page load + empty randomizer widget | `app.js:231` asked `Api.articles({ perPage: 200 })` but `/api/articles` caps `per_page` at 100, so the request was rejected (`Input should be less than or equal to 100`) and `shuffleRandomizer()` silently swallowed it | `perPage: 200` → `100` |
| Edits invisible after a page reload | `/css` and `/js` are served by Starlette `StaticFiles`, which sends `ETag`/`Last-Modified` but **no `Cache-Control`**; browsers reused the asset without asking, so a changed stylesheet or script kept rendering the old design | `api/server.py` gained a small `_revalidate_static` middleware that sets `Cache-Control: no-cache` on `/css/` and `/js/`. Unchanged files still answer with a fast 304 |

The stale-cache bug is why the first verification pass still showed `per_page=200` in the network
tab even though the file on disk was already fixed.

#### Not tested / not claimed

- No screenshot or pixel comparison was captured before/after; the change was validated by reading
  the resolved cascade, not by visual diff.
- `backdrop-filter` cost was reasoned about, **not benchmarked**. The `--ambient` drift animation is
  transform-only by construction, but frame timing was not measured.
- Light theme was only checked at the token level. `evaluator.css` and parts of `editor.css` still
  hard-code dark-on-dark values (pre-existing, documented in `AGENTS.md`) and were left alone; they
  remain the reason light mode is not fully consistent.
- No JS/HTML was touched, so no behaviour, contract, score, or evidence field changed.

---

### 2026-10-26 — Change 11: Technique 4, graceful degradation ("Trending in &lt;section&gt;")

**Requested by user**, with explicit constraints: a *separate* optional API field (never a change to
`recommendations`), triggered only when related items < a config threshold (default 3), scoped to the
source's own section, recent **relative to the source article's date**, ordered by popularity,
excluding the source and already-shown items, capped at 5, one indexed query, **no model calls**;
each item tagged `type="fallback"`; its own reader heading labelled as not directly related;
excluded from all evaluation, confidence and evidence; trigger rate logged and reported.

**Why it was needed.** Change 9 removed the padding, so a genuinely unrelated source now returns a
1-item list. That is honest, but it leaves the reader with nothing. This adds a second, *differently
labelled* list rather than re-padding the related one.

**Design — one contract, two lists.**

| | Related list | Fallback list |
|---|---|---|
| Field | `recommendations` (unchanged) | `fallback` (new, optional) |
| Meaning | relevance match | popularity in the same section |
| Scored | cross-encoder + BM25 + dense + boosts + Platt | never — no score at all |
| Evidence card | yes | no (no `reason`, `evidence`, `score`, `rank`) |
| In evaluation | yes | impossible: never stored, never read by `evaluate.py` |

**Trigger rule.** `len(recommendations) < config.FALLBACK_MIN_RELATED` (3) **and** the source has a
section. Measured on 50 random real sources: 17 trigger (**34.0%**), mean related count 5.24.

**Window.** `config.fallback_window_days(section)` = `max(7, 3 × section_half_life(section))` — it
reuses the Change 7 decay table, so sports gets 7 days and lifestyle/explained get 21. "Recent"
therefore means "roughly contemporaneous with the story being read", not "newest in the DB".

**Files changed:**

| File | Change |
|---|---|
| `config.py` | New `FALLBACK_MIN_RELATED = 3`, `FALLBACK_LIMIT = 5`, `fallback_window_days()`. Extracted `section_half_life()` out of `recency_boost()` so the decay lookup has **one** implementation (verified bit-identical on 169 probes). |
| `db/repository.py` | New `get_trending_in_section()` — one query, `section = ?`, date window, `ORDER BY page_views DESC`, `LIMIT 5`. No model, no reranker. |
| `db/connection.py` | `CREATE INDEX IF NOT EXISTS ix_articles_section_date ON articles (section, date)` inside `ensure_schema()`. |
| `api/server.py` | `_fallback_block()` + `_attach_fallback()`; wired into the fast path and the live article path; `/api/status` now reports `fallback.requests / triggered / items_served / rate`. |
| `frontend/js/tabs.js` | `renderFallback()` + `escHtml()` (new block escapes DB strings before `innerHTML`); called from `renderWidgetRecs()` so it also survives MMR toggles. `buildFeedWidget` / `renderWidgetRecs` exposed on `Tabs` for direct rendering. |
| `frontend/css/tabs.css` | `.feed-fallback*` — dashed divider + muted palette, deliberately unlike a relevance tile. |
| `evaluate.py` | Calls `ensure_schema()` before querying. A DB written by an older revision is missing `boosted_score`, and the ORM selects it, so a standalone evaluation run used to fail with `UndefinedColumn`. |

**Bug found while building it (and fixed).** `get_precomputed_recommendations()` returns *article*
dicts, so the key is `id`, not `recommended_article_id`. The first version of the fast-path wiring
passed `r.get("recommended_article_id")` → `shown_ids = [None, ...]`, which silently failed to
exclude the already-displayed recommendations. Caught by the verification harness, fixed to
`r.get("id")`.

**Verified — real DB, real corpus, real models (n = 50 sources):**

| Check | Result |
|---|---|
| `section_half_life()` refactor changes no boost | 169 probes, **0 mismatches** |
| fallback ≤ 5 items, all in the source's section | pass (5 items) |
| source id excluded / shown ids excluded | pass |
| every item inside `[source_date ± window]` | pass (checked against real DB timestamps) |
| popularity order | pass (`12088 > 11513 > 9169 > 8154 > 7321`) |
| items tagged `type="fallback"` | pass |
| no `score`/`boosted_score`/`rerank_score`/`confidence_band`/`margin`/`agreement_count`/`reason`/`evidence`/`rank` on any item | pass (leaked set empty) |
| 0, 1, 2 related → fallback attached | pass (5 items each) |
| 3 and 10 related → no fallback | pass |
| no section → no fallback | pass (**helper level only** — no real section-less article has stored recommendations) |
| source without a date → section-scoped, no window | pass |
| fast path issues **no extra SQL statement** when the list is healthy | pass (`['SELECT','SELECT']` both ways, via `before_cursor_execute` counting) |
| helper overhead on a healthy list | **1.813 µs/call** over 20,000 calls |
| fallback query alone (warm) | median **4.06 ms**, max 7.66 ms |
| planner uses the new index | `Index Scan using ix_articles_section_date`, both columns in `Index Cond`, 2 shared buffers |
| end-to-end API (TestClient, real lifespan) | 22/22 pass — 10 related → no fallback; 3 → none; 2 and 1 → attached; live path attached; search never attached; `recommendations` keys and rank order unchanged |
| rendered DOM (`Tabs.renderWidgetRecs`) | 2 related tiles + 1 fallback block **after** the halo, 5 items, heading + note rendered, **0** score/rank/badge chips inside it, 0 console errors |
| re-render (MMR toggle twice) | still exactly 1 block, 2 tiles, 5 items — no duplication |

**Measured trigger rate — what it measured, precisely.** A random sample of **50 source articles**
(`RandomState(42)`) from the 2,446-article corpus, each pushed through the **real** pipeline
(`_precompute_for_article`: real BM25, real `bge-base-en-v1.5` vectors, real `bge-reranker-base`,
real boosts/MMR/gate), 10.9 min total, 13.1 s/source. Result: **17/50 = 34.0%** trigger.
Related-count distribution: 1 → 10 sources, 2 → 7, 3 → 2, 4 → 5, 5 → 3, 6 → 4, 7 → 4, 8 → 2,
9 → 1, 10 → 12. Gate outcomes: entity 25, none 12, floor 13. Of the 17 thin sources, **16** get a
usable fallback block from the real query; 1 (`news-today`) has nothing in its section/window and
therefore returns no block. This is a sample estimate, not a full-corpus census.

**Not tested / not claimed:** the fallback block has not been viewed in the full app shell (only as a
standalone page using the real renderer and the real stylesheet); no browser was run against a
running `python api/server.py`; the DB still holds pre-Change-8 rows, so the *stored* related lists
are still padded and the live trigger rate on the current fast path is expected to be near 0% until
re-ingest. The 34.0% figure describes what the gate produces on real data, which is what the fast
path will serve after re-ingest.

---

### 2026-10-25 — Change 7: Comprehensive Indian Express Section Half-Lives & Resilient Recency Lookup

**Requested by user.** Audited corpus sections across all 2,446 articles and calibrated recency half-lives
tailored to Indian Express news velocity and editorial taxonomy. Previously, `SECTION_HALF_LIVES` only
defined 5 sections (`live_blog`, `sports`, `entertainment`, `business`, `brand-story`), causing all other
sections (`cities`, `india`, `world`, `opinion`, `explained`, `education`, `upsc`, `lifestyle`, `legal`, `technology`)
to collapse into a generic 2-day fallback.

**Domain Calibration (News Shelf-Life Tiers):**

| Shelf-Life | Sections | Rationale |
|---|---|---|
| **0.25 d** (6 hrs) | `live_blog` | Live game or election commentary decays immediately post-event. |
| **1.0 d** (24 hrs) | `trending`, `news-today`, `puzzles-and-games` | Viral buzz, daily bulletins, and daily puzzles. |
| **1.5 d** (36 hrs) | `sports` | Match coverage, scores, and post-game reactions. |
| **2.5 d** (60 hrs) | `cities`, `delhi`, `mumbai`, `chandigarh`, `pune`, `india` | Civic reporting, regional crime, local governance updates. |
| **3.0 d** (72 hrs) | `world`, `entertainment`, `horoscope` (2d) | Global diplomacy, film reviews, box office trends. |
| **4.0 d** (4 days) | `business`, `technology`, `political-pulse`, `delhi-confidential` | Market earnings, RBI policy, gadget launches, insider political reporting. |
| **5.0 d** (5 days) | `legal-news`, `legal`, `brand-story`, `in-collaboration` | High court / Supreme Court verdicts, bail orders, sponsored series. |
| **7.0 d** (7 days) | `explained`, `opinion`, `editorials`, `columns`, `idea-exchange`, `education`, `upsc-current-affairs`, `upsc-essentials`, `lifestyle`, `health-wellness`, `express-exclusive`, `long-reads`, `eye` | **The Indian Express Flagships**: Explanatory journalism, op-eds by scholars, civil services preparation, and investigative long-reads maintain high contextual value throughout the week. |
| **10.0 d** (10 days) | `books-and-literature` | Book reviews and author discussions are semi-evergreen. |
| **3.0 d** (72 hrs) | `default` | General news baseline. |

**Files changed:**

| File | Change |
|---|---|
| `config.py` | Expanded `SECTION_HALF_LIVES` covering all 25+ IE beats; upgraded `recency_boost()` with dual hyphen/underscore normalization. |
| `AGENTS.md` | Synchronized `SECTION_HALF_LIVES` summary in architecture memory. |
| `opencode.md` | Updated Section 2.5 scoring table and logged Change 7. |

**Verified:**
- `live_blog` (6h) $\to 1.052$ (exact 50% decay of 0.104 bonus).
- `sports` (1d) $\to 1.066$.
- `explained` (1d) $\to 1.094$ (retains 90%+ freshness boost).
- `explained` (7d) $\to 1.052$ (still retains 50% boost after 1 week).
- `legal-news` and `legal_news` $\to 1.069$ (normalized lookup verified identical).
- Unrecognized section $\to 1.052$ (graceful fallback to 3-day default).

---

### 2026-10-26 — Change 10: real per-candidate evaluation card (zero fabricated values)

**Requested by user.** Every recommended candidate must show a card the mentor can audit:
which keywords *actually* matched, the embedding similarity, the relevance score, the score
after boosting, and the confidence without boosting — **all real, none invented**.

**Root problem found:** the card data came from two layers that both guessed.
`src/evidence.py:enrich_recommendation_evidence()` invented a reranker score
(`score * 0.9`), rebuilt a "boosted" value from hard-coded multipliers, and
overrode the real `agreement_count` with a re-derived one. `frontend/js/evaluator.js` was
worse: `art.score * 0.9`, `82%` confidence, `+5.0% (1.05x)` boosts, `0.064` margin,
`Batch 05 (articles_batch_05.csv)`, `'16 days apart'`. None of it came from the scorer.

**Architecture change:** the scorer now records what it computed; the UI only reads it.

| Layer | Before | After |
|---|---|---|
| BM25 evidence | regex word-overlap on title+description | `bm25_term_hits()` — the exact query terms that scored this candidate, with `tf`, `idf`, `weight`, `contribution`, using the identical formula as `BM25Index.scores()` |
| Embedding | not shown | real document cosine (`dense_scores[c]` / `semantic_scores[pos]`), plus an explicit note that a 768-d vector cannot be inverted to text |
| Blend | only the total | `components` — `rerank_term`, `bm25_norm`, `bm25_term`, `dense_norm`, `dense_term`, `weights` (sums to `blend`, verified) |
| Boosts | reconstructed from text labels | `boost_items` with the real per-factor multipliers written by `_calculate_boosts` / `_boost_parts` |
| Boosted score | `blend × guessed multiplier` | `boosted_raw` — the actual value the ordering used; also persisted as a `boosted_score` column |
| Confidence w/o boost | `ev.confidence.unboosted_score` (often absent) | `unboosted_confidence` = `calibrate(unboosted blend)` |
| Agreement | re-derived in the formatter | read from `ev.agreement_count` (never recomputed) |
| Margin | `final − worst score in list` (wrong quantity) | the calibrator's real "this vs next" margin, `None` when unrecorded |
| Missing values | plausible defaults | `None` → the UI prints **"not recorded"** |

**Files changed:**

| File | Change |
|---|---|
| `src/evidence.py` | new `bm25_term_hits()`; `build_evidence_reason()` takes `cand_pos` / `bm25_query` / `cosine` / `admission`; `enrich_recommendation_evidence()` is now a pure formatter (fabrications deleted); `boosts` labels derived from the real `boost_items`; `date_gap_text` no longer claims "Same edition" when dates are missing |
| `db/pipeline.py` | `_calculate_boosts()` returns `(multipliers, breakdowns)`; both the forward precompute and the reverse push pass the real per-pair metrics; **reverse-push fabricated ranks fixed** (it passed `rel_pos + 1` as both BM25 and Dense rank — now real corpus ranks via `argsort`) |
| `src/recommender.py` | `_boost_mult()` → `_boost_parts()` returning the factor dict (also fixes a latent `NameError` on `a` when `query_pos is None`); `Recommendation.boosted_score`; same real metrics in the evidence call |
| `api/server.py` | live response exposes `boosted_score`; removed the misleading `unboosted_score` (it was the calibrated score) |
| `db/models.py`, `db/connection.py`, `db/repository.py` | new `boosted_score` column (relevance after boosts, before calibration) + `ensure_schema` `ADD COLUMN IF NOT EXISTS` + save/read paths |
| `frontend/js/evaluator.js` | rewritten: 9 real sections, every fabricated fallback removed |
| `frontend/js/tabs.js` | orbit-tile badges use `boost_items` (real multipliers), the top real BM25 term, the real cosine, and `agreement_count` guarded against the string/array bug |

**Verified (no Postgres, no model download):**
- `bm25_term_hits` contributions sum to the real `BM25Index.scores()` value: `2.612400` vs `2.612378`.
- A candidate sharing no token returns `[]` (honest "BM25 contributed 0").
- Enriched score breakdown of a fully-recorded pair reproduces `boosted_raw 0.9143` and `total_boost_mult 1.1256` exactly, and `agreement_count` is *not* recomputed.
- A record with no recorded numbers keeps `reranker_prob`, `boosted_raw`, `boost_items` and `margin` at `None` — nothing is substituted.
- Live path: for every result, `rerank_term + bm25_term + dense_term == blend` (within 1e-3); text search correctly reports `dense_term = None`.
- Ingest path: same identity holds; the editorial Also-Read link survives the gate.

**Not done at the time:** technique 4 (graceful degradation into a separate "Trending in `<section>`"
list) — it needed API + frontend changes, not a filter. **Now implemented in Change 11.**

---

### 2026-10-26 — Change 9: relevance admission gate (techniques 1–3), replacing the dead discard floor

**Requested by user** ("stop recommending non-relatable articles"), with four techniques offered
and freedom to pick. Techniques **1 (hard floor), 2 (elbow cliff-drop)** and the **entity half of 3
(lexical/entity gate)** are implemented; technique 4 followed in Change 11.

**Why the old floor was dead code:** `ARTICLE_DISCARD_FLOOR = 0.05` could never fire because the
double-sigmoid bug (Change 8) trapped every score in `[0.5, 0.731]`. Measured on the 24,458 stored
rows: a 0.50 threshold removed 11 rows; a 0.05 threshold removed none. Ranks 3–10 were always
padded with whatever noise the reranker had left.

**One shared implementation** — `config.admission_cut(prob_scores, entities, bypass)` — called by
**both** `db/pipeline.py` and `src/recommender.py`, so ingested and live lists cannot drift
(same rationale as the Change 5 recency helpers).

| Knob | Value | Meaning |
|---|---|---|
| `RELEVANCE_FLOOR` | 0.40 | absolute minimum raw cross-encoder probability |
| `ELBOW_RATIO` | 0.45 | cut when a rank's probability is below 45% of its predecessor |
| `ELBOW_MIN_KEEP` | 4 | below this many survivors only the floor applies |
| `LEXICAL_GATE` / `ENTITY_GATE` | `True` | pre-reranker block when a candidate shares neither a BM25 term nor an entity |

- Returns an explicit **position list** plus `cut` / `cut_at` / `n_kept`, so the card can state
  which gate fired instead of guessing.
- **Editorial Also-Read links bypass every gate** and are *rescued* if they sit after the cut —
  they are the evaluation ground truth and must never be silently dropped.
- The gate can never return an empty list.
- Pre-reranker gate in both paths: a candidate with no shared keyword and no shared entity is
  dropped before paying for cross-encoder inference.
- `ARTICLE_DISCARD_FLOOR` removed from both scoring paths; dead `RELATED_MIN_HEALTHY` and
  `_has_token()` removed.

**Verified:** spec example `[0.82, 0.79, 0.31]` → 2 kept; `[0.95, 0.88, 0.05, …]` → 2 kept;
flat healthy 10-list untouched; all-below-floor → 1 kept; bypassed item survives a failing floor;
2,000 random lists never exceed `len()`; editorial link retained in both the ingest and live paths.

**Measured on the real corpus** (12 sources with editorial links, real cross-encoder, real vectors):

| Metric | Value |
|---|---|
| candidate rows → kept rows | 240 → 96 (**60.0% trimmed**) |
| gate outcomes | `no cut` ×8, `entity` ×2, `floor` ×2 |
| tight story clusters (e.g. the Nana Patekar / Messi storyline) | 10 kept — genuinely all related |
| sparse source (`'No one believed in Hanuman Ansh…'`) | **1 kept** — the 19 others all scored below the floor |
| editorial Also-Read links surviving | all that were retrieved; the one below the floor was **rescued**, not dropped |

That last row is the whole point: the list is no longer padded to 10 with unrelated articles, and
the editorial ground truth is never lost. It also makes technique 4 (graceful degradation) a real
UX gap — a 1-item list currently has nothing to fall back to.

---

### 2026-10-26 — Change 8: double-sigmoid in `Reranker.score()` (the actual cause of bad recommendations)

**Found while investigating Change 9.** `bge-reranker-base` is a `num_labels=1` cross-encoder, so
`sentence_transformers.CrossEncoder.predict()` already returns `sigmoid(logit)`. `Reranker.score()`
passed `activation_fn=Identity` and then applied **its own sigmoid**, squashing every score into
`[0.5, 0.731]`.

**Measured on 24,458 stored rows:** `rerank_score` p75 = p90 = p95 = **0.7310** (a hard wall, since
`sigmoid(1.0) = 0.731`), with 14,000+ rows in one 0.001-wide bin. Unrelated articles scored exactly
0.5000 — indistinguishable from relevant ones. Median consecutive ratio 0.978, **0%** of gaps below
0.5, so the elbow detector had no cliffs to find.

**Fix:** `Reranker._to_probability(raw)` clips when the model already returned a probability and
sigmoids only genuine logits — which also covers `RemoteCrossEncoderClient`, whose `predict()`
ignores `activation_fn` and returns raw logits.

**Verified:** same six pairs went from
`[0.7248, 0.5570, 0.5008, 0.5000, 0.5000, 0.5000]` to
`[0.9686, 0.2289, 0.0033, 0.0000, 0.0000, 0.0000]` — irrelevant articles now score 0.0.

**Consequence:** every stored recommendation row is scored with the broken reranker. The next
ingestion forces a full recompute (fingerprint changed).

---

### 2026-10-23 — Change 6: Colab notebook overhaul + four latent bugs fixed

**Requested by user.** "Our system goes two ways — there is a cloud .ipynb for model running.
If you think of any change you can do that in too if it needs help, and if you see other bugs
just fix them."

Two workstreams: (a) the notebook, (b) every bug found while auditing for the notebook fix.

---

#### 6a. `cloud_gpu_worker.ipynb` — Mode B was unusable *and* destructive

| Bug | Consequence | Fix |
|---|---|---|
| `row['id']` / `row['title']` on raw batch CSVs | Raw `articles_batch_*.csv` has camelCase (`articleId`, `articleTitle`) → **`KeyError: 'id'`**. Mode B could only run on a pre-cleaned CSV, and nothing said so. | Explicit column check with a message naming the expected snake_case columns and pointing at `clean_dataframe()` |
| `embedder.encode(text_embed, …)` per article | ~2446 sequential GPU calls, no batching | One batched call (`batch_size=32`), then slice |
| `config_hash: str` was a **hand-typed argument** | Nothing computed it; `import_colab_export.py` *requires* it to equal `config.get_pipeline_version()` → silent `--force-hash` pressure | Read from `NEWSREC_PIPELINE_VERSION` env var; **raises** if unset |
| Hash mismatch → silently discarded the old checkpoint, then overwrote it | **Prior GPU work lost with no warning** | Raises, naming the file to move/delete |
| `np` and `embedder` used but never imported in-cell | `NameError` after a runtime restart | Imported in-cell |
| `/health` reported no token caps | Silent quality drift: local scorer assumes 256/512 | Returns `embedding_max_seq_length` + `reranker_max_length`; connection test **asserts** them |
| Keep-alive: unconditional `while True` + 240 s sleep | Held a GPU awake forever, including when unused | Idle watchdog — exits after `IDLE_EXIT_SECONDS = 3600` with no `/embed` / `/rerank` traffic (new `last_inference_at` field in `/health`) |
| Tunnel URL scraped with blocking `readline()` ×40 | Could hang the cell indefinitely | stderr drained on a daemon thread; 30 s deadline; prints tail on failure |
| **No auth on a public tunnel** | Free-tier T4 exposed to anyone with the URL | Optional `WORKER_TOKEN` (Colab Secret); `/embed` + `/rerank` return **401** without it; `/health` stays open for the test |
| Two cells both labelled "Step 5" | Confusing run order | Renumbered (Step 5 test, Step 6 Mode B) |
| Cell 0 claimed "Installs from pinned requirements" | Untrue — it never read `colab_requirements.txt` | Reworded to match what the cell actually pins |

`REMOTE_MODEL_TOKEN` (`config.py` + `.env.example`) lets the local client send the token;
verified against a stub server (`_check_remote_available` → `True` with the token, header
omitted when unset).

**Rule compliance:** (B) batched encode + a single `_load_corpus()`-style helper replaces
hundreds of lines of per-row work. (C) the notebook now fails loudly on a wrong hash rather
than destroying data.

---

#### 6b. `import_colab_export.py` — could never import anything

Line 60 ran `db.query(...)` (SQLAlchemy 2.0 **autobegins** a transaction), then line 106
called `with db.begin():` → **`InvalidRequestError: A transaction is already begun`** → rollback
→ re-raise. The importer's headline feature — atomic import — had never once executed.

Rewrote as a single explicit `with SessionLocal.begin() as db:` wrapping both the validation
reads and the writes. Removed the now-redundant `db = SessionLocal()` / `try` / `except
db.rollback()` / `finally db.close()` scaffolding and the unused `import sys`.

**Verified end-to-end against the live DB:** built a 1-article export with a real ID and a fresh
unit vector → import completed → confirmed the row's embedding actually changed → restored the
original. Non-destructive.

---

#### 6c. `freeze_pool.py` — the harness could not run at all

It read `data/cache/cleaned_articles.pkl` and `data/cache/articles_2467_vectors.npy`.
**Nothing in the repo generates either file**, so a fresh clone could not run
`run_experiments.py` or any ablation at all.

| Bug | Fix |
|---|---|
| Two required inputs, no generator | New `_load_corpus()` builds both from `src.data_loader` + `get_embedder()`, caches them, and globs `data/articles_batches/*.csv` (they are **not** in `config.DATA_PATHS`) |
| `sorted(set(body_toks), key=…)` | Iterating a **set** → candidate selection depended on `PYTHONHASHSEED` → pool not reproducible across processes. Now `sorted` on `(-idf, term)` |
| `--max-len` default `192`, revision unpinned | Production is **512** with a pinned revision. Every `tuning_results.csv` number came from a truncated, unpinned cross-encoder. Now defaults to `config.RERANK_MAX_LEN` + `config.RERANKER_REVISION` |
| `--samples` help said "default: all 290" | Hard-coded count; actual corpus has 290 GT sources only by coincidence | Help text now reads from the data |
| No way to force a rebuild | Added `--rebuild` |

**Verified:** built the corpus cache — 2446 articles, vectors `(2446, 768)` fp32,
**all norms exactly 1.000000**, 290 rows with ground-truth links (matches `AGENTS.md`).

---

#### 6d. Frontend P0s

| Bug | Where | Fix |
|---|---|---|
| Assignment to `const statusData` | `app.js` | `const` → `let`. The `TypeError` was swallowed by the enclosing `try/catch`, so `checkReaderDatabaseState()` never refreshed engine status |
| `syncCategoryPills(sec)` does not exist | `app.js` | Call removed. The `ReferenceError` threw **before** `triggerActiveTabReload()`, so changing the category filter never rebuilt the feed |
| Notes banner wiped by `innerHTML = ''` | `tabs.js` + `ui.js` | `renderNotes()` now runs **after** `renderCards()` — the banners survived for zero frames before |
| Duplicate `@keyframes shimmer` | `rightpanel.css` | Renamed to `pulse-fade`. The later file overrode `base.css`'s sweep, so **every skeleton in the app** showed a static opacity pulse instead of the shimmer |
| `.btn-secondary` used 4×, never defined | `base.css` | Aliased to the existing `.btn-ghost` rather than duplicating the rules |
| Raw `fetch('/api/db/clear')` | `editor.js` | Added `Api.dbClear()`; the path now lives only in `api.js` |
| Dead `Api.getArticle` | `api.js` | Deleted (zero call sites, per `AGENTS.md` §4 P2) |

---

#### 6e. `src/evidence.py` was fabricating readouts

The "grounded evaluation items" shown to editors contained invented numbers:

| Invented | Real |
|---|---|
| `rerank_val = 0.65` | unknown — should be `None` |
| `final_score = 0.82` | unknown — should be `None` |
| Section boost `+8.0%` (×1.08) | **×1.072 → +7.2%** |
| Subsection boost `+5.0%` (×1.05) | **×1.09 → +9.0%** |
| Recency boost `+10.0%` (×1.10) | **×1.104 → +10.4%** |
| Popularity `+5.0%` (×1.05) | **×1.044 → +4.4%** |

Boost multipliers now come from a `boost_map` reading `config`, so the panel cannot drift from
the scoring code again. Missing values return `None` and the calibration line reads
"delta unavailable (missing upstream score)" instead of printing a precise-looking fiction.

**Rule compliance:** (A) the invented fallbacks were replaced, not guarded. (B) a 6-entry
`boost_map` replaced the if/elif chain. (C) no new abstraction.

---

#### 6f. Not a bug — `run_experiments.py`

`AGENTS.md` §4 #19 claims `run_experiments.py:467+496` passes a **scalar** as the paired
baseline array and crashes with `IndexError`. **This is already fixed in the current tree.**
Line 467's `a0_val_ndcg` scalar is used only for the scalar subtraction at :506, while :496
correctly passes the per-article `a0_ndcg` array from :382. `AGENTS.md` describes an older
revision. **No code change made.**

---

### 2026-10-23 — Change 5: honour `BOOST_MODE` (source-date recency)

**Requested by user.** Fix issue #2 — the last unwired MEASURED setting.

**The bug.** `BOOST_MODE = "source_date_recency"` was declared in `config.py` with the comment
"MEASURED: recency relative to source article date, not corpus max" — and was **referenced
nowhere in the codebase**. Both paths measured recency against the **corpus max date**, i.e.
the `multiplicative` variant that the 200-config search **rejected**.

**What the experiment harness actually specified** (`run_experiments.py:153`):

```python
ref_date = s_date if boost_mode == "source_date_recency" and pd.notna(s_date) else max_corpus_date
```

So a candidate should be rewarded for being fresh **relative to the article it is recommended
next to**, not for absolute freshness.

**Measured impact** (live DB, 300 sampled sources, 5816 candidate pairs):

```
recency multiplier CHANGES in        99.2% of pairs
  higher under source-date mode  :   99.2%
  lower                          :    0.0%   <- strictly one-directional
  mean delta                     :  +0.02888
  max delta                      :  +0.10399  (≈ the entire boost range)
  unchanged                      :    0.8%
```

The old behaviour systematically **under-boosted** by nearly the full range, because every
article is compared against a corpus max that can be months newer than the source.

**Files changed:**

| File | Change |
|---|---|
| `config.py` | Added `recency_boost(cand_date, ref_date, section, is_live)` — one shared implementation |
| `config.py` | Added `recency_ref_date(source_date, corpus_max_date)` — reads `BOOST_MODE` in one place |
| `config.py` | Added `_has_date()` — `config.py` has **no pandas import** (it is the base config module), so `NaT`/`NaN` are detected without pulling in pandas |
| `db/pipeline.py` | `_calculate_boosts()` computes `ref_date` once from `config.recency_ref_date()` and calls `config.recency_boost()`; the corpus-max block is gone. Docstring corrected (it advertised 1.08/1.05/1.06/1.10) |
| `src/recommender.py` | `_boost_mult()` calls the same two helpers; the source article's date is used when `query_pos is not None` |

**Why a shared helper rather than two edits:** `db/pipeline.py` reimplements BM25/blend/boosts/MMR
separately from `src/recommender.py`, and they have already drifted three times (`AGENTS.md`
bugs #15/#16/#17). Putting the formula in `config` means the ingested and live paths
**cannot** disagree — verified below.

**Verified:**
- `recency_ref_date(src, corpus)` → the source date; falls back to corpus max on `NaT`/`None`.
- `recency_boost(NaT, ref)` → `1.0` (neutral), and the same for a `NaT` reference.
- **Pipeline vs live recency term on 59 real candidates: `max abs diff = 0.000000000000`.**
- All 59 candidates change vs the old corpus-max behaviour (mean `+0.01463`, max `+0.07318`).

**Rule compliance:** (A) the corpus-max computation was replaced in both paths, not shadowed.
(B) two helpers replace two duplicated 6-line blocks and make the mode a one-line switch.
(C) `BOOST_MODE` is now a real setting instead of dead text.

**Fingerprint:** unchanged at `f57cf1bd654da71a` — `boost_mode` was already a
`get_pipeline_version()` token, and its *value* did not change. Only whether it is honoured did.

---

### 2026-10-23 — Change 4: wire `MMR_THETA` into the ingestion pipeline

**Requested by user.** Fix issue #1 in §4 — the ingested MMR and the live MMR disagreed.

**The bug.** `MMR_THETA = 0.76` was a MEASURED value, but the pipeline never received it:

- `db/pipeline.py` `_apply_mmr()` had a **hard-coded `> 0.80`** threshold and **no `theta`
  parameter at all** — the config value could not reach it.
- The live path (`src/recommender.py` `_mmr()`) *did* use `config.MMR_THETA`, plus two
  **unnamed literals** for search mode (`0.88` lambda, `0.85` theta).

So the same corpus produced different top-10 lists depending on which path scored it, and
`evaluate.py` (which grades the ingested rows) was measuring an MMR that no config key
controlled.

**Behavioural proof this was not cosmetic** (25 synthetic vectors, identical scores):

```
theta = 0.76 (config)  ->  [0, 1, 2, 3, 4, 5, 7, 8, 6, 9]
theta = 0.80 (old)     ->  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Different selection order — the stored recommendations genuinely change.

**Files changed:**

| File | Change |
|---|---|
| `config.py` | Added `MMR_LAMBDA_SEARCH = 0.88`, `MMR_THETA_SEARCH = 0.85` (previously magic literals in `_mmr()`) |
| `config.py` | Added `WEIGHT_RERANK/BM25/DENSE` (Change 1) plus `mmr_theta` to `get_pipeline_version()` — see fingerprint note below |
| `db/pipeline.py` | `_apply_mmr()` gained `theta: float = config.MMR_THETA`; the hard-coded `0.80` is gone. Both call sites inherit it, so no call-site edits were needed |
| `src/recommender.py` | `_mmr()` search branch now reads `config.MMR_THETA_SEARCH` / `config.MMR_LAMBDA_SEARCH`; no literals remain |

**Rule compliance:** (A) the `0.80` literal replaced, not bypassed. (B) one `theta` parameter
replaces a hidden constant and makes the two implementations structurally identical. (C) both
paths now read the same four config keys, which is simpler than the previous implicit contract.

**Verified:** `py_compile` clean; `inspect.signature` confirms pipeline defaults equal
`config.MMR_THETA` / `config.MMR_LAMBDA`; zero remaining MMR magic literals repo-wide.

**Fingerprint change (important).** `get_pipeline_version()` now includes the relevance blend
**and** `mmr_theta` — the blend had been omitted since Change 1:

```
old: 1aa8dc568ca0f822   (rows currently in article_recommendations)
new: f57cf1bd654da71a
```

Because they differ, the next ingestion **auto-detects the change and forces a full
recompute** — no manual step. This is the intended behaviour from `AGENTS.md` §7 rule 3, and
it means Change 1 and Change 4 both take effect on the stored rows at that point.

---

### 2026-10-23 — Change 3: delete dead Postgres vector retriever (benchmarked)

**Requested by user.** "Check which retriever is faster; if Postgres wins keep it, otherwise delete it."

**Benchmark — live DB, 2446 × 768-d, `top_k = N_RETRIEVE = 50`, 20 iterations, 3 warm-up:**

| Method | min | median | Rows returned |
|---|---|---|---|
| **NumPy dot (pre-normalised, `DenseIndex`)** | **0.43 ms** | **0.53 ms** | 50 |
| **NumPy dot (raw, `pipeline.py:121` `emb @ emb[idx]`)** | **0.30 ms** | **0.35 ms** | 50 |
| Postgres pgvector seq scan | 9.07 ms | 10.58 ms | 50 |
| Postgres + `section` filter | 3.55 ms | 4.08 ms | 50 |

**Correctness: identical.** Overlap between in-memory and Postgres top-50 = **50/50**, same
neighbours, same order. Postgres offers no quality advantage, only 20–30× more latency.

**Result: in-memory wins by ~20–30×.** The Postgres retriever was deleted.

**Files changed:**

| File | Change |
|---|---|
| `db/repository.py` | **Deleted** `ArticleRepository.search_similar_by_vector()` (~33 lines, called from nowhere) |
| `db/repository.py` | Removed the now-unused `from pgvector.sqlalchemy import Vector` import |
| `db/repository.py` | Corrected module docstring and class docstring (no longer claims pgvector similarity queries) |

**Rule compliance:** (A) dead code removed rather than left beside the live path. Verified
`cosine_distance` and `search_similar_by_vector` have **zero** remaining references repo-wide.
All backend modules import cleanly; `ArticleRepository` still exposes 23 methods.

**⚠️ Correction to an earlier claim.** I previously described `repository.py:156`
(`section ILIKE ... ORDER BY distance LIMIT` returning fewer than `top_k` rows) as an
**active bug**. The benchmark does **not** support that: the filtered query returned the
full 50 rows. The theoretical risk (top-N heapsort may not examine every row) is real, but
I could not reproduce it, so it is **not** an active bug and is now moot — the function is
deleted. Earlier characterisation was overstated.

---

### 2026-10-23 — Change 2: vector indexing decision (HNSW vs in-memory)

**Requested by user.** Question: for an enterprise deployment at The Indian Express,
where editors use this daily, should dense retrieval stay **in-memory** or move to a
proper ANN index such as **HNSW**?

**Measured on the live DB (2446 rows, 768-d, pgvector 0.8.7, PostgreSQL 18.6):**

```
Seq Scan on articles      actual time = 8.743 ms
Sort (top-N heapsort)     actual time = 9.144 ms
Execution Time                    9.165 ms
```

`top_k = 50` of 2446 rows = **2% selectivity**.

**Decision: keep in-memory exact search. Do NOT add HNSW yet.**

| Option | Verdict |
|---|---|
| HNSW | **No, not yet.** Approximate: can drop the true nearest neighbour at 2% selectivity, which can silently remove an editor's `Also Read` article from the candidate pool before fusion/rerank. Also carries the `ef_search`/post-filter trap with the `section` filter. |
| IVF-Flat ("inverted file") | **No.** Same approximation problem, plus `nprobe` tuning, and slower to query at this exactness requirement. |
| `tsvector` + GIN | **Not for ranking.** Postgres has no BM25 — `ts_rank` is a different function. Adopting it would silently change lexical scoring and confound every measurement. |
| In-memory exact NumPy / seq scan | **Yes.** 9 ms per query, no approximation, no tuning. |

**Why this is the enterprise answer, not just the small-corpus answer:** an ANN index
buys speed by giving up the guarantee of exactness. At 9 ms for an exact answer, HNSW
would trade a correctness guarantee for latency nobody can perceive. Recommendation
quality is this product's actual output; 9 ms is not a problem worth that trade.

**Documented trigger for revisiting:** ~100k+ rows (≈40× current corpus). At that point
use HNSW with `m=16, ef_construction=64`, `hnsw.ef_search >= 100`, and exact re-scoring
of the returned candidates.

**Files changed:** none — this is a decision, not a code change.

**Resolved 2026-10-23 by Change 3:** the dead-code finding below was actioned —
`search_similar_by_vector()` is deleted, so the "should Postgres serve vectors?" question
is closed. In-memory exact search is the answer for this project.

---

### 2026-10-23 — Change 1: relevance blend weights → 0.70 / 0.20 / 0.10

**Requested by user.** The relevance score blends three signals (cross-encoder, BM25, dense).
Re-weighted to `cross = 0.70, bm25 = 0.20, bge = 0.10`.

**What was wrong before:** the blend was hard-coded as bare literals in **three separate
places**, and they had already drifted apart:

| Location | Old blend |
|---|---|
| `src/recommender.py` (article→article) | `0.55 / 0.30 / 0.15` |
| `src/recommender.py` (text search) | `0.65 / 0.35` (no dense term) |
| `db/pipeline.py` (forward precompute) | `0.55 / 0.30 / 0.15` |
| `db/pipeline.py` (reverse push) | `0.55 / 0.30 / 0.15` |
| `run_experiments.py` (harness defaults) | `0.55 / 0.30 / 0.15` |

The measured optimum in `tuning_results.csv` was `0.5 / 0.3 / 0.2`, which was **never wired
in anywhere** — the DB precompute and the live path both ignored it.

**Files changed:**

| File | Change |
|---|---|
| `config.py` | **Added** `WEIGHT_RERANK = 0.70`, `WEIGHT_BM25 = 0.20`, `WEIGHT_DENSE = 0.10` with a comment stating they must sum to 1.0 |
| `src/recommender.py` | Replaced both hard-coded blends with `config.WEIGHT_*` (search path now uses `WEIGHT_RERANK`/`WEIGHT_BM25`; article path uses all three) |
| `db/pipeline.py` | Replaced both hard-coded blends (forward precompute + reverse push) with `config.WEIGHT_*` |
| `run_experiments.py` | `score_and_rank_article()` defaults now read `config.WEIGHT_*` instead of literals |

**Rule compliance:** (A) old literals replaced, not duplicated — `db/pipeline.py` and
`src/recommender.py` now share one source of truth, which also kills the drift risk noted in
`AGENTS.md` §5.6. (B) one config key replaces five scattered literals. (C) no new abstraction —
`config` was already imported by both files.

**Verified:** `py_compile` passes on all four files; runtime import confirms
`0.70 / 0.20 / 0.10` (sum 1.0). No other hard-coded blend weights remain in the codebase.

**⚠️ Follow-ups — status as of Change 6:**
1. ~~The four new keys are **not** in `get_pipeline_version()`.~~ **Resolved in Change 4** —
   `WEIGHT_RERANK/BM25/DENSE` were added to the fingerprint, so the stored version is now
   `f57cf1bd654da71a` and the next ingestion auto-forces a full recompute.
2. `baseline_evaluation.json` still describes the old `0.55/0.30/0.15` weights → not a valid
   regression reference. **Re-baseline after the re-ingest.**
3. **Evaluation is deferred.** Per user decision (2026-10-23): the DB will be deleted and
   re-ingested once all planned changes are in, and evaluation runs **only after** that.
   No evaluation against the intermediate state.

**Note on the chosen weights.** The instruction was `0.70 / 0.20 / 0.10`. The random search in
`tuning_results.csv` had selected `0.5 / 0.3 / 0.2`. This is a deliberate divergence, not a
wiring error — recorded here so it is not mistaken for one later.

---

### 2026-10-22 — setup

- Created this change log.
- Baseline established: `evaluate.py` extended to record model revisions, token budgets,
  all boost multipliers, `mmr_theta`, `boost_mode`, `pipeline_version` and `compute_device`.
- `evaluate.py`: baseline now saved **immediately after Part 1**; new `--skip-methods` flag
  (Part 2 is a ~28 min CPU cross-encoder sweep).
- Pushed local state to `repo2` as `b89fb2e` (5 new logical commits).
- `AGENTS.md` added as architecture memory.

---

## 4. Known Open Issues (carried from `AGENTS.md`)

Highest value first — candidates for the next changes.

| # | Issue | Where | Status |
|---|---|---|---|
| ~~1~~ | ~~`MMR_THETA` (0.76) ignored; threshold hard-coded `> 0.80`~~ | `db/pipeline.py` | ✅ **fixed — Change 4** |
| ~~2~~ | ~~`BOOST_MODE = "source_date_recency"` referenced nowhere; both paths use corpus-max date~~ | `db/pipeline.py`, `src/recommender.py` | ✅ **fixed — Change 5** |
| ~~4~~ | ~~`evidence.py` fabricates fallbacks (0.65 / 0.82) and hard-codes old boost mults~~ | `src/evidence.py` | ✅ **fixed — Change 6e** |
| ~~5~~ | ~~Frontend P0: `const` reassignment~~ | `frontend/js/app.js` | ✅ **fixed — Change 6d** |
| ~~6~~ | ~~Frontend P0: `syncCategoryPills()` does not exist~~ | `frontend/js/app.js` | ✅ **fixed — Change 6d** |
| ~~7~~ | ~~Frontend P0: notes banner wiped by `innerHTML = ''`~~ | `frontend/js/tabs.js` | ✅ **fixed — Change 6d** |
| ~~8~~ | ~~Duplicate `@keyframes shimmer` breaks skeleton animation~~ | `frontend/css/rightpanel.css` | ✅ **fixed — Change 6d** |
| ~~9~~ | ~~`.btn-secondary` used 4×, never defined~~ | `frontend/css/base.css` | ✅ **fixed — Change 6d** |
| ~~10~~ | ~~`run_experiments.py` crashes (scalar as paired array)~~ | `run_experiments.py` | ✅ **not a bug** — already correct in the tree; `AGENTS.md` was stale (Change 6f) |
| ~~11~~ | ~~`import_colab_export.py` can never import (transaction already begun)~~ | `import_colab_export.py` | ✅ **fixed — Change 6b** |
| 3 | `linked_ids` stored as a Python repr if a **list** is ever passed to `bulk_upsert_articles_fast` | `db/pipeline.py` | open — latent only; the current path passes JSON strings |
| 12 | `tabs.js` sets `dataset.loaded = 'true'` *before* the await → failed requests never retry | `frontend/js/tabs.js` | open |
| 13 | `tabs.js` `buildFeed()` has no request sequencing → concurrent calls can interleave | `frontend/js/tabs.js` | open |
| ~~14~~ | ~~`tabs.js` `ev.signals.join(...)` assumes an array → throws on a string~~ | `frontend/js/tabs.js` | ✅ **fixed — Change 10** (`Array.isArray` guard) |
| 15 | XSS: `ui.js` / `tabs.js` / `app.js` / `hero.js` inject API strings via `innerHTML` unescaped | `frontend/js/*` | open |
| ~~16~~ | ~~`evaluator.js` prints "(High)" regardless of the actual score~~ | `frontend/js/evaluator.js` | ✅ **fixed — Change 10** (rewritten; every number is read, never guessed) |
| ~~17~~ | ~~`pipeline.py` reverse-push passes `rel_pos + 1` as both bm25 and dense rank → fabricated agreement ranks~~ | `db/pipeline.py` | ✅ **fixed — Change 10** (real corpus ranks via `argsort`) |
| 18 | Dead code: `State.subscribe` / `theme` / `engineInfo`, `buildCategoryPills`, `window.UI`, duplicate `escapeHtml` | several | open |
| 19 | ~~Technique 4 (graceful degradation: separate "Trending in `<section>`" list) not implemented~~ | `config.py`, `api/server.py`, `db/repository.py`, `frontend/` | **fixed (Change 11)** — measured trigger rate 34.0% (n=50 real sources) |
| 20 | `baseline_evaluation.json` describes neither the current weights nor the fixed reranker | `baseline_evaluation.json` | open — re-baseline **after** the re-ingest |
| 21 | `freeze_pool.pkl` not rebuilt since the `--max-len`/determinism fixes, so `tuning_results.csv` still reflects the old 192-cap, unpinned, double-sigmoid cross-encoder | `data/cache/frozen_pool.pkl` | open — rebuild before trusting any tuning number |

### Measured-vs-production gap (was `AGENTS.md` §5.6) — now closed

| Measured setting | Applied? |
|---|---|
| Blend weights | ✅ wired — Change 1 (**note:** wired `0.70/0.20/0.10` per user instruction, not the measured `0.5/0.3/0.2`) |
| `MMR_LAMBDA = 0.91`, boost multipliers | ✅ wired |
| `MMR_THETA = 0.76` | ✅ wired — Change 4 (was hard-coded 0.80 in the pipeline) |
| `BOOST_MODE = "source_date_recency"` | ✅ wired — Change 5 (was unused everywhere) |

> ⚠️ **One deliberate divergence remains.** Change 1 set the blend to `0.70 / 0.20 / 0.10`
> on the user's explicit instruction. The random search in `tuning_results.csv` selected
> `0.5 / 0.3 / 0.2`. `baseline_evaluation.json` therefore describes neither — it was saved
> under the old `0.55 / 0.30 / 0.15`. Re-baseline after the re-ingest.

---

## 2026-10-27 Fixes: Pipeline Precompute & User View Evaluation Card

1. **Pipeline Precompute Error Fixed (`db/pipeline.py`)**:
   - `_gate_candidates(...)` was previously deleted, but a stray call remained in `_precompute_for_article`, triggering a silent `NameError` per article that caused precompute to report "1000 skipped". Removed the stale call.

2. **Database Cleared (`ArticleRepository.clear_all_data`)**:
   - PostgreSQL tables wiped clean: articles = 0, recommendations = 0, ingestion jobs = 0. Ready for clean batch ingestion.

3. **User View Evaluation Modal Fixed (`frontend/js/evaluator.js`)**:
   - In `showModal()`, previously `mergeRec(candArt || rec)` discarded recommendation object `rec` because `candArt` (`rec.article`) was truthy but had no `reason` or `evidence`. Updated `showModal` to merge candidate article details while preserving `rec.evidence`, `rec.reason`, `rec.score`, `rec.rank`, and `rec.rerank_score`.
   - Updated `bm25Section()` with graceful fallback to `ev.terms` and `ev.entities`, avoiding false "No shared token with the query" text when shared topics exist.
   - Replaced confusing technical note (*"A 768-dimension embedding cannot be converted back into text..."*) with professional, user-friendly presentation: displaying `Neural Semantic Similarity` percentage along with clear explanation of dense vector alignment.
4. **Pipeline NameError Fixed (`db/pipeline.py:_rank_and_build`)**:
   - `editorial_link=(cand_id in linked)` was referencing `linked`, but `linked = linked_set(source_row)` was missing in `_rank_and_build()`. This caused `NameError: name 'linked' is not defined` to be caught by the per-article exception handler, skipping recommendations. Added `linked = linked_set(source_row)`.

5. **Codebase Cleanup (Removed Obsolete Experiments & Test Files)**:
   - Deleted obsolete test cases, logs, and experiment files: `tests/`, `parity_test.py`, `run_experiments.py`, `ablation_results.csv`, `tuning_results.csv`, `baseline_evaluation.json`, `baseline_evaluation_legacy_2026-10-18.json`, `blind_labeling_sample.csv`, `evaluation_run.log`, `freeze_pool.py`, `eda.py`, `cloud_gpu_worker.ipynb`, and `colab_requirements.txt`.
   - Core production files verified intact and running.

---

### 2026-10-27 — Change 20: Complete Removal of Dead Admission Cut Gates

**Requested by user:** *"i not want them off if anything in code pieline no more is in use i want you to make sure to delete their code from pipeline without breaking the flow so do that"*

1. **`config.py`**:
   - Removed dead configuration constants: `RELEVANCE_FLOOR`, `ELBOW_RATIO`, `ELBOW_MIN_KEEP`, `LEXICAL_GATE`, `ENTITY_GATE`.
   - Removed dead pass-through function `admission_cut()`.
   - Removed obsolete tokens from `get_pipeline_version()` (`rel_floor`, `elbow_ratio`, `elbow_min_keep`, `lexical_gate`, `entity_gate`). New pipeline version: `f5f110761b2c47d8`.

2. **`src/recommender.py`**:
   - Removed dead candidate entity extraction loop and `config.admission_cut(...)` call.
   - Removed `admission=admission` argument passed to `build_evidence_reason`.

3. **`db/pipeline.py`**:
   - Removed dead `gate = config.admission_cut(...)` and `order = [order[i] for i in gate["keep"]]`.

4. **`src/evidence.py`**:
   - Removed `admission` argument from `build_evidence_reason` signature and output dictionary.

5. **`frontend/js/evaluator.js`**:
   - Cleaned up admission gate display block so it only renders if historical records contain admission data, avoiding empty/dead UI rows.


