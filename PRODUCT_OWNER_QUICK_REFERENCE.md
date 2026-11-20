# Indian Express AI News Recommendation Engine (NewsRec v2.0)
## Product Owner's Executive Quick Reference & Technical Cheat Sheet

> **Companion Document:** For the complete 15-part master specification, see [PRODUCT_OWNER_MASTER_HANDBOOK.md](PRODUCT_OWNER_MASTER_HANDBOOK.md).  
> **Active Target Branch:** `migration/pipeline-c-bge-small-384d`  
> **Production Pipeline Fingerprint:** `9bbf5a9ae461a14e`

---

## 1. System Vital Signs & Architecture at a Glance

```
+------------------------------------+---------------------------------------------------------------+
| System Parameter                   | Production Specification                                      |
+------------------------------------+---------------------------------------------------------------+
| Active Dense Embedding Model       | BAAI/bge-small-en-v1.5 (384 Dimensions, L2-Normalized)       |
| Pinned Embedding Git Revision      | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a                      |
| Neural Cross-Encoder Reranker      | BAAI/bge-reranker-base (num_labels=1, 512-Token Hard Ceiling) |
| Pinned Reranker Git Revision       | 2cfc18c993282b04f7a55ad8c1d0442387e4ea6a                      |
| Primary Database Engine            | PostgreSQL 16+ with pgvector 0.8.7 extension                   |
| Active Production Tables           | articles (vector 384), article_recommendations, ingestion_jobs |
| Historical Archive Tables          | articles_768d (vector 768), article_recommendations_768d       |
| Live Fast-Path Read Latency        | 2 - 5 milliseconds (Precomputed PostgreSQL indexed scan)      |
| In-Memory Live Search Latency      | 25 - 65 milliseconds (NumPy dot-product + PyTorch CE)         |
| Active Automated Test Suite        | 56/56 Passing Tests (15 Strategy E, 41 Pipeline C Migration)  |
+------------------------------------+---------------------------------------------------------------+
```

---

## 2. End-to-End Recommendation Pipeline Flow

```
[Raw CMS Article CSV]
        │
        ▼ 1. Preprocessing (data_loader.py): NFKC, Unicode quotes, 'Also Read' stripping, 254-token budget
[Clean Budgeted Document]
        │
        ├──► 2A. Sparse Lexical Index (BM25Index): Okapi BM25 + Indian Entity Fuzzy Matcher
        └──► 2B. Dense Semantic Index (DenseIndex): BGE-Small 384D Unit Vectors + In-Memory NumPy Scan
        │
        ▼ 3. Quota Candidate Selector (select_quota_candidates()):
             - 12 Reciprocal Rank Fusion (RRF k=60)
             - 5 Dense-Only Semantic Hits (Dense ranks 1..5 without BM25 overlap)
             - 3 BM25-Only Named Entity Hits (BM25 ranks 1..5 without Dense overlap)
        │
        ▼ 4. Pre-Rerank Gate: Lexical + Entity Veto (Must share >=1 BM25 term OR Capitalized Entity)
        │
        ▼ 5. Cross-Encoder Neural Reranking: BGE-Reranker-Base (Probability sigmoiding via _to_probability)
        │
        ▼ 6. Fixed-Scale Score Blending:
             0.70 * Rerank + 0.20 * BM25_norm (Self-normalized) + 0.10 * Dense_norm (Cosine Quantiles)
        │
        ▼ 7. Strategy E Editorial Boosts:
             Editorial (1.40x) * Section (1.02x/1.10x) * Recency (Tiered Half-Lives) * Popularity (1.044x)
             + Additive Signal Agreement Delta [-0.03, +0.03]
        │
        ▼ 8. Diversity & Redundancy Filter (MMR):
             lambda=0.91, theta=0.76 (Redundancy penalty applied ONLY when Cosine Sim > 0.76)
        │
        ▼ 9. Quality Filtering & Low-Confidence Detection:
             Candidate Quotas (12 RRF + 5 Dense + 3 BM25) + MIN_RAW_RERANK_THRESHOLD (0.12)
        │
        ▼ 10. Platt Scaling Calibration & Zero-LLM Evidence Assembly:
              P = sigmoid(4.85 * Score - 2.60) -> Confidence Band (HIGH/MED/LOW) + JSON Evidence Card
        │
        ▼ [OUTPUT: Top-10 Recommendations] (Plus Technique 4 Trending Fallback if recs < 3)
```

---

## 3. Master Tunables & Hyperparameters Cheat Sheet

```
+------------------------------------+--------------------------+-------------------------------------------------------------+
| Tunable Constant                   | Configured Value         | Architectural Responsibility                                |
+------------------------------------+--------------------------+-------------------------------------------------------------+
| EMBED_TOKEN_BUDGET                 | 254 tokens               | Maximum subwords for dense embedding generation             |
| RERANK_PER_ARTICLE_TOKENS          | 254 tokens               | Budget per side for Cross-Encoder (512 total pair limit)    |
| N_RETRIEVE                         | 50 candidates            | Depth of initial BM25 and Dense candidate retrieval         |
| N_CANDIDATES                       | 20 candidates            | Total candidate pool fed to Cross-Encoder reranker          |
| RRF_K                              | 60                       | Reciprocal Rank Fusion smoothing constant                   |
| TOP_K                              | 10 articles              | Final number of recommendations delivered to reader         |
| WEIGHT_RERANK                      | 0.70                     | Weight of Cross-Encoder neural probability in blend         |
| WEIGHT_BM25                        | 0.20                     | Weight of self-normalized BM25 keyword score in blend       |
| WEIGHT_DENSE                       | 0.10                     | Weight of empirical cosine quantile percentile in blend     |
| EDITORIAL_BOOST                    | 1.40 (+40%)              | Multiplier for ground-truth "Also Read" editorial links     |
| SECTION_BOOST_FOCUSED              | 1.100 (+10%)             | Multiplier for focused beats (Politics, Business, Tech)     |
| SECTION_BOOST_UMBRELLA             | 1.020 (+2%)              | Multiplier for broad umbrella sections (India, Cities)      |
| SUBSECTION_BOOST_FOCUSED           | 1.050 (+5%)              | Multiplier for matching focused subsections                 |
| POPULARITY_BOOST_MULT              | 1.044 (Max +4.4%)        | Logarithmic page-view multiplier                            |
| RECENCY_MAX_BOOST_MULT             | 1.104 (Max +10.4%)       | Multiplier for fresh articles relative to source date       |
| MMR_LAMBDA                         | 0.91                     | Relevance vs diversity weighting in MMR                     |
| MMR_THETA                          | 0.76                     | Redundancy threshold: penalties apply ONLY above 0.76 sim   |
| MIN_RAW_RERANK_THRESHOLD           | 0.12                     | Raw CE prob threshold for low-confidence flag detection     |
| CANDIDATE_QUOTAS                   | 12 RRF / 5 Dense / 3 BM25| Guaranteed retrieval quota allocation across branches       |
| FALLBACK_MIN_RELATED               | 3 articles               | If fewer than 3 survive, Technique 4 fallback is appended   |
+------------------------------------+--------------------------+-------------------------------------------------------------+
```

---

## 4. Section Recency Half-Lives ($T_{1/2}$)

Decay formula: $	ext{Boost} = 1.0 + 0.104 \cdot \exp\left(-rac{\ln(2) \cdot \Delta t}{T_{1/2}}ight)$ relative to **source article date**:

- **Live Blogs:** `0.25 days` (6 hours)
- **Trending News:** `1.0 day`
- **Sports:** `1.5 days`
- **Cities / India:** `2.5 days`
- **World / Entertainment:** `3.0 days`
- **Business / Technology:** `4.0 days`
- **Legal / Courts:** `5.0 days`
- **Explained / Opinion / UPSC:** `7.0 days` (Long-tail educational value)
- **Books & Literary:** `10.0 days`
- **Default Fallback:** `3.0 days`

---

## 5. Dual-Track Serving & Database Table Matrix

```
+------------------------------------+------------------------------------+------------------------------------+
| Feature Dimension                  | Active 384D Pipeline (Production)  | Historical 768D Pipeline (Archive) |
+------------------------------------+------------------------------------+------------------------------------+
| Embedding Model & Dimension        | BGE-Small-en-v1.5 (384D)           | BGE-Base-en-v1.5 (768D)            |
| Primary Articles Table             | articles                           | articles_768d                      |
| Precomputed Recs Table             | article_recommendations            | article_recommendations_768d       |
| Row Count in Clean State           | 0 rows (Awaiting fresh ingestion)  | 4,177 articles (41,770 recs)       |
| API Parameter Switch               | ?pipeline=384d (Default)           | ?pipeline=768d                     |
| Serving Speed                      | <5ms (Fast-Path) / <50ms (Live)    | <5ms (Fast-Path)                   |
| Foreign Key Integrity              | Strict isolation (Zero crossover)  | Strict isolation (Zero crossover)  |
+------------------------------------+------------------------------------+------------------------------------+
```

---

## 6. Top 20 Must-Know Product Owner Interview Q&As

1. **What is NewsRec v2.0?** An editorial-first, 4-stage hybrid AI recommendation engine built for The Indian Express, combining BM25 keyword search, 384D dense neural embeddings, a Cross-Encoder reranker, and Strategy E business rules.
2. **Why not collaborative filtering?** Breaking news has zero clicks at publication (cold start). Content-grounded retrieval enables instant recommendations upon publication.
3. **Why BGE-Small (384D) over BGE-Base (768D)?** 50% less RAM, 50% less database storage, 2x faster vector dot products, and 99.6% retention of ground-truth editorial recall.
4. **Why pin model revisions?** Locking Hugging Face commit SHAs prevents upstream model weight changes from silently corrupting vector alignments in the database.
5. **What is the dual-track serving path?** Precomputed articles serve from PostgreSQL in under 5ms; ad-hoc search and custom filter queries run on an in-memory NumPy/PyTorch engine in under 50ms.
6. **Why do we need BM25 if we have embeddings?** Embeddings blur specific proper nouns (e.g., confusing two political rallies). BM25 guarantees rare Indian entities and statutory terms are retrieved.
7. **What is Quota Candidate Selection?** Reserves 12 seats for RRF consensus, 5 seats for pure semantic leaps (Dense top-5), and 3 seats for pure entity hits (BM25 top-5), preventing compromise candidates from monopolizing the pool.
8. **What was the double-sigmoid bug?** Applying `sigmoid()` to Cross-Encoder outputs that were already bounded in $[0, 1]$ compressed scores into $[0.50, 0.731]$. Solved via `_to_probability()`.
9. **How is BM25 normalized in the blend?** Self-normalized against the source article's self-score ($rac{	ext{BM25}(q,d)}{	ext{BM25}(q,q)}$), guaranteeing an exact $[0, 1]$ range.
10. **How is Dense cosine normalized in the blend?** Mapped to empirical percentiles via an invariant 101-point quantile distribution calibrated on 50,000 random document pairs.
11. **Why is recency relative to the source article date?** Prevents historical archive stories from having their contemporaneous companions penalized by modern clock time.
12. **What does the $	heta = 0.76$ threshold do in MMR?** Diversity penalties apply **only** when pairwise cosine similarity exceeds $0.76$. Standard topical relatedness is never penalized; only near-duplicate wire copy is penalized.
13. **What is the Additive Signal Agreement Adjustment?** An additive adjustment $[-0.03, +0.03]$ rewarding multi-model consensus and penalizing keyword flukes or embedding drift.
14. **What is Technique 4?** If fewer than 3 recommendations survive, the API automatically appends up to 5 trending articles from the same section within a dynamic time window.
15. **What are the ground-truth evaluation pairs?** Editorial "ALSO READ" links extracted directly from article text by `strip_also_read()`, representing verified human journalist curation.
16. **Why use an in-memory NumPy matrix over pgvector HNSW?** In-memory NumPy executes a 5,000-article vector scan in 1.8ms with 100% exact recall, beating pgvector socket and query overhead (8–14ms).
17. **What is the reverse-push update?** When new articles arrive, the pipeline inverts candidate pairings to re-score and update older archive articles affected by the new content.
18. **How are hallucinations prevented?** Zero generative LLMs are used. All explanations are 100% deterministic, assembled from verified text overlaps, named entities, and boost multipliers.
19. **How are duplicate articles handled?** Deduplicated by headline and ID; the latest `postPublishDate` strictly wins, ensuring fresh editions overwrite breaking drafts.
20. **What is the pipeline fingerprint?** A SHA-256 hash of all model names, token limits, and weights. If configuration changes, the database automatically detects it and recomputes scores.

---

## 7. Operational Command Reference

```bash
# 1. Run the production API server (Port 5000)
python api/server.py

# 2. Run the server in light CPU mode (TF-IDF fallback, no neural models)
NEWSREC_NEURAL=0 python api/server.py

# 3. Execute the full automated test suite (56 tests)
pytest tests/test_strategy_e.py tests/test_pipeline_c_migration.py -v

# 4. Run offline evaluation against ground truth editorial links
python evaluate.py

# 5. Check database table counts and pipeline status
python -c "from db.connection import SessionLocal; from sqlalchemy import text; 
with SessionLocal() as s:
    for t in ['articles', 'articles_768d', 'article_recommendations', 'article_recommendations_768d', 'ingestion_jobs']:
        print(f'{t}: {s.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()}')"
```

---
