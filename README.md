# 📰 Indian Express AI News Recommendation Platform (NewsRec v2.0)

> **High-Precision Multi-Modal News Recommendation System**  
> Powered by **FastAPI + PostgreSQL/pgvector + BGE-Small (384-D) + BGE Cross-Encoder + Okapi BM25**.

---

## ⚡ Overview

The **NewsRec v2.0** engine delivers real-time, personalized, and editorially grounded news recommendations for the *Indian Express* digital newsroom.

It combines lexical keyword retrieval (**Okapi BM25**), dense semantic vector retrieval (**BAAI/bge-small-en-v1.5, 384-D**), and a deep Transformer Cross-Encoder reranker (**BAAI/bge-reranker-base**) into a unified ranking cascade with editorial boosting, Maximum Marginal Relevance (MMR) diversification, and Platt-scaled calibrated confidence scores.

### Key Architectural Highlights

- **384-Dimensional Dense Embeddings (Pipeline C)**: Migrated from 768-D to 384-D (`BAAI/bge-small-en-v1.5`), yielding a **50% storage reduction**, **2× faster vector scans**, and sharper topical clustering without semantic drift.
- **Candidate Quota Architecture (12 RRF + 5 Dense + 3 BM25)**: Ensures balanced multi-modal representation across cross-encoder scoring pools, preventing dense or lexical mono-cultures.
- **Calibrated Tri-Modal Scoring Blend**:
  $$\text{Relevance} = 0.55 \cdot \text{Reranker}_{\text{prob}} + 0.15 \cdot \text{BM25}_{\text{norm}} + 0.30 \cdot \text{Dense}_{\text{norm}}$$
- **Strategy E Beat-Aware Domain Multipliers**: Distinguishes **Umbrella beats** (e.g. *cities*, *world*, *india*) from **Focused beats** (e.g. *explained*, *business*, *technology*, *lifestyle*), preventing local municipal drift while reinforcing domain-specific reporting depth.
- **Signal Corroboration**: Evaluates multi-modal signal consensus (BM25, Dense, Cross-Encoder) to award agreement adjustments (1 to 3 signals corroborated).
- **Dual Pipeline Reader View**: Seamlessly switch between the **Active 384-D Engine** and the **Historical 768-D Archive** directly from the frontend UI with category-filtered reader exploration.
- **Reverse-Push Incremental Ingestion**: When new batches are uploaded, old articles are automatically and symmetrically refreshed without requiring a full corpus rebuild.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/vaibhavsharma731/News_recommender.git
cd News_recommender

# Create and activate Python virtual environment
python -m venv venv
# On Windows PowerShell:
venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Configuration

Ensure PostgreSQL (with the `pgvector` extension) is running. Configure your connection in `.env`:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/news_recommender
NEWSREC_NEURAL=1
```

### 3. Launch Server

```bash
python api/server.py
```

Open **http://localhost:5000** in your web browser. The modern vanilla JS frontend is served automatically by FastAPI.

> **Lightweight Fallback (No GPU / No Neural Downloads)**:  
> If you wish to run in lightweight mode without downloading PyTorch neural models:
> ```bash
> $env:NEWSREC_NEURAL="0"; python api/server.py
> ```

---

## ⚙️ Core Tunables & Configuration (`config.py`)

All production hyperparameters are centrally controlled in [`config.py`](file:///d:/indian%20express/Recommendation%20system/news%20recommender%20base/config.py):

| Parameter | Current Value | Description |
|---|:---:|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 384-d unit-normalized dense embeddings (pinned SHA `5c38ec7c...`) |
| `EMBEDDING_DIM` | `384` | Dense vector dimension stored in PostgreSQL `vector(384)` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Transformer Cross-Encoder for pair scoring (pinned SHA `2cfc18c9...`) |
| `WEIGHT_RERANK` | `0.55` | Primary weight assigned to Cross-Encoder probability |
| `WEIGHT_BM25` | `0.15` | Weight for Okapi BM25 keyword score |
| `WEIGHT_DENSE` | `0.30` | Weight for dense cosine similarity |
| `N_RETRIEVE` | `50` | Initial retrieval candidate count per branch |
| `N_CANDIDATES` | `20` | Total candidates fed into the Cross-Encoder |
| `RRF_K` | `60` | Reciprocal Rank Fusion smoothing constant |
| `TOP_K` | `10` | Final recommendations surfaced to readers |
| `MMR_LAMBDA` | `0.91` | Diversity trade-off parameter (relevance vs novelty) |
| `MMR_THETA` | `0.76` | Cosine similarity threshold for MMR diversity penalty |
| `SECTION_BOOST_FOCUSED` | `1.10` | Same-section multiplier for focused vertical beats (+10%) |
| `SECTION_BOOST_UMBRELLA` | `1.02` | Same-section multiplier for umbrella/city beats (+2%) |
| `SUBSECTION_BOOST_FOCUSED` | `1.05` | Same-subsection multiplier for focused beats (+5%) |
| `POPULARITY_BOOST_MULT` | `1.015` | Log-scaled page view popularity boost (+1.5% max) |
| `RECENCY_MAX_BOOST_MULT` | `1.040` | Source-date recency boost with section-tiered half-lives (+4.0% max) |

---

## 🏗️ Project Layout

```
News_recommender/
├── api/
│   ├── server.py              # FastAPI endpoints (Dual serving path, precomputed + live fallback)
│   ├── editor_routes.py       # Ingestion endpoints (CSV upload, jobs, status)
│   └── dataset_routes.py      # Dataset/schema management endpoints
├── db/
│   ├── connection.py          # SQLAlchemy engine, pooled connections, declarative schema migrations
│   ├── models.py              # Articles, recommendations, ingestion jobs, datasets ORM models
│   ├── repository.py          # Unified SQL queries, batch inserts, and trending section lookups
│   ├── dataset_repository.py  # Multi-dataset schema isolation CRUD
│   └── pipeline.py            # End-to-end ingestion pipeline, reverse-push, precomputation
├── frontend/
│   ├── index.html             # Single-page UI shell
│   ├── css/                   # Modular CSS files (tabs, theme, cards, modal, typography)
│   └── js/                    # Vanilla JS client modules (api, state, tabs, editor, evaluator, app)
├── src/
│   ├── data_loader.py         # Authoritative text cleaner: NFKC, dedup, sentence token-budgeting
│   ├── retrieval.py           # Okapi BM25, Indian-entity fuzzy correction, 384-D DenseIndex, RRF
│   ├── reranker.py            # Budget-safe Cross-Encoder pair inference (512 token budget)
│   ├── scoring.py             # Canonical scoring cascade: norm, blend, boost_parts, mmr
│   ├── evidence.py            # Deterministic, zero-LLM editorial evidence and explainability cards
│   ├── calibration.py         # Platt scaling calibrator fitting probabilities to confidence bands
│   ├── recommender.py         # In-memory Recommender engine for live search and user history
│   └── model_registry.py      # Thread-safe singleton model cache (local/remote fallback)
├── monitoring/                # Resource monitor and stage timer utilities
├── tests/                     # Comprehensive test suites (Pipeline C, Strategy E, isolation, regression)
├── config.py                  # Single source of truth for all pipeline tunables and version hash
├── evaluate.py                # Standalone evaluation & validation script
└── requirements.txt           # Production Python dependencies
```

---

## 🔄 Dual-Path Recommendation Engine

The system operates on an intelligent **dual-serving path**:

1. **Fast Path (< 5ms)**:
   - When a user views an article without search filters, recommendations are served directly from the precomputed PostgreSQL table `article_recommendations` (or `article_recommendations_768d` if the reader toggles the 768-D model switcher).
   - Serves pre-calculated Cross-Encoder scores, calibrated Platt confidence, multi-signal evidence cards, and boost breakdowns instantly.

2. **Live Dynamic Path**:
   - Fires during real-time free-text search queries, user browsing history personalization, or custom section filters.
   - Executes live Okapi BM25 + in-memory Dense vector scan → RRF → Cross-Encoder reranking → MMR diversification.

---

## 📈 Evaluation & Testing

Run the full evaluation suite:

```bash
# Run the full test suite
python -m pytest tests/ -v --ignore=tests/test_stage2_regression.py

# Run interactive evaluation across beats
python evaluate.py
```

Architecture reference:
- [`AGENTS.md`](file:///d:/indian%20express/Recommendation%20system/news%20recommender%20base/AGENTS.md): Authoritative engineering and architectural specification memory.

---

## 🛡️ License

Proprietary — Developed for Indian Express AI News Recommendation Platform.
