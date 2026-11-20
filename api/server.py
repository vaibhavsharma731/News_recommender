"""
FastAPI REST API — News Recommender Backend
Exposes all recommender functionality as JSON endpoints for the frontend.
Run:  python api/server.py
"""
import sys
import os
import time
import math
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import config
from src.recommender import METHODS, Recommender
from api.editor_routes import router as editor_router
from api.monitoring_routes import router as monitoring_router, record_live_query
from api.dataset_routes import router as dataset_router

from db.connection import SessionLocal, ensure_schema, resolve_schema, set_search_path
from db.repository import ArticleRepository
from db.dataset_repository import DatasetRepository
from db.models import Article

# ─── App lifecycle and setup ──────────────────────────────────────────────────
_reload_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        ensure_schema()  # idempotent: adds new columns to existing tables if needed
    except Exception as e:
        print(f"[Startup] Schema check skipped: {e}")

    try:
        with SessionLocal() as db:
            closed = ArticleRepository.fail_orphaned_jobs(db)
            if closed:
                print(f"[Startup] Marked {closed} interrupted ingestion job(s) as failed.")
    except Exception as e:
        print(f"[Startup] Could not clean up old jobs: {e}")

    # Initial loading is heavy; don't block the event loop.
    await run_in_threadpool(reload_engine_from_postgres)
    yield


app = FastAPI(
    title="News Recommender API",
    description="News recommendation engine built on Indian Express articles.",
    version="2.0.0",
    lifespan=lifespan,
)

# Allowed origins configuration: read from ALLOWED_ORIGINS env or default to localhost
_env_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
allowed_origins = [o.strip() for o in _env_origins.split(",") if o.strip()] if _env_origins else [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Database, Editor, Monitoring, and Dataset Routers
app.include_router(editor_router)
app.include_router(monitoring_router)
app.include_router(dataset_router)

# ─── Globals & PostgreSQL Sync ────────────────────────────────────────────────
rec: Optional[Recommender] = None
articles: pd.DataFrame = pd.DataFrame()
_recommenders: Dict[str, Recommender] = {}


def get_recommender(dataset_id: Optional[str] = "default") -> Optional[Recommender]:
    """Retrieves or lazily loads the Recommender instance for a specific dataset."""
    ds_key = (dataset_id or "default").strip()
    if ds_key in _recommenders:
        return _recommenders[ds_key]
    if ds_key == "default" and rec is not None:
        return rec
    reload_engine_from_postgres(dataset_id=ds_key)
    return _recommenders.get(ds_key) or rec


def reload_engine_from_postgres(dataset_id: str = "default"):
    """Build the live search engine from PostgreSQL for a specific dataset, reusing stored texts and vectors.
    Uses stale-while-rebuild: builds in local memory without holding _reload_lock,
    acquiring lock only for the instantaneous reference swap so active traffic is never blocked.
    """
    global rec, articles, _recommenders
    ds_key = (dataset_id or "default").strip()
    try:
        with SessionLocal() as db:
            schema = resolve_schema(db, ds_key)
            set_search_path(db, schema)
            if ArticleRepository.count_articles(db) == 0:
                print(f"[INFO] Dataset '{ds_key}' ({schema}) articles table is empty.")
                with _reload_lock:
                    if ds_key in ("default", "public"):
                        articles, rec = pd.DataFrame(), None
                    _recommenders.pop(ds_key, None)
                return

            arts = db.query(Article).order_by(Article.date.desc().nulls_last()).all()
            df_db = pd.DataFrame(ArticleRepository.corpus_rows(arts, camel_keys=True))
            df_db["date"] = pd.to_datetime(df_db["date"], utc=True, errors="coerce")
            dim = config.EMBEDDING_DIM
            vectors = np.asarray(
                [a.embedding if a.embedding is not None else [0.0] * dim for a in arts],
                dtype=np.float32,
            )

            # Heavy index & semantic builds happen completely outside the global lock
            new_rec = Recommender(df_db, vectors=vectors, use_disk_cache=False)
            
            with _reload_lock:
                _recommenders[ds_key] = new_rec
                if ds_key in ("default", "public"):
                    rec, articles = new_rec, new_rec.articles  # instant reference swap
            print(f"[READY] Engine initialized for dataset '{ds_key}' ({schema}): {len(articles)} articles.")
    except Exception as e:
        print(f"[DB Warning] Could not load dataset '{ds_key}' from PostgreSQL: {e}")
        with _reload_lock:
            if ds_key in ("default", "public") and rec is None:
                articles, rec = pd.DataFrame(), None


# ─── Pydantic request/response models ──────────────────────────────────────
class RecommendRequest(BaseModel):
    query: Optional[str] = None
    article_id: Optional[str] = None
    dataset: Optional[str] = "default"
    top_k: int = config.TOP_K
    method: str = "hybrid+rerank"
    use_boosts: bool = True
    diversify: bool = True
    calibrate: bool = True
    section: Optional[str] = None
    model_version: Optional[str] = "384d"
    candidate_strategy: Optional[str] = "quota"


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _clean_str(val) -> str:
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()


def _clean_float(val, default: float = 0.0) -> float:
    try:
        if val is None or pd.isna(val) or math.isnan(float(val)):
            return default
        return round(float(val), 4)
    except (TypeError, ValueError):
        return default


def _article_dict(a: dict) -> dict:
    """Serialise one article dict to a safe JSON-able dict with no NaNs."""
    return {
        "id":         _clean_str(a.get("id")),
        "title":      _clean_str(a.get("title")),
        "snippet":    _clean_str(a.get("snippet") or a.get("description")),
        "section":    _clean_str(a.get("section")),
        "subsection": _clean_str(a.get("subsection")),
        "date":       _clean_str(a.get("date")),
        "url":        _clean_str(a.get("url")),
        "image":      _clean_str(a.get("image")),
        "is_live":    bool(a.get("is_live", False)),
        # When the article was ingested (its "upload" time), so the reader feed
        # can show how fresh each recommendation is.
        "created_at": _clean_str(a.get("created_at")),
    }


# ─── Graceful degradation (technique 4) ─────────────────────────────────────
# Thin related lists are filled by a SEPARATE "Trending in <section>" block, never
# by padding the related list. Fallback items carry no score, confidence, reason or
# evidence, so nothing downstream can mistake popularity for relevance.
_fallback_stats = {"requests": 0, "triggered": 0, "items": 0}


def _fallback_block(source_art, shown_ids) -> Optional[dict]:
    """
    Popular stories in the source's own section, around the source's own date.

    ``source_art`` is the ORM row, so the date is a real timestamp (the serialised
    dict only carries a display string). Model-free: one indexed query, no reranker,
    no embeddings. Returns None when the section is unknown or nothing qualifies.
    """
    section = (getattr(source_art, "section", "") or "").strip()
    if not section:
        return None
    window = config.fallback_window_days(section)
    try:
        with SessionLocal() as db:
            rows = ArticleRepository.get_trending_in_section(
                db,
                section,
                exclude_ids=[str(getattr(source_art, "id", ""))] + [str(i) for i in shown_ids],
                ref_date=getattr(source_art, "date", None),
                window_days=window,
                limit=config.FALLBACK_LIMIT,
            )
            items = [
                {
                    **_article_dict(a.to_dict()),
                    "type": "fallback",
                    "pageViews": int(a.page_views or 0),
                }
                for a in rows
            ]
    except Exception as e:
        print(f"[Fallback] lookup failed: {e}")
        return None
    if not items:
        return None
    return {
        "heading": f"Trending in {section.replace('-', ' ').replace('_', ' ').title()}",
        "note": ("Popular in this section around the same dates. Not directly matched "
                 "to this story, and not counted as related reading."),
        "type": "fallback",
        "count": len(items),
        "window_days": round(window, 1),
        "items": items,
    }


def _attach_fallback(payload: dict, source_art, shown_ids) -> dict:
    """Count one article-to-article request; attach the fallback block only when thin."""
    _fallback_stats["requests"] += 1
    if len(payload.get("recommendations") or []) >= config.FALLBACK_MIN_RELATED:
        return payload
    block = _fallback_block(source_art, shown_ids)
    if not block:
        return payload
    payload["fallback"] = block
    _fallback_stats["triggered"] += 1
    _fallback_stats["items"] += block["count"]
    print(f"[Fallback] {block['heading']!r} related={len(payload.get('recommendations') or [])} "
          f"window={block['window_days']}d items={block['count']}")
    return payload


def _result_json(result, method: str) -> dict:
    is_gpu = False
    if rec and getattr(rec, "reranker", None):
        is_gpu = bool(getattr(rec.reranker, "last_used_remote", False))
    else:
        from src.model_registry import get_cross_encoder
        cross_enc = get_cross_encoder()
        is_remote_client = getattr(cross_enc, "__class__", None).__name__ == "RemoteCrossEncoderClient"
        is_gpu = is_remote_client and getattr(cross_enc, "last_used_remote", False)

    engine_desc = rec.engine_summary if rec else "PostgreSQL Engine"
    
    return {
        "method":         method,
        "engine":         engine_desc,
        "device":         "Remote Cloud GPU" if is_gpu else "Local CPU",
        "is_gpu":         is_gpu,
        # "postgresql" recommendations are served from the precomputed table (see
        # /api/recommend); anything that reaches here was computed live in memory.
        "source":         "live",
        "candidate_strategy": getattr(result, "candidate_strategy", "quota"),
        "low_confidence": bool(result.low_confidence),
        "best_relevance": _clean_float(result.best_relevance),
        "timings":        getattr(result, "timings", {}),
        "notes":          list(result.notes) if result.notes else [],
        "recommendations": [
            {
                "article": _article_dict(r.article),
                "score":   _clean_float(getattr(r, "calibrated_score", r.score)),
                # Relevance after boosts, before calibration: the value the ordering
                # used. Kept separately so `score` (confidence) is never confused
                # with the boosted relevance that produced the rank.
                "boosted_score": _clean_float(getattr(r, "boosted_score", 0.0)),
                "confidence_band": getattr(r, "confidence_band", "MEDIUM"),
                "margin": _clean_float(getattr(r, "margin", 0.0)),
                "agreement_count": getattr(r, "agreement_count", 1),
                "rank": getattr(r, "rank", idx + 1),
                "rank_raw": getattr(r, "rank_raw", idx + 1),
                "reason":  _clean_str(r.reason),
            }
            for idx, r in enumerate(result.recommendations)
        ],
    }


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/api/status")
def status(pipeline: str = Query("384d"), dataset: str = Query("default")):
    """Engine health-check + metadata directly from PostgreSQL."""
    with SessionLocal() as db:
        schema = resolve_schema(db, dataset)
        set_search_path(db, schema)
        total = ArticleRepository.count_articles(db)
        secs = ArticleRepository.get_sections(db)
        ds_info = DatasetRepository.get_dataset(db, dataset)
    active_rec = get_recommender(dataset)
    return {
        "engine": active_rec.engine_summary if (active_rec and total > 0) else "PostgreSQL Engine (384D BGE-Small)",
        "pipeline": "384d",
        "dataset": dataset,
        "dataset_info": ds_info,
        "total_articles": total,
        "database_empty": total == 0,
        "methods":        METHODS,
        "sections":       secs,
        "notes":          active_rec.notes if (active_rec and total > 0) else ["Database is active; ready for dataset ingestion in Editor Studio."],
        "config": {
            "top_k":        config.TOP_K,
            "n_retrieve":   config.N_RETRIEVE,
            "n_candidates": config.N_CANDIDATES,
            "mmr_lambda":   config.MMR_LAMBDA,
            "use_neural":   config.USE_NEURAL_MODELS,
        },
        "fallback": {
            "min_related": config.FALLBACK_MIN_RELATED,
            "limit":       config.FALLBACK_LIMIT,
            "requests":    _fallback_stats["requests"],
            "triggered":   _fallback_stats["triggered"],
            "items_served": _fallback_stats["items"],
            "rate":        (round(_fallback_stats["triggered"] / _fallback_stats["requests"], 4)
                            if _fallback_stats["requests"] else None),
        },
    }


@app.get("/api/articles")
def get_articles(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    section: str = Query(""),
    sort: str = Query("date"),
    pipeline: str = Query("384d"),
    search: str = Query(""),
    dataset: str = Query("default"),
):
    """Return a paginated list of articles from PostgreSQL."""
    m_ver = pipeline.lower()
    search_term = search.strip() if search else None
    try:
        with SessionLocal() as db:
            schema = resolve_schema(db, dataset)
            set_search_path(db, schema)
            total = ArticleRepository.count_articles(
                db, model_version=m_ver, section=section or None, search=search_term
            )
            if total == 0:
                return {
                    "total": 0,
                    "page": page,
                    "per_page": per_page,
                    "articles": [],
                    "database_empty": True,
                    "pipeline": m_ver,
                    "dataset": dataset,
                }
            start = (page - 1) * per_page
            if sort == "confidence":
                arts = ArticleRepository.list_articles_ranked_by_confidence(
                    db, limit=per_page, offset=start, section=section or None, model_version=m_ver
                )
            else:
                arts = ArticleRepository.list_articles(
                    db, limit=per_page, offset=start, section=section or None, search=search_term, model_version=m_ver
                )
            return {
                "total": total,
                "page": page,
                "per_page": per_page,
                "articles": [a.to_dict() for a in arts],
                "database_empty": False,
                "pipeline": m_ver,
                "dataset": dataset,
            }
    except Exception as e:
        print(f"Error listing articles: {e}")
        return {"total": 0, "page": page, "per_page": per_page, "articles": [], "error": str(e), "pipeline": m_ver, "dataset": dataset}


@app.get("/api/articles/trending")
def trending(n: int = Query(6, ge=1, le=20), pipeline: str = Query("384d"), dataset: str = Query("default")):
    """Return top-N most recent articles directly from PostgreSQL."""
    m_ver = pipeline.lower()
    try:
        with SessionLocal() as db:
            schema = resolve_schema(db, dataset)
            set_search_path(db, schema)
            arts = ArticleRepository.list_articles(db, limit=n, model_version=m_ver)
            return {"articles": [a.to_dict() for a in arts], "pipeline": m_ver, "dataset": dataset}
    except Exception as e:
        return {"articles": [], "pipeline": m_ver, "dataset": dataset}


@app.get("/api/articles/{article_id}")
def get_article(article_id: str, pipeline: str = Query("384d"), dataset: str = Query("default")):
    """Fetch a single article by ID directly from PostgreSQL."""
    try:
        with SessionLocal() as db:
            schema = resolve_schema(db, dataset)
            set_search_path(db, schema)
            art = ArticleRepository.get_article(db, str(article_id))
            if not art:
                raise HTTPException(status_code=404, detail="Article not found")
            return art.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend")
def recommend(body: RecommendRequest):
    if not body.query and not body.article_id:
        raise HTTPException(
            status_code=400,
            detail="Please provide either 'query' (for search) or 'article_id' (for similar articles).",
        )

    # Precomputed recommendations represent similar-article requests with or without MMR diversity.
    use_precomputed = bool(
        body.article_id
        and not body.query
        and not body.section
        and body.method == "hybrid+rerank"
        and body.use_boosts
        and body.top_k <= config.TOP_K
    )
    if use_precomputed:
        t_pre_start = time.perf_counter()
        try:
            m_ver = (body.model_version or "384d").lower()
            with SessionLocal() as db:
                schema = resolve_schema(db, body.dataset)
                set_search_path(db, schema)
                source_art = ArticleRepository.get_article(db, str(body.article_id))
                source_dict = source_art.to_dict() if source_art else None
                precomputed = ArticleRepository.get_precomputed_recommendations(
                    db, str(body.article_id), limit=body.top_k, diversify=body.diversify, model_version=m_ver
                )
            if precomputed and len(precomputed) >= body.top_k:
                pre_ms = round((time.perf_counter() - t_pre_start) * 1000, 2)
                from src.evidence import enrich_recommendation_evidence
                enrich_recommendation_evidence(source_dict, precomputed)
                mode_str = "MMR Diversified" if body.diversify else "Pure Relevance (No MMR)"
                payload = {
                    "method": "postgresql_precomputed_rerank",
                    "engine": "PostgreSQL pgvector (384D BGE-Small)",
                    "source": "postgresql",
                    "dataset": body.dataset or "default",
                    "model_version": "384d",
                    "source_article": source_dict,
                    "diversify": body.diversify,
                    "low_confidence": False,
                    "best_relevance": precomputed[0].get("score", 0.0),
                    "timings": {
                        "total_ms": pre_ms,
                        "db_lookup_ms": pre_ms,
                        "throughput_rps": round(1000.0 / max(pre_ms, 0.1), 1),
                        "source": "postgresql_precomputed",
                    },
                    "notes": [f"⚡ Served directly from PostgreSQL ({pre_ms}ms, {mode_str})"],
                    "recommendations": [{
                        "article": {
                            **_article_dict(r),
                            "rank": r.get("rank", 1),
                            "rank_raw": r.get("rank_raw", 1),
                            "score": r.get("score", 0.0),
                            "rerank_score": r.get("rerank_score", 0.0),
                        },
                        "score": r.get("score", 0.0),
                        "boosted_score": r.get("boosted_score"),
                        "rank": r.get("rank", 1),
                        "rank_raw": r.get("rank_raw", 1),
                        "rerank_score": r.get("rerank_score", 0.0),
                        "reason": r.get("reason") or f"Rank #{r.get('rank', 1)} precomputed",
                        "evidence": r.get("evidence"),
                    } for r in precomputed],

                }
                record_live_query(
                    method="postgresql_precomputed_rerank",
                    query_type="similar",
                    timings=payload["timings"],
                    article_id=str(body.article_id),
                    source="postgresql",
                )
                return _attach_fallback(
                    payload,
                    source_art,
                    # These dicts are article payloads (their key is `id`), not raw rows.
                    [r.get("id") for r in precomputed],
                )
        except Exception as e:
            print(f"Error checking precomputed recommendations: {e}")

    active_rec = get_recommender(body.dataset)
    if active_rec is None:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{body.dataset}' has no articles indexed yet. Please upload a dataset in the Editor Studio.",
        )

    try:
        result = active_rec.recommend(
            query=body.query,
            article_id=str(body.article_id) if body.article_id else None,
            top_k=body.top_k,
            method=body.method,
            use_boosts=body.use_boosts,
            diversify=body.diversify,
            calibrate=body.calibrate,
            section=body.section,
            candidate_strategy=body.candidate_strategy or "quota",
        )
        payload = _result_json(result, body.method)
        payload["dataset"] = body.dataset or "default"
        record_live_query(
            method=body.method,
            query_type="search" if body.query else "similar",
            timings=getattr(result, "timings", {}),
            query_str=body.query,
            article_id=str(body.article_id) if body.article_id else None,
            source="live",
        )
        # Only article-to-article requests have a section to scope "trending" to.
        if body.article_id:
            try:
                with SessionLocal() as db:
                    schema = resolve_schema(db, body.dataset)
                    set_search_path(db, schema)
                    source_art = ArticleRepository.get_article(db, str(body.article_id))
            except Exception as e:
                print(f"[Fallback] source lookup failed: {e}")
                source_art = None
            if source_art is not None:
                payload["source_article"] = source_art.to_dict()
                payload = _attach_fallback(
                    payload,
                    source_art,
                    [r.article.get("id") for r in result.recommendations],
                )
        return payload
    except KeyError:
        raise HTTPException(status_code=404, detail="Article not found in the search index.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sections")
def sections(pipeline: str = Query("384d")):
    """Return all distinct section names directly from PostgreSQL."""
    m_ver = pipeline.lower()
    try:
        with SessionLocal() as db:
            secs = ArticleRepository.get_sections(db, model_version=m_ver)
            return {"sections": secs, "pipeline": m_ver}
    except Exception:
        secs = sorted(articles["section"].dropna().unique().tolist()) if not articles.empty and "section" in articles.columns else []
        return {"sections": secs, "pipeline": m_ver}


# ─── Serve Frontend Static Files ────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/")
def serve_index():
    """Serve the frontend index.html."""
    return FileResponse(FRONTEND_DIR / "index.html")


def _icon_response():
    """Serve the SVG app icon, or a silent 204 if it is missing."""
    icon = FRONTEND_DIR / "favicon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    return Response(status_code=204)


@app.get("/favicon.svg", include_in_schema=False)
def serve_favicon_svg():
    return _icon_response()


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon():
    """
    Browsers request /favicon.ico automatically. Without a handler this logs a
    noisy 404 on every page load, so we answer it with the SVG icon (or a 204).
    """
    return _icon_response()


# Mount static dirs for CSS and JS
app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js",  StaticFiles(directory=FRONTEND_DIR / "js"),  name="js")


@app.middleware("http")
async def _revalidate_static(request, call_next):
    """Force revalidation of CSS/JS so edits are not masked by a stale browser cache.

    Starlette's StaticFiles sends ETag/Last-Modified but no Cache-Control, so browsers
    reuse the asset without asking. `no-cache` still answers with a fast 304 when the
    file is unchanged.
    """
    response = await call_next(request)
    if request.url.path.startswith(("/css/", "/js/")):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import socket
    import sys

    port = int(os.environ.get("PORT", 5000))
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    # Pre-flight check: is port already occupied?
    def _is_port_in_use(port_num: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port_num)) == 0

    if _is_port_in_use(port):
        print("\n" + "=" * 70, flush=True)
        print(f" [PORT ALREADY IN USE] Port {port} is ALREADY running the backend server!", flush=True)
        print(f" You do not need to start it again. Open your browser:", flush=True)
        print(f"   👉 http://localhost:{port}", flush=True)
        print(f"", flush=True)
        print(" If you want to restart it fresh, stop the running instance first:", flush=True)
        print("   Get-Process python | Stop-Process -Force", flush=True)
        print(" Then run: python api/server.py", flush=True)
        print("=" * 70 + "\n", flush=True)
        sys.exit(0)

    print("\n" + "=" * 70, flush=True)
    print("  Indian Express AI News Recommendation Platform (NewsRec v2.0)", flush=True)
    print(f"  Backend server running on: http://localhost:{port}", flush=True)
    print(f"  Open in your web browser:  http://localhost:{port}", flush=True)
    print("  Press CTRL+C to stop the server.", flush=True)
    print("=" * 70 + "\n", flush=True)

    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=False)
