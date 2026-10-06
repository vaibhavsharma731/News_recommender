"""
FastAPI REST API — News Recommender Backend
Exposes all recommender functionality as JSON endpoints for the frontend.
Run:  python api/server.py
"""
import sys
import os
import math
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from src.data_loader import load_articles
from src.recommender import METHODS, Recommender

# ─── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI News Recommender API",
    description="AI-powered news recommendation engine built on Indian Express articles.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Globals (loaded once at startup) ───────────────────────────────────────
print("[INFO] Loading articles...")
df = load_articles(config.DATA_PATH)
print("[INFO] Building recommender (loading embeddings & models)...")
rec = Recommender(df)
articles = rec.articles
print(f"[READY] Engine ready: {rec.engine_summary} | {len(articles)} articles indexed")


# ─── Pydantic request/response models ──────────────────────────────────────
class RecommendRequest(BaseModel):
    query: Optional[str] = None
    article_id: Optional[str] = None
    user_history: Optional[List[str]] = None
    top_k: int = config.TOP_K
    method: str = "hybrid+rerank"
    use_boosts: bool = True
    diversify: bool = True
    section: Optional[str] = None


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
        "snippet":    _clean_str(a.get("snippet")),
        "section":    _clean_str(a.get("section")),
        "subsection": _clean_str(a.get("subsection")),
        "date":       _clean_str(a.get("date")),
        "url":        _clean_str(a.get("url")),
        "image":      _clean_str(a.get("image")),
        "is_live":    bool(a.get("is_live", False)),
    }


def _row_to_dict(row) -> dict:
    """Convert a DataFrame row to a serialisable article dict."""
    desc = _clean_str(row.get("description", ""))
    body = _clean_str(row.get("body", ""))
    snippet = desc if desc else body[:200]
    
    date_val = row.get("date")
    if hasattr(date_val, "strftime") and pd.notna(date_val):
        date_str = date_val.strftime("%d %b %Y")
    else:
        date_str = _clean_str(date_val)

    return {
        "id":         _clean_str(row.get("id")),
        "title":      _clean_str(row.get("title")),
        "snippet":    snippet,
        "section":    _clean_str(row.get("section")),
        "subsection": _clean_str(row.get("subsection")),
        "date":       date_str,
        "url":        _clean_str(row.get("url")),
        "image":      _clean_str(row.get("image")),
        "is_live":    bool(row.get("is_live", False)),
    }


def _result_json(result, method: str) -> dict:
    return {
        "method":         method,
        "engine":         rec.engine_summary,
        "low_confidence": bool(result.low_confidence),
        "best_relevance": _clean_float(result.best_relevance),
        "notes":          list(result.notes) if result.notes else [],
        "recommendations": [
            {
                "article": _article_dict(r.article),
                "score":   _clean_float(r.score),
                "reason":  _clean_str(r.reason),
            }
            for r in result.recommendations
        ],
    }


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    """Engine health-check + metadata."""
    secs = sorted(articles["section"].dropna().unique().tolist())
    return {
        "engine":         rec.engine_summary,
        "total_articles": int(len(articles)),
        "methods":        METHODS,
        "sections":       secs,
        "notes":          rec.notes,
        "config": {
            "top_k":        config.TOP_K,
            "n_retrieve":   config.N_RETRIEVE,
            "n_candidates": config.N_CANDIDATES,
            "mmr_lambda":   config.MMR_LAMBDA,
            "use_neural":   config.USE_NEURAL_MODELS,
        },
    }


@app.get("/api/articles")
def get_articles(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    section: str = Query(""),
    sort: str = Query("date"),
):
    """Return a paginated list of articles."""
    df_view = articles.copy()
    if section:
        df_view = df_view[df_view["section"] == section]

    if sort == "date":
        df_view = df_view.sort_values("date", ascending=False)

    total = len(df_view)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df_view.iloc[start:end]

    return {
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "articles": [_row_to_dict(row) for _, row in page_df.iterrows()],
    }


@app.get("/api/articles/trending")
def trending(n: int = Query(6, ge=1, le=20)):
    """Return top-N most recent articles as trending stories."""
    top_df = articles.sort_values("date", ascending=False).head(n)
    return {
        "articles": [_row_to_dict(row) for _, row in top_df.iterrows()]
    }


@app.get("/api/articles/{article_id}")
def get_article(article_id: str):
    """Fetch a single article by ID."""
    match = articles[articles["id"].astype(str) == str(article_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail="Article not found")
    return _row_to_dict(match.iloc[0])


@app.post("/api/recommend")
def recommend(body: RecommendRequest):
    """
    Unified recommendation endpoint.
    Pass query, article_id, or user_history.
    If none provided (cold start), returns top stories with welcoming note.
    """
    # Cold start: user has no history and gave no search query or article ID
    if not body.query and not body.article_id and (not body.user_history or len(body.user_history) == 0):
        recent_df = articles.sort_values("date", ascending=False)
        if body.section:
            recent_df = recent_df[recent_df["section"] == body.section]
        recent_df = recent_df.head(body.top_k)
        recs = [
            {
                "article": _row_to_dict(row),
                "score": 1.0,
                "reason": "Top trending story -- start reading to build your personalized feed",
            }
            for _, row in recent_df.iterrows()
        ]
        return {
            "method": "trending",
            "engine": rec.engine_summary,
            "low_confidence": False,
            "best_relevance": 1.0,
            "notes": ["Welcome! Here are top stories to get you started. Read articles to build your personalized feed."],
            "recommendations": recs,
        }

    try:
        aid = int(body.article_id) if (body.article_id and str(body.article_id).isdigit()) else body.article_id
        hist = [int(x) if str(x).isdigit() else x for x in body.user_history] if body.user_history else None
        result = rec.recommend(
            query=body.query,
            article_id=aid,
            user_history=hist,
            top_k=body.top_k,
            method=body.method,
            use_boosts=body.use_boosts,
            diversify=body.diversify,
            section=body.section,
        )
        return _result_json(result, body.method)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sections")
def sections():
    """Return all distinct section names."""
    secs = sorted(articles["section"].dropna().unique().tolist())
    return {"sections": secs}


# ─── Serve Frontend Static Files ────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/")
def serve_index():
    """Serve the frontend index.html."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount static dirs for CSS and JS
app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js",  StaticFiles(directory=FRONTEND_DIR / "js"),  name="js")


# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=5000, reload=False)
