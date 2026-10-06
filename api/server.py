"""
FastAPI REST API — News Recommender Backend
Exposes all recommender functionality as JSON endpoints for the frontend.
Run:  uvicorn api.server:app --reload --port 5000
"""
import sys
import os
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
print("📂 Loading articles...")
df = load_articles(config.DATA_PATH)
print("🧠 Building recommender (may take a moment for embeddings)...")
rec = Recommender(df)
articles = rec.articles
print(f"✅ Engine ready: {rec.engine_summary} | {len(articles)} articles indexed")


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
def _article_dict(a: dict) -> dict:
    """Serialise one article dict to a safe JSON-able dict."""
    return {
        "id":         a["id"],
        "title":      a["title"],
        "snippet":    a["snippet"],
        "section":    a["section"],
        "subsection": a["subsection"],
        "date":       a["date"],
        "url":        a["url"],
        "image":      a["image"],
        "is_live":    a["is_live"],
    }


def _row_to_dict(row) -> dict:
    """Convert a DataFrame row to a serialisable article dict."""
    snippet = row["description"] or row["body"][:200]
    return {
        "id":         row["id"],
        "title":      row["title"],
        "snippet":    snippet,
        "section":    row["section"],
        "subsection": row["subsection"],
        "date":       row["date"].strftime("%d %b %Y") if hasattr(row["date"], "strftime") else str(row["date"]),
        "url":        row["url"],
        "image":      row["image"],
        "is_live":    bool(row["is_live"]),
    }


def _result_json(result, method: str) -> dict:
    return {
        "method":         method,
        "engine":         rec.engine_summary,
        "low_confidence": result.low_confidence,
        "best_relevance": round(result.best_relevance, 4),
        "notes":          result.notes,
        "recommendations": [
            {
                "article": _article_dict(r.article),
                "score":   round(float(r.score), 4),
                "reason":  r.reason,
            }
            for r in result.recommendations
        ],
    }


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    """Engine health-check + metadata."""
    sections = sorted(articles["section"].dropna().unique().tolist())
    return {
        "engine":         rec.engine_summary,
        "total_articles": int(len(articles)),
        "methods":        METHODS,
        "sections":       sections,
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

    if sort == "title":
        df_view = df_view.sort_values("title")
    else:
        df_view = df_view.sort_values("date", ascending=False)

    total = len(df_view)
    start = (page - 1) * per_page
    end   = start + per_page
    subset = df_view.iloc[start:end]

    rows = [_row_to_dict(row) for _, row in subset.iterrows()]
    return {"total": total, "page": page, "per_page": per_page, "articles": rows}


@app.get("/api/articles/trending")
def trending(n: int = Query(6, ge=1, le=20)):
    """Return the N most recent articles (hero / spotlight)."""
    top = articles.sort_values("date", ascending=False).head(n)
    rows = [_row_to_dict(row) for _, row in top.iterrows()]
    return {"articles": rows}


@app.get("/api/articles/{article_id}")
def get_article(article_id: str):
    """Return a single article by ID."""
    try:
        return {"article": _article_dict(rec.get_article(article_id))}
    except KeyError:
        raise HTTPException(status_code=404, detail="Article not found")


@app.post("/api/recommend")
def recommend(body: RecommendRequest):
    """
    Unified recommendation endpoint.
    Pass exactly one of: query, article_id, user_history.
    """
    if not body.query and not body.article_id and not body.user_history:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: query, article_id, user_history",
        )

    try:
        result = rec.recommend(
            query=body.query,
            article_id=body.article_id,
            user_history=body.user_history,
            top_k=body.top_k,
            method=body.method,
            use_boosts=body.use_boosts,
            diversify=body.diversify,
            section=body.section,
        )
        return _result_json(result, body.method)
    except Exception as e:
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


# Mount static dirs for CSS and JS (must come after explicit routes)
app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js",  StaticFiles(directory=FRONTEND_DIR / "js"),  name="js")


# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=5000, reload=True)
