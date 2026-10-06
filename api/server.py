"""
Flask REST API - News Recommender Backend
Exposes all recommender functionality as JSON endpoints for the React frontend.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import config
from src.data_loader import load_articles
from src.recommender import METHODS, Recommender

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# ─── Globals ────────────────────────────────────────────────────────────────
print("📂 Loading articles...")
df = load_articles(config.DATA_PATH)
print("🧠 Building recommender (may take a moment for embeddings)...")
rec = Recommender(df)
articles = rec.articles
print(f"✅ Engine ready: {rec.engine_summary} | {len(articles)} articles indexed")


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


def _result_json(result, method: str) -> dict:
    return {
        "method":        method,
        "engine":        rec.engine_summary,
        "low_confidence": result.low_confidence,
        "best_relevance": round(result.best_relevance, 4),
        "notes":         result.notes,
        "recommendations": [
            {
                "article": _article_dict(r.article),
                "score":   round(float(r.score), 4),
                "reason":  r.reason,
            }
            for r in result.recommendations
        ],
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the React frontend."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def status():
    """Engine health-check + metadata."""
    sections = sorted(articles["section"].dropna().unique().tolist())
    return jsonify({
        "engine":       rec.engine_summary,
        "total_articles": int(len(articles)),
        "methods":      METHODS,
        "sections":     sections,
        "notes":        rec.notes,
        "config": {
            "top_k":           config.TOP_K,
            "n_retrieve":      config.N_RETRIEVE,
            "n_candidates":    config.N_CANDIDATES,
            "mmr_lambda":      config.MMR_LAMBDA,
            "use_neural":      config.USE_NEURAL_MODELS,
        }
    })


@app.route("/api/articles")
def get_articles():
    """
    Return a paginated list of articles.
    Query params: page (default 1), per_page (default 20), section (optional), sort (date|title)
    """
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    section  = request.args.get("section", "")
    sort     = request.args.get("sort", "date")

    df_view = articles.copy()
    if section:
        df_view = df_view[df_view["section"] == section]

    if sort == "title":
        df_view = df_view.sort_values("title")
    else:
        df_view = df_view.sort_values("date", ascending=False)

    total   = len(df_view)
    start   = (page - 1) * per_page
    end     = start + per_page
    subset  = df_view.iloc[start:end]

    rows = []
    for _, row in subset.iterrows():
        snippet = row["description"] or row["body"][:200]
        rows.append({
            "id":         row["id"],
            "title":      row["title"],
            "snippet":    snippet,
            "section":    row["section"],
            "subsection": row["subsection"],
            "date":       row["date"].strftime("%d %b %Y") if hasattr(row["date"], "strftime") else str(row["date"]),
            "url":        row["url"],
            "image":      row["image"],
            "is_live":    bool(row["is_live"]),
        })

    return jsonify({"total": total, "page": page, "per_page": per_page, "articles": rows})


@app.route("/api/articles/trending")
def trending():
    """Return the N most recent articles (hero/spotlight)."""
    n = int(request.args.get("n", 6))
    top = articles.sort_values("date", ascending=False).head(n)
    rows = []
    for _, row in top.iterrows():
        snippet = row["description"] or row["body"][:200]
        rows.append({
            "id":         row["id"],
            "title":      row["title"],
            "snippet":    snippet,
            "section":    row["section"],
            "subsection": row["subsection"],
            "date":       row["date"].strftime("%d %b %Y") if hasattr(row["date"], "strftime") else str(row["date"]),
            "url":        row["url"],
            "image":      row["image"],
            "is_live":    bool(row["is_live"]),
        })
    return jsonify({"articles": rows})


@app.route("/api/articles/<article_id>")
def get_article(article_id):
    """Return a single article by ID."""
    try:
        return jsonify({"article": _article_dict(rec.get_article(article_id))})
    except KeyError:
        return jsonify({"error": "Article not found"}), 404


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Unified recommendation endpoint.
    Body (JSON):
      - query:        str   (free-text semantic search)
      - article_id:   str   (item-to-item similarity)
      - user_history: list  (personalised feed — list of article IDs)
      - top_k:        int   (default from config)
      - method:       str   (bm25 | semantic | hybrid | hybrid+rerank)
      - use_boosts:   bool
      - diversify:    bool
      - section:      str   (category filter, empty = all)
    """
    data = request.get_json(force=True, silent=True) or {}

    top_k        = int(data.get("top_k", config.TOP_K))
    method       = data.get("method", "hybrid+rerank")
    use_boosts   = bool(data.get("use_boosts", True))
    diversify    = bool(data.get("diversify", True))
    section      = data.get("section") or None
    query        = data.get("query") or None
    article_id   = data.get("article_id") or None
    user_history = data.get("user_history") or None

    if not query and not article_id and not user_history:
        return jsonify({"error": "Provide at least one of: query, article_id, user_history"}), 400

    try:
        result = rec.recommend(
            query=query,
            article_id=article_id,
            user_history=user_history,
            top_k=top_k,
            method=method,
            use_boosts=use_boosts,
            diversify=diversify,
            section=section,
        )
        return jsonify(_result_json(result, method))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sections")
def sections():
    """Return all distinct section names."""
    secs = sorted(articles["section"].dropna().unique().tolist())
    return jsonify({"sections": secs})


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
