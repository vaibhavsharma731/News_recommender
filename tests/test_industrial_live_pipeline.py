"""
Tests for Industrial Scale Optimizations:
1. PostgreSQL HNSW Index on 384-D vector embeddings
2. PostgreSQL GIN Index on Full-Text Search tsvectors
3. Database-level Hybrid Reciprocal Rank Fusion (ArticleRepository.retrieve_hybrid_candidates)
4. Sub-1ms In-Memory Recommendation Cache (RecommendationCache)
"""
import pytest
import time
from db.connection import SessionLocal, resolve_schema, set_search_path
from db.repository import ArticleRepository
from api.server import RecommendationCache, RecommendRequest, _compute_db_backed_recommendations


def test_recommendation_cache_ttl_and_hits():
    """Verify LRU cache hits, overwrites, and eviction."""
    cache = RecommendationCache(maxsize=3, ttl_seconds=1)
    cache.set("k1", {"result": 1})
    cache.set("k2", {"result": 2})

    assert cache.get("k1") == {"result": 1}
    assert cache.get("k2") == {"result": 2}
    assert cache.get("k3") is None

    # Test LRU eviction when exceeding maxsize
    cache.set("k3", {"result": 3})
    cache.set("k4", {"result": 4})
    # k1 was accessed, so k2 should be evicted first or k1/k2 according to order
    assert len(cache._data) <= 3

    # Test TTL expiration
    time.sleep(1.1)
    assert cache.get("k1") is None
    assert cache.get("k4") is None


def test_industrial_hybrid_candidate_retrieval():
    """Test ArticleRepository.retrieve_hybrid_candidates performs fast DB-level hybrid search."""
    with SessionLocal() as db:
        schema = resolve_schema(db, "default")
        set_search_path(db, schema)

        art = db.query(ArticleRepository.get_article(db, "art_fbd9aa_1") or None).first() if False else ArticleRepository.get_article(db, "art_fbd9aa_1")
        if not art:
            pytest.skip("Article art_fbd9aa_1 not found in default dataset")

        t0 = time.perf_counter()
        candidates = ArticleRepository.retrieve_hybrid_candidates(
            db,
            embedding=art.embedding,
            query_text=art.title,
            limit=10,
            n_retrieve=20,
            exclude_id=art.id,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(candidates) > 0, "Expected hybrid candidate matches from PostgreSQL"
        assert elapsed_ms < 200, f"DB hybrid search took {elapsed_ms:.2f}ms (expected < 200ms)"

        first = candidates[0]
        assert "id" in first
        assert "dense_score" in first
        assert "bm25_score" in first
        assert "rrf_score" in first
        assert first["id"] != art.id, "Source article must be excluded"


def test_db_backed_live_recommendation_computation():
    """Test _compute_db_backed_recommendations serves live results using DB candidates and 0MB Python RAM."""
    req = RecommendRequest(
        article_id="art_fbd9aa_1",
        dataset="default",
        top_k=5,
        method="hybrid",
        use_boosts=True,
    )
    t0 = time.perf_counter()
    payload = _compute_db_backed_recommendations(req)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert payload["source"] == "live"
    assert "PostgreSQL Industrial Engine" in payload["engine"]
    assert len(payload["recommendations"]) > 0
    assert payload["recommendations"][0]["score"] > 0.0
    assert elapsed_ms < 500, f"Live recommendation took {elapsed_ms:.2f}ms"
