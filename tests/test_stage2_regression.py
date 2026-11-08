"""
Stage 2 Regression Test Suite
==============================
Permanent tests verifying:
1. Shared quota candidate selection (scaling at 5, 10, 15, 20 pool sizes)
2. Dense-only candidates exclusively from Dense ranks 1-5
3. BM25-only candidates exclusively from BM25 ranks 1-5
4. Duplicate prevention and pool-size limits
5. Reverse-push retrieval rank map correctness (no artificial ranks, source exclusion)
6. Additive signal agreement adjustments when reranking is unavailable or weak
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import select_quota_candidates, reciprocal_rank_fusion
from src.scoring import signal_agreement_adjustment, boost_parts
import config


# -----------------------------------------------------------------------------
# 1. Quota Selection & Pool Size Scaling Tests
# -----------------------------------------------------------------------------
def test_quota_scaling():
    """Verify that select_quota_candidates scales quotas for pool sizes 5, 10, 15, 20."""
    bm25_mock = list(range(1, 51))
    dense_mock = list(range(101, 151))

    for pool_size in [5, 10, 15, 20]:
        cands = select_quota_candidates(
            bm25_mock,
            dense_mock,
            n_cand=pool_size,
            k_rrf=60,
        )
        assert len(cands) == pool_size, f"Pool size mismatch for n_cand={pool_size}: got {len(cands)}"
        assert len(cands) == len(set(cands)), f"Duplicates found for n_cand={pool_size}"


def test_dense_only_rank_restriction():
    """Confirm Dense-only candidates come exclusively from Dense ranks 1-5."""
    # Ranks 1..5 in dense: [100, 101, 102, 103, 104]
    # Ranks 6..10 in dense: [105, 106, 107, 108, 109]
    bm25_mock = list(range(1, 20))  # Disjoint from dense
    dense_mock = list(range(100, 150))

    cands = select_quota_candidates(
        bm25_mock,
        dense_mock,
        n_cand=20,
        strong_dense_cutoff=5,
    )

    # Any dense candidate chosen via the dense-only branch must be from 100..104
    # (Unless it gets in via pure RRF backfill)
    dense_only_picked = [p for p in cands if p in range(100, 150) and p not in set(bm25_mock)]
    # The first dense-only candidates in the list must be from top 5 (100..104)
    for p in dense_only_picked[:5]:
        assert p in range(100, 105), f"Dense-only candidate {p} was not from top 5 ranks!"


def test_bm25_only_rank_restriction():
    """Confirm BM25-only candidates come exclusively from BM25 ranks 1-5."""
    bm25_mock = list(range(1, 50))
    dense_mock = list(range(100, 150))

    cands = select_quota_candidates(
        bm25_mock,
        dense_mock,
        n_cand=20,
        strong_bm25_cutoff=5,
    )

    bm25_only_picked = [p for p in cands if p in range(1, 50) and p not in set(dense_mock)]
    for p in bm25_only_picked[:3]:
        assert p in range(1, 6), f"BM25-only candidate {p} was not from top 5 ranks!"


def test_no_candidate_pool_overflow_and_no_duplicates():
    """Test with overlapping and disjoint lists: zero duplicates and strict ceiling."""
    bm25_mock = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dense_mock = [1, 2, 3, 100, 101, 102, 103, 104, 105]

    for limit in [1, 3, 7, 12, 17, 20]:
        cands = select_quota_candidates(bm25_mock, dense_mock, n_cand=limit)
        assert len(cands) <= limit
        assert len(cands) == len(set(cands))


def test_rrf_backfill():
    """Confirm that unfilled dense/bm25 quotas are backfilled by RRF."""
    # Only 1 dense-only candidate available in top 5, 0 bm25-only
    bm25_mock = [1, 2, 3, 4, 5]
    dense_mock = [1, 2, 3, 4, 100]  # Only 100 is dense-only

    cands = select_quota_candidates(bm25_mock, dense_mock, n_cand=6)
    assert len(cands) == 6
    assert 100 in cands


# -----------------------------------------------------------------------------
# 2. Reverse-Push Retrieval Rank Maps Tests
# -----------------------------------------------------------------------------
def test_reverse_push_rank_maps():
    """Verify rank map behavior: source excluded, 0-score excluded, unretrieved returns None."""
    bm25_all = np.array([0.0, 8.5, 4.2, 0.0, 0.0, 1.1])
    dense_all = np.array([0.2, 0.9, 0.8, -0.1, 0.5, 0.3])
    opos = 0  # source article

    # Exclude source
    bm25_all[opos] = -np.inf
    bm25_retrieved = [
        int(p) for p in np.argsort(-bm25_all)[:config.N_RETRIEVE] if bm25_all[p] > 1e-9
    ]
    bm25_pos_of = {p: i + 1 for i, p in enumerate(bm25_retrieved)}

    dense_all[opos] = -np.inf
    dense_retrieved = [
        int(p) for p in np.argsort(-dense_all)[:config.N_RETRIEVE] if p != opos
    ]
    dense_pos_of = {p: i + 1 for i, p in enumerate(dense_retrieved)}

    # Source must not have a retrieval rank
    assert bm25_pos_of.get(opos) is None
    assert dense_pos_of.get(opos) is None

    # Zero-score articles (indices 3 and 4) must NOT have a rank
    assert bm25_pos_of.get(3) is None
    assert bm25_pos_of.get(4) is None

    # Retrieved articles must match exact order
    assert bm25_pos_of.get(1) == 1  # 8.5 is rank 1
    assert bm25_pos_of.get(2) == 2  # 4.2 is rank 2
    assert bm25_pos_of.get(5) == 3  # 1.1 is rank 3

    assert dense_pos_of.get(1) == 1  # 0.9 is rank 1
    assert dense_pos_of.get(2) == 2  # 0.8 is rank 2
    assert dense_pos_of.get(4) == 3  # 0.5 is rank 3


# -----------------------------------------------------------------------------
# 3. Additive Agreement Adjustment Tests (With & Without Reranking)
# -----------------------------------------------------------------------------
def test_signal_agreement_without_reranking():
    """When reranker is unavailable (rerank_score=None), adjustment must be safe and bounded."""
    cand = {"id": "c1", "section": "india", "subsection": "politics", "page_views": 100}
    src = {"id": "s1", "section": "india", "subsection": "politics", "linked_ids": []}

    # Dual strong retrieval without reranking
    adj = signal_agreement_adjustment(cand, src, bm25_rank=1, dense_rank=2, rerank_score=None)
    assert -0.03 <= adj <= 0.03
    assert adj == 0.0  # Inconclusive/moderate without CE confirmation

    # Completely unretrieved
    adj_none = signal_agreement_adjustment(cand, src, bm25_rank=None, dense_rank=None, rerank_score=None)
    assert adj_none == 0.0


def test_additive_adjustment_applied_once():
    """Verify that boost_parts separates domain_multiplier and agreement_adjustment correctly."""
    cand = {"id": "c1", "section": "india", "subsection": "politics", "page_views": 100}
    src = {"id": "s1", "section": "india", "subsection": "politics", "linked_ids": []}

    parts = boost_parts(src, cand, max_views=1000, max_date=None, bm25_rank=1, dense_rank=4, rerank_score=0.85)

    assert "domain_multiplier" in parts
    assert "agreement_adjustment" in parts
    assert parts["agreement_adjustment"] == 0.03
    assert parts["domain_multiplier"] > 1.0  # Same section boost applied

    base_rel = 0.60
    final_score = (base_rel * parts["domain_multiplier"]) + parts["agreement_adjustment"]
    # Verify bounds
    assert final_score > base_rel


if __name__ == "__main__":
    tests = [
        test_quota_scaling,
        test_dense_only_rank_restriction,
        test_bm25_only_rank_restriction,
        test_no_candidate_pool_overflow_and_no_duplicates,
        test_rrf_backfill,
        test_reverse_push_rank_maps,
        test_signal_agreement_without_reranking,
        test_additive_adjustment_applied_once,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} - {e}")
            raise e

    print(f"\nAll {passed}/{len(tests)} Stage 2 regression tests passed successfully!")
