"""
Comprehensive Article-to-Article Recommendation Quality Evaluation Suite
========================================================================
Focuses on HOW GOOD the recommended articles are for each source article.

Key Metric Groups:
  A. Semantic Coherence  – Are recommendations topically aligned? (embedding cosine similarity)
  B. Ranking Quality     – Are the BEST matches at the top? (nDCG@10, MRR, MAP@5)
  C. Ground-Truth Recall – Do editor "Also Read" links appear in the list? (Precision, Hit@K)
  D. Section Alignment   – Same news section / subsection? (Section P@5, Sub P@5)
  E. Diversity Health    – Are recommendations varied enough? (ILD, unique articles coverage)
  F. Popularity Bias     – Is the system unfairly favoring popular articles?
  G. Catalog Coverage    – What % of articles ever get recommended?
  H. Latency             – Precomputed SQL lookup speed

Run with:
  python evaluate.py                    # Full evaluation against PostgreSQL
  python evaluate.py --source api       # Benchmark live HTTP API endpoints
  python evaluate.py --csv data/file.csv  # Evaluate a specific CSV
  python evaluate.py --samples 100     # Limit sample size
  python evaluate.py --save-baseline   # Save results as JSON baseline for comparison
"""
import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from src.data_loader import load_articles
from src.recommender import METHODS, Recommender

# Safe UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEMO_QUERIES = [
    "flight crash",
    "actress Leony dies",
    "Ujjain mosque demolition",
    "Iran US war",
    "cricket world cup India",
    "health insurance claim",
    "skhilesh yadav election",  # Typo test
]


def print_banner(title: str):
    print("\n" + "=" * 78, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 78, flush=True)


def print_section(title: str):
    print(f"\n  {'─' * 70}", flush=True)
    print(f"  {title}", flush=True)
    print(f"  {'─' * 70}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Metric Calculations
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (assumes L2-normalized)."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def calculate_ndcg(ranked_ids: List[str], ground_truth: set, k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at K."""
    hits = [1.0 if aid in ground_truth else 0.0 for aid in ranked_ids[:k]]
    if not any(hits):
        return 0.0
    dcg = sum(h / math.log2(idx + 2) for idx, h in enumerate(hits))
    ideal_hits = [1.0] * min(len(ground_truth), k)
    idcg = sum(h / math.log2(idx + 2) for idx, h in enumerate(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def calculate_mrr(ranked_ids: List[str], ground_truth: set) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant item."""
    for rank, aid in enumerate(ranked_ids, start=1):
        if aid in ground_truth:
            return 1.0 / rank
    return 0.0


def calculate_map(ranked_ids: List[str], ground_truth: set, k: int = 5) -> float:
    """Mean Average Precision at K."""
    hits = 0
    sum_precisions = 0.0
    for i, aid in enumerate(ranked_ids[:k]):
        if aid in ground_truth:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / min(len(ground_truth), k) if ground_truth else 0.0


def calculate_ild(candidate_ids: List[str], vectors_map: Dict[str, np.ndarray]) -> float:
    """
    Intra-List Diversity (ILD): 1 - average pairwise cosine similarity.
    Higher ILD indicates better diversity (MMR is actively suppressing duplicates).
    """
    vecs = [vectors_map[aid] for aid in candidate_ids if aid in vectors_map]
    if len(vecs) < 2:
        return 0.0
    mat = np.array(vecs)
    sims = mat @ mat.T
    n = len(vecs)
    triu_indices = np.triu_indices(n, k=1)
    if len(triu_indices[0]) == 0:
        return 0.0
    mean_sim = float(np.mean(sims[triu_indices]))
    return max(0.0, 1.0 - mean_sim)


def calculate_semantic_coherence(
    source_id: str,
    rec_ids: List[str],
    vectors_map: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Measures how semantically aligned each recommendation is with the source article.
    Returns dict with avg, min, max, and rank-weighted average cosine similarity.
    """
    if source_id not in vectors_map:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "rank_weighted": 0.0}
    source_vec = vectors_map[source_id]
    sims = []
    weighted_sum = 0.0
    weight_total = 0.0
    for rank, rid in enumerate(rec_ids, start=1):
        if rid in vectors_map:
            sim = cosine_similarity(source_vec, vectors_map[rid])
            sims.append(sim)
            # Higher weight for top ranks: weight = 1/log2(rank+1)
            w = 1.0 / math.log2(rank + 1)
            weighted_sum += sim * w
            weight_total += w
    if not sims:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "rank_weighted": 0.0}
    return {
        "avg": float(np.mean(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "rank_weighted": weighted_sum / weight_total if weight_total > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL Precomputed Table Evaluation (Article-to-Article Focus)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_postgres_article_quality(
    db_session,
    all_articles: List[dict],
    valid_linked_rows: List[dict],
    vectors_map: dict,
    sample_size: int = 100
) -> dict:
    """
    Deep evaluation of article-to-article recommendation quality from PostgreSQL.
    Focuses on: How good are the recommended articles for each source article?
    """
    from db.repository import ArticleRepository

    total_articles = len(all_articles)
    total_recs = ArticleRepository.count_total_recommendations(db_session)

    # ── A. Semantic Coherence across ALL articles ──
    print("    [A] Computing Semantic Coherence (source ↔ recommendations)...")
    coherence_scores = []
    rank_weighted_scores = []
    min_sims_all = []
    per_article_results = []
    rec_target_counter = Counter()  # Track how often each article gets recommended
    articles_with_recs = 0
    latencies = []

    # Build article lookup
    art_lookup = {a["id"]: a for a in all_articles}

    for art in all_articles:
        aid = art["id"]
        t0 = time.perf_counter()
        recs = ArticleRepository.get_precomputed_recommendations(db_session, aid, limit=10)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        rec_ids = [r["id"] for r in recs]
        if not rec_ids:
            continue
        articles_with_recs += 1

        for rid in rec_ids:
            rec_target_counter[rid] += 1

        coherence = calculate_semantic_coherence(aid, rec_ids, vectors_map)
        coherence_scores.append(coherence["avg"])
        rank_weighted_scores.append(coherence["rank_weighted"])
        min_sims_all.append(coherence["min"])

        # Section alignment for this article
        source_section = art.get("section", "")
        source_subsection = art.get("subsection", "")
        sec_match = sum(1 for r in recs[:5] if r.get("section") == source_section and source_section) / 5.0
        subsec_match = sum(1 for r in recs[:5] if r.get("subsection") == source_subsection and source_subsection) / 5.0

        per_article_results.append({
            "id": aid,
            "title": art.get("title", "")[:80],
            "section": source_section,
            "n_recs": len(rec_ids),
            "semantic_coherence_avg": coherence["avg"],
            "semantic_coherence_rank_weighted": coherence["rank_weighted"],
            "semantic_coherence_min": coherence["min"],
            "section_p5": sec_match,
            "subsection_p5": subsec_match,
        })

    # ── B. Ground-Truth Editorial Links ──
    print("    [B] Evaluating Editorial Ground-Truth (Also Read links)...")
    p5_list, hit5_list, hit10_list, ndcg_list, mrr_list, map_list = [], [], [], [], [], []

    for row in valid_linked_rows:
        aid = row["id"]
        relevant = set(row["linked_ids"])
        recs = ArticleRepository.get_precomputed_recommendations(db_session, aid, limit=10)
        ranked = [r["id"] for r in recs]

        p5_list.append(sum(1 for x in ranked[:5] if x in relevant) / 5.0)
        hit5_list.append(1.0 if any(x in relevant for x in ranked[:5]) else 0.0)
        hit10_list.append(1.0 if any(x in relevant for x in ranked[:10]) else 0.0)
        ndcg_list.append(calculate_ndcg(ranked, relevant, k=10))
        mrr_list.append(calculate_mrr(ranked, relevant))
        map_list.append(calculate_map(ranked, relevant, k=5))

    # ── C. Section / Subsection Alignment (sample) ──
    print("    [C] Calculating Section & Subsection Alignment...")
    random.seed(42)
    sample_arts = random.sample(all_articles, min(len(all_articles), sample_size))
    sec_hits = total_sec = 0
    subsec_hits = total_subsec = 0
    ild_list = []
    score_distributions = []

    for art in sample_arts:
        recs = ArticleRepository.get_precomputed_recommendations(db_session, art["id"], limit=10)
        rec_ids = [r["id"] for r in recs]
        scores = [r.get("score", 0.0) for r in recs]
        if scores:
            score_distributions.extend(scores)

        for r in recs[:5]:
            if r.get("section") == art.get("section") and art.get("section"):
                sec_hits += 1
            total_sec += 1
            if r.get("subsection") == art.get("subsection") and art.get("subsection"):
                subsec_hits += 1
            total_subsec += 1

        ild_list.append(calculate_ild(rec_ids, vectors_map))

    # ── D. Catalog Coverage & Popularity Bias ──
    print("    [D] Analyzing Catalog Coverage & Popularity Bias...")
    unique_recommended = set(rec_target_counter.keys())
    catalog_coverage = len(unique_recommended) / total_articles if total_articles > 0 else 0.0

    # Gini coefficient of recommendation frequency (0 = perfect equality, 1 = one article gets all)
    if rec_target_counter:
        counts = sorted(rec_target_counter.values())
        n = len(counts)
        cumulative = np.cumsum(counts)
        gini = (2.0 * np.sum((np.arange(1, n + 1) * counts)) / (n * np.sum(counts))) - (n + 1) / n
        gini = max(0.0, min(1.0, gini))

        # Top-heavy analysis: what % of recommendations go to top 5% of articles?
        top_5pct_count = max(1, int(0.05 * len(counts)))
        top_5pct_rec_share = sum(counts[-top_5pct_count:]) / sum(counts) if sum(counts) > 0 else 0
    else:
        gini = 0.0
        top_5pct_rec_share = 0.0

    # ── E. Score Quality Analysis ──
    print("    [E] Analyzing Score Distributions...")
    score_arr = np.array(score_distributions) if score_distributions else np.array([0.0])

    # ── Worst Performing Articles (lowest semantic coherence) ──
    per_article_sorted = sorted(per_article_results, key=lambda x: x["semantic_coherence_avg"])
    worst_10 = per_article_sorted[:10]

    # ── Best Performing Articles ──
    best_10 = per_article_sorted[-10:][::-1]

    return {
        "method": "PostgreSQL Precomputed (Article-to-Article)",
        "total_articles": total_articles,
        "articles_with_recs": articles_with_recs,
        "total_recommendations": total_recs,
        "recs_per_article": total_recs / max(articles_with_recs, 1),

        # Semantic Coherence (THE key metric for article-to-article quality)
        "semantic_coherence_avg": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
        "semantic_coherence_rank_weighted": float(np.mean(rank_weighted_scores)) if rank_weighted_scores else 0.0,
        "semantic_coherence_min_avg": float(np.mean(min_sims_all)) if min_sims_all else 0.0,
        "semantic_coherence_std": float(np.std(coherence_scores)) if coherence_scores else 0.0,

        # Editorial Ground Truth
        "gt_articles": len(valid_linked_rows),
        "gt_p5": float(np.mean(p5_list)) if p5_list else 0.0,
        "gt_hit5": float(np.mean(hit5_list)) if hit5_list else 0.0,
        "gt_hit10": float(np.mean(hit10_list)) if hit10_list else 0.0,
        "gt_ndcg10": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        "gt_mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
        "gt_map5": float(np.mean(map_list)) if map_list else 0.0,

        # Section Alignment
        "section_p5": sec_hits / max(total_sec, 1),
        "subsection_p5": subsec_hits / max(total_subsec, 1),

        # Diversity
        "ild_avg": float(np.mean(ild_list)) if ild_list else 0.0,
        "ild_std": float(np.std(ild_list)) if ild_list else 0.0,

        # Catalog Health
        "catalog_coverage": catalog_coverage,
        "unique_articles_recommended": len(unique_recommended),
        "gini_coefficient": gini,
        "top_5pct_rec_share": top_5pct_rec_share,

        # Score Distribution
        "score_mean": float(np.mean(score_arr)),
        "score_median": float(np.median(score_arr)),
        "score_std": float(np.std(score_arr)),
        "score_min": float(np.min(score_arr)),
        "score_max": float(np.max(score_arr)),
        "score_below_0.3": float(np.mean(score_arr < 0.3)),

        # Latency
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,

        # Drill-down (not saved to baseline)
        "_worst_10": worst_10,
        "_best_10": best_10,
    }


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Method Comparison
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_recommender_method(
    rec: Recommender,
    method_name: str,
    valid_linked_rows: List[dict],
    sample_articles: List[dict],
    use_boosts: bool = False,
    vectors_map: dict = None
) -> dict:
    """Evaluates a live recommender search method."""
    p5_list, ndcg_list, mrr_list, hit10_list, map_list = [], [], [], [], []

    for row in valid_linked_rows:
        aid = row["id"]
        relevant = set(row["linked_ids"])
        try:
            res = rec.recommend(article_id=aid, top_k=10, method=method_name, use_boosts=use_boosts)
            ranked = [r.article["id"] for r in res.recommendations]
            p5_list.append(sum(1 for x in ranked[:5] if x in relevant) / 5.0)
            hit10_list.append(1.0 if any(x in relevant for x in ranked[:10]) else 0.0)
            ndcg_list.append(calculate_ndcg(ranked, relevant, k=10))
            mrr_list.append(calculate_mrr(ranked, relevant))
            map_list.append(calculate_map(ranked, relevant, k=5))
        except Exception:
            continue

    # Semantic coherence, section alignment, diversity
    sec_hits = total_sec = 0
    ild_list = []
    coherence_list = []
    latencies = []

    for art in sample_articles:
        t_start = time.perf_counter()
        try:
            res = rec.recommend(article_id=art["id"], top_k=10, method=method_name, use_boosts=use_boosts)
            latencies.append((time.perf_counter() - t_start) * 1000.0)

            rec_ids = [r.article["id"] for r in res.recommendations]

            for r in res.recommendations[:5]:
                sec_hits += 1 if r.article.get("section") == art.get("section") and art.get("section") else 0
                total_sec += 1

            if vectors_map:
                ild_list.append(calculate_ild(rec_ids, vectors_map))
                coh = calculate_semantic_coherence(art["id"], rec_ids, vectors_map)
                coherence_list.append(coh["avg"])
        except Exception:
            continue

    display_name = f"{method_name} (+boosts)" if use_boosts else method_name

    return {
        "method": display_name,
        "semantic_coherence_avg": float(np.mean(coherence_list)) if coherence_list else 0.0,
        "gt_p5": float(np.mean(p5_list)) if p5_list else 0.0,
        "gt_hit10": float(np.mean(hit10_list)) if hit10_list else 0.0,
        "gt_ndcg10": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        "gt_mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
        "gt_map5": float(np.mean(map_list)) if map_list else 0.0,
        "section_p5": sec_hits / max(total_sec, 1),
        "ild_avg": float(np.mean(ild_list)) if ild_list else 0.0,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Live API Server Evaluation (HTTP)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_live_api(api_url: str = "http://localhost:5000", n_queries: int = 20):
    """Benchmarks the running FastAPI server endpoints over HTTP."""
    print_banner(f"BENCHMARKING LIVE API SERVER ({api_url})")

    # 1. Health check
    try:
        req = urllib.request.Request(f"{api_url}/api/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"  • Engine:         {data.get('engine')}")
            print(f"  • Total Articles: {data.get('total_articles')}")
            print(f"  • Sections:       {len(data.get('sections', []))} categories")
    except Exception as e:
        print(f"  [ERROR] Cannot reach API at {api_url}: {e}")
        print("  Ensure 'python api/server.py' is running!")
        return

    # 2. Fetch sample article IDs
    try:
        req = urllib.request.Request(f"{api_url}/api/articles?per_page={n_queries}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            arts = json.loads(resp.read().decode("utf-8")).get("articles", [])
    except Exception as e:
        print(f"  [ERROR] Could not fetch articles: {e}")
        return

    if not arts:
        print("  Database is currently empty. Ingest articles first via Editor Studio.")
        return

    # 3. Test Precomputed Endpoint Latency
    print(f"\n  Testing Sub-5ms Precomputed SQL Endpoint on {len(arts)} articles...")
    pre_lats = []
    for a in arts:
        aid = a["id"]
        t0 = time.perf_counter()
        req = urllib.request.Request(f"{api_url}/api/db/recommendations/{aid}")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                _ = json.loads(resp.read().decode("utf-8"))
                pre_lats.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            pass

    # 4. Test Search Endpoint Latency
    print(f"  Testing Search Endpoint on Demo Queries...")
    search_lats = []
    for q in DEMO_QUERIES:
        t0 = time.perf_counter()
        payload = json.dumps({"query": q, "top_k": 5}).encode("utf-8")
        req = urllib.request.Request(
            f"{api_url}/api/recommend",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                search_lats.append((time.perf_counter() - t0) * 1000.0)
                flag = "  <-- (No close match)" if res.get("low_confidence") else ""
                best = res.get("best_relevance", 0.0)
                first_title = res.get("recommendations", [{}])[0].get("article", {}).get("title", "")[:50]
                print(f"    • '{q:25s}' -> Best: {best:.2f} | Latency: {search_lats[-1]:.1f}ms{flag} ({first_title}...)")
        except Exception as e:
            print(f"    • Query '{q}' failed: {e}")

    print("\n  " + "-" * 60)
    print(f"  ⚡ Precomputed SQL Direct Lookup:  avg {np.mean(pre_lats):.2f} ms  (min {np.min(pre_lats):.2f} ms)")
    print(f"  🛰️ Live Hybrid Neural Search:     avg {np.mean(search_lats):.2f} ms")
    print("  " + "-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Report Rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_article_quality_report(result: dict):
    """Renders the deep article-to-article quality report."""

    print_section("A. SEMANTIC COHERENCE (Source ↔ Recommendations)")
    print(f"    Avg Cosine Similarity (source → recs):     {result['semantic_coherence_avg']:.4f}")
    print(f"    Rank-Weighted Avg Similarity:               {result['semantic_coherence_rank_weighted']:.4f}")
    print(f"    Avg Worst-Match Similarity:                 {result['semantic_coherence_min_avg']:.4f}")
    print(f"    Std Dev (consistency across articles):      {result['semantic_coherence_std']:.4f}")
    print(f"    Interpretation: {'✅ Good' if result['semantic_coherence_avg'] > 0.55 else '⚠️ Could be improved' if result['semantic_coherence_avg'] > 0.40 else '❌ Low'} topical alignment")

    print_section("B. EDITORIAL GROUND-TRUTH (Also Read Links)")
    if result['gt_articles'] > 0:
        print(f"    Articles with Editor Curated Links:          {result['gt_articles']}")
        print(f"    Precision@5 (Gold):                          {result['gt_p5']:.4f}  ({result['gt_p5']*100:.1f}% of top-5 match editor picks)")
        print(f"    Hit@5:                                       {result['gt_hit5']:.1%}  (any editor pick in top 5)")
        print(f"    Hit@10:                                      {result['gt_hit10']:.1%}  (any editor pick in top 10)")
        print(f"    nDCG@10 (ranking quality):                   {result['gt_ndcg10']:.4f}")
        print(f"    MRR (first relevant rank):                   {result['gt_mrr']:.4f}")
        print(f"    MAP@5 (precision-recall trade-off):          {result['gt_map5']:.4f}")
    else:
        print("    No editorial ground-truth links available in this dataset.")

    print_section("C. SECTION & SUBSECTION ALIGNMENT")
    print(f"    Section Match P@5:                           {result['section_p5']:.1%}")
    print(f"    Subsection Match P@5:                        {result['subsection_p5']:.1%}")
    print(f"    Interpretation: {'✅ Strong' if result['section_p5'] > 0.50 else '⚠️ Mixed' if result['section_p5'] > 0.30 else '❌ Weak'} category alignment")

    print_section("D. DIVERSITY & REDUNDANCY")
    print(f"    Intra-List Diversity (ILD):                  {result['ild_avg']:.4f}  (0=identical, 1=maximally diverse)")
    print(f"    ILD Std Dev:                                 {result['ild_std']:.4f}")
    print(f"    Interpretation: {'✅ Good diversity' if result['ild_avg'] > 0.35 else '⚠️ Some near-duplicates' if result['ild_avg'] > 0.20 else '❌ Too many duplicates'}")

    print_section("E. CATALOG COVERAGE & POPULARITY BIAS")
    print(f"    Unique Articles Recommended:                 {result['unique_articles_recommended']} / {result['total_articles']}")
    print(f"    Catalog Coverage:                            {result['catalog_coverage']:.1%}")
    print(f"    Gini Coefficient (0=equal, 1=monopoly):      {result['gini_coefficient']:.4f}")
    print(f"    Top 5% Articles' Recommendation Share:       {result['top_5pct_rec_share']:.1%}")
    print(f"    Interpretation: {'✅ Healthy' if result['catalog_coverage'] > 0.60 else '⚠️ Moderate' if result['catalog_coverage'] > 0.30 else '❌ Low'} catalog utilization")

    print_section("F. SCORE DISTRIBUTION")
    print(f"    Mean Relevance Score:                        {result['score_mean']:.4f}")
    print(f"    Median Score:                                {result['score_median']:.4f}")
    print(f"    Std Dev:                                     {result['score_std']:.4f}")
    print(f"    Min / Max:                                   {result['score_min']:.4f} / {result['score_max']:.4f}")
    print(f"    % Scores Below 0.3 (low quality):            {result['score_below_0.3']:.1%}")

    print_section("G. LATENCY")
    print(f"    Avg SQL Lookup Latency:                      {result['avg_latency_ms']:.2f} ms")
    print(f"    P95 Latency:                                 {result['p95_latency_ms']:.2f} ms")

    # Worst and Best articles
    if result.get("_worst_10"):
        print_section("H. WORST PERFORMING ARTICLES (Lowest Semantic Coherence)")
        for i, w in enumerate(result["_worst_10"], 1):
            print(f"    {i:2d}. [{w['section']:<12s}] Coherence={w['semantic_coherence_avg']:.3f}  Min={w['semantic_coherence_min']:.3f}  SecP5={w['section_p5']:.0%}  | {w['title']}")

    if result.get("_best_10"):
        print_section("I. BEST PERFORMING ARTICLES (Highest Semantic Coherence)")
        for i, b in enumerate(result["_best_10"], 1):
            print(f"    {i:2d}. [{b['section']:<12s}] Coherence={b['semantic_coherence_avg']:.3f}  Min={b['semantic_coherence_min']:.3f}  SecP5={b['section_p5']:.0%}  | {b['title']}")


def render_method_comparison(results: List[dict], has_ground_truth: bool):
    """Renders comparison table across all methods."""
    print_banner("METHOD COMPARISON MATRIX")

    header = (
        f"  {'Method':<32} | {'Coherence':>9} | {'P@5':>6} | {'Hit@10':>6} | {'nDCG@10':>7} | "
        f"{'MRR':>5} | {'MAP@5':>6} | {'Sec%':>5} | {'ILD':>5} | {'Latency':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        m_name = r["method"]
        coh = f"{r.get('semantic_coherence_avg', 0):.4f}"
        p5 = f"{r['gt_p5']:.3f}" if has_ground_truth else "N/A"
        hit10 = f"{r['gt_hit10']:.1%}" if has_ground_truth else "N/A"
        ndcg = f"{r['gt_ndcg10']:.3f}" if has_ground_truth else "N/A"
        mrr = f"{r['gt_mrr']:.3f}" if has_ground_truth else "N/A"
        map5 = f"{r.get('gt_map5', 0):.3f}" if has_ground_truth else "N/A"
        sec = f"{r.get('section_p5', 0):.0%}"
        ild = f"{r.get('ild_avg', 0):.3f}"
        lat = f"{r['avg_latency_ms']:.1f} ms"

        prefix = "⚡ " if "Precomputed" in m_name or "PostgreSQL" in m_name else "  "
        print(
            f"{prefix}{m_name:<30} | {coh:>9} | {p5:>6} | {hit10:>6} | {ndcg:>7} | "
            f"{mrr:>5} | {map5:>6} | {sec:>5} | {ild:>5} | {lat:>8}"
        )

    print("  " + "-" * (len(header) - 2))
    print("  Legend:")
    print("    • Coherence : Avg cosine similarity between source article and its recommendations")
    print("    • P@5 (Gold): Precision@5 on Editor 'Also Read' links (Curated Ground Truth)")
    print("    • Hit@10   : % of queries with at least 1 editor-linked story in Top-10")
    print("    • nDCG@10  : Normalized Discounted Cumulative Gain (Ranking quality)")
    print("    • MAP@5    : Mean Average Precision at 5")
    print("    • Sec%     : Category alignment (% of recommendations in same section)")
    print("    • ILD      : Intra-List Distance (higher = better diversity)")
    print("    • Latency  : Average response time per recommendation request")


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Save / Compare
# ─────────────────────────────────────────────────────────────────────────────

def _detect_device() -> str:
    """Best-effort description of where scoring actually ran, so baselines
    produced on CPU are never compared against ones produced on a GPU."""
    if config.REMOTE_MODEL_URL:
        return f"remote:{config.REMOTE_MODEL_URL}"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def save_baseline(result: dict, filepath: str = "baseline_evaluation.json"):
    """Saves evaluation results as a JSON baseline for future comparison."""
    # Remove internal drill-down data
    saveable = {k: v for k, v in result.items() if not k.startswith("_")}
    saveable["timestamp"] = datetime.now().isoformat()
    saveable["version"] = "1.0"
    saveable["config"] = {
        # ── Models (pinned) ──────────────────────────────────────────────
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_revision": config.EMBEDDING_REVISION,
        "reranker_model": config.RERANKER_MODEL,
        "reranker_revision": config.RERANKER_REVISION,
        # ── Token budgeting: 512 pair cap = 254 + 254 content + 4 special ──
        "embed_max_seq_len": config.EMBED_MAX_SEQ_LEN,
        "embed_token_budget": config.EMBED_TOKEN_BUDGET,
        "rerank_max_len": config.RERANK_MAX_LEN,
        "rerank_per_article_tokens": config.RERANK_PER_ARTICLE_TOKENS,
        "rerank_token_budget_per_side": config.RERANK_TOKEN_BUDGET,
        "text_recipe_version": config.TEXT_RECIPE_VERSION,
        # ── Retrieval ────────────────────────────────────────────────────
        "n_retrieve": config.N_RETRIEVE,
        "n_candidates": config.N_CANDIDATES,
        "precompute_candidates": config.PRECOMPUTE_CANDIDATES,
        "rrf_k": config.RRF_K,
        "top_k": config.TOP_K,
        # ── Diversity + boosts (the "MEASURED" set) ───────────────────────
        "mmr_lambda": config.MMR_LAMBDA,
        "mmr_theta": config.MMR_THETA,
        "section_boost_mult": config.SECTION_BOOST_MULT,
        "subsection_boost_mult": config.SUBSECTION_BOOST_MULT,
        "popularity_boost_mult": config.POPULARITY_BOOST_MULT,
        "recency_max_boost_mult": config.RECENCY_MAX_BOOST_MULT,
        "boost_mode": config.BOOST_MODE,
        "section_half_lives": dict(config.SECTION_HALF_LIVES),
        # ── Relevance gates ──────────────────────────────────────────────
        "min_raw_rerank_threshold": config.MIN_RAW_RERANK_THRESHOLD,
        "article_discard_floor": config.ARTICLE_DISCARD_FLOOR,
        # ── Retired word-based settings (kept so old/new diffs line up) ───
        "body_words_embed": config.BODY_WORDS_EMBED,
        "body_words_rerank": config.BODY_WORDS_RERANK,
        # ── Provenance ───────────────────────────────────────────────────
        "pipeline_version": config.get_pipeline_version(),
        "neural_enabled": config.USE_NEURAL_MODELS,
        "compute_device": _detect_device(),
    }

    save_path = os.path.join(config.ROOT, filepath)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(saveable, f, indent=2, ensure_ascii=False)
    print(f"\n  📁 Baseline saved to: {save_path}")
    return save_path


def compare_baselines(current: dict, baseline_path: str):
    """Compares current results against a saved baseline."""
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print(f"  No baseline found at {baseline_path}. Run with --save-baseline first.")
        return

    print_banner("BASELINE COMPARISON (Current vs Saved)")
    print(f"  Baseline timestamp: {baseline.get('timestamp', 'unknown')}")
    print(f"  Baseline version:   {baseline.get('version', 'unknown')}\n")

    compare_keys = [
        ("semantic_coherence_avg", "Semantic Coherence", True),
        ("semantic_coherence_rank_weighted", "Rank-Weighted Coherence", True),
        ("gt_p5", "Precision@5 (Ground Truth)", True),
        ("gt_hit10", "Hit@10 (Ground Truth)", True),
        ("gt_ndcg10", "nDCG@10", True),
        ("gt_mrr", "MRR", True),
        ("gt_map5", "MAP@5", True),
        ("section_p5", "Section P@5", True),
        ("ild_avg", "ILD Diversity", True),
        ("catalog_coverage", "Catalog Coverage", True),
        ("gini_coefficient", "Gini (lower=better)", False),
        ("score_mean", "Avg Relevance Score", True),
        ("avg_latency_ms", "Avg Latency (lower=better)", False),
    ]

    header = f"  {'Metric':<35} | {'Baseline':>10} | {'Current':>10} | {'Delta':>10} | {'Verdict':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key, label, higher_is_better in compare_keys:
        old_val = baseline.get(key, 0.0)
        new_val = current.get(key, 0.0)
        if old_val is None:
            old_val = 0.0
        if new_val is None:
            new_val = 0.0
        delta = new_val - old_val
        if higher_is_better:
            verdict = "✅ ↑" if delta > 0.005 else "❌ ↓" if delta < -0.005 else "━ Same"
        else:
            verdict = "✅ ↓" if delta < -0.005 else "❌ ↑" if delta > 0.005 else "━ Same"

        print(f"  {label:<35} | {old_val:>10.4f} | {new_val:>10.4f} | {delta:>+10.4f} | {verdict:>8}")

    print("  " + "-" * (len(header) - 2))


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Human Evaluation Benchmark (Blind Labels)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_human_labels(csv_path: str = "blind_labeling_sample.csv"):
    """
    Evaluates human relevance judgements from a blind labeling CSV.
    Expected format: columns [pair_id, source_id, source_title, candidate_id, candidate_title, label]
    Valid labels: 'relevant' (1.0), 'somewhat' (0.5), 'not' (0.0).
    """
    print_banner("PART 4: HUMAN EVALUATION BENCHMARK (BLIND LABELS)")
    if not os.path.exists(csv_path):
        print(f"  [Notice] Blind labeling file '{csv_path}' not found.")
        print("  Run 'python run_experiments.py' to generate candidate pairs for human evaluation.\n")
        return

    try:
        df_labels = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [Error] Failed to read '{csv_path}': {e}\n")
        return

    if "label" not in df_labels.columns:
        print(f"  [Error] '{csv_path}' missing 'label' column.\n")
        return

    valid_mask = df_labels["label"].notna() & (df_labels["label"].astype(str).str.strip() != "")
    labeled = df_labels[valid_mask].copy()
    total_pairs = len(df_labels)
    n_labeled = len(labeled)

    print(f"  • Source file: '{csv_path}'")
    print(f"  • Total pairs pooled: {total_pairs} | Labeled by human: {n_labeled}")

    if n_labeled == 0:
        print("  • Status: All pairs are currently UNLABELED.")
        print("    👉 To benchmark against human perception, open 'blind_labeling_sample.csv' and")
        print("       fill the 'label' column with: 'relevant', 'somewhat', or 'not'.\n")
        return

    norm_labels = labeled["label"].astype(str).str.strip().str.lower()
    rel_count = int((norm_labels == "relevant").sum())
    some_count = int((norm_labels == "somewhat").sum())
    not_count = int((norm_labels == "not").sum())

    score_map = {"relevant": 1.0, "somewhat": 0.5, "not": 0.0}
    num_scores = norm_labels.map(score_map).fillna(0.0)

    print(f"\n  • Human Rating Breakdown:")
    print(f"      - Relevant (1.0) : {rel_count:3d} ({rel_count/n_labeled*100:5.1f}%)")
    print(f"      - Somewhat (0.5) : {some_count:3d} ({some_count/n_labeled*100:5.1f}%)")
    print(f"      - Not (0.0)      : {not_count:3d} ({not_count/n_labeled*100:5.1f}%)")
    print(f"\n  • Summary Metrics:")
    print(f"      - Human Relevance Score (avg) : {num_scores.mean():.4f}")
    print(f"      - Strict Precision@K          : {rel_count / n_labeled:.4f}")
    print(f"      - Relaxed (Strict+Partial)    : {(rel_count + some_count) / n_labeled:.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main Suite Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation_suite(
    source: str = "auto",
    csv_path: Optional[str] = None,
    sample_size: int = 100,
    save_as_baseline: bool = False,
    compare_to: Optional[str] = None,
    labels_path: Optional[str] = "blind_labeling_sample.csv",
    skip_methods: bool = False,
):
    print_banner("ARTICLE-TO-ARTICLE RECOMMENDATION QUALITY EVALUATION")

    # 1. Resolve Data Source
    has_db = False
    db_session = None
    df = None
    stored_vectors = None

    if source in ("auto", "postgres"):
        try:
            from db.connection import SessionLocal, check_db_health
            from db.models import Article
            from db.repository import ArticleRepository

            health = check_db_health()
            if health.get("connected"):
                db = SessionLocal()
                count = ArticleRepository.count_articles(db)
                if count > 0:
                    print(f"  [SOURCE] Connected to PostgreSQL: {count} articles indexed in database.")
                    has_db = True
                    db_session = db

                    # Load articles from DB
                    arts = db.query(Article).all()
                    rows, vecs = [], []
                    for a in arts:
                        try:
                            linked = json.loads(a.linked_ids) if a.linked_ids else []
                        except Exception:
                            linked = []

                        rows.append({
                            "id": str(a.id),
                            "title": str(a.title),
                            "description": str(a.description or ""),
                            "section": str(a.section or ""),
                            "subsection": str(a.subsection or ""),
                            "date": a.date,
                            "url": str(a.url or ""),
                            "image": str(a.image or ""),
                            "is_live": bool(a.is_live),
                            "body": str(a.body or ""),
                            "pageViews": int(a.page_views or 0),
                            "uniqueUsers": int(a.unique_users or 0),
                            "linked_ids": linked,
                            "text_embed": a.text_embed or f"{a.title}. {a.description or ''}",
                            "text_bm25": a.text_bm25 or f"{a.title}. {a.title}. {a.title}. {a.description or ''}. {a.body or ''}",
                            "text_rerank": a.text_rerank or f"{a.title}. {a.description or ''}.",
                        })
                        vecs.append(a.embedding if a.embedding is not None else [0.0] * 768)

                    df = pd.DataFrame(rows)
                    stored_vectors = np.asarray(vecs, dtype=np.float32)
        except Exception as db_err:
            print(f"  [Notice] PostgreSQL not available or empty ({db_err}). Falling back to CSV.")

    if df is None:
        if csv_path:
            print(f"  [SOURCE] Loading custom CSV from: {csv_path}")
            df = load_articles(csv_path)
        else:
            print("  [SOURCE] Loading default dataset CSVs from config...")
            df = load_articles()

    print(f"  • Total Corpus: {len(df)} unique articles.")

    # 2. Build In-Memory Engine
    print("\n  Initializing Recommender Engine...")
    t_start = time.time()
    rec = Recommender(df, vectors=stored_vectors, use_disk_cache=True)
    print(f"  • Engine: {rec.engine_summary} (Loaded in {time.time() - t_start:.2f}s)")

    # Vector lookup dictionary for similarity calculations
    vectors_map = {}
    for idx, row in df.iterrows():
        vec = rec.semantic.get_vector(idx)
        if vec is not None:
            vectors_map[row["id"]] = vec

    # 3. Partition Ground-Truth Samples
    valid_linked = [
        {"id": row["id"], "linked_ids": row["linked_ids"]}
        for _, row in df.iterrows()
        if len(row["linked_ids"]) > 0
    ]
    random.seed(42)
    sample_articles = df.sample(min(len(df), sample_size), random_state=42).to_dict("records")
    all_articles = df.to_dict("records")

    print(f"  • Ground-Truth 'Also Read' Linked Articles: {len(valid_linked)}")
    print(f"  • Random Sample for Section & Diversity: {len(sample_articles)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 1: Deep Article-to-Article Quality Analysis (PostgreSQL)
    # ═══════════════════════════════════════════════════════════════════════════
    primary_result = None
    if has_db and db_session:
        print_banner("PART 1: DEEP ARTICLE-TO-ARTICLE QUALITY ANALYSIS")
        print("  Evaluating every precomputed recommendation against its source article...\n")
        primary_result = evaluate_postgres_article_quality(
            db_session, all_articles, valid_linked, vectors_map, sample_size
        )
        render_article_quality_report(primary_result)

        # Persist the baseline IMMEDIATELY after Part 1.
        # The baseline is *defined* as the Part 1 result; Part 2 is slow enough
        # (~30 min of CPU cross-encoding) that a crash there must not lose the run.
        if save_as_baseline:
            save_baseline(primary_result, "baseline_evaluation.json")

        if compare_to:
            compare_baselines(primary_result, compare_to)

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 2: Method Comparison (BM25 vs Semantic vs Hybrid vs Rerank)
    # ═══════════════════════════════════════════════════════════════════════════
    if skip_methods:
        print_banner("PART 2: METHOD COMPARISON BENCHMARKS")
        print("  Skipped (--skip-methods) — baseline already saved after Part 1.\n")
    else:
        print_banner("PART 2: METHOD COMPARISON BENCHMARKS")
        print("  Running all retrieval methods on ground-truth articles...\n")
        results = []

        if primary_result:
            # Add PostgreSQL precomputed as a method result for comparison
            results.append({
                "method": "PostgreSQL (Precomputed Top-10)",
                "semantic_coherence_avg": primary_result["semantic_coherence_avg"],
                "gt_p5": primary_result["gt_p5"],
                "gt_hit10": primary_result["gt_hit10"],
                "gt_ndcg10": primary_result["gt_ndcg10"],
                "gt_mrr": primary_result["gt_mrr"],
                "gt_map5": primary_result["gt_map5"],
                "section_p5": primary_result["section_p5"],
                "ild_avg": primary_result["ild_avg"],
                "avg_latency_ms": primary_result["avg_latency_ms"],
            })

        for idx, m in enumerate(METHODS, start=2 if primary_result else 1):
            n_total = len(METHODS) + (2 if primary_result else 1)
            print(f"  • Benchmarking [{idx}/{n_total}]: {m}...")
            res = evaluate_recommender_method(rec, m, valid_linked, sample_articles, use_boosts=False, vectors_map=vectors_map)
            results.append(res)

        print(f"  • Benchmarking [{len(METHODS) + (2 if primary_result else 1)}/{len(METHODS) + (2 if primary_result else 1)}]: hybrid+rerank (+News Boosts)...")
        res_boosted = evaluate_recommender_method(rec, "hybrid+rerank", valid_linked, sample_articles, use_boosts=True, vectors_map=vectors_map)
        results.append(res_boosted)

        render_method_comparison(results, has_ground_truth=bool(valid_linked))

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3: Free-Text Search Demo
    # ═══════════════════════════════════════════════════════════════════════════
    print_banner("PART 3: FREE-TEXT SEARCH DEMO (SEMANTIC + BM25 FUZZY)")
    for q in DEMO_QUERIES:
        res = rec.recommend(query=q, top_k=3)
        flag = "  <-- (Low Confidence / No Close Match)" if res.low_confidence else ""
        print(f"\n  Query: '{q}'  (Best Relevance: {res.best_relevance:.2f}){flag}")
        for r in res.recommendations:
            print(f"     [{int(r.score * 100)}%] {r.article['title'][:80]}")
            print(f"            Reason: {r.reason}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 4: Human Evaluation Benchmark (Blind Labels)
    # ═══════════════════════════════════════════════════════════════════════════
    if labels_path:
        evaluate_human_labels(labels_path)

    # ═══════════════════════════════════════════════════════════════════════════
    # Save / Compare are handled right after Part 1 (see above).
    # ═══════════════════════════════════════════════════════════════════════════
    if has_db and db_session:
        db_session.close()

    print("\n" + "=" * 78)
    print("  ✅ EVALUATION COMPLETE")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate News Recommendation System (Article-to-Article Focus)")
    parser.add_argument("--source", choices=["auto", "postgres", "csv", "api"], default="auto",
                        help="Data source to evaluate: auto, postgres, csv, or api")
    parser.add_argument("--csv", type=str, default=None,
                        help="Custom CSV file path to evaluate")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of articles to sample for section & diversity tests")
    parser.add_argument("--api-url", type=str, default="http://localhost:5000",
                        help="Base URL of running API server when using --source api")
    parser.add_argument("--labels", type=str, default="blind_labeling_sample.csv",
                        help="Path to blind human labeling CSV for Part 4")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save results as baseline_evaluation.json for future comparison")
    parser.add_argument("--compare", type=str, default=None, metavar="BASELINE_FILE",
                        help="Compare current results against a saved baseline JSON file")
    parser.add_argument("--skip-methods", action="store_true",
                        help="Skip Part 2 (4x live retrieval benchmarks, ~30 min on CPU). "
                             "Part 1 — the part that --save-baseline records — still runs and is saved first.")

    args = parser.parse_args()

    if args.source == "api":
        evaluate_live_api(args.api_url)
    else:
        run_evaluation_suite(
            source=args.source,
            csv_path=args.csv,
            sample_size=args.samples,
            save_as_baseline=args.save_baseline,
            compare_to=args.compare,
            labels_path=args.labels,
            skip_methods=args.skip_methods,
        )
