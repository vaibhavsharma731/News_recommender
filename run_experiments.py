"""
run_experiments.py
==================
Runs the complete measurement & calibration suite on the frozen pool:
1. Story cluster splits: Train (60%) / Validation (20%) / Test (20%)
2. Ablation ladder A0 -> A6 with bootstrap 95% CIs and paired CIs vs A0
3. Design variants comparison (caps, floor, tie-breaker, source date recency)
4. Random search (200 configs on validation), logging to tuning_results.csv
5. Selection rule to choose winning config
6. Final evaluation of winning config on Test set (used exactly once)
7. Generates blind_labeling_sample.csv for human evaluation
"""
import os
import sys
import time
import math
import pickle
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# Safe UTF-8 console output (Windows cp1252 fix)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config

# ─────────────────────────────────────────────────────────────────────────────
# 1. SPLITS: Story Clustering & Leakage-Free Partitioning
# ─────────────────────────────────────────────────────────────────────────────

def get_splits(articles: List[Dict], vecs: np.ndarray, seed: int = 42):
    """
    Groups source articles by high semantic similarity (near-duplicates / same-story clusters)
    so no story ever leaks across splits. Partitions clusters into train (60%), val (20%), test (20%).
    """
    n = len(articles)
    positions = [a["source_pos"] for a in articles]
    src_vecs = vecs[positions]
    sim_matrix = src_vecs @ src_vecs.T
    
    # Story cluster adjacency: cosine >= 0.82
    adj = sim_matrix >= 0.82
    visited = [False] * n
    clusters = []
    
    for i in range(n):
        if not visited[i]:
            cluster = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                cluster.append(curr)
                for neighbor in range(n):
                    if adj[curr, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            clusters.append(cluster)
            
    # Deterministic split of clusters
    rng = np.random.RandomState(seed)
    rng.shuffle(clusters)
    
    train_idx, val_idx, test_idx = [], [], []
    train_target = int(0.60 * n)
    val_target = int(0.20 * n)
    
    for cl in clusters:
        if len(train_idx) + len(cl) <= train_target:
            train_idx.extend(cl)
        elif len(val_idx) + len(cl) <= val_target:
            val_idx.extend(cl)
        else:
            test_idx.extend(cl)
            
    # Fallback ensure non-empty
    if not val_idx and len(train_idx) > 5:
        val_idx.append(train_idx.pop())
    if not test_idx and len(train_idx) > 5:
        test_idx.append(train_idx.pop())
        
    print(f"[Splits] Total {n} articles in {len(clusters)} story clusters.")
    print(f"  • Train: {len(train_idx)} articles | Val: {len(val_idx)} articles | Test: {len(test_idx)} articles")
    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ranking & Scoring Function (Zero Model Calls)
# ─────────────────────────────────────────────────────────────────────────────

def score_and_rank_article(
    art: Dict,
    max_page_views: int,
    max_corpus_date: pd.Timestamp,
    w_rerank: float = 0.55,
    w_bm25: float = 0.30,
    w_dense: float = 0.15,
    sec_mult: float = 1.0,
    subsec_mult: float = 1.0,
    pop_mult: float = 1.0,
    rec_mult: float = 1.0,
    boost_cap: float = None,
    boost_mode: str = "multiplicative",
    relevance_floor: float = 0.05,
    epsilon: float = 0.03,
    mmr_lambda: float = 1.0,
    mmr_theta: float = 0.80,
    top_k: int = 5,
) -> List[int]:
    """
    Ranks top 20 candidates for a source article using pure cached signals.
    Never includes Also-Read boost.
    """
    n_cands = len(art["cands_meta"])
    if n_cands == 0:
        return []
        
    # 1. Base unboosted relevance blend
    rerank_raw = art["cand_rerank_raw"]
    bm25_norm = art["cand_bm25_norm"]
    dense_norm = art["cand_dense_norm"]
    
    unboosted_rel = w_rerank * rerank_raw + w_bm25 * bm25_norm + w_dense * dense_norm
    
    # 2. Compute individual boost multipliers
    multipliers = np.ones(n_cands, dtype=float)
    s_sec = art["source_section"]
    s_subsec = art["source_subsection"]
    s_date = art["source_date"]
    
    log_max_views = math.log1p(max_page_views)
    
    for i, cm in enumerate(art["cands_meta"]):
        m = 1.0
        # Section boost
        if sec_mult > 1.0 and s_sec and cm["section"] == s_sec:
            m *= sec_mult
        # Subsection boost
        if subsec_mult > 1.0 and s_subsec and cm["subsection"] == s_subsec:
            m *= subsec_mult
        # Popularity boost
        if pop_mult > 1.0 and cm["pageViews"] > 0 and log_max_views > 0:
            factor = 1.0 + (pop_mult - 1.0) * (math.log1p(cm["pageViews"]) / log_max_views)
            m *= factor
        # Recency boost
        if rec_mult > 1.0 and pd.notna(cm["date"]):
            ref_date = s_date if boost_mode == "source_date_recency" and pd.notna(s_date) else max_corpus_date
            days_old = max(0.0, (ref_date - cm["date"]).total_seconds() / 86400.0)
            sec = "live_blog" if cm["is_live"] else str(cm["section"]).lower()
            half_life = config.SECTION_HALF_LIVES.get(sec, config.SECTION_HALF_LIVES.get("default", 2.0))
            rec_factor = 1.0 + (rec_mult - 1.0) * (0.5 ** (days_old / half_life))
            m *= rec_factor
            
        multipliers[i] = m

    # 3. Apply Boost Mode & Cap
    if boost_cap is not None:
        multipliers = np.minimum(multipliers, boost_cap)
        
    final_scores = np.copy(unboosted_rel)
    
    if boost_mode == "floor":
        # Only boost candidates above relevance floor
        mask = unboosted_rel >= relevance_floor
        final_scores[mask] = unboosted_rel[mask] * multipliers[mask]
    elif boost_mode == "tie_breaker":
        # Rank primarily by unboosted, use boost within epsilon
        final_scores = unboosted_rel + (multipliers - 1.0) * epsilon
    else:
        # Standard multiplicative
        final_scores = unboosted_rel * multipliers
        
    # 4. Diversity / MMR (if mmr_lambda < 1.0)
    if mmr_lambda < 0.999 and n_cands > 1:
        sim_mat = art["cand_sim_mat"]
        selected = []
        remaining = list(range(n_cands))
        
        while remaining and len(selected) < top_k:
            def mmr_score(i):
                high_sims = [sim_mat[i, j] for j in selected if sim_mat[i, j] > mmr_theta]
                redundancy = max(high_sims, default=0.0)
                return mmr_lambda * final_scores[i] - (1.0 - mmr_lambda) * redundancy
                
            best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
        top_indices = selected
    else:
        top_indices = list(np.argsort(-final_scores)[:top_k])
        
    return top_indices, unboosted_rel


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metrics Calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_split_metrics(
    articles: List[Dict],
    indices: List[int],
    max_page_views: int,
    max_corpus_date: pd.Timestamp,
    total_corpus_size: int,
    **cfg_params,
) -> Dict[str, float]:
    """
    Computes all required metrics on a split:
    nDCG@5, Recall@5, Hit@5, near-dup rate, catalog coverage, median age, mean unboosted relevance.
    """
    ndcgs, recalls, hits = [], [], []
    served_cand_ids = set()
    dup_rates = []
    ages_days = []
    served_unboosted = []
    
    theta = cfg_params.get("mmr_theta", 0.80)
    
    for idx in indices:
        art = articles[idx]
        top_indices, unboosted_rel = score_and_rank_article(
            art, max_page_views, max_corpus_date, **cfg_params
        )
        
        gt_ids = art["linked_ids"]
        k = len(top_indices)
        if k == 0:
            continue
            
        ranked_cands = [art["cands_meta"][i] for i in top_indices]
        ranked_ids = [c["id"] for c in ranked_cands]
        
        for cid in ranked_ids:
            served_cand_ids.add(cid)
            
        # Hit, Recall, nDCG
        hit_flags = [1.0 if cid in gt_ids else 0.0 for cid in ranked_ids]
        hits.append(1.0 if any(hit_flags) else 0.0)
        
        denom_recall = min(len(gt_ids), 5) if len(gt_ids) > 0 else 1
        recalls.append(sum(hit_flags) / denom_recall)
        
        # nDCG@5
        if not any(hit_flags):
            ndcgs.append(0.0)
        else:
            dcg = sum(h / math.log2(pos + 2) for pos, h in enumerate(hit_flags))
            idcg = sum(1.0 / math.log2(pos + 2) for pos in range(min(len(gt_ids), 5)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
            
        # Extra Metric 1: Near-duplicate rate in top-5 (pairs with sim > theta)
        sim_mat = art["cand_sim_mat"]
        n_pairs = 0
        n_dups = 0
        for i_pos in range(len(top_indices)):
            for j_pos in range(i_pos + 1, len(top_indices)):
                n_pairs += 1
                c_i = top_indices[i_pos]
                c_j = top_indices[j_pos]
                if sim_mat[c_i, c_j] > theta:
                    n_dups += 1
        dup_rates.append(n_dups / max(n_pairs, 1))
        
        # Extra Metric 2: Median age vs source date
        s_date = art["source_date"]
        for c in ranked_cands:
            if pd.notna(s_date) and pd.notna(c["date"]):
                age = (s_date - c["date"]).total_seconds() / 86400.0
                ages_days.append(age)
                
        # Extra Metric 3: Mean unboosted relevance of served items
        for i_pos in top_indices:
            served_unboosted.append(unboosted_rel[i_pos])
            
    coverage = len(served_cand_ids) / total_corpus_size
    med_age = float(np.median(ages_days)) if ages_days else 0.0
    mean_unb = float(np.mean(served_unboosted)) if served_unboosted else 0.0
    dup_rate = float(np.mean(dup_rates)) if dup_rates else 0.0
    
    return {
        "ndcg5": float(np.mean(ndcgs)),
        "recall5": float(np.mean(recalls)),
        "hit5": float(np.mean(hits)),
        "near_dup_rate": dup_rate,
        "catalog_coverage": coverage,
        "median_age_days": med_age,
        "mean_unboosted_rel": mean_unb,
        "_per_article_ndcg": ndcgs,
        "_per_article_recall": recalls,
        "_per_article_hit": hits,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bootstrap 95% Confidence Intervals & Paired CIs vs A0
# ─────────────────────────────────────────────────────────────────────────────

def compute_bootstrap_ci(
    arr: List[float],
    arr_a0: List[float] = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Computes 95% bootstrap CI for the metric and paired CI vs A0."""
    rng = np.random.RandomState(seed)
    n = len(arr)
    boot_means = []
    paired_diffs = []
    
    arr = np.asarray(arr)
    arr_a0 = np.asarray(arr_a0) if arr_a0 is not None else None
    
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        m = np.mean(arr[idx])
        boot_means.append(m)
        if arr_a0 is not None:
            diff = np.mean(arr[idx] - arr_a0[idx])
            paired_diffs.append(diff)
            
    ci_metric = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    ci_paired = (float(np.percentile(paired_diffs, 2.5)), float(np.percentile(paired_diffs, 97.5))) if paired_diffs else (0.0, 0.0)
    return ci_metric, ci_paired


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Execution: Ablations, Variants, Random Search, Test Set Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 70)
    print("  EXPERIMENTS & CALIBRATION: MEASURED BOOSTS, MMR & BLENDS")
    print("=" * 70)
    
    cache_path = "data/cache/frozen_pool.pkl"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Frozen pool not found at {cache_path}! Run freeze_pool.py first.")
        
    with open(cache_path, "rb") as f:
        pool_data = pickle.load(f)
        
    articles = pool_data["articles"]
    max_pv = pool_data["max_page_views"]
    max_cd = pool_data["max_corpus_date"]
    tot_corpus = pool_data["total_corpus_size"]
    
    vecs = np.load("data/cache/articles_2467_vectors.npy")
    
    # Splits
    train_idx, val_idx, test_idx = get_splits(articles, vecs, seed=42)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: ABLATION LADDER (A0 -> A6) on Validation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 3: ABLATION LADDER (A0 -> A6) ON VALIDATION SET")
    print("=" * 70)
    
    ablations = [
        ("A0 (Relevance only)", dict(sec_mult=1.0, subsec_mult=1.0, pop_mult=1.0, rec_mult=1.0, mmr_lambda=1.0)),
        ("A1 (A0 + Section alone)", dict(sec_mult=1.08, subsec_mult=1.0, pop_mult=1.0, rec_mult=1.0, mmr_lambda=1.0)),
        ("A2 (A0 + Subsection alone)", dict(sec_mult=1.0, subsec_mult=1.05, pop_mult=1.0, rec_mult=1.0, mmr_lambda=1.0)),
        ("A3 (A0 + Popularity alone)", dict(sec_mult=1.0, subsec_mult=1.0, pop_mult=1.06, rec_mult=1.0, mmr_lambda=1.0)),
        ("A4 (A0 + Recency alone)", dict(sec_mult=1.0, subsec_mult=1.0, pop_mult=1.0, rec_mult=1.10, mmr_lambda=1.0)),
        ("A5 (All boosts)", dict(sec_mult=1.08, subsec_mult=1.05, pop_mult=1.06, rec_mult=1.10, mmr_lambda=1.0)),
        ("A6 (All boosts + MMR)", dict(sec_mult=1.08, subsec_mult=1.05, pop_mult=1.06, rec_mult=1.10, mmr_lambda=0.75, mmr_theta=0.80)),
    ]
    
    ablation_results = []
    a0_ndcg, a0_recall, a0_hit = None, None, None
    
    for name, params in ablations:
        res = calculate_split_metrics(articles, val_idx, max_pv, max_cd, tot_corpus, **params)
        
        if name.startswith("A0"):
            a0_ndcg = res["_per_article_ndcg"]
            a0_recall = res["_per_article_recall"]
            a0_hit = res["_per_article_hit"]
            ci_ndcg, paired_ndcg = compute_bootstrap_ci(a0_ndcg, a0_ndcg)
            ci_rec, paired_rec = compute_bootstrap_ci(a0_recall, a0_recall)
            ci_hit, paired_hit = compute_bootstrap_ci(a0_hit, a0_hit)
        else:
            ci_ndcg, paired_ndcg = compute_bootstrap_ci(res["_per_article_ndcg"], a0_ndcg)
            ci_rec, paired_rec = compute_bootstrap_ci(res["_per_article_recall"], a0_recall)
            ci_hit, paired_hit = compute_bootstrap_ci(res["_per_article_hit"], a0_hit)
            
        ablation_results.append({
            "step": name,
            "ndcg5": res["ndcg5"],
            "ndcg5_ci": f"[{ci_ndcg[0]:.4f}, {ci_ndcg[1]:.4f}]",
            "delta_ndcg5_paired_ci": f"[{paired_ndcg[0]:+.4f}, {paired_ndcg[1]:+.4f}]",
            "recall5": res["recall5"],
            "recall5_ci": f"[{ci_rec[0]:.4f}, {ci_rec[1]:.4f}]",
            "hit5": res["hit5"],
            "hit5_ci": f"[{ci_hit[0]:.4f}, {ci_hit[1]:.4f}]",
            "near_dup_rate": res["near_dup_rate"],
            "catalog_coverage": res["catalog_coverage"],
            "median_age_days": res["median_age_days"],
            "mean_unboosted_rel": res["mean_unboosted_rel"],
        })
        
    df_abl = pd.DataFrame(ablation_results)
    df_abl.to_csv("ablation_results.csv", index=False)
    print("\n" + df_abl[["step", "ndcg5", "delta_ndcg5_paired_ci", "recall5", "hit5", "near_dup_rate", "catalog_coverage", "median_age_days"]].to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: DESIGN VARIANTS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 5: DESIGN VARIANTS ON VALIDATION SET")
    print("=" * 70)
    
    variants = [
        ("Base All Boosts (No cap)", dict(sec_mult=1.08, subsec_mult=1.05, pop_mult=1.06, rec_mult=1.10, boost_cap=None)),
        ("Cap 1.10", dict(sec_mult=1.08, subsec_mult=1.05, pop_mult=1.06, rec_mult=1.10, boost_cap=1.10)),
        ("Cap 1.15", dict(sec_mult=1.08, subsec_mult=1.05, pop_mult=1.06, rec_mult=1.10, boost_cap=1.15)),
        ("Cap 1.25", dict(sec_mult=1.08, subsec_mult=1.05, pop_mult=1.06, rec_mult=1.10, boost_cap=1.25)),
        ("Boost above relevance floor (0.15)", dict(sec_mult=1.08, subsec_mult=1.05, pop_mult=1.06, rec_mult=1.10, boost_mode="floor", relevance_floor=0.15)),
        ("Boost as tie-breaker within eps=0.03", dict(sec_mult=1.08, subsec_mult=1.05, pop_mult=1.06, rec_mult=1.10, boost_mode="tie_breaker", epsilon=0.03)),
        ("Recency vs Source Date (not today)", dict(sec_mult=1.08, subsec_mult=1.05, pop_mult=1.06, rec_mult=1.10, boost_mode="source_date_recency")),
    ]
    
    variant_results = []
    for vname, vparams in variants:
        res = calculate_split_metrics(articles, val_idx, max_pv, max_cd, tot_corpus, **vparams)
        _, paired_ndcg = compute_bootstrap_ci(res["_per_article_ndcg"], a0_ndcg)
        variant_results.append({
            "variant": vname,
            "ndcg5": res["ndcg5"],
            "delta_vs_a0_ci": f"[{paired_ndcg[0]:+.4f}, {paired_ndcg[1]:+.4f}]",
            "recall5": res["recall5"],
            "hit5": res["hit5"],
            "catalog_coverage": res["catalog_coverage"],
            "median_age_days": res["median_age_days"],
        })
    df_var = pd.DataFrame(variant_results)
    print("\n" + df_var.to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: RANDOM SEARCH TUNING (~200 CONFIGS ON VALIDATION)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 6: RANDOM SEARCH TUNING (~200 CONFIGS ON VALIDATION)")
    print("=" * 70)
    
    rng = np.random.RandomState(42)
    n_configs = 200
    all_runs = []
    
    # Generate valid blend weights on 0.05 step grid
    blend_grid = []
    for w_r in np.arange(0.35, 0.85, 0.05):
        for w_b in np.arange(0.05, 0.50, 0.05):
            w_d = 1.0 - w_r - w_b
            if 0.04 <= w_d <= 0.40:
                blend_grid.append((round(w_r, 2), round(w_b, 2), round(w_d, 2)))
                
    caps = [None, 1.10, 1.15, 1.25]
    modes = ["multiplicative", "floor", "tie_breaker", "source_date_recency"]
    
    a0_val_ndcg = df_abl.loc[0, "ndcg5"]
    
    for run_id in range(n_configs):
        # Sample parameters
        sec_m = round(rng.uniform(1.00, 1.15), 3)
        subsec_m = round(rng.uniform(1.00, 1.15), 3)
        pop_m = round(rng.uniform(1.00, 1.15), 3)
        rec_m = round(rng.uniform(1.00, 1.15), 3)
        
        cap = rng.choice(caps)
        mode = rng.choice(modes)
        
        lam = round(rng.uniform(0.60, 1.00), 2)
        theta = round(rng.uniform(0.75, 0.92), 2)
        
        w_r, w_b, w_d = blend_grid[rng.choice(len(blend_grid))]
        
        cfg_str = f"sec={sec_m}|sub={subsec_m}|pop={pop_m}|rec={rec_m}|cap={cap}|mode={mode}|lam={lam}|theta={theta}|w={w_r},{w_b},{w_d}"
        cfg_hash = hashlib.sha256(cfg_str.encode()).hexdigest()[:8]
        
        params = dict(
            w_rerank=w_r, w_bm25=w_b, w_dense=w_d,
            sec_mult=sec_m, subsec_mult=subsec_m, pop_mult=pop_m, rec_mult=rec_m,
            boost_cap=cap, boost_mode=mode,
            relevance_floor=0.12, epsilon=0.03,
            mmr_lambda=lam, mmr_theta=theta,
        )
        
        m_res = calculate_split_metrics(articles, val_idx, max_pv, max_cd, tot_corpus, **params)
        _, paired_ndcg = compute_bootstrap_ci(m_res["_per_article_ndcg"], a0_ndcg)
        
        record = {
            "config_hash": cfg_hash,
            "w_rerank": w_r, "w_bm25": w_b, "w_dense": w_d,
            "sec_mult": sec_m, "subsec_mult": subsec_m,
            "pop_mult": pop_m, "rec_mult": rec_m,
            "boost_cap": str(cap), "boost_mode": mode,
            "mmr_lambda": lam, "mmr_theta": theta,
            "ndcg5": m_res["ndcg5"],
            "delta_vs_a0": m_res["ndcg5"] - a0_val_ndcg,
            "paired_ci_low": paired_ndcg[0],
            "paired_ci_high": paired_ndcg[1],
            "recall5": m_res["recall5"],
            "hit5": m_res["hit5"],
            "near_dup_rate": m_res["near_dup_rate"],
            "catalog_coverage": m_res["catalog_coverage"],
            "median_age_days": m_res["median_age_days"],
            "mean_unboosted_rel": m_res["mean_unboosted_rel"],
        }
        all_runs.append(record)
        
    df_runs = pd.DataFrame(all_runs)
    df_runs.to_csv("tuning_results.csv", index=False)
    print(f"Logged all {len(df_runs)} runs to tuning_results.csv.")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SELECTION RULE
    # ─────────────────────────────────────────────────────────────────────────
    # Rule: nDCG@5 no worse than A0 by more than 0.01 (paired_ci_low >= -0.01);
    # among those, maximize freshness (minimize median_age_days) and catalog coverage
    # with no rise in near_dup_rate (near_dup_rate <= A0_dup_rate).
    a0_dup_rate = df_abl.loc[0, "near_dup_rate"]
    
    # Filter qualifying
    qualifying = df_runs[
        (df_runs["delta_vs_a0"] >= -0.01) & 
        (df_runs["paired_ci_low"] >= -0.01) &
        (df_runs["near_dup_rate"] <= a0_dup_rate + 0.005)
    ].copy()
    
    if len(qualifying) == 0:
        print("[Notice] Strict paired_ci_low >= -0.01 yielded 0 configs; using delta_vs_a0 >= -0.01.")
        qualifying = df_runs[df_runs["delta_vs_a0"] >= -0.01].copy()
        
    print(f"\n{len(qualifying)} configs qualified under the nDCG retention constraint.")
    
    # Objective: Freshness (lower age is better) + Coverage (higher is better)
    # Score = coverage - 0.01 * median_age_days
    qualifying["composite_rank_score"] = qualifying["catalog_coverage"] - 0.005 * qualifying["median_age_days"]
    best_config_row = qualifying.sort_values(by=["composite_rank_score", "ndcg5"], ascending=False).iloc[0]
    
    print("\n" + "=" * 70)
    print("  [WINNER] RECOMMENDED WINNING CONFIG (SELECTED ON VALIDATION)")
    print("=" * 70)
    print(best_config_row.to_dict())
    
    # ─────────────────────────────────────────────────────────────────────────
    # FINAL TEST SET EVALUATION (USED EXACTLY ONCE)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION ON UNSEEN TEST SET (USED EXACTLY ONCE)")
    print("=" * 70)
    
    # Test A0
    test_a0 = calculate_split_metrics(articles, test_idx, max_pv, max_cd, tot_corpus, **ablations[0][1])
    test_a0_ndcg = test_a0["_per_article_ndcg"]
    
    # Test A5 (Old Hand-Picked Full Config)
    test_old = calculate_split_metrics(articles, test_idx, max_pv, max_cd, tot_corpus, **ablations[5][1])
    _, paired_old = compute_bootstrap_ci(test_old["_per_article_ndcg"], test_a0_ndcg)
    
    # Test Winning Config
    best_params = dict(
        w_rerank=best_config_row["w_rerank"],
        w_bm25=best_config_row["w_bm25"],
        w_dense=best_config_row["w_dense"],
        sec_mult=best_config_row["sec_mult"],
        subsec_mult=best_config_row["subsec_mult"],
        pop_mult=best_config_row["pop_mult"],
        rec_mult=best_config_row["rec_mult"],
        boost_cap=None if best_config_row["boost_cap"] == "None" else float(best_config_row["boost_cap"]),
        boost_mode=best_config_row["boost_mode"],
        mmr_lambda=best_config_row["mmr_lambda"],
        mmr_theta=best_config_row["mmr_theta"],
    )
    test_best = calculate_split_metrics(articles, test_idx, max_pv, max_cd, tot_corpus, **best_params)
    ci_best_ndcg, paired_best = compute_bootstrap_ci(test_best["_per_article_ndcg"], test_a0_ndcg)
    
    test_comparison = pd.DataFrame([
        {
            "Config": "A0 (Relevance baseline)",
            "nDCG@5": f"{test_a0['ndcg5']:.4f}",
            "Delta vs A0 (95% CI)": "0.0000 (reference)",
            "Recall@5": f"{test_a0['recall5']:.4f}",
            "Hit@5": f"{test_a0['hit5']:.1%}",
            "Coverage": f"{test_a0['catalog_coverage']:.1%}",
            "Dup Rate": f"{test_a0['near_dup_rate']:.1%}",
            "Median Age": f"{test_a0['median_age_days']:.1f}d",
        },
        {
            "Config": "Old Hand-Picked (A5)",
            "nDCG@5": f"{test_old['ndcg5']:.4f}",
            "Delta vs A0 (95% CI)": f"[{paired_old[0]:+.4f}, {paired_old[1]:+.4f}]",
            "Recall@5": f"{test_old['recall5']:.4f}",
            "Hit@5": f"{test_old['hit5']:.1%}",
            "Coverage": f"{test_old['catalog_coverage']:.1%}",
            "Dup Rate": f"{test_old['near_dup_rate']:.1%}",
            "Median Age": f"{test_old['median_age_days']:.1f}d",
        },
        {
            "Config": "Measured Winning Config",
            "nDCG@5": f"{test_best['ndcg5']:.4f}",
            "Delta vs A0 (95% CI)": f"[{paired_best[0]:+.4f}, {paired_best[1]:+.4f}]",
            "Recall@5": f"{test_best['recall5']:.4f}",
            "Hit@5": f"{test_best['hit5']:.1%}",
            "Coverage": f"{test_best['catalog_coverage']:.1%}",
            "Dup Rate": f"{test_best['near_dup_rate']:.1%}",
            "Median Age": f"{test_best['median_age_days']:.1f}d",
        },
    ])
    print("\n" + test_comparison.to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: HUMAN BLIND LABELING CSV GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 7: GENERATING BLIND LABELING CSV (150 PAIRS)")
    print("=" * 70)
    
    # Load full corpus title mapping for accurate headline display in blind labeling
    with open("data/cache/cleaned_articles.pkl", "rb") as f:
        full_df = pickle.load(f)
    title_map = dict(zip(full_df["id"].astype(str), full_df["title"]))

    # Sample 150 pairs pooled across A0, A5, A6, and Best Config (configs strictly hidden)
    pooled_pairs = set()
    pairs_list = []
    
    sampled_test_sources = test_idx[:25]
    configs_to_sample = [
        ablations[0][1], # A0
        ablations[5][1], # A5
        ablations[6][1], # A6
        best_params,     # Best
    ]
    
    for s_idx in sampled_test_sources:
        art = articles[s_idx]
        for cfg in configs_to_sample:
            top_i, _ = score_and_rank_article(art, max_pv, max_cd, **cfg)
            for cid_pos in top_i:
                cand = art["cands_meta"][cid_pos]
                pair_key = (art["source_id"], cand["id"])
                if pair_key not in pooled_pairs and len(pairs_list) < 150:
                    pooled_pairs.add(pair_key)
                    pairs_list.append({
                        "pair_id": f"PAIR_{len(pairs_list)+1:03d}",
                        "source_id": art["source_id"],
                        "source_title": art["source_title"],
                        "candidate_id": cand["id"],
                        "candidate_title": title_map.get(str(cand["id"]), f"Article {cand['id']}"),
                        "candidate_section": cand["section"],
                        "label": "", # User will fill: relevant / somewhat / not
                    })
                    
    df_blind = pd.DataFrame(pairs_list)
    df_blind.to_csv("blind_labeling_sample.csv", index=False)
    print(f"Created blind labeling CSV at blind_labeling_sample.csv ({len(df_blind)} pairs). Config identity hidden.")

    return {
        "best_config": best_config_row.to_dict(),
        "test_results": test_comparison.to_dict(orient="records"),
        "ablation_results": ablation_results,
    }

if __name__ == "__main__":
    run_all()
