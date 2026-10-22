"""
freeze_pool.py
==============
Retrieves top-20 candidates for all 290 ground-truth source articles once,
computes raw signals (BM25, dense, CrossEncoder rerank, metadata, 20x20 similarity matrix),
and caches the pool to data/cache/frozen_pool.pkl with incremental checkpointing.
Zero model calls will be needed for any subsequent tuning or ablation sweeps.
"""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

import config
from src.retrieval import reciprocal_rank_fusion, tokenize
from src.recommender import BM25Index

def freeze_pool(sample_size: int = None, max_length: int = 192):
    print("=" * 70, flush=True)
    print("  STEP 1: FREEZE THE CANDIDATE POOL (TOP-20 PER SOURCE ARTICLE)", flush=True)
    print("=" * 70, flush=True)
    
    t0 = time.time()
    cache_path = "data/cache/frozen_pool.pkl"
    ckpt_path = "data/cache/frozen_pool_ckpt.pkl"
    
    if os.path.exists(cache_path):
        print(f"Found existing complete frozen pool at {cache_path}. Loading...", flush=True)
        with open(cache_path, "rb") as f:
            pool = pickle.load(f)
        print(f"Loaded {len(pool['articles'])} source articles from cache in {time.time()-t0:.2f}s.", flush=True)
        return pool

    print("Loading cleaned articles and precomputed vectors...", flush=True)
    with open("data/cache/cleaned_articles.pkl", "rb") as f:
        df = pickle.load(f)
    vecs = np.load("data/cache/articles_2467_vectors.npy")
    
    print("Building BM25 index...", flush=True)
    bm25 = BM25Index(df["text_bm25"])
    
    max_page_views = int(df["pageViews"].max()) if "pageViews" in df.columns else 1
    max_corpus_date = df["date"].max()
    
    # Ground truth articles (source articles with editorial links)
    gt_sources = []
    for i, (_, row) in enumerate(df.iterrows()):
        links = row["linked_ids"]
        if isinstance(links, list) and len(links) > 0:
            gt_sources.append((i, row))
            
    if sample_size and sample_size < len(gt_sources):
        # Deterministic sample if sample_size is requested
        np.random.seed(42)
        idx_sample = sorted(np.random.choice(len(gt_sources), size=sample_size, replace=False))
        gt_sources = [gt_sources[i] for i in idx_sample]
        print(f"Sampled {len(gt_sources)} source articles (seed=42).", flush=True)
    else:
        print(f"Total source articles with ground truth: {len(gt_sources)}", flush=True)
    
    # 1. Retrieve top 20 candidates per source article
    print("Retrieving BM25 + Dense fused top-20 candidates...", flush=True)
    t_ret = time.time()
    
    pool_records = []
    all_rerank_pairs = []
    
    for q_pos, s_row in gt_sources:
        title_toks = tokenize(s_row["title"])
        body_toks = tokenize(f"{s_row['description']} {s_row['body']}")
        top_body = sorted(set(body_toks), key=lambda t: -bm25.idf.get(t, 0.0))[:15]
        focused_text = " ".join(title_toks + top_body)
        
        bm25_all = bm25.scores(focused_text)
        dense_all = vecs @ vecs[q_pos]
        
        # Block source itself
        bm25_all[q_pos] = -1e9
        dense_all[q_pos] = -1e9
        
        # Rank top 50
        b_top50 = [int(i) for i in np.argsort(-bm25_all)[:50] if bm25_all[i] > 1e-9]
        d_top50 = [int(i) for i in np.argsort(-dense_all)[:50]]
        
        # RRF fusion
        fused = reciprocal_rank_fusion([b_top50, d_top50], k=config.RRF_K)
        top20_cand_indices = sorted(fused, key=fused.get, reverse=True)[:config.N_CANDIDATES]
        
        # Collect candidate data
        cand_bm25 = bm25_all[top20_cand_indices]
        bm25_max = cand_bm25.max() if len(cand_bm25) and cand_bm25.max() > 0 else 1.0
        cand_bm25_norm = cand_bm25 / bm25_max
        
        cand_dense = dense_all[top20_cand_indices]
        dense_max = cand_dense.max() if len(cand_dense) and cand_dense.max() > 0 else 1.0
        cand_dense_norm = cand_dense / dense_max
        
        # Candidate similarity matrix (20x20)
        cand_vecs = vecs[top20_cand_indices]
        cand_sim_mat = cand_vecs @ cand_vecs.T
        
        # Pair text for reranker
        s_rerank_text = s_row["text_rerank"]
        c_passages = df["text_rerank"].iloc[top20_cand_indices].tolist()
        
        pair_start_idx = len(all_rerank_pairs)
        for p in c_passages:
            all_rerank_pairs.append((s_rerank_text, p))
            
        cands_meta = []
        for c_idx in top20_cand_indices:
            c_row = df.iloc[c_idx]
            cands_meta.append({
                "id": str(c_row["id"]),
                "section": str(c_row["section"]),
                "subsection": str(c_row["subsection"]),
                "pageViews": int(c_row["pageViews"]),
                "date": c_row["date"],
                "is_live": bool(c_row["is_live"]),
                "pos": c_idx,
            })
            
        pool_records.append({
            "source_id": str(s_row["id"]),
            "source_title": str(s_row["title"]),
            "source_section": str(s_row["section"]),
            "source_subsection": str(s_row["subsection"]),
            "source_pageViews": int(s_row["pageViews"]),
            "source_date": s_row["date"],
            "source_pos": q_pos,
            "linked_ids": set(str(x) for x in s_row["linked_ids"]),
            "cand_indices": top20_cand_indices,
            "cand_bm25_raw": cand_bm25,
            "cand_bm25_norm": cand_bm25_norm,
            "cand_dense_raw": cand_dense,
            "cand_dense_norm": cand_dense_norm,
            "cand_sim_mat": cand_sim_mat,
            "cands_meta": cands_meta,
            "pair_slice": (pair_start_idx, pair_start_idx + len(c_passages)),
        })
        
    print(f"Retrieved candidate pools for {len(pool_records)} articles in {time.time()-t_ret:.2f}s.", flush=True)
    print(f"Total pairs to score with Cross-Encoder: {len(all_rerank_pairs)}", flush=True)
    
    # 2. Check for checkpoint
    completed_probs = []
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "rb") as f:
                completed_probs = pickle.load(f)
            print(f"Resuming from checkpoint: {len(completed_probs)}/{len(all_rerank_pairs)} pairs already scored.", flush=True)
        except Exception as e:
            print(f"Could not load checkpoint ({e}), starting fresh.", flush=True)
            completed_probs = []

    # Score remaining pairs
    remaining_pairs = all_rerank_pairs[len(completed_probs):]
    if remaining_pairs:
        print(f"Scoring {len(remaining_pairs)} remaining pairs with CrossEncoder (threads=8, max_len={max_length})...", flush=True)
        torch.set_num_threads(8)
        ce = CrossEncoder(config.RERANKER_MODEL, max_length=max_length)
        
        batch_size = 128
        chunk_save = 512  # Save checkpoint every 512 pairs (~4 articles)
        
        t_start = time.time()
        for i in range(0, len(remaining_pairs), chunk_save):
            sub_pairs = remaining_pairs[i : i + chunk_save]
            with torch.inference_mode():
                sub_logits = ce.predict(sub_pairs, batch_size=batch_size, show_progress_bar=False)
            sub_probs = 1.0 / (1.0 + np.exp(-np.clip(sub_logits, -30.0, 30.0)))
            completed_probs.extend(sub_probs.tolist())
            
            # Save checkpoint
            with open(ckpt_path, "wb") as f:
                pickle.dump(completed_probs, f)
                
            elapsed = time.time() - t_start
            rate = (i + len(sub_pairs)) / max(elapsed, 0.001)
            eta = (len(remaining_pairs) - (i + len(sub_pairs))) / max(rate, 0.001)
            print(f"  • Progress: {len(completed_probs)}/{len(all_rerank_pairs)} pairs ({len(completed_probs)/len(all_rerank_pairs)*100:.1f}%) | Rate: {rate:.1f} pairs/s | ETA: {eta/60:.1f} min", flush=True)

    probs = np.asarray(completed_probs, dtype=np.float32)
    
    # Assign scores back to each pool record
    for rec in pool_records:
        start, end = rec["pair_slice"]
        rec["cand_rerank_raw"] = probs[start:end]
        del rec["pair_slice"]
        
    frozen_payload = {
        "articles": pool_records,
        "max_page_views": max_page_views,
        "max_corpus_date": max_corpus_date,
        "created_at": time.time(),
        "total_corpus_size": len(df),
    }
    
    with open(cache_path, "wb") as f:
        pickle.dump(frozen_payload, f)
        
    if os.path.exists(ckpt_path):
        try:
            os.remove(ckpt_path)
        except Exception:
            pass
            
    print(f"\n[SUCCESS] Frozen pool saved to {cache_path} ({len(pool_records)} articles, size: {os.path.getsize(cache_path)/1024:.1f} KB in {time.time()-t0:.1f}s).", flush=True)
    return frozen_payload

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None, help="Sample size of source articles (default: all 290)")
    parser.add_argument("--max-len", type=int, default=192, help="CrossEncoder max token length")
    args = parser.parse_args()
    freeze_pool(sample_size=args.samples, max_length=args.max_len)
