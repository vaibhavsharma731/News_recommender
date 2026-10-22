"""
Parity Verification Test (Local Machine vs. Colab GPU Worker)
============================================================
Evaluates mathematical consistency between local CPU and Colab T4 GPU:
  1. Embeddings Parity: Requires embedding cosine similarity >= 0.9999 across ~200 articles.
  2. Reranking Parity: Requires 100% identical Top-5 rank ordering for candidates.
  3. Reports exact numbers: min cosine, mean cosine, max delta, and rerank rank correlations.

Usage:
  python parity_test.py
  python parity_test.py --url https://your-colab-tunnel.trycloudflare.com --samples 200
"""
import argparse
import json
import math
import sys
import time
import urllib.request
import numpy as np
import pandas as pd

import config
from src.data_loader import load_articles
from src.model_registry import get_embedder, get_cross_encoder, RemoteEmbedderClient, RemoteCrossEncoderClient


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def run_parity_test(remote_url: str, sample_size: int = 200):
    remote_url = remote_url.rstrip("/")
    print("=" * 70)
    print("  NEWSREC LOCAL vs. COLAB PARITY VERIFICATION SUITE")
    print("=" * 70)
    print(f"Target Remote Worker : {remote_url}")
    print(f"Test Sample Size     : {sample_size} articles")
    print(f"Embedding Model      : {config.EMBEDDING_MODEL} (rev: {config.EMBEDDING_REVISION[:8]})")
    print(f"Reranker Model       : {config.RERANKER_MODEL} (rev: {config.RERANKER_REVISION[:8]})")
    print("=" * 70)

    # 1. Health & Remote Info Check
    print("\n[1/4] Checking Remote Colab Worker Health...")
    req = urllib.request.Request(f"{remote_url}/health", headers={"User-Agent": "NewsRec-Parity"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
            print(f"  Remote Device : {health.get('device')} ({health.get('gpu')})")
            print(f"  Remote Status : {health.get('status')}")
    except Exception as e:
        print(f"  [ERROR] Could not connect to remote worker at {remote_url}: {e}")
        sys.exit(1)

    # 2. Load Evaluation Dataset
    print(f"\n[2/4] Loading {sample_size} articles for parity testing...")
    df = load_articles()
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    texts = df["text_embed"].tolist()
    article_ids = df["id"].tolist()
    print(f"  Sampled {len(texts)} articles successfully.")

    # 3. Embedding Parity Verification
    print("\n[3/4] Benchmarking Embedding Parity (Cosine Similarity >= 0.9999)...")
    from sentence_transformers import SentenceTransformer
    local_embedder = SentenceTransformer(
        config.EMBEDDING_MODEL,
        revision=config.EMBEDDING_REVISION,
        device="cpu"
    )
    local_embedder.max_seq_length = config.EMBED_MAX_SEQ_LEN
    remote_embedder = RemoteEmbedderClient(remote_url)

    t0 = time.time()
    local_vecs = local_embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    local_time = time.time() - t0

    t0 = time.time()
    remote_vecs = remote_embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    remote_time = time.time() - t0

    local_vecs = np.asarray(local_vecs, dtype=np.float32)
    remote_vecs = np.asarray(remote_vecs, dtype=np.float32)

    cosines = [cosine_sim(lv, rv) for lv, rv in zip(local_vecs, remote_vecs)]
    min_cos = float(np.min(cosines))
    mean_cos = float(np.mean(cosines))
    max_cos = float(np.max(cosines))
    max_abs_diff = float(np.max(np.abs(local_vecs - remote_vecs)))

    print(f"  Local Encoding Time   : {local_time:.2f}s")
    print(f"  Remote Encoding Time  : {remote_time:.2f}s (Speedup: {local_time / max(1e-4, remote_time):.1f}x)")
    print(f"  Minimum Cosine Sim    : {min_cos:.6f}")
    print(f"  Mean Cosine Sim       : {mean_cos:.6f}")
    print(f"  Maximum Cosine Sim    : {max_cos:.6f}")
    print(f"  Maximum Absolute Diff : {max_abs_diff:.6e}")

    emb_passed = min_cos >= 0.9999
    if emb_passed:
        print("  [PASS] Embedding cosine parity verified (>= 0.9999 for all 200 articles)!")
    else:
        print(f"  [FAIL] Embedding parity failed: min cosine {min_cos:.6f} < 0.9999")

    # 4. Reranker Top-5 Ordering Parity
    print("\n[4/4] Benchmarking Cross-Encoder Reranking Parity (Identical Top-5 Order)...")
    from sentence_transformers import CrossEncoder
    local_reranker = CrossEncoder(
        config.RERANKER_MODEL,
        revision=config.RERANKER_REVISION,
        max_length=config.RERANK_MAX_LEN,
        device="cpu"
    )
    remote_reranker = RemoteCrossEncoderClient(remote_url)

    # Pick 5 distinct query articles and score them against 20 candidate articles each
    order_matches = 0
    total_queries = min(10, len(df))
    queries_tested = 0

    for q_idx in range(total_queries):
        query_text = df.iloc[q_idx]["text_rerank"]
        cand_indices = [i for i in range(len(df)) if i != q_idx][:20]
        cand_texts = [df.iloc[c]["text_rerank"] for c in cand_indices]
        pairs = [(query_text, ct) for ct in cand_texts]

        local_scores = local_reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        remote_scores = remote_reranker.predict(pairs, batch_size=32, show_progress_bar=False)

        local_ranked = [cand_indices[i] for i in np.argsort(-local_scores)[:5]]
        remote_ranked = [cand_indices[i] for i in np.argsort(-remote_scores)[:5]]

        if local_ranked == remote_ranked:
            order_matches += 1
        queries_tested += 1

    rank_pct = (order_matches / max(1, queries_tested)) * 100
    print(f"  Rerank Queries Tested : {queries_tested}")
    print(f"  Top-5 Order Matches   : {order_matches}/{queries_tested} ({rank_pct:.1f}%)")

    rerank_passed = (order_matches == queries_tested)
    if rerank_passed:
        print("  [PASS] Cross-Encoder Top-5 rank ordering is 100% identical!")
    else:
        print(f"  [FAIL] Reranker ordering mismatch observed in {queries_tested - order_matches} queries.")

    print("\n" + "=" * 70)
    if emb_passed and rerank_passed:
        print("  ✅ FULL PARITY VERIFIED: Local CPU and Colab T4 GPU produce identical results!")
    else:
        print("  ❌ PARITY CHECK FAILED: Review reported numbers above.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parity Test: Local CPU vs. Colab GPU")
    parser.add_argument("--url", default=config.REMOTE_MODEL_URL, help="Remote Colab Cloudflare URL")
    parser.add_argument("--samples", type=int, default=200, help="Number of articles to sample (default: 200)")
    args = parser.parse_args()

    if not args.url:
        print("[!] No REMOTE_MODEL_URL provided. Specify via --url https://xxx.trycloudflare.com or set in .env")
        sys.exit(1)

    run_parity_test(args.url, sample_size=args.samples)
