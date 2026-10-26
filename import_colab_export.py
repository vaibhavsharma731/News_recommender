"""
Safe Colab Export Importer
==========================
Validates and imports neural vector embeddings and precomputed recommendations
exported from Google Colab GPU into PostgreSQL.

Validation guarantees:
  1. Pipeline config hash in the export strictly equals the active config fingerprint.
  2. Article IDs in the export exist in the PostgreSQL database.
  3. Every vector dimension is strictly 768.
  4. Zero NaNs or Infinite values in embeddings and recommendation scores.
  5. Vector L2 norms are normalized to approximately 1.0 (0.990 <= norm <= 1.010).
  6. Atomic transaction: validation reads and all article/recommendation writes run in a
     single explicit transaction. If ANY row fails validation, nothing is committed.

Usage:
  python import_colab_export.py path/to/newsrec_export.json
"""
import argparse
import json
import math
import numpy as np

import config
from db.connection import SessionLocal
from db.models import Article, ArticleRecommendation


def validate_and_import(export_path: str, force_hash: bool = False):
    print("=" * 70)
    print("  SAFE COLAB EXPORT VALIDATION & ATOMIC IMPORTER")
    print("=" * 70)
    print(f"Loading export file: {export_path}")

    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Config Hash Check
    current_hash = config.get_pipeline_version()
    exported_hash = data.get("config_hash", "")
    print(f"  Current Config Hash  : {current_hash}")
    print(f"  Exported Config Hash : {exported_hash}")

    if exported_hash != current_hash and not force_hash:
        raise ValueError(
            f"Config hash mismatch! Exported hash '{exported_hash}' does not match "
            f"current pipeline config '{current_hash}'. Models or budgeting parameters differ. "
            f"(Use --force-hash to override if intentional)."
        )

    articles_data = data.get("articles", {})
    if not articles_data:
        raise ValueError("Export file contains 0 articles.")
    print(f"  Total Articles to Import : {len(articles_data)}")

    # 2. Database Pre-Validation
    # NOTE: SQLAlchemy 2.0 auto-begins a transaction on the first query. The whole
    # read+write therefore runs inside ONE explicit transaction that we open up front,
    # so the validation reads and the writes commit atomically (nothing to roll back
    # separately). Calling db.begin() after a query would raise InvalidRequestError.
    with SessionLocal.begin() as db:
        existing_ids = {r[0] for r in db.query(Article.id).all()}
        print(f"  Existing DB Articles     : {len(existing_ids)}")

        # Validate article IDs
        missing_ids = [aid for aid in articles_data.keys() if aid not in existing_ids]
        if missing_ids:
            raise ValueError(
                f"ID mismatch: {len(missing_ids)} articles in export do not exist in the database! "
                f"Sample missing: {missing_ids[:5]}. Please ingest article metadata first."
            )

        # 3. Vector & Data Integrity Validation
        print("\nValidating vectors (Dimension=768, No NaNs, Norm ~= 1.0)...")
        validated_payload = []
        for aid, item in articles_data.items():
            emb = np.asarray(item.get("embedding", []), dtype=np.float32)

            # Check dimension
            if len(emb) != 768:
                raise ValueError(f"Article {aid} has invalid embedding dimension: {len(emb)} (expected 768).")

            # Check NaNs and Infs
            if np.isnan(emb).any() or np.isinf(emb).any():
                raise ValueError(f"Article {aid} embedding contains NaN or Inf values!")

            # Check L2 Norm
            norm = float(np.linalg.norm(emb))
            if not (0.990 <= norm <= 1.010):
                raise ValueError(f"Article {aid} vector norm ({norm:.4f}) is not normalized to ~1.0!")

            # Validate recommendations if present
            recs = item.get("recommendations", [])
            for rec in recs:
                score = rec.get("score")
                rerank = rec.get("rerank_score")
                if score is not None and (math.isnan(score) or math.isinf(score)):
                    raise ValueError(f"Article {aid} recommendation contains NaN/Inf score!")
                if rerank is not None and (math.isnan(rerank) or math.isinf(rerank)):
                    raise ValueError(f"Article {aid} recommendation contains NaN/Inf rerank score!")

            validated_payload.append((aid, emb.tolist(), recs))

        print(f"  All {len(validated_payload)} vectors successfully passed mathematical integrity checks!")

        # 4. Write phase -- still inside the same transaction opened above
        print("\nExecuting atomic database transaction...")
        for aid, emb_list, recs in validated_payload:
            art = db.query(Article).filter(Article.id == aid).first()
            if art:
                art.embedding = emb_list

            # If precomputed recommendations are included, replace them
            if recs:
                db.query(ArticleRecommendation).filter(ArticleRecommendation.article_id == aid).delete()
                for rank_idx, rec in enumerate(recs[:10], start=1):
                    db.add(ArticleRecommendation(
                        article_id=aid,
                        recommended_article_id=rec["target_id"],
                        rank=rank_idx,
                        score=float(rec.get("score", 0.0)),
                        rerank_score=float(rec.get("rerank_score", 0.0)) if rec.get("rerank_score") is not None else None,
                        reason=rec.get("reason", "Colab GPU precomputed recommendation"),
                        pipeline_version=exported_hash,
                        method="colab_gpu_precompute"
                    ))

    print("  Database transaction committed successfully!")
    print("=" * 70)
    print("  IMPORT SUCCESSFUL: All embeddings and recommendations live in PostgreSQL.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safely import verified Colab export into PostgreSQL")
    parser.add_argument("file", help="Path to newsrec_export.json")
    parser.add_argument("--force-hash", action="store_true", help="Bypass config hash verification")
    args = parser.parse_args()

    validate_and_import(args.file, force_hash=args.force_hash)
