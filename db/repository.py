"""
Database Repository providing clean CRUD, recommendation persistence,
and ingestion job management.
"""
from typing import List, Optional, Dict, Any, Union
import re
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, or_, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .models import Article, ArticleRecommendation, IngestionJob
from src.scoring import parse_ids


class ArticleRepository:
    """Encapsulates all PostgreSQL queries."""

    # ──────────────────────────────────────────────────────────────────────────
    # Article CRUD
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_article(db: Session, article_id: str, model_version: str = "384d") -> Optional[Any]:
        return db.query(Article).filter(Article.id == article_id).first()

    @staticmethod
    def count_articles(
        db: Session,
        model_version: str = "384d",
        section: Optional[str] = None,
        search: Optional[str] = None,
    ) -> int:
        q = db.query(func.count(Article.id))
        if section:
            q = q.filter(Article.section.ilike(f"%{section}%"))
        if search:
            search_pattern = f"%{search}%"
            q = q.filter(
                or_(
                    Article.title.ilike(search_pattern),
                    Article.description.ilike(search_pattern),
                )
            )
        return q.scalar() or 0

    @staticmethod
    def count_embedded_articles(db: Session) -> int:
        return db.query(func.count(Article.id)).filter(Article.embedding.isnot(None)).scalar() or 0

    @staticmethod
    def list_articles(
        db: Session,
        limit: int = 50,
        offset: int = 0,
        section: Optional[str] = None,
        search: Optional[str] = None,
        model_version: str = "384d",
    ) -> List[Any]:
        query = db.query(Article)
        if section:
            query = query.filter(Article.section.ilike(f"%{section}%"))
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    Article.title.ilike(search_pattern),
                    Article.description.ilike(search_pattern)
                )
            )
        return query.order_by(desc(Article.date).nulls_last()).offset(offset).limit(limit).all()

    @staticmethod
    def list_articles_ranked_by_confidence(
        db: Session,
        limit: int = 50,
        offset: int = 0,
        section: Optional[str] = None,
        model_version: str = "384d",
    ) -> List[Any]:
        """
        List articles ordered by how CONFIDENT their precomputed recommendations are.
        """
        confidence = func.coalesce(func.avg(ArticleRecommendation.score), 0.0).label("confidence")
        rec_count = func.count(ArticleRecommendation.id).label("rec_count")

        query = db.query(Article).outerjoin(
            ArticleRecommendation, ArticleRecommendation.article_id == Article.id
        )
        if section:
            query = query.filter(Article.section.ilike(f"%{section}%"))
        return (
            query.group_by(Article.id)
            .order_by(desc(confidence), desc(rec_count), desc(Article.date).nulls_last(), Article.id)
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_sections(db: Session, model_version: str = "384d") -> List[str]:
        results = db.query(Article.section).distinct().filter(Article.section.isnot(None)).all()
        return sorted([r[0] for r in results if r[0]])

    @staticmethod
    def get_trending_in_section(
        db: Session,
        section: str,
        exclude_ids: List[str] = (),
        ref_date=None,
        window_days: float = 7.0,
        limit: int = 5,
        model_version: str = "384d",
    ) -> List[Any]:
        """
        Popular stories from one section, for the graceful-degradation list.
        """
        query = db.query(Article).filter(Article.section == section)
        ids = [str(i) for i in exclude_ids if i]
        if ids:
            query = query.filter(Article.id.notin_(ids))
        if ref_date is not None:
            query = query.filter(
                Article.date >= ref_date - timedelta(days=window_days),
                Article.date <= ref_date + timedelta(days=window_days),
            )
        return (
            query.order_by(
                desc(Article.page_views).nulls_last(),
                desc(Article.date).nulls_last(),
                Article.id,
            )
            .limit(limit)
            .all()
        )

    @staticmethod
    def corpus_rows(arts: List[Article], camel_keys: bool = False) -> List[dict]:
        """
        ORM articles -> plain rows for the retrieval engines.

        Used by BOTH corpus loaders (ingestion pipeline and live engine reload) so
        a field can never be mapped one way in one place and another way in the
        other. ``camel_keys`` matches the CSV column names the Recommender
        DataFrame uses (``pageViews``); the pipeline works in snake_case.
        """
        rows = []
        for a in arts:
            row = {
                "id": str(a.id),
                "title": str(a.title),
                "description": str(a.description or ""),
                "body": str(a.body or ""),
                "section": str(a.section or ""),
                "subsection": str(a.subsection or ""),
                "date": a.date,
                "url": str(a.url or ""),
                "image": str(a.image or ""),
                "is_live": bool(a.is_live),
                "premium": bool(a.premium),
                "linked_ids": parse_ids(a.linked_ids),
                "text_embed": a.text_embed or f"{a.title}. {a.description or ''}",
                "text_bm25": a.text_bm25 or f"{a.title}. {a.title}. {a.title}. {a.description or ''}. {a.body or ''}",
                "text_rerank": a.text_rerank or f"{a.title}. {a.description or ''}.",
            }
            views, users = int(a.page_views or 0), int(a.unique_users or 0)
            if camel_keys:
                row["pageViews"], row["uniqueUsers"] = views, users
                row["created_at"] = a.created_at.isoformat() if a.created_at else ""
            else:
                row["page_views"], row["unique_users"] = views, users
            rows.append(row)
        return rows

    @staticmethod
    def bulk_upsert_articles_fast(db: Session, rows: List[dict], chunk: int = 500):
        """One INSERT ... ON CONFLICT per chunk instead of a SELECT per article."""
        if not rows:
            return
        update_cols = [column for column in rows[0].keys() if column != "id"]
        for i in range(0, len(rows), chunk):
            stmt = pg_insert(Article).values(rows[i:i + chunk])
            set_ = {column: stmt.excluded[column] for column in update_cols}
            set_["updated_at"] = func.now()
            db.execute(stmt.on_conflict_do_update(index_elements=[Article.id], set_=set_))
        db.commit()

    @staticmethod
    def bulk_set_embeddings(db: Session, ids: List[str], vectors, chunk: int = 500):
        for i in range(0, len(ids), chunk):
            db.bulk_update_mappings(Article, [
                {"id": article_id, "embedding": vector.tolist()}
                for article_id, vector in zip(ids[i:i + chunk], vectors[i:i + chunk])
            ])
        db.commit()

    # ──────────────────────────────────────────────────────────────────────────
    # Precomputed Recommendations
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def save_recommendations(
        db: Session,
        article_id: str,
        recommendations: List[Dict[str, Any]],
        commit: bool = True,
    ):
        """
        Atomically saves precomputed recommendations for an article.
        recommendations: list of dicts with keys:
            'recommended_article_id', 'rank', 'score', 'rerank_score', 'method', 'reason'
        'rerank_score' is the raw cross-encoder probability for this (source -> recommended)
        pair, computed with the source article as the query. It is used by the incremental
        reverse-push to compare a new article against an existing article's stored entries.
        """
        db.query(ArticleRecommendation).filter(
            ArticleRecommendation.article_id == article_id
        ).delete(synchronize_session=False)
        db.flush()

        for item in recommendations:
            rec = ArticleRecommendation(
                article_id=article_id,
                recommended_article_id=item["recommended_article_id"],
                rank=item["rank"],
                rank_raw=item.get("rank_raw"),
                score=item.get("score", 0.0),
                boosted_score=item.get("boosted_score"),
                rerank_score=item.get("rerank_score"),
                pipeline_version=item.get("pipeline_version"),
                method=item.get("method", "neural_hybrid_rerank"),
                reason=item.get("reason", "High topic and semantic relevance"),
            )
            db.add(rec)
        if commit:
            db.commit()

    @staticmethod
    def get_precomputed_rows(
        db: Session,
        article_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Lightweight read of an article's own stored recommendations (no join).

        Returns dicts with the ids, scores AND the raw rerank_score, which the
        incremental reverse-push needs in order to merge new candidates into an
        existing article's list.
        """
        rows = db.query(ArticleRecommendation).filter(
            ArticleRecommendation.article_id == article_id
        ).order_by(ArticleRecommendation.rank).limit(limit).all()
        return [{
            "recommended_article_id": r.recommended_article_id,
            "rank": r.rank,
            "rank_raw": r.rank_raw,
            "score": r.score,
            "boosted_score": r.boosted_score,
            "rerank_score": r.rerank_score,
            "pipeline_version": r.pipeline_version,
            "method": r.method,
            "reason": r.reason,
        } for r in rows]

    @staticmethod
    def get_precomputed_recommendations(
        db: Session,
        article_id: str,
        limit: int = 10,
        diversify: bool = True,
        model_version: str = "384d"
    ) -> List[Dict[str, Any]]:
        """
        Fast lookup: Retrieves precomputed top recommendations for an article
        in sub-5ms by joining with the articles table.
        If diversify=True, orders by MMR-diversified rank.
        If diversify=False, orders by pure relevance (rank_raw / score) without MMR penalty.
        """
        from sqlalchemy import func
        order_col = ArticleRecommendation.rank if diversify else func.coalesce(ArticleRecommendation.rank_raw, ArticleRecommendation.rank)
        recs = db.query(ArticleRecommendation, Article).join(
            Article, ArticleRecommendation.recommended_article_id == Article.id
        ).filter(
            ArticleRecommendation.article_id == article_id
        ).order_by(
            order_col
        ).limit(limit).all()

        results = []
        for rec, art in recs:
            art_dict = art.to_dict()
            art_dict["rank"] = rec.rank
            art_dict["rank_raw"] = rec.rank_raw or rec.rank
            art_dict["score"] = rec.score
            art_dict["boosted_score"] = rec.boosted_score
            art_dict["rerank_score"] = rec.rerank_score
            art_dict["method"] = rec.method
            art_dict["reason"] = rec.reason or "High topic and semantic relevance"
            art_dict["pipeline_version"] = rec.pipeline_version
            results.append(art_dict)
        return results

    @staticmethod
    def count_total_recommendations(db: Session, model_version: str = "384d") -> int:
        return db.query(func.count(ArticleRecommendation.id)).scalar() or 0

    @staticmethod
    def clear_all_data(db: Session) -> Dict[str, int]:
        """
        Clears all articles, precomputed recommendations, and ingestion jobs from PostgreSQL.
        """
        n_recs = db.query(ArticleRecommendation).delete()
        n_articles = db.query(Article).delete()
        n_jobs = db.query(IngestionJob).delete()
        db.commit()
        return {
            "articles_deleted": n_articles,
            "recommendations_deleted": n_recs,
            "jobs_deleted": n_jobs
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Ingestion Jobs (Editor Side Tracking)
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def create_job(
        db: Session,
        job_id: str,
        filename: str,
        total_articles: int = 0,
        pipeline_version: Optional[str] = None,
        pipeline_type: Optional[str] = None,
    ) -> IngestionJob:
        job = IngestionJob(
            id=job_id,
            filename=filename,
            pipeline_version=pipeline_version,
            pipeline_type=pipeline_type,
            total_articles=total_articles,
            processed_articles=0,
            status="pending",
            message="Job created, awaiting worker..."
        )
        db.add(job)
        db.commit()
        try:
            db.refresh(job)
        except Exception:
            pass
        return job

    @staticmethod
    def update_job(
        db: Session,
        job_id: str,
        status: Optional[str] = None,
        total_articles: Optional[int] = None,
        processed_articles: Optional[int] = None,
        refreshed_articles: Optional[int] = None,
        pipeline_version: Optional[str] = None,
        pipeline_type: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ) -> Optional[IngestionJob]:
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if not job:
            return None
        changes = {
            "status": status,
            "total_articles": total_articles,
            "processed_articles": processed_articles,
            "refreshed_articles": refreshed_articles,
            "pipeline_version": pipeline_version,
            "pipeline_type": pipeline_type,
            "message": message,
            "error": error,
        }
        for field, value in changes.items():
            if value is not None:
                setattr(job, field, value)
        db.commit()
        try:
            db.refresh(job)
        except Exception:
            pass
        return job


    @staticmethod
    def get_job(db: Session, job_id: str) -> Optional[IngestionJob]:
        return db.query(IngestionJob).filter(IngestionJob.id == job_id).first()

    @staticmethod
    def list_recent_jobs(db: Session, limit: int = 10, pipeline_type: Optional[str] = None) -> List[IngestionJob]:
        query = db.query(IngestionJob)
        if pipeline_type:
            query = query.filter(IngestionJob.pipeline_type == pipeline_type)
        return query.order_by(desc(IngestionJob.created_at)).limit(limit).all()

    @staticmethod
    def get_ingestion_summary(db: Session) -> Dict[str, Any]:
        """Aggregate upload history: job counts and articles/recommendations ingested."""
        done = IngestionJob.status == "completed"
        row = db.query(
            func.count(IngestionJob.id),
            func.count(IngestionJob.id).filter(done),
            func.count(IngestionJob.id).filter(IngestionJob.status == "failed"),
            func.coalesce(func.sum(IngestionJob.total_articles).filter(done), 0),
            func.coalesce(func.sum(IngestionJob.refreshed_articles).filter(done), 0),
        ).one()
        keys = ("total_jobs", "completed_jobs", "failed_jobs",
                "articles_ingested", "refreshed_articles")
        return {key: int(value or 0) for key, value in zip(keys, row)}

    @staticmethod
    def get_active_job(db: Session) -> Optional[IngestionJob]:
        return (
            db.query(IngestionJob)
            .filter(IngestionJob.status.in_(["processing", "pending"]))
            .order_by(desc(IngestionJob.created_at))
            .first()
        )

    @staticmethod
    def fail_orphaned_jobs(db: Session) -> int:
        """Mark jobs left in pending/processing after a server restart as failed across all schemas."""
        total_failed = 0
        # 1. Clean public schema
        jobs = (
            db.query(IngestionJob)
            .filter(IngestionJob.status.in_(["processing", "pending"]))
            .all()
        )
        for job in jobs:
            job.status = "failed"
            job.error = "Interrupted because the API server restarted."
            job.message = "Processing encountered an error"
        db.commit()
        total_failed += len(jobs)

        # 2. Clean all isolated dataset schemas
        try:
            schemas = db.execute(text("SELECT schema_name FROM public.datasets WHERE schema_name != 'public'")).fetchall()
            for (sch,) in schemas:
                if not sch:
                    continue
                safe_sch = re.sub(r"[^a-zA-Z0-9_]", "", str(sch))
                res = db.execute(text(
                    f"UPDATE {safe_sch}.ingestion_jobs SET status = 'failed', "
                    f"error = 'Interrupted because the API server restarted.', "
                    f"message = 'Processing encountered an error' "
                    f"WHERE status IN ('processing', 'pending')"
                ))
                db.commit()
                total_failed += (res.rowcount or 0)
        except Exception:
            pass

        return total_failed

    @staticmethod
    def retrieve_hybrid_candidates(
        db: Session,
        embedding: Optional[Union[List[float], np.ndarray, str]] = None,
        query_text: Optional[str] = None,
        limit: int = 20,
        n_retrieve: int = 50,
        section: Optional[str] = None,
        min_date: Optional[datetime] = None,
        exclude_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Industrial-Scale Fast Candidate Retrieval directly in PostgreSQL.
        Uses HNSW index on 384-D embeddings + GIN full-text index on text expressions.
        Takes ~20-35ms across 600,000+ articles without loading raw docs into Python RAM.
        """
        vec_str = None
        if embedding is not None:
            if isinstance(embedding, str):
                vec_str = embedding
            elif hasattr(embedding, "tolist"):
                vec_str = "[" + ",".join(str(float(x)) for x in embedding.tolist()) + "]"
            elif isinstance(embedding, (list, tuple)):
                vec_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

        q_clean = (query_text or "").strip()
        if not vec_str and not q_clean:
            return []

        # If only vector is present
        if vec_str and not q_clean:
            sql = """
                SELECT a.id, a.title, a.description, a.section, a.subsection, a.date, a.url, a.image,
                       a.page_views, a.unique_users, a.text_rerank, a.linked_ids,
                       (1.0 - (a.embedding <=> CAST(:v AS vector))) AS dense_score,
                       0.0 AS bm25_score,
                       (1.0 / (60.0 + ROW_NUMBER() OVER (ORDER BY a.embedding <=> CAST(:v AS vector)))) AS rrf_score
                FROM articles a
                WHERE a.embedding IS NOT NULL
                  AND (:exclude_id IS NULL OR a.id != :exclude_id)
                  AND (:section IS NULL OR a.section = :section)
                  AND (:min_date IS NULL OR a.date >= :min_date)
                ORDER BY a.embedding <=> CAST(:v AS vector)
                LIMIT :limit;
            """
            rows = db.execute(
                text(sql),
                {
                    "v": vec_str,
                    "exclude_id": exclude_id,
                    "section": section,
                    "min_date": min_date,
                    "limit": limit,
                }
            ).fetchall()
        # If only text query is present
        elif q_clean and not vec_str:
            sql = """
                SELECT a.id, a.title, a.description, a.section, a.subsection, a.date, a.url, a.image,
                       a.page_views, a.unique_users, a.text_rerank, a.linked_ids,
                       0.0 AS dense_score,
                       ts_rank_cd(to_tsvector('english', coalesce(a.title, '') || ' ' || coalesce(a.description, '')), plainto_tsquery('english', :q)) AS bm25_score,
                       (1.0 / (60.0 + ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', coalesce(a.title, '') || ' ' || coalesce(a.description, '')), plainto_tsquery('english', :q)) DESC))) AS rrf_score
                FROM articles a
                WHERE to_tsvector('english', coalesce(a.title, '') || ' ' || coalesce(a.description, '')) @@ plainto_tsquery('english', :q)
                  AND (:exclude_id IS NULL OR a.id != :exclude_id)
                  AND (:section IS NULL OR a.section = :section)
                  AND (:min_date IS NULL OR a.date >= :min_date)
                ORDER BY bm25_score DESC
                LIMIT :limit;
            """
            rows = db.execute(
                text(sql),
                {
                    "q": q_clean,
                    "exclude_id": exclude_id,
                    "section": section,
                    "min_date": min_date,
                    "limit": limit,
                }
            ).fetchall()
        # Both vector + text are present (Full Hybrid Reciprocal Rank Fusion)
        else:
            sql = """
            WITH dense_matches AS (
                SELECT id, 
                       (1.0 - (embedding <=> CAST(:v AS vector))) AS dense_score,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:v AS vector)) AS dense_rank
                FROM articles
                WHERE embedding IS NOT NULL
                  AND (:exclude_id IS NULL OR id != :exclude_id)
                  AND (:section IS NULL OR section = :section)
                  AND (:min_date IS NULL OR date >= :min_date)
                ORDER BY embedding <=> CAST(:v AS vector)
                LIMIT :n_retrieve
            ),
            lexical_matches AS (
                SELECT id,
                       ts_rank_cd(to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, '')), plainto_tsquery('english', :q)) AS bm25_score,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, '')), plainto_tsquery('english', :q)) DESC) AS bm25_rank
                FROM articles
                WHERE to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, '')) @@ plainto_tsquery('english', :q)
                  AND (:exclude_id IS NULL OR id != :exclude_id)
                  AND (:section IS NULL OR section = :section)
                  AND (:min_date IS NULL OR date >= :min_date)
                LIMIT :n_retrieve
            ),
            combined AS (
                SELECT COALESCE(d.id, l.id) AS id,
                       COALESCE(d.dense_score, 0.0) AS dense_score,
                       COALESCE(l.bm25_score, 0.0) AS bm25_score,
                       (COALESCE(1.0 / (60.0 + d.dense_rank), 0.0) + COALESCE(1.0 / (60.0 + l.bm25_rank), 0.0)) AS rrf_score
                FROM dense_matches d
                FULL OUTER JOIN lexical_matches l ON d.id = l.id
            )
            SELECT a.id, a.title, a.description, a.section, a.subsection, a.date, a.url, a.image,
                   a.page_views, a.unique_users, a.text_rerank, a.linked_ids,
                   c.dense_score, c.bm25_score, c.rrf_score
            FROM combined c
            JOIN articles a ON a.id = c.id
            ORDER BY c.rrf_score DESC
            LIMIT :limit;
            """
            rows = db.execute(
                text(sql),
                {
                    "v": vec_str,
                    "q": q_clean,
                    "exclude_id": exclude_id,
                    "section": section,
                    "min_date": min_date,
                    "n_retrieve": n_retrieve,
                    "limit": limit,
                }
            ).fetchall()

        results = []
        for r in rows:
            results.append({
                "id": str(r[0]),
                "title": str(r[1] or ""),
                "description": str(r[2] or ""),
                "section": str(r[3] or ""),
                "subsection": str(r[4] or ""),
                "date": r[5],
                "url": str(r[6] or ""),
                "image": str(r[7] or ""),
                "page_views": int(r[8] or 0),
                "unique_users": int(r[9] or 0),
                "text_rerank": str(r[10] or r[1] or ""),
                "linked_ids": r[11],
                "dense_score": float(r[12] or 0.0),
                "bm25_score": float(r[13] or 0.0),
                "rrf_score": float(r[14] or 0.0),
            })
        return results
