"""
Database Repository providing clean CRUD, recommendation persistence,
and ingestion job management.
"""
from typing import List, Optional, Dict, Any
from datetime import timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, or_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .models import Article, ArticleRecommendation, ArticleRecommendation768D, IngestionJob
from src.scoring import parse_ids


class ArticleRepository:
    """Encapsulates all PostgreSQL queries."""

    # ──────────────────────────────────────────────────────────────────────────
    # Article CRUD
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_article(db: Session, article_id: str) -> Optional[Article]:
        return db.query(Article).filter(Article.id == article_id).first()

    @staticmethod
    def count_articles(db: Session) -> int:
        return db.query(func.count(Article.id)).scalar() or 0

    @staticmethod
    def count_embedded_articles(db: Session) -> int:
        return db.query(func.count(Article.id)).filter(Article.embedding.isnot(None)).scalar() or 0

    @staticmethod
    def list_articles(
        db: Session,
        limit: int = 50,
        offset: int = 0,
        section: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[Article]:
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
    ) -> List[Article]:
        """
        List articles ordered by how CONFIDENT their precomputed recommendations are.

        Confidence = the average ``score`` of the article's stored top-10
        recommendations (the same 0.0-1.0 blend the reader sees as a percentage).
        Highest first, so stories whose recommendations are strong (70%, 80% …)
        surface ahead of weak ones (10%, 20% …). Articles with no recommendations
        sort last; ties fall back to newest-first.
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
    def get_sections(db: Session) -> List[str]:
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
    ) -> List[Article]:
        """
        Popular stories from one section, for the graceful-degradation list.

        Deliberately model-free: one indexed query, ordered by real page views.
        ``window_days`` is measured against the SOURCE article's date, so "recent"
        means roughly contemporaneous with the story being read rather than newest
        in the database. Served by ``ix_articles_section_date``.
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
        Supports switching between:
          - '384d': Pipeline C (BGE-small 384D, active table `article_recommendations`)
          - '768d': Strategy E baseline (BGE-base 768D, archive table `article_recommendations_768d`)
        If diversify=True, orders by MMR-diversified rank.
        If diversify=False, orders by pure relevance (rank_raw / score) without MMR penalty.
        """
        from sqlalchemy import func
        RecModel = ArticleRecommendation768D if model_version == "768d" else ArticleRecommendation
        order_col = RecModel.rank if diversify else func.coalesce(RecModel.rank_raw, RecModel.rank)
        recs = db.query(RecModel, Article).join(
            Article, RecModel.recommended_article_id == Article.id
        ).filter(
            RecModel.article_id == article_id
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
    def count_total_recommendations(db: Session) -> int:
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
        pipeline_version: Optional[str] = None
    ) -> IngestionJob:
        job = IngestionJob(
            id=job_id,
            filename=filename,
            pipeline_version=pipeline_version,
            total_articles=total_articles,
            processed_articles=0,
            status="pending",
            message="Job created, awaiting worker..."
        )
        db.add(job)
        db.commit()
        db.refresh(job)
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
            "message": message,
            "error": error,
        }
        for field, value in changes.items():
            if value is not None:
                setattr(job, field, value)
        db.commit()
        db.refresh(job)
        return job


    @staticmethod
    def get_job(db: Session, job_id: str) -> Optional[IngestionJob]:
        return db.query(IngestionJob).filter(IngestionJob.id == job_id).first()

    @staticmethod
    def list_recent_jobs(db: Session, limit: int = 10) -> List[IngestionJob]:
        return db.query(IngestionJob).order_by(desc(IngestionJob.created_at)).limit(limit).all()

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
        """Mark jobs left in pending/processing after a server restart as failed."""
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
        return len(jobs)
