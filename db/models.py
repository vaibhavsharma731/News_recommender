"""
SQLAlchemy ORM models for news articles, precomputed recommendations, and ingestion jobs.
"""
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    func
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from .connection import Base
import config


class Article(Base):
    """
    Core storage for news articles and their neural vector embeddings.
    """
    __tablename__ = "articles"

    id = Column(String(100), primary_key=True, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    body = Column(Text, nullable=True)
    section = Column(String(100), index=True, nullable=True)
    subsection = Column(String(100), nullable=True)
    date = Column(DateTime(timezone=True), index=True, nullable=True)
    url = Column(Text, nullable=True)
    image = Column(Text, nullable=True)
    is_live = Column(Boolean, default=False)
    premium = Column(Boolean, default=False)
    page_views = Column(Integer, default=0)
    unique_users = Column(Integer, default=0)
    linked_ids = Column(Text, nullable=True)    # JSON string of editor "Also Read" article IDs
    text_embed = Column(Text, nullable=True)    # Preprocessed text passed to embedding model
    text_bm25 = Column(Text, nullable=True)     # Weighted text for BM25 keyword matching
    text_rerank = Column(Text, nullable=True)   # Truncated lead text for Cross-Encoder reranker

    # Token budgeting tracking metadata
    embed_token_count = Column(Integer, default=0)
    rerank_token_count = Column(Integer, default=0)
    is_truncated = Column(Boolean, default=False)
    recipe_version = Column(String(32), nullable=True)

    # BAAI/bge-small-en-v1.5 produces 384-dimensional dense vector embeddings
    embedding = Column(Vector(config.EMBEDDING_DIM), nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Precomputed recommendations where this article is the source
    recommendations = relationship(
        "ArticleRecommendation",
        foreign_keys="ArticleRecommendation.article_id",
        back_populates="source_article",
        cascade="all, delete-orphan",
        order_by="ArticleRecommendation.rank",
    )

    def to_dict(self, include_embedding: bool = False) -> dict:
        description = self.description or ""
        body = self.body or ""
        if self.date:
            try:
                date_str = self.date.strftime("%d %b %Y")
            except Exception:
                date_str = str(self.date)
        else:
            date_str = ""
        data = {
            "id": self.id,
            "title": self.title,
            "description": description,
            "snippet": (description or body)[:200],
            "section": self.section or "",
            "subsection": self.subsection or "",
            "date": date_str,
            "url": self.url or "",
            "image": self.image or "",
            "is_live": bool(self.is_live),
            "premium": bool(self.premium),
            "pageViews": self.page_views or 0,
            "uniqueUsers": self.unique_users or 0,
            "embed_token_count": self.embed_token_count or 0,
            "rerank_token_count": self.rerank_token_count or 0,
            "is_truncated": bool(self.is_truncated),
            "recipe_version": self.recipe_version or "",
            "has_embedding": self.embedding is not None,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = list(self.embedding)
        return data


class ArticleRecommendation(Base):
    """
    Precomputed recommendations for fast sub-5ms reader lookups.
    Stores the Top 10 recommendations for each article generated via Vector Search + Reranker.
    """
    __tablename__ = "article_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(
        String(100),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    recommended_article_id = Column(
        String(100),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    rank = Column(Integer, nullable=False)   # 1 to 10 (MMR diversified rank)
    rank_raw = Column(Integer, nullable=True, index=True) # 1 to 10 (Pure relevance rank before MMR)
    score = Column(Float, nullable=True)     # Final blended relevance shown to the user (0.0 - 1.0)
    # Raw cross-encoder probability for THIS (source -> recommended) pair, computed
    # O-centrically (query = source article). It is the only score that is directly
    # comparable across lists, so the incremental "reverse push" uses it to decide
    # whether a newly ingested article can displace an old entry.
    rerank_score = Column(Float, nullable=True)
    # Relevance AFTER domain boosts, BEFORE Platt calibration: `boosted_score` is
    # what actually decided the ordering, while `score` is the calibrated
    # confidence on the unboosted blend. Storing both makes the difference
    # inspectable instead of implicit.
    boosted_score = Column(Float, nullable=True)
    pipeline_version = Column(String(32), nullable=True)
    method = Column(String(50), default="neural_hybrid_rerank")
    reason = Column(Text, nullable=True)     # Human-readable explanation why recommended
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    source_article = relationship("Article", foreign_keys=[article_id], back_populates="recommendations")
    recommended_article = relationship("Article", foreign_keys=[recommended_article_id])

    __table_args__ = (
        UniqueConstraint("article_id", "rank", name="uq_source_article_rank"),
        Index("idx_rec_source_rank", "article_id", "rank"),
    )


class IngestionJob(Base):
    """
    Tracks CSV upload and recommendation precomputation jobs for the Editor side.
    """
    __tablename__ = "ingestion_jobs"

    id = Column(String(64), primary_key=True)
    filename = Column(String(255), nullable=False)
    pipeline_version = Column(String(32), nullable=True)
    pipeline_type = Column(String(32), default="precomputed", nullable=True) # precomputed | live
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    total_articles = Column(Integer, default=0)      # articles in THIS upload
    processed_articles = Column(Integer, default=0)
    # Number of EXISTING (previously uploaded) articles whose precomputed
    # recommendations were refreshed by the incremental reverse-push from this job.
    refreshed_articles = Column(Integer, default=0)
    message = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        progress = 0.0
        if self.total_articles and self.total_articles > 0:
            progress = round((self.processed_articles / self.total_articles) * 100, 1)

        duration_seconds = None
        if self.created_at and self.updated_at:
            try:
                duration_seconds = round((self.updated_at - self.created_at).total_seconds(), 1)
            except Exception:
                duration_seconds = None

        return {
            "id": self.id,
            "filename": self.filename,
            "pipeline_version": self.pipeline_version or "",
            "pipeline_type": self.pipeline_type or "precomputed",
            "status": self.status,
            "total_articles": self.total_articles,
            "processed_articles": self.processed_articles,
            "refreshed_articles": self.refreshed_articles or 0,
            "progress_percent": progress,
            "duration_seconds": duration_seconds,
            "message": self.message or "",
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
        }


class Dataset(Base):
    """
    Metadata registry for isolated datasets (PostgreSQL schemas).
    Always resides in the public schema.
    """
    __tablename__ = "datasets"

    id = Column(String(64), primary_key=True)               # e.g. "default", "ds_experiment_1"
    name = Column(String(120), nullable=False)              # Human-readable display label
    schema_name = Column(String(63), nullable=False, unique=True) # Valid PostgreSQL schema name
    pipeline_type = Column(String(32), default="precomputed", nullable=False) # "precomputed" | "live"
    description = Column(Text, nullable=True)
    pipeline_version = Column(String(32), nullable=True)
    is_protected = Column(Boolean, default=False)           # True for default/production dataset
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "schema_name": self.schema_name,
            "pipeline_type": self.pipeline_type or "precomputed",
            "description": self.description or "",
            "pipeline_version": self.pipeline_version or "",
            "is_protected": bool(self.is_protected),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ─── Execution Records (Performance Monitoring) ──────────────────────────────

class ExecutionRecord(Base):
    """
    One row per significant pipeline operation (ingestion job, live recommendation
    batch, embedding run, recompute pass).

    Linked to IngestionJob via job_id when the operation is an ingestion.
    """
    __tablename__ = "execution_records"

    id = Column(String(50), primary_key=True)           # e.g. "exec_<hex8>"
    dataset_id = Column(String(64), nullable=True, default="default")  # isolated dataset target
    job_type = Column(String(50), nullable=False)        # ingestion | live_recommend | recompute | embedding
    job_id = Column(String(64), nullable=True)           # FK to ingestion_jobs.id (if applicable)
    pipeline_version = Column(String(32), nullable=True)
    model_version = Column(String(20), default="384d")
    status = Column(String(20), default="running")       # running | completed | failed | partial

    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)

    articles_total = Column(Integer, default=0)
    articles_success = Column(Integer, default=0)
    articles_failed = Column(Integer, default=0)
    throughput_rps = Column(Float, nullable=True)        # articles per second

    # Resource snapshots
    cpu_start_pct = Column(Float, nullable=True)
    cpu_peak_pct = Column(Float, nullable=True)
    cpu_avg_pct = Column(Float, nullable=True)
    ram_start_mb = Column(Float, nullable=True)
    ram_peak_mb = Column(Float, nullable=True)
    ram_end_mb = Column(Float, nullable=True)
    gpu_used = Column(Boolean, default=False)
    gpu_peak_mem_mb = Column(Float, nullable=True)
    gpu_peak_util_pct = Column(Float, nullable=True)
    disk_read_mb = Column(Float, nullable=True)
    disk_write_mb = Column(Float, nullable=True)
    cpu_time_user_s = Column(Float, nullable=True)
    cpu_time_sys_s = Column(Float, nullable=True)

    # Stage timings as compact JSON {"parse_ms": 120, "embed_ms": 4500, ...}
    stage_timings = Column(Text, nullable=True)

    error_message = Column(Text, nullable=True)
    # JSON blob: extra context (batch_size, n_retrieve, n_candidates, etc.)
    metadata_json = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self) -> dict:
        import json
        stage_t = {}
        try:
            stage_t = json.loads(self.stage_timings) if self.stage_timings else {}
        except Exception:
            pass
        meta = {}
        try:
            meta = json.loads(self.metadata_json) if self.metadata_json else {}
        except Exception:
            pass
        return {
            "id": self.id,
            "dataset_id": self.dataset_id or "default",
            "job_type": self.job_type,
            "job_id": self.job_id,
            "pipeline_version": self.pipeline_version or "",
            "model_version": self.model_version or "384d",
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "duration_s": round(self.duration_ms / 1000, 2) if self.duration_ms else None,
            "articles_total": self.articles_total or 0,
            "articles_success": self.articles_success or 0,
            "articles_failed": self.articles_failed or 0,
            "throughput_rps": self.throughput_rps,
            "cpu_start_pct": self.cpu_start_pct,
            "cpu_peak_pct": self.cpu_peak_pct,
            "cpu_avg_pct": self.cpu_avg_pct,
            "ram_start_mb": self.ram_start_mb,
            "ram_peak_mb": self.ram_peak_mb,
            "ram_end_mb": self.ram_end_mb,
            "gpu_used": bool(self.gpu_used),
            "gpu_peak_mem_mb": self.gpu_peak_mem_mb,
            "gpu_peak_util_pct": self.gpu_peak_util_pct,
            "disk_read_mb": self.disk_read_mb,
            "disk_write_mb": self.disk_write_mb,
            "cpu_time_user_s": self.cpu_time_user_s,
            "cpu_time_sys_s": self.cpu_time_sys_s,
            "stage_timings": stage_t,
            "error_message": self.error_message,
            "metadata": meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ExecutionStage(Base):
    """
    One row per pipeline stage within an ExecutionRecord.
    Provides granular per-stage timing, resource, and throughput data.
    """
    __tablename__ = "execution_stages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(
        String(50),
        ForeignKey("execution_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage_name = Column(String(100), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)
    articles_processed = Column(Integer, default=0)
    cpu_pct = Column(Float, nullable=True)
    ram_mb = Column(Float, nullable=True)
    extra_json = Column(Text, nullable=True)   # JSON for stage-specific data

    def to_dict(self) -> dict:
        import json
        extra = {}
        try:
            extra = json.loads(self.extra_json) if self.extra_json else {}
        except Exception:
            pass
        return {
            "id": self.id,
            "record_id": self.record_id,
            "stage_name": self.stage_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "duration_ms": self.duration_ms,
            "articles_processed": self.articles_processed or 0,
            "cpu_pct": self.cpu_pct,
            "ram_mb": self.ram_mb,
            "extra": extra,
        }
