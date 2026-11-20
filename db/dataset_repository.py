"""
DatasetRepository — CRUD and schema isolation manager for NewsRec datasets.

Every dataset has an isolated PostgreSQL schema:
  - public (default, protected)
  - ds_<slug>_<hex> (experimental datasets)

Each schema contains its own:
  - articles
  - article_recommendations
  - ingestion_jobs

Shared across schemas in public:
  - datasets (metadata registry)
  - execution_records & execution_stages (centralised monitoring)
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

import config
from .connection import resolve_schema, set_search_path
from .models import Dataset


def _slugify(text_val: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text_val.strip().lower())
    return slug.strip("_")[:30] or "dataset"


def _bytes_pretty(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f} KB"
    elif n_bytes < 1024 * 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):.1f} MB"
    return f"{n_bytes / (1024 * 1024 * 1024):.2f} GB"


class DatasetRepository:
    """Manages datasets and schema-level isolation in PostgreSQL."""

    @staticmethod
    def ensure_default_dataset(db: Session) -> Dataset:
        """Guarantees the default/production public dataset exists in the metadata registry."""
        ds = db.query(Dataset).filter(Dataset.id == "default").first()
        if not ds:
            ds = Dataset(
                id="default",
                name="Default (Production)",
                schema_name="public",
                pipeline_type="precomputed",
                description="Original production dataset in public schema",
                pipeline_version=config.get_pipeline_version(),
                is_protected=True,
            )
            db.add(ds)
            db.commit()
            db.refresh(ds)
        elif not getattr(ds, "pipeline_type", None):
            ds.pipeline_type = "precomputed"
            db.commit()
            db.refresh(ds)
        return ds

    @staticmethod
    def get_dataset_stats(db: Session, schema_name: str) -> Dict[str, Any]:
        """Calculates live row counts and disk storage usage for a given schema."""
        safe_schema = re.sub(r"[^a-zA-Z0-9_]", "", schema_name)
        stats = {
            "article_count": 0,
            "embedded_count": 0,
            "recommendation_count": 0,
            "total_bytes": 0,
            "size_pretty": "0 B",
        }
        try:
            # Check if articles table exists in schema
            table_exists = db.execute(
                text("SELECT to_regclass(:tbl)"),
                {"tbl": f"{safe_schema}.articles"}
            ).scalar()
            if not table_exists:
                return stats

            # 1. Row counts
            art_cnt = db.execute(text(f"SELECT count(*) FROM {safe_schema}.articles")).scalar() or 0
            emb_cnt = db.execute(text(f"SELECT count(*) FROM {safe_schema}.articles WHERE embedding IS NOT NULL")).scalar() or 0
            
            # Check if recommendations table exists (present in precomputed, omitted/empty in live)
            rec_tbl_exists = db.execute(
                text("SELECT to_regclass(:tbl)"),
                {"tbl": f"{safe_schema}.article_recommendations"}
            ).scalar()
            rec_cnt = (
                db.execute(text(f"SELECT count(*) FROM {safe_schema}.article_recommendations")).scalar() or 0
                if rec_tbl_exists else 0
            )

            stats["article_count"] = int(art_cnt)
            stats["embedded_count"] = int(emb_cnt)
            stats["recommendation_count"] = int(rec_cnt)

            # 2. Table sizes in this schema
            size_query = text("""
                SELECT COALESCE(SUM(pg_total_relation_size(c.oid)), 0)
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = :schema AND c.relkind = 'r'
            """)
            total_bytes = db.execute(size_query, {"schema": safe_schema}).scalar() or 0
            stats["total_bytes"] = int(total_bytes)
            stats["size_pretty"] = _bytes_pretty(int(total_bytes))
        except Exception as e:
            stats["error"] = str(e)

        return stats

    @staticmethod
    def list_datasets(db: Session, pipeline_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all registered datasets with live storage, embedding, and recommendation metrics."""
        DatasetRepository.ensure_default_dataset(db)
        query = db.query(Dataset)
        if pipeline_type:
            query = query.filter(Dataset.pipeline_type == pipeline_type)
        datasets = query.order_by(Dataset.is_protected.desc(), Dataset.created_at.asc()).all()
        result = []
        for d in datasets:
            d_dict = d.to_dict()
            stats = DatasetRepository.get_dataset_stats(db, d.schema_name)
            d_dict.update(stats)
            result.append(d_dict)
        return result

    @staticmethod
    def get_dataset(db: Session, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Finds a dataset by ID or schema_name and returns it with live statistics."""
        DatasetRepository.ensure_default_dataset(db)
        target = dataset_id.strip() if dataset_id else "default"
        ds = db.query(Dataset).filter((Dataset.id == target) | (Dataset.schema_name == target)).first()
        if not ds:
            return None
        d_dict = ds.to_dict()
        stats = DatasetRepository.get_dataset_stats(db, ds.schema_name)
        d_dict.update(stats)
        return d_dict

    @staticmethod
    def create_dataset(
        db: Session,
        name: str,
        description: Optional[str] = None,
        dataset_id: Optional[str] = None,
        pipeline_type: str = "precomputed",
    ) -> Dict[str, Any]:
        """
        Creates a new isolated schema with exact clones of articles, article_recommendations,
        and ingestion_jobs tables, complete with vector(384) embeddings, constraints, and indexes.
        For pipeline_type='live', the article_recommendations table is omitted to save storage.
        """
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Dataset name cannot be empty.")

        clean_type = "live" if pipeline_type == "live" else "precomputed"
        slug = _slugify(clean_name)
        hex_suffix = uuid.uuid4().hex[:6]
        schema_name = f"ds_{clean_type[:4]}_{slug}_{hex_suffix}"[:63]
        ds_id = dataset_id.strip() if dataset_id else f"{clean_type}_{slug}_{hex_suffix}"[:64]

        # 1. Create PostgreSQL schema
        db.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))

        # 2. Clone articles and ingestion_jobs tables from public schema
        db.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {schema_name}.articles (
                LIKE public.articles INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING INDEXES
            )
        """))
        db.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {schema_name}.ingestion_jobs (
                LIKE public.ingestion_jobs INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING INDEXES
            )
        """))

        # 3. For precomputed pipeline, also create article_recommendations table and FK constraints
        if clean_type == "precomputed":
            db.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {schema_name}.article_recommendations (
                    LIKE public.article_recommendations INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING INDEXES
                )
            """))
            try:
                db.execute(text(f"""
                    ALTER TABLE {schema_name}.article_recommendations
                    ADD CONSTRAINT fk_{schema_name}_rec_article
                    FOREIGN KEY (article_id) REFERENCES {schema_name}.articles(id) ON DELETE CASCADE
                """))
                db.execute(text(f"""
                    ALTER TABLE {schema_name}.article_recommendations
                    ADD CONSTRAINT fk_{schema_name}_rec_recommended
                    FOREIGN KEY (recommended_article_id) REFERENCES {schema_name}.articles(id) ON DELETE CASCADE
                """))
            except Exception:
                pass

        # 4. Register in public.datasets
        new_ds = Dataset(
            id=ds_id,
            name=clean_name,
            schema_name=schema_name,
            pipeline_type=clean_type,
            description=description or "",
            pipeline_version=config.get_pipeline_version(),
            is_protected=False,
        )
        db.add(new_ds)
        db.commit()
        db.refresh(new_ds)

        result = new_ds.to_dict()
        result.update(DatasetRepository.get_dataset_stats(db, schema_name))
        return result

    @staticmethod
    def delete_dataset(db: Session, dataset_id: str) -> bool:
        """
        Deletes an isolated experimental dataset and drops its PostgreSQL schema.
        Protected datasets (such as 'default' / public) are permanently protected.
        """
        target = dataset_id.strip()
        ds = db.query(Dataset).filter((Dataset.id == target) | (Dataset.schema_name == target)).first()
        if not ds:
            raise ValueError(f"Dataset '{target}' not found.")

        if ds.is_protected or ds.schema_name == "public" or ds.id == "default":
            raise ValueError(f"Cannot delete protected dataset '{ds.name}' ({ds.schema_name}).")

        schema_to_drop = ds.schema_name
        # 1. Drop schema CASCADE
        db.execute(text(f"DROP SCHEMA IF EXISTS {schema_to_drop} CASCADE"))

        # 2. Remove registry entry
        db.delete(ds)
        db.commit()
        return True

    @staticmethod
    def clone_articles(
        db: Session,
        source_id: str,
        target_id: str,
        with_embeddings: bool = True,
        limit: Optional[int] = None,
    ) -> int:
        """
        Copies article records from one dataset schema to another without copying
        recommendations. Enables testing alternative recommendation algorithms
        on identical article corpus.
        """
        src_schema = resolve_schema(db, source_id)
        dst_schema = resolve_schema(db, target_id)

        if src_schema == dst_schema:
            raise ValueError("Source and target dataset schemas must be different.")

        emb_expr = "embedding" if with_embeddings else "NULL"
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

        insert_sql = text(f"""
            INSERT INTO {dst_schema}.articles (
                id, title, description, body, section, subsection, date, url, image,
                is_live, page_views, unique_users, linked_ids, text_embed, text_bm25,
                text_rerank, embed_token_count, rerank_token_count, is_truncated,
                recipe_version, embedding
            )
            SELECT
                id, title, description, body, section, subsection, date, url, image,
                is_live, page_views, unique_users, linked_ids, text_embed, text_bm25,
                text_rerank, embed_token_count, rerank_token_count, is_truncated,
                recipe_version, {emb_expr}
            FROM {src_schema}.articles
            ORDER BY date DESC NULLS LAST
            {limit_clause}
            ON CONFLICT (id) DO NOTHING
        """)
        res = db.execute(insert_sql)
        db.commit()
        return res.rowcount or 0
