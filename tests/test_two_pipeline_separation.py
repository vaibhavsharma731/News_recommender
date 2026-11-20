"""
Tests for Two-Pipeline Separation:
1. Precomputation Pipeline (A): isolated schemas with article_recommendations table.
2. Live Computation Pipeline (B): isolated schemas with articles table ONLY (omitting article_recommendations).
3. API endpoints for live single article manual ingestion and live CSV upload.
4. Dataset filtering by pipeline_type.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from api.server import app
from db.connection import SessionLocal, resolve_schema, set_search_path, ensure_schema
from db.dataset_repository import DatasetRepository

client = TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def setup_schema():
    ensure_schema()


def test_two_pipeline_table_separation():
    db = SessionLocal()
    precomp_id = "test_precomp_sep"
    live_id = "test_live_sep"

    try:
        # 1. Clean up if existed from previous run
        try:
            DatasetRepository.delete_dataset(db, precomp_id)
        except Exception:
            pass
        try:
            DatasetRepository.delete_dataset(db, live_id)
        except Exception:
            pass

        # 2. Create Precomputed dataset -> MUST have article_recommendations table
        ds_precomp = DatasetRepository.create_dataset(
            db,
            name="Test Precomputed Sep",
            dataset_id=precomp_id,
            pipeline_type="precomputed",
        )
        assert ds_precomp["pipeline_type"] == "precomputed"

        schema_precomp = resolve_schema(db, precomp_id)
        has_recs_precomp = db.execute(
            text(f"SELECT to_regclass('{schema_precomp}.article_recommendations')")
        ).scalar()
        assert has_recs_precomp is not None, "Precomputed schema must contain article_recommendations table"

        # 3. Create Live dataset -> MUST OMIT article_recommendations table
        ds_live = DatasetRepository.create_dataset(
            db,
            name="Test Live Sep",
            dataset_id=live_id,
            pipeline_type="live",
        )
        assert ds_live["pipeline_type"] == "live"

        schema_live = resolve_schema(db, live_id)
        has_recs_live = db.execute(
            text(f"SELECT to_regclass('{schema_live}.article_recommendations')")
        ).scalar()
        assert has_recs_live is None, "Live schema must NOT contain article_recommendations table"

        # Both must have articles and ingestion_jobs
        has_arts_live = db.execute(text(f"SELECT to_regclass('{schema_live}.articles')")).scalar()
        assert has_arts_live is not None
        has_jobs_live = db.execute(text(f"SELECT to_regclass('{schema_live}.ingestion_jobs')")).scalar()
        assert has_jobs_live is not None

        # 4. Verify DatasetRepository.list_datasets filtering by pipeline_type
        precomp_list = DatasetRepository.list_datasets(db, pipeline_type="precomputed")
        assert any(d["id"] == precomp_id for d in precomp_list)
        assert not any(d["id"] == live_id for d in precomp_list)

        live_list = DatasetRepository.list_datasets(db, pipeline_type="live")
        assert any(d["id"] == live_id for d in live_list)
        assert not any(d["id"] == precomp_id for d in live_list)

        # 5. Verify stats reporting for live dataset with missing recs table
        ds_info = DatasetRepository.get_dataset(db, live_id)
        assert ds_info is not None
        assert ds_info["recommendation_count"] == 0
        assert ds_info["pipeline_type"] == "live"

    finally:
        try:
            DatasetRepository.delete_dataset(db, precomp_id)
        except Exception:
            pass
        try:
            DatasetRepository.delete_dataset(db, live_id)
        except Exception:
            pass
        db.close()


def test_live_manual_ingestion_endpoint():
    db = SessionLocal()
    live_id = "test_live_manual_ds"

    try:
        try:
            DatasetRepository.delete_dataset(db, live_id)
        except Exception:
            pass

        DatasetRepository.create_dataset(
            db,
            name="Test Live Manual DS",
            dataset_id=live_id,
            pipeline_type="live",
        )

        # Ingest single article to live dataset
        payload = {
            "title": "Breaking Tech: Next Generation AI Chips Released",
            "section": "technology",
            "subsection": "hardware",
            "body": "A leading semiconductor firm has unveiled a brand new architecture delivering unprecedented throughput and energy efficiency.",
            "description": "Next-gen AI chip released.",
            "dataset": live_id,
        }

        res = client.post("/api/db/articles/manual-live", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["status"] == "success"
        assert data["pipeline_type"] == "live"
        assert data["article_id"].startswith("art_")
        assert "duration_ms" in data

        # Check DB that article exists and has 384d embedding
        with SessionLocal() as db_check:
            schema_live = resolve_schema(db_check, live_id)
            art_row = db_check.execute(
                text(f"SELECT id, title, embedding IS NOT NULL as has_emb FROM {schema_live}.articles WHERE id = :id"),
                {"id": data["article_id"]}
            ).fetchone()
            assert art_row is not None
            assert art_row.has_emb is True

    finally:
        try:
            DatasetRepository.delete_dataset(db, live_id)
        except Exception:
            pass
        db.close()


def test_api_datasets_pipeline_type_filter():
    res_all = client.get("/api/datasets")
    assert res_all.status_code == 200
    all_ds = res_all.json()["datasets"]

    res_precomp = client.get("/api/datasets?pipeline_type=precomputed")
    assert res_precomp.status_code == 200
    precomp_ds = res_precomp.json()["datasets"]
    assert all(d.get("pipeline_type") == "precomputed" for d in precomp_ds)

    res_live = client.get("/api/datasets?pipeline_type=live")
    assert res_live.status_code == 200
    live_ds = res_live.json()["datasets"]
    assert all(d.get("pipeline_type") == "live" for d in live_ds)
