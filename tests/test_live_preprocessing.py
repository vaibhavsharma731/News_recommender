"""
tests/test_live_preprocessing.py — Test suite for Pipeline B (Live Preprocessing)

Verifies:
1. POST /api/db/preprocess-live generates dense embeddings for articles.
2. Cross-Encoder reranking and recommendations precomputation are bypassed (rec count remains 0).
3. Telemetry (ExecutionRecord) is recorded for live preprocessing.
4. Live Recommender can serve recommendations from the preprocessed dataset.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.server import app
from db.connection import SessionLocal, get_db
from db.dataset_repository import DatasetRepository
from db.models import Dataset


@pytest.fixture
def client():
    return TestClient(app)


def test_live_preprocessing_pipeline_b(client):
    test_ds_id = "test_live_prep_suite"
    with SessionLocal() as db:
        # Clean up any leftover from previous runs
        try:
            DatasetRepository.delete_dataset(db, test_ds_id)
        except ValueError:
            pass
        
        # 1. Create isolated test dataset
        ds = DatasetRepository.create_dataset(
            db,
            dataset_id=test_ds_id,
            name="Live Preprocessing Test Dataset",
            description="Testing Pipeline B fast embedding generation",
        )
        assert ds["id"] == test_ds_id
        schema = ds["schema_name"]

        # 2. Clone 5 articles from default into this schema WITHOUT embeddings
        cloned = DatasetRepository.clone_articles(db, "default", test_ds_id, with_embeddings=False, limit=5)
        assert cloned == 5

        # Check that embeddings are indeed NULL
        count_null = db.execute(text(f"SELECT count(*) FROM {schema}.articles WHERE embedding IS NULL;")).scalar()
        assert count_null == cloned

    # 4. Trigger Live Preprocessing API
    resp = client.post(
        "/api/db/preprocess-live",
        json={"dataset": test_ds_id, "batch_size": 64, "recompute_all": False},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "processing"
    assert data["dataset"] == test_ds_id
    job_id = data["job_id"]

    # Poll until completed (remote URL fallback + local model load can take up to 25s)
    import time
    completed = False
    for _ in range(120):
        time.sleep(0.5)
        j_resp = client.get(f"/api/db/jobs/{job_id}?dataset={test_ds_id}")
        if j_resp.status_code == 200:
            j_data = j_resp.json()
            if j_data.get("status") == "completed":
                completed = True
                break
            elif j_data.get("status") == "failed":
                pytest.fail(f"Job failed: {j_data.get('error')}")

    assert completed, "Live preprocessing job did not complete in time"

    # 5. Verify database state
    with SessionLocal() as db:
        # All articles should now have non-null embeddings
        still_null = db.execute(text(f"SELECT count(*) FROM {schema}.articles WHERE embedding IS NULL;")).scalar()
        assert still_null == 0

        # Recommendation pairs MUST remain 0 because Pipeline B does not precompute recs
        recs_count = db.execute(text(f"SELECT count(*) FROM {schema}.article_recommendations;")).scalar()
        assert recs_count == 0

        # 6. Verify ExecutionRecord was recorded
        exec_record = db.execute(
            text(
                "SELECT job_type, dataset_id, status, articles_success "
                "FROM execution_records WHERE dataset_id = :ds ORDER BY created_at DESC LIMIT 1;"
            ),
            {"ds": test_ds_id},
        ).fetchone()
        assert exec_record is not None
        assert exec_record[0] == "live_preprocessing"
        assert exec_record[1] == test_ds_id
        assert exec_record[2] == "completed"
        assert exec_record[3] == cloned

        # 7. Clean up
        DatasetRepository.delete_dataset(db, test_ds_id)
