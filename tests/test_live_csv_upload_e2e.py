import pytest
from fastapi.testclient import TestClient
from api.server import app
from db.connection import SessionLocal, resolve_schema, set_search_path
from db.repository import ArticleRepository
import io

def test_live_csv_upload_and_status():
    client = TestClient(app)

    # Clean any prior jobs so dataset is idle
    client.post("/api/db/jobs/reset-stuck")
    
    # Simple CSV content
    csv_content = (
        "id,title,description,body,section,subsection,date,page_views,unique_users\n"
        "live_test_art_001,Breaking AI News,Summary of AI news,Full article body text about AI,technology,ai,2026-11-20T10:00:00Z,500,300\n"
        "live_test_art_002,Global Market Updates,Stock markets rise,Markets rise across Asia,business,markets,2026-11-20T11:00:00Z,800,450\n"
    )
    
    file_bytes = io.BytesIO(csv_content.encode("utf-8"))
    
    # Test upload to live_data_1 dataset
    response = client.post(
        "/api/db/upload-live-csv?dataset=live_live_data_1_c8f6fa&batch_size=64",
        files={"file": ("test_articles.csv", file_bytes, "text/csv")}
    )
    
    assert response.status_code == 200, f"Upload failed: {response.text}"
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "processing"
    assert data["pipeline_type"] == "live"
    job_id = data["job_id"]
    
    # Test poll job status
    poll_res = client.get(f"/api/db/jobs/{job_id}?dataset=live_live_data_1_c8f6fa")
    assert poll_res.status_code == 200, f"Poll failed: {poll_res.text}"
    job_info = poll_res.json()
    assert job_info["id"] == job_id
    
    # Test poll without passing dataset query param (fallback search across schemas)
    fallback_res = client.get(f"/api/db/jobs/{job_id}")
    assert fallback_res.status_code == 200, f"Fallback poll failed: {fallback_res.text}"
    assert fallback_res.json()["id"] == job_id
