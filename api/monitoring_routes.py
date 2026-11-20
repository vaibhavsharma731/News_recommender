"""
REST API endpoints for Centralised Performance Monitoring and Execution History.
Exposes hardware resource telemetry, stage timings, persistent execution history,
database storage diagnostics, and pipeline version comparisons.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

import config
from db.connection import SessionLocal, get_db
from db.monitoring_repository import MonitoringRepository
from monitoring.resource_monitor import take_snapshot

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

# ─── Live Recommendation Rolling Metrics (In-Memory Buffer) ─────────────────
# Thread-safe ring buffer for the last 150 live queries to compute latency percentiles
_live_metrics_lock = threading.Lock()
_live_queries_buffer: Deque[Dict[str, Any]] = deque(maxlen=150)
_live_total_count = 0


def record_live_query(
    method: str,
    query_type: str,  # "search" or "similar"
    timings: dict,
    query_str: Optional[str] = None,
    article_id: Optional[str] = None,
    source: str = "live",
):
    """Called after a recommend() call to update in-memory rolling stats."""
    global _live_total_count
    with _live_metrics_lock:
        _live_total_count += 1
        entry = {
            "timestamp": time.time(),
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "query_type": query_type,
            "query": query_str[:60] if query_str else None,
            "article_id": article_id,
            "source": source,
            "total_ms": timings.get("total_ms", 0.0),
            "prep_ms": timings.get("prep_ms", 0.0),
            "bm25_ms": timings.get("bm25_ms", 0.0),
            "dense_ms": timings.get("dense_ms", 0.0),
            "rerank_ms": timings.get("rerank_ms", 0.0),
            "blend_ms": timings.get("blend_ms", 0.0),
            "boosts_ms": timings.get("boosts_ms", 0.0),
            "mmr_ms": timings.get("mmr_ms", 0.0),
            "calibration_ms": timings.get("calibration_ms", 0.0),
            "throughput_rps": timings.get("throughput_rps", 0.0),
        }
        _live_queries_buffer.append(entry)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/resources/current")
def current_resources():
    """
    Instantaneous point-in-time snapshot of CPU, RAM, GPU, and process resources.
    Values are None when genuinely unavailable (never reported as false zeros).
    """
    snap = take_snapshot()
    data = snap.to_dict()
    data["active_threads"] = threading.active_count()
    data["pid"] = os.getpid()
    return data


@router.get("/summary")
def monitoring_summary(db: Session = Depends(get_db)):
    """
    High-level aggregate summary of all recorded runs, average durations,
    resource usage peaks, and recent status counts.
    """
    summary = MonitoringRepository.get_summary_stats(db)

    # Add live throughput info
    with _live_metrics_lock:
        live_count = _live_total_count
        buf = list(_live_queries_buffer)

    avg_live_ms = round(sum(q["total_ms"] for q in buf) / len(buf), 2) if buf else None
    summary["live_recommendations"] = {
        "total_requests": live_count,
        "recent_window_size": len(buf),
        "recent_avg_latency_ms": avg_live_ms,
    }
    summary["current_pipeline_version"] = config.get_pipeline_version()
    summary["embedding_model"] = config.EMBEDDING_MODEL
    summary["reranker_model"] = config.RERANKER_MODEL
    return summary


@router.get("/live/stats")
def live_stats():
    """
    Latency distribution and stage timing breakdown for live recommendations.
    Computes p50, p90, p95, p99 percentiles and average stage breakdown over the rolling window.
    """
    with _live_metrics_lock:
        buf = list(_live_queries_buffer)
        total_reqs = _live_total_count

    if not buf:
        return {
            "total_requests": total_reqs,
            "window_size": 0,
            "message": "No live recommendations recorded yet in this server session.",
            "percentiles": {},
            "stage_breakdown_avg_ms": {},
            "recent_queries": [],
        }

    durations = sorted(q["total_ms"] for q in buf)
    n = len(durations)

    def _pct(p: float) -> float:
        idx = min(int(math.ceil(p * n)) - 1, n - 1)
        return round(durations[max(0, idx)], 2)

    # Average stage timings
    stages = ["prep_ms", "bm25_ms", "dense_ms", "rerank_ms", "blend_ms", "boosts_ms", "mmr_ms", "calibration_ms"]
    avg_stages = {}
    for stg in stages:
        vals = [q.get(stg, 0.0) for q in buf if q.get(stg) is not None]
        avg_stages[stg] = round(sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "total_requests": total_reqs,
        "window_size": n,
        "percentiles": {
            "p50_ms": _pct(0.50),
            "p90_ms": _pct(0.90),
            "p95_ms": _pct(0.95),
            "p99_ms": _pct(0.99),
            "min_ms": round(durations[0], 2),
            "max_ms": round(durations[-1], 2),
            "avg_ms": round(sum(durations) / n, 2),
        },
        "stage_breakdown_avg_ms": avg_stages,
        "recent_queries": list(reversed(buf[-25:])),
    }


@router.get("/records")
def list_execution_records(
    job_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    pipeline_version: Optional[str] = Query(None),
    dataset_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Paginated execution history with comprehensive filtering by job_type,
    status, pipeline_version, dataset_id, and date range.
    """
    offset = (page - 1) * per_page
    records, total = MonitoringRepository.list_records(
        db,
        job_type=job_type,
        status=status,
        pipeline_version=pipeline_version,
        dataset_id=dataset_id,
        start_date=start_date,
        end_date=end_date,
        limit=per_page,
        offset=offset,
    )
    pages = math.ceil(total / per_page) if total > 0 else 1
    return {
        "records": [r.to_dict() for r in records],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    }


@router.get("/records/{record_id}")
def get_execution_record(record_id: str, db: Session = Depends(get_db)):
    """Fetch complete execution record including granular stages and resource telemetry."""
    rec = MonitoringRepository.get_record(db, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Execution record {record_id} not found")
    stages = MonitoringRepository.get_stages(db, record_id)
    return {
        "record": rec.to_dict(),
        "stages": [s.to_dict() for s in stages],
    }


@router.get("/comparison")
def compare_executions(
    record_ids: str = Query(..., description="Comma-separated list of record IDs to compare"),
    db: Session = Depends(get_db),
):
    """
    Side-by-side comparison of 2 or more execution runs.
    Enables comparing old vs new pipelines, datasets, resource consumption, and stage timings.
    """
    ids = [i.strip() for i in record_ids.split(",") if i.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="Provide at least one record ID")

    records = []
    for rid in ids:
        rec = MonitoringRepository.get_record(db, rid)
        if rec:
            stages = MonitoringRepository.get_stages(db, rid)
            records.append({
                "record": rec.to_dict(),
                "stages": [s.to_dict() for s in stages],
            })

    if not records:
        raise HTTPException(status_code=404, detail="None of the requested records were found")

    # Generate comparison summary
    comparison_summary = {
        "count": len(records),
        "durations_ms": {r["record"]["id"]: r["record"]["duration_ms"] for r in records},
        "throughputs_rps": {r["record"]["id"]: r["record"]["throughput_rps"] for r in records},
        "ram_peaks_mb": {r["record"]["id"]: r["record"]["ram_peak_mb"] for r in records},
        "cpu_peaks_pct": {r["record"]["id"]: r["record"]["cpu_peak_pct"] for r in records},
        "gpu_peaks_mem_mb": {r["record"]["id"]: r["record"]["gpu_peak_mem_mb"] for r in records},
        "stage_timings": {r["record"]["id"]: r["record"]["stage_timings"] for r in records},
    }

    return {
        "records": records,
        "summary": comparison_summary,
    }


@router.get("/database/storage")
def database_storage(db: Session = Depends(get_db)):
    """
    Database storage usage and growth tracking.
    Queries PostgreSQL system catalogs to report exact table sizes, index sizes,
    total database volume, and coverage metrics.
    """
    try:
        # 1. Total database size
        db_size_res = db.execute(
            text("SELECT pg_database_size(current_database()), pg_size_pretty(pg_database_size(current_database()))")
        ).fetchone()
        db_size_bytes = db_size_res[0] if db_size_res else 0
        db_size_pretty = db_size_res[1] if db_size_res else "0 bytes"

        # 2. Per-table and index sizes
        tables_query = text("""
            SELECT
                c.relname AS table_name,
                c.reltuples::bigint AS approx_rows,
                pg_total_relation_size(c.oid) AS total_bytes,
                pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size_pretty,
                pg_relation_size(c.oid) AS table_bytes,
                pg_size_pretty(pg_relation_size(c.oid)) AS table_size_pretty,
                pg_indexes_size(c.oid) AS index_bytes,
                pg_size_pretty(pg_indexes_size(c.oid)) AS index_size_pretty
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relkind = 'r'
              AND c.relname IN ('articles', 'article_recommendations', 'ingestion_jobs', 'execution_records', 'execution_stages')
            ORDER BY pg_total_relation_size(c.oid) DESC;
        """)
        table_rows = db.execute(tables_query).fetchall()
        tables = [
            {
                "table_name": r[0],
                "approx_rows": max(0, int(r[1])),
                "total_bytes": int(r[2]),
                "total_size": r[3],
                "table_bytes": int(r[4]),
                "table_size": r[5],
                "index_bytes": int(r[6]),
                "index_size": r[7],
            }
            for r in table_rows
        ]

        # 3. Exact table counts & coverage
        total_articles = db.execute(text("SELECT count(*) FROM articles")).scalar() or 0
        embedded_articles = db.execute(
            text("SELECT count(*) FROM articles WHERE embedding IS NOT NULL")
        ).scalar() or 0
        total_recs = db.execute(text("SELECT count(*) FROM article_recommendations")).scalar() or 0
        distinct_sources_with_recs = db.execute(
            text("SELECT count(DISTINCT article_id) FROM article_recommendations")
        ).scalar() or 0

        embed_pct = round((embedded_articles / total_articles * 100), 1) if total_articles > 0 else 0.0
        rec_coverage_pct = round((distinct_sources_with_recs / total_articles * 100), 1) if total_articles > 0 else 0.0

        return {
            "database_size_bytes": db_size_bytes,
            "database_size_pretty": db_size_pretty,
            "total_articles": total_articles,
            "embedded_articles": embedded_articles,
            "embedding_coverage_pct": embed_pct,
            "total_precomputed_recs": total_recs,
            "articles_with_precomputed_recs": distinct_sources_with_recs,
            "recommendation_coverage_pct": rec_coverage_pct,
            "tables": tables,
        }
    except Exception as e:
        return {"error": str(e)}


class RetentionRequest(BaseModel):
    days: int = 90
    keep_job_types: List[str] = ["ingestion"]


@router.post("/cleanup")
def cleanup_old_records(body: RetentionRequest, db: Session = Depends(get_db)):
    """
    Retention policy execution: safely prune execution records older than N days.
    By default preserves all 'ingestion' records to maintain permanent upload history.
    """
    deleted_count = MonitoringRepository.delete_old_records(
        db,
        days_older_than=body.days,
        keep_job_types=body.keep_job_types,
    )
    return {
        "status": "success",
        "deleted_records": deleted_count,
        "retention_days": body.days,
        "preserved_types": body.keep_job_types,
    }
