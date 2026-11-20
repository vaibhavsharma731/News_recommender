"""
MonitoringRepository — CRUD for execution_records and execution_stages.

All methods are static and accept a SQLAlchemy Session.
Every field that is unavailable is stored as NULL (not 0) so the UI can
display "N/A" rather than a misleading zero.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func as sql_func, text

from .models import ExecutionRecord, ExecutionStage


def _hex8() -> str:
    return os.urandom(4).hex()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _round(v, n=2):
    if v is None:
        return None
    try:
        return round(float(v), n)
    except Exception:
        return None


class MonitoringRepository:
    # ── Create ──────────────────────────────────────────────────────────────

    @staticmethod
    def create_record(
        db: Session,
        job_type: str,
        pipeline_version: Optional[str] = None,
        job_id: Optional[str] = None,
        model_version: str = "384d",
        dataset_id: str = "default",
        articles_total: int = 0,
        metadata: Optional[dict] = None,
    ) -> ExecutionRecord:
        """
        Open a new execution record and immediately persist it so the UI can
        see the job as "running" even before it completes.
        """
        rec = ExecutionRecord(
            id=f"exec_{_hex8()}",
            dataset_id=dataset_id or "default",
            job_type=job_type,
            job_id=job_id,
            pipeline_version=pipeline_version,
            model_version=model_version,
            status="running",
            started_at=_now(),
            articles_total=articles_total,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec

    # ── Update (mid-job) ─────────────────────────────────────────────────────

    @staticmethod
    def update_record(db: Session, record_id: str, **kwargs) -> None:
        """Partial-update any fields on an open record (e.g. mid-job progress)."""
        db.query(ExecutionRecord).filter(ExecutionRecord.id == record_id).update(kwargs)
        db.commit()

    # ── Finish ───────────────────────────────────────────────────────────────

    @staticmethod
    def finish_record(
        db: Session,
        record_id: str,
        status: str,
        duration_ms: float,
        articles_success: int = 0,
        articles_failed: int = 0,
        resource_summary: Optional[dict] = None,
        stage_timings: Optional[dict] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Close out an execution record once the job is done/failed.
        resource_summary should come from BackgroundResourceMonitor.get_summary().
        stage_timings should come from ExecutionContext.flat_timings().
        """
        rs = resource_summary or {}
        throughput = None
        if duration_ms and duration_ms > 0 and articles_success > 0:
            throughput = _round(articles_success / (duration_ms / 1000), 3)

        updates = {
            "status": status,
            "completed_at": _now(),
            "duration_ms": _round(duration_ms),
            "articles_success": articles_success,
            "articles_failed": articles_failed,
            "throughput_rps": throughput,
            "error_message": error_message,
            "stage_timings": json.dumps(stage_timings) if stage_timings else None,
            # Resources
            "cpu_start_pct": _round(rs.get("cpu_start_pct")),
            "cpu_peak_pct": _round(rs.get("cpu_peak_pct")),
            "cpu_avg_pct": _round(rs.get("cpu_avg_pct")),
            "ram_start_mb": _round(rs.get("ram_start_mb")),
            "ram_peak_mb": _round(rs.get("ram_peak_mb")),
            "ram_end_mb": _round(rs.get("ram_end_mb")),
            "gpu_used": bool(rs.get("gpu_available", False)),
            "gpu_peak_mem_mb": _round(rs.get("gpu_peak_mem_mb")),
            "gpu_peak_util_pct": _round(rs.get("gpu_peak_util_pct")),
            "disk_read_mb": _round(rs.get("disk_read_delta_mb")),
            "disk_write_mb": _round(rs.get("disk_write_delta_mb")),
            "cpu_time_user_s": _round(rs.get("cpu_time_user_delta")),
            "cpu_time_sys_s": _round(rs.get("cpu_time_sys_delta")),
        }
        db.query(ExecutionRecord).filter(ExecutionRecord.id == record_id).update(updates)
        db.commit()

    # ── Stages ───────────────────────────────────────────────────────────────

    @staticmethod
    def add_stage(
        db: Session,
        record_id: str,
        stage_name: str,
        duration_ms: float,
        started_at: Optional[datetime] = None,
        articles_processed: int = 0,
        cpu_pct: Optional[float] = None,
        ram_mb: Optional[float] = None,
        extra: Optional[dict] = None,
        commit: bool = True,
    ) -> ExecutionStage:
        stage = ExecutionStage(
            record_id=record_id,
            stage_name=stage_name,
            started_at=started_at or _now(),
            duration_ms=_round(duration_ms),
            articles_processed=articles_processed,
            cpu_pct=_round(cpu_pct),
            ram_mb=_round(ram_mb),
            extra_json=json.dumps(extra) if extra else None,
        )
        db.add(stage)
        if commit:
            db.commit()
        return stage

    @staticmethod
    def add_stages_from_context(db: Session, record_id: str, ctx_dict: dict) -> None:
        """
        Persist all stages from an ExecutionContext.to_dict() result.
        ctx_dict["stages"] is the list of StageResult dicts.
        """
        from datetime import timezone as _tz
        for s in ctx_dict.get("stages", []):
            started = None
            try:
                started = datetime.fromtimestamp(s["start_ms"] / 1000, tz=_tz.utc)
            except Exception:
                pass
            MonitoringRepository.add_stage(
                db, record_id,
                stage_name=s["name"],
                duration_ms=s["elapsed_ms"],
                started_at=started,
                articles_processed=s.get("articles_processed", 0),
                extra=s.get("extra"),
                commit=False,
            )
        db.commit()

    # ── Query ─────────────────────────────────────────────────────────────────

    @staticmethod
    def get_record(db: Session, record_id: str) -> Optional[ExecutionRecord]:
        return db.query(ExecutionRecord).filter(ExecutionRecord.id == record_id).first()

    @staticmethod
    def get_stages(db: Session, record_id: str) -> List[ExecutionStage]:
        return (
            db.query(ExecutionStage)
            .filter(ExecutionStage.record_id == record_id)
            .order_by(ExecutionStage.id)
            .all()
        )

    @staticmethod
    def list_records(
        db: Session,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        pipeline_version: Optional[str] = None,
        dataset_id: Optional[str] = None,
        start_date: Optional[str] = None,   # ISO date string "YYYY-MM-DD"
        end_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[ExecutionRecord], int]:
        """Return (records, total_count) with optional filters."""
        q = db.query(ExecutionRecord)
        if dataset_id:
            q = q.filter(ExecutionRecord.dataset_id == dataset_id)
        if job_type:
            q = q.filter(ExecutionRecord.job_type == job_type)
        if status:
            q = q.filter(ExecutionRecord.status == status)
        if pipeline_version:
            q = q.filter(ExecutionRecord.pipeline_version == pipeline_version)
        if start_date:
            try:
                from datetime import date
                d = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
                q = q.filter(ExecutionRecord.created_at >= d)
            except Exception:
                pass
        if end_date:
            try:
                d = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
                q = q.filter(ExecutionRecord.created_at <= d)
            except Exception:
                pass
        total = q.count()
        records = q.order_by(ExecutionRecord.created_at.desc()).limit(limit).offset(offset).all()
        return records, total

    @staticmethod
    def get_records_for_job(db: Session, job_id: str) -> List[ExecutionRecord]:
        return (
            db.query(ExecutionRecord)
            .filter(ExecutionRecord.job_id == job_id)
            .order_by(ExecutionRecord.created_at.desc())
            .all()
        )

    # ── Summary stats ─────────────────────────────────────────────────────────

    @staticmethod
    def get_summary_stats(db: Session) -> dict:
        """Aggregate counts and averages for the monitoring dashboard."""
        try:
            total = db.query(sql_func.count(ExecutionRecord.id)).scalar() or 0
            by_type = {}
            for row in db.query(
                ExecutionRecord.job_type,
                sql_func.count(ExecutionRecord.id),
                sql_func.avg(ExecutionRecord.duration_ms),
                sql_func.sum(ExecutionRecord.articles_success),
            ).group_by(ExecutionRecord.job_type).all():
                by_type[row[0]] = {
                    "count": row[1],
                    "avg_duration_ms": _round(row[2]),
                    "total_articles": int(row[3] or 0),
                }
            by_status = {}
            for row in db.query(
                ExecutionRecord.status,
                sql_func.count(ExecutionRecord.id),
            ).group_by(ExecutionRecord.status).all():
                by_status[row[0]] = row[1]

            avg_ram = _round(
                db.query(sql_func.avg(ExecutionRecord.ram_peak_mb))
                .filter(ExecutionRecord.ram_peak_mb.isnot(None))
                .scalar()
            )
            avg_cpu = _round(
                db.query(sql_func.avg(ExecutionRecord.cpu_peak_pct))
                .filter(ExecutionRecord.cpu_peak_pct.isnot(None))
                .scalar()
            )
            last_ingestion = (
                db.query(ExecutionRecord)
                .filter(ExecutionRecord.job_type == "ingestion")
                .order_by(ExecutionRecord.created_at.desc())
                .first()
            )
            return {
                "total_records": total,
                "by_type": by_type,
                "by_status": by_status,
                "avg_ram_peak_mb": avg_ram,
                "avg_cpu_peak_pct": avg_cpu,
                "last_ingestion": last_ingestion.to_dict() if last_ingestion else None,
            }
        except Exception as e:
            return {"error": str(e), "total_records": 0}

    # ── Retention ─────────────────────────────────────────────────────────────

    @staticmethod
    def delete_old_records(
        db: Session,
        days_older_than: int = 90,
        keep_job_types: Optional[list] = None,
    ) -> int:
        """
        Delete execution records older than N days.
        Ingestion records are always kept by default.
        Returns the number of rows deleted.
        """
        keep = keep_job_types or ["ingestion"]
        cutoff = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=days_older_than)
        q = (
            db.query(ExecutionRecord)
            .filter(ExecutionRecord.created_at < cutoff)
            .filter(ExecutionRecord.job_type.notin_(keep))
        )
        count = q.count()
        q.delete(synchronize_session=False)
        db.commit()
        return count
