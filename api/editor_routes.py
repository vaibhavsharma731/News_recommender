"""
Editor and PostgreSQL API routes:
- Database health check & table setup
- CSV upload & background ingestion task triggering
- Ingestion progress polling
- Instant precomputed recommendations endpoint
"""
import os
import uuid
import threading
from pathlib import Path
from typing import Optional, List, Any, Union
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

import config
import time
from db.connection import get_db, check_db_health, init_db, resolve_schema, set_search_path
from db.models import Article
from db.repository import ArticleRepository
from db.monitoring_repository import MonitoringRepository
from monitoring.resource_monitor import take_snapshot
from db.pipeline import (
    run_ingestion_pipeline,
    _load_full_corpus,
    _precompute_for_article,
    _reverse_push_for_old,
)
from src.data_loader import clean_dataframe, clean_dataframe_live
from src.model_registry import get_embedder
from src.reranker import Reranker
from src.retrieval import BM25Index
from src.scoring import json_ids
from src.evidence import enrich_recommendation_evidence

router = APIRouter(prefix="/api/db", tags=["Database & Editor"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Cancel must actually stop the background worker, not just flip a DB flag.
# The worker polls this set between stages and between embedding batches.
_cancelled_jobs = set()
_cancel_lock = threading.Lock()


def _request_cancel(job_id: str) -> None:
    with _cancel_lock:
        _cancelled_jobs.add(job_id)


def _is_cancelled(job_id: str) -> bool:
    with _cancel_lock:
        return job_id in _cancelled_jobs


def _clear_cancel(job_id: str) -> None:
    with _cancel_lock:
        _cancelled_jobs.discard(job_id)


def _check_cancel(job_id: str) -> None:
    """Raise inside the worker the moment a user cancels the job."""
    if _is_cancelled(job_id):
        raise InterruptedError("Job cancelled by user.")


class _LiveProgress:
    """Thread-safe progress state persisted to the job row by a heartbeat.

    Some operations (pandas CSV read, per-article token budgeting, model load)
    cannot report sub-progress themselves.  Without a heartbeat the job row goes
    quiet for minutes and the UI correctly reports "no new update" — which looks
    identical to a dead worker.  The heartbeat rewrites the message with a live
    elapsed counter every 2s so activity is always visible, while the worker
    updates the real counts between chunks/batches.
    """

    # Each stage owns a slice of the bar so it only ever moves forward.
    STAGE_WEIGHTS = {
        "PARSE": (0.00, 0.08),
        "CLEAN": (0.08, 0.45),
        "STORAGE": (0.45, 0.60),
        "EMBED": (0.60, 1.00),
    }

    def __init__(self, job_id: str, dataset_id: str, t_start: float):
        self.job_id = job_id
        self.dataset_id = dataset_id
        self.t_start = t_start
        self._lock = threading.Lock()
        self.stage = "PARSE"
        self.detail = "Reading and validating uploaded CSV..."
        self.done = 0
        self.total = 0
        self.base_total = 0          # authoritative article count once known
        self.w0, self.w1 = self.STAGE_WEIGHTS["PARSE"]
        self._stop = threading.Event()
        self._thread = None

    def set(self, stage=None, detail=None, done=None, total=None, base_total=None):
        with self._lock:
            if stage is not None and stage != self.stage:
                self.stage = stage
                if stage in self.STAGE_WEIGHTS:
                    self.w0, self.w1 = self.STAGE_WEIGHTS[stage]
            if detail is not None:
                self.detail = detail
            if done is not None:
                self.done = int(done)
            if total is not None:
                self.total = int(total)
            if base_total is not None:
                self.base_total = max(1, int(base_total))

    def counts(self):
        """Weighted, monotonic (processed, total) pair for the progress bar."""
        with self._lock:
            frac = min(1.0, max(0.0, self.done / self.total)) if self.total else 0.0
            pct = self.w0 + (self.w1 - self.w0) * frac
            return round(pct * self.base_total), self.base_total

    def message(self):
        with self._lock:
            return f"STAGE:{self.stage}|{self.detail} · {int(time.time() - self.t_start)}s elapsed"

    def start(self):
        self.flush()
        self._thread = threading.Thread(
            target=self._beat, daemon=True, name=f"progress-{self.job_id}"
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def _beat(self):
        while not self._stop.wait(2.0):
            self.flush()

    def flush(self):
        from db.connection import SessionLocal, resolve_schema, set_search_path
        msg = self.message()
        done, total = self.counts()
        hb = None
        try:
            hb = SessionLocal()
            set_search_path(hb, resolve_schema(hb, self.dataset_id))
            ArticleRepository.update_job(
                hb, self.job_id,
                message=msg, processed_articles=done, total_articles=total or None,
            )
        except Exception as e:
            print(f"[LiveProgress] heartbeat failed: {e}")
        finally:
            if hb is not None:
                hb.close()


# The precomputed path runs a cross-encoder over every article, so uploads stay small.
# The live path only cleans + embeds, so it accepts ~100k-row CSVs (300-400 MB).
# Both are env-overridable without a code change.
MAX_FILE_SIZE = int(os.getenv("NEWSREC_MAX_UPLOAD_MB", "50")) * 1024 * 1024
MAX_LIVE_FILE_SIZE = int(os.getenv("NEWSREC_MAX_LIVE_UPLOAD_MB", "600")) * 1024 * 1024


def _require_idle(db: Session, dataset_id: str = "default") -> None:
    """Only one ingestion job at a time per dataset: two would fight for CPU and the model cache."""
    schema = resolve_schema(db, dataset_id)
    set_search_path(db, schema)
    active = ArticleRepository.get_active_job(db)
    if active:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        job_time = active.updated_at or active.created_at
        is_stale = False
        if job_time:
            if job_time.tzinfo is None:
                diff_sec = (datetime.now() - job_time).total_seconds()
            else:
                diff_sec = (now - job_time).total_seconds()
            if diff_sec > 600:  # 10 minutes without progress
                is_stale = True

        if is_stale:
            active.status = "failed"
            active.error = "Auto-failed stale job: no progress recorded for > 10 minutes."
            active.message = "Job timed out or interrupted."
            db.commit()
        else:
            raise HTTPException(
                status_code=409,
                detail=f"Ingestion job '{active.id}' ({active.filename}) in dataset '{dataset_id}' is currently running. Please wait for it to finish.",
            )


def _launch(db: Session, job_id: str, filename: str, csv_source: Path, dataset_id: str = "default", **kwargs) -> None:
    """Record the job, then run the (long) ingestion pipeline off the request thread."""
    schema = resolve_schema(db, dataset_id)
    set_search_path(db, schema)
    ArticleRepository.create_job(db, job_id=job_id, filename=filename, pipeline_type="precomputed")
    threading.Thread(
        target=run_ingestion_pipeline,
        args=(csv_source, job_id),
        kwargs={**kwargs, "dataset_id": dataset_id},
        daemon=True,
    ).start()


def _reload_live_engine(dataset_id: str = "default") -> None:
    """Re-sync the in-process engine after the corpus changed (late import: circular)."""
    from api.server import reload_engine_from_postgres
    reload_engine_from_postgres(dataset_id=dataset_id)


@router.get("/health")
def db_health(dataset: Optional[str] = Query("default")):
    """Returns PostgreSQL connection status, pgvector extension readiness, and article counts for the dataset."""
    health = check_db_health()
    if health["connected"]:
        try:
            from db.connection import SessionLocal
            with SessionLocal() as db:
                schema = resolve_schema(db, dataset)
                set_search_path(db, schema)
                health["dataset"] = dataset or "default"
                health["schema"] = schema
                health["total_articles"] = ArticleRepository.count_articles(db)
                health["embedded_articles"] = ArticleRepository.count_embedded_articles(db)
                health["total_recommendations"] = ArticleRepository.count_total_recommendations(db)
        except Exception as e:
            health["stats_error"] = str(e)
    return health


@router.post("/init")
def trigger_db_init():
    """Initializes pgvector extension and creates all database tables."""
    success = init_db()
    if not success:
        raise HTTPException(status_code=500, detail="Database initialization failed. Check logs.")
    return {"status": "success", "message": "PostgreSQL tables and extensions verified and ready."}


@router.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    dataset: Optional[str] = Query("default"),
    db: Session = Depends(get_db)
):
    """
    Accepts CSV upload with security validation (path sanitization & size limit),
    launches background pipeline (Cleaning -> Postgres -> Embeddings -> pgvector Top 50 -> Reranker Top 10)
    targeted at the selected isolated dataset.
    """
    target_ds = dataset or "default"
    _require_idle(db, dataset_id=target_ds)  # before writing 50 MB, not after

    safe_filename = Path(file.filename or "articles.csv").name
    if not safe_filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    saved_path = UPLOAD_DIR / f"{job_id}_{safe_filename}"

    # Safe chunked write with max size enforcement
    size = 0
    with open(saved_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                buffer.close()
                saved_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE // (1024 * 1024)}MB.")
            buffer.write(chunk)

    _launch(db, job_id, safe_filename, saved_path, dataset_id=target_ds)

    return {
        "job_id": job_id,
        "filename": safe_filename,
        "dataset": target_ds,
        "status": "processing",
        "message": f"File validated and uploaded to dataset '{target_ds}'. Ingestion pipeline running in background."
    }


class LivePreprocessPayload(BaseModel):
    dataset: Optional[str] = Field("default", description="Target dataset")
    batch_size: Optional[int] = Field(64, ge=16, le=256, description="Embedding batch size")
    recompute_all: Optional[bool] = Field(False, description="Re-encode all articles even if they already have embeddings")


def _run_live_preprocessing_worker(dataset_id: str, job_id: str, batch_size: int = 64, recompute_all: bool = False):
    """Background worker for live preprocessing: embeds articles without Cross-Encoder precomputation."""
    from db.connection import SessionLocal
    db = SessionLocal()
    t_start = time.time()
    resource_monitor = None
    exec_record_id = None
    schema = resolve_schema(db, dataset_id)
    set_search_path(db, schema)

    try:
        try:
            from monitoring.resource_monitor import BackgroundResourceMonitor
            resource_monitor = BackgroundResourceMonitor(interval_s=0.5)
            resource_monitor.start()
            exec_rec = MonitoringRepository.create_record(
                db,
                job_type="live_preprocessing",
                pipeline_version=config.get_pipeline_version(),
                job_id=job_id,
                dataset_id=dataset_id,
                metadata={"batch_size": batch_size, "recompute_all": recompute_all, "schema": schema},
            )
            exec_record_id = exec_rec.id
        except Exception as mon_err:
            print(f"[Live Preprocessing Monitoring] {mon_err}")

        # Find candidate articles
        query = db.query(Article)
        if not recompute_all:
            query = query.filter(Article.embedding.is_(None))
        articles_to_process = query.all()
        total = len(articles_to_process)

        ArticleRepository.update_job(
            db,
            job_id,
            total_articles=total,
            processed_articles=0,
            status="processing",
            message=f"STAGE:EMBED|Found {total} articles needing live embedding in '{dataset_id}'...",
        )

        if total == 0:
            ArticleRepository.update_job(
                db,
                job_id,
                status="completed",
                message=f"STAGE:COMPLETE|All articles in dataset '{dataset_id}' are already embedded and ready for live recommendations.",
            )
            return

        embedder = get_embedder()
        processed_count = 0

        # Process in batches
        for i in range(0, total, batch_size):
            chunk = articles_to_process[i:i + batch_size]
            texts = []
            ids = []
            for a in chunk:
                t = a.text_embed
                if not t:
                    t = f"{a.title}. {a.description or ''} {a.body or ''}"[:config.EMBED_TOKEN_BUDGET * 4]
                texts.append(t)
                ids.append(a.id)

            vecs = embedder.encode(texts, batch_size=len(texts), normalize_embeddings=True)
            ArticleRepository.bulk_set_embeddings(db, ids, vecs)
            processed_count += len(chunk)

            ArticleRepository.update_job(
                db,
                job_id,
                processed_articles=processed_count,
                message=f"STAGE:EMBED|Embedded {processed_count} / {total} articles ({round(processed_count / total * 100, 1)}%)...",
            )

        elapsed = round(time.time() - t_start, 2)
        if exec_record_id:
            from db.pipeline import _finish_monitoring
            _finish_monitoring(
                db, exec_record_id, resource_monitor, "completed",
                t_start, total, 0, {"embed_total_ms": elapsed * 1000}
            )

        ArticleRepository.update_job(
            db,
            job_id,
            status="completed",
            processed_articles=total,
            message=f"STAGE:COMPLETE|Successfully preprocessed {total} articles in {elapsed}s for live recommendations.",
        )

        # Reload engine for live search
        _reload_live_engine(dataset_id=dataset_id)

    except Exception as e:
        print(f"[Live Preprocessing Error] {e}")
        ArticleRepository.update_job(
            db,
            job_id,
            status="failed",
            message=f"Live preprocessing failed: {e}",
            error=str(e),
        )
        if exec_record_id:
            try:
                from db.pipeline import _finish_monitoring
                _finish_monitoring(
                    db, exec_record_id, resource_monitor, "failed",
                    t_start, 0, 0, {}, error_message=str(e)
                )
            except Exception:
                pass
    finally:
        db.close()


@router.post("/preprocess-live")
def preprocess_live(payload: LivePreprocessPayload, db: Session = Depends(get_db)):
    """
    Pipeline B: Fast Live Preprocessing
    Generates BGE-Small dense embeddings and lexical indexes for articles in the targeted dataset
    WITHOUT running expensive Cross-Encoder reranking or generating full-archive recommendation tables.
    """
    target_ds = (payload.dataset or "default").strip()
    _require_idle(db, dataset_id=target_ds)
    job_id = f"job_live_{uuid.uuid4().hex[:6]}"
    schema = resolve_schema(db, target_ds)
    set_search_path(db, schema)
    ArticleRepository.create_job(db, job_id=job_id, filename=f"live_preprocess_{target_ds}", pipeline_type="live")
    threading.Thread(
        target=_run_live_preprocessing_worker,
        args=(target_ds, job_id, payload.batch_size, payload.recompute_all),
        daemon=True,
    ).start()
    return {
        "job_id": job_id,
        "dataset": target_ds,
        "status": "processing",
        "pipeline_type": "live",
        "message": f"Started live embedding generation for dataset '{target_ds}'."
    }


def _run_live_csv_worker(dataset_id: str, job_id: str, csv_source: Path, batch_size: int = 64):
    """Background worker for live CSV ingestion: cleans, stores and embeds (never reranks)."""
    from db.connection import SessionLocal
    db = SessionLocal()
    t_start = time.time()
    resource_monitor = None
    exec_record_id = None
    total_clean = 0
    schema = resolve_schema(db, dataset_id)
    set_search_path(db, schema)

    # The heartbeat owns message/count columns so the UI always sees real
    # activity, even while pandas or the tokenizer is busy for minutes.
    progress = _LiveProgress(job_id, dataset_id, t_start)

    try:
        progress.start()
        ArticleRepository.update_job(db, job_id, status="processing")

        try:
            from monitoring.resource_monitor import BackgroundResourceMonitor
            resource_monitor = BackgroundResourceMonitor(interval_s=0.5)
            resource_monitor.start()
            exec_rec = MonitoringRepository.create_record(
                db,
                job_type="live_csv_ingestion",
                pipeline_version=config.get_pipeline_version(),
                job_id=job_id,
                dataset_id=dataset_id,
                metadata={"csv_file": csv_source.name, "batch_size": batch_size, "schema": schema},
            )
            exec_record_id = exec_rec.id
        except Exception as mon_err:
            print(f"[Live CSV Monitoring] {mon_err}")

        # 1. PARSE — chunked read so the row count visibly climbs on a 132 MB file.
        est_rows = 0
        try:
            with open(csv_source, "rb") as fh:
                est_rows = max(1, sum(1 for _ in fh) - 1)
        except Exception:
            pass
        if est_rows:
            progress.set(base_total=est_rows)
        frames = []
        rows_read = 0
        for chunk_df in pd.read_csv(csv_source, low_memory=False, chunksize=5000):
            _check_cancel(job_id)
            frames.append(chunk_df)
            rows_read += len(chunk_df)
            progress.set(
                done=rows_read, total=est_rows or None,
                detail=f"Reading CSV — {rows_read:,} / {est_rows:,} rows",
            )
        raw_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        del frames
        _check_cancel(job_id)

        # 2. CLEAN — the authoritative cleaner reports sub-progress via callback.
        def on_clean(done, total, label):
            _check_cancel(job_id)
            progress.set(done=done, total=total, detail=f"{label} — {done:,} / {total:,}")

        progress.set(stage="CLEAN", done=0, total=len(raw_df), detail="Preparing text...")
        clean_df = clean_dataframe_live(raw_df, progress_callback=on_clean)
        total_clean = len(clean_df)
        progress.set(base_total=total_clean)
        del raw_df
        _check_cancel(job_id)

        # 3. STORAGE — report every 500-row upsert chunk (and during row build).
        progress.set(
            stage="STORAGE", done=0, total=total_clean,
            detail=f"Preparing 0 / {total_clean:,} rows for storage",
        )
        rows_to_insert = []
        for idx, (_, r) in enumerate(clean_df.iterrows(), start=1):
            rows_to_insert.append({
                "id": str(r["id"]),
                "title": str(r["title"]),
                "description": str(r["description"]),
                "body": str(r["body"]),
                "section": str(r["section"]),
                "subsection": str(r["subsection"]),
                "date": r["date"].to_pydatetime() if pd.notna(r["date"]) else None,
                "url": str(r["url"]),
                "image": str(r["image"]),
                "is_live": bool(r["is_live"]),
                "premium": bool(r.get("premium", False)),
                "page_views": int(r["page_views"]),
                "unique_users": int(r["unique_users"]),
                "linked_ids": json_ids(r["linked_ids"]),
                "text_embed": str(r["text_embed"]),
                "text_bm25": str(r["text_bm25"]),
                "text_rerank": str(r["text_rerank"]),
                "embed_token_count": int(r.get("embed_token_count", 0)),
                "rerank_token_count": int(r.get("rerank_token_count", 0)),
                "is_truncated": bool(r.get("is_truncated", False)),
                "recipe_version": str(r.get("recipe_version", config.TEXT_RECIPE_VERSION)),
            })
            if idx % 5000 == 0:
                progress.set(done=idx, detail=f"Preparing {idx:,} / {total_clean:,} rows for storage")

        for chunk_start in range(0, len(rows_to_insert), 500):
            _check_cancel(job_id)
            ArticleRepository.bulk_upsert_articles_fast(db, rows_to_insert[chunk_start:chunk_start + 500])
            done = min(chunk_start + 500, len(rows_to_insert))
            progress.set(done=done, detail=f"Upserting {done:,} / {len(rows_to_insert):,} articles into PostgreSQL")

        # 4. EMBED — keep the exact "Embedded N / M" wording the UI also parses.
        progress.set(
            stage="EMBED", done=0, total=total_clean,
            detail="Loading BGE embedding model...",
        )
        embedder = get_embedder()
        progress.set(detail=f"Generating embeddings — 0 / {total_clean} articles (0.0%)")
        processed_count = 0
        for i in range(0, total_clean, batch_size):
            _check_cancel(job_id)
            chunk = clean_df.iloc[i:i + batch_size]
            texts = chunk["text_embed"].tolist()
            ids = [str(x) for x in chunk["id"].tolist()]
            vecs = embedder.encode(texts, batch_size=len(texts), normalize_embeddings=True)
            ArticleRepository.bulk_set_embeddings(db, ids, vecs)
            processed_count += len(chunk)
            progress.set(
                done=processed_count,
                detail=f"Embedded {processed_count} / {total_clean} articles ({round(processed_count / max(total_clean, 1) * 100, 1)}%)",
            )

        # Stop the heartbeat before writing the terminal row so it cannot be
        # overwritten by a late beat.
        progress.stop()
        elapsed = round(time.time() - t_start, 2)
        throughput = round(total_clean / max(elapsed, 0.001), 1)

        if exec_record_id:
            from db.pipeline import _finish_monitoring
            _finish_monitoring(
                db, exec_record_id, resource_monitor, "completed",
                t_start, total_clean, 0, {"embed_total_ms": elapsed * 1000, "throughput_art_per_sec": throughput}
            )

        ArticleRepository.update_job(
            db, job_id, status="completed", processed_articles=total_clean,
            message=f"STAGE:COMPLETE|Successfully ingested and embedded {total_clean} articles in {elapsed}s ({throughput} art/s). Ready for live search."
        )

        _reload_live_engine(dataset_id=dataset_id)

    except Exception as e:
        progress.stop()
        print(f"[Live CSV Ingestion Error] {e}")
        ArticleRepository.update_job(
            db, job_id, status="failed", message=f"Live CSV ingestion failed: {e}", error=str(e)
        )
        if exec_record_id:
            try:
                from db.pipeline import _finish_monitoring
                _finish_monitoring(db, exec_record_id, resource_monitor, "failed", t_start, 0, 0, {}, error_message=str(e))
            except Exception:
                pass
    finally:
        progress.stop()
        db.close()


@router.post("/upload-live-csv")
async def upload_live_csv(
    file: UploadFile = File(...),
    dataset: Optional[str] = Query("default"),
    batch_size: Optional[int] = Query(64, ge=16, le=256),
    db: Session = Depends(get_db),
):
    """
    Pipeline B: Fast Live CSV Batch Ingestion.
    Uploads news articles, cleans them, stores them in PostgreSQL, and encodes 384d BGE dense embeddings.
    Completely skips Cross-Encoder reranking and recommendation table generation.
    """
    target_ds = dataset or "default"
    _require_idle(db, dataset_id=target_ds)

    safe_filename = Path(file.filename or "articles.csv").name
    if not safe_filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    job_id = f"job_live_csv_{uuid.uuid4().hex[:6]}"
    saved_path = UPLOAD_DIR / f"{job_id}_{safe_filename}"

    size = 0
    with open(saved_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_LIVE_FILE_SIZE:
                buffer.close()
                saved_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum allowed size is {MAX_LIVE_FILE_SIZE // (1024 * 1024)}MB.",
                )
            buffer.write(chunk)

    schema = resolve_schema(db, target_ds)
    set_search_path(db, schema)
    ArticleRepository.create_job(db, job_id=job_id, filename=safe_filename, pipeline_type="live")
    _clear_cancel(job_id)  # fresh job id; drop any stale cancel marker
    threading.Thread(
        target=_run_live_csv_worker,
        args=(target_ds, job_id, saved_path, batch_size or 64),
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "filename": safe_filename,
        "dataset": target_ds,
        "status": "processing",
        "pipeline_type": "live",
        "message": f"File uploaded to live dataset '{target_ds}'. Fast embedding pipeline running in background.",
    }



@router.post("/ingest-default")
def ingest_default_dataset(
    limit: Optional[int] = Query(None, description="Max rows to process (e.g. 100 for fast demo)"),
    db: Session = Depends(get_db)
):
    """
    One-click ingestion of the existing sample dataset into PostgreSQL.
    """
    _require_idle(db)

    default_csv = Path(__file__).resolve().parent.parent / "data" / "articles_clean_train_80.csv"
    if not default_csv.exists():
        raise HTTPException(status_code=404, detail="Default dataset file not found on server.")

    job_id = f"job_demo_{uuid.uuid4().hex[:6]}"
    _launch(db, job_id, "articles_clean_train_80.csv", default_csv, limit=limit)

    return {
        "job_id": job_id,
        "filename": "articles_clean_train_80.csv",
        "status": "processing",
        "message": f"Started ingestion of sample dataset into PostgreSQL (limit={limit or 'all'})."
    }



@router.get("/jobs/{job_id}")
def get_job_status(job_id: str, dataset: Optional[str] = Query(None), db: Session = Depends(get_db)):
    """Poll progress of an ingestion job for the Editor progress bar."""
    # 1. Target dataset if provided
    if dataset:
        schema = resolve_schema(db, dataset)
        set_search_path(db, schema)
        job = ArticleRepository.get_job(db, job_id)
        if job:
            return job.to_dict()

    # 2. Check public schema
    set_search_path(db, "public")
    job = ArticleRepository.get_job(db, job_id)
    if job:
        return job.to_dict()

    # 3. Check all other registered dataset schemas
    try:
        schemas = db.execute(text("SELECT schema_name FROM public.datasets WHERE schema_name != 'public'")).fetchall()
        for (sch,) in schemas:
            if not sch:
                continue
            set_search_path(db, str(sch))
            job = ArticleRepository.get_job(db, job_id)
            if job:
                return job.to_dict()
    except Exception:
        pass

    raise HTTPException(status_code=404, detail="Job not found")


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, dataset: Optional[str] = Query(None), db: Session = Depends(get_db)):
    """Cancel or reset a running/stuck ingestion job."""
    def try_cancel(sch: str):
        set_search_path(db, sch)
        j = ArticleRepository.get_job(db, job_id)
        if j and j.status in ("pending", "processing"):
            # Tell the live daemon thread to stop at its next chunk/batch. A DB
            # flag alone never stopped the worker — it kept embedding for minutes.
            _request_cancel(job_id)
            j.status = "failed"
            j.error = "Canceled by user request."
            j.message = "Job manually stopped."
            db.commit()
            return j
        return None

    if dataset:
        cancelled = try_cancel(resolve_schema(db, dataset))
        if cancelled:
            return {"status": "ok", "job_id": job_id, "message": "Job cancelled"}

    cancelled = try_cancel("public")
    if cancelled:
        return {"status": "ok", "job_id": job_id, "message": "Job cancelled"}

    try:
        schemas = db.execute(text("SELECT schema_name FROM public.datasets WHERE schema_name != 'public'")).fetchall()
        for (sch,) in schemas:
            cancelled = try_cancel(str(sch))
            if cancelled:
                return {"status": "ok", "job_id": job_id, "message": "Job cancelled"}
    except Exception:
        pass

    return {"status": "ok", "job_id": job_id, "message": "Job updated or already finished"}


@router.post("/jobs/reset-stuck")
def reset_stuck_jobs(db: Session = Depends(get_db)):
    """Reset all pending/processing jobs across all schemas."""
    total = ArticleRepository.fail_orphaned_jobs(db)
    return {"status": "ok", "reset_count": total}


@router.get("/jobs")
def list_jobs(
    limit: int = 10,
    dataset: Optional[str] = Query("default"),
    pipeline_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """List recent ingestion jobs (upload history) plus an aggregate summary for the Editor dashboard."""
    schema = resolve_schema(db, dataset)
    set_search_path(db, schema)
    jobs = ArticleRepository.list_recent_jobs(db, limit=limit, pipeline_type=pipeline_type)
    summary = ArticleRepository.get_ingestion_summary(db)
    return {
        "jobs": [j.to_dict() for j in jobs],
        "summary": summary,
        "dataset": dataset or "default",
        "pipeline_type": pipeline_type,
    }


@router.get("/recommendations/{article_id}")
def get_precomputed_recs(
    article_id: str,
    limit: int = 10,
    diversify: bool = True,
    dataset: Optional[str] = Query("default"),
    db: Session = Depends(get_db),
):
    """
    Ultra-fast sub-5ms endpoint: retrieves precomputed top-10 recommendations directly
    from PostgreSQL article_recommendations table with complete 7-item evaluation evidence.
    Supports diversify=True (MMR diversified) and diversify=False (Pure relevance rank).
    """
    schema = resolve_schema(db, dataset)
    set_search_path(db, schema)
    source_art = db.query(Article).filter(Article.id == article_id).first()
    source_dict = source_art.to_dict() if source_art else None
    recs = ArticleRepository.get_precomputed_recommendations(db, article_id, limit=limit, diversify=diversify)
    enrich_recommendation_evidence(source_dict, recs)

    return {
        "article_id": article_id,
        "source_article": source_dict,
        "source": "postgresql_precomputed",
        "dataset": dataset or "default",
        "diversify": diversify,
        "count": len(recs),
        "recommendations": recs,
        "scoring_metadata": {
            "weights": {
                "rerank": config.WEIGHT_RERANK,
                "bm25": config.WEIGHT_BM25,
                "dense": config.WEIGHT_DENSE,
            },
            "mmr": {
                "lambda": config.MMR_LAMBDA,
                "theta": config.MMR_THETA,
                "top_k": config.TOP_K,
                "applied": bool(diversify),
            },
            "boost_caps": {
                "editorial": 1.40,
                "section_focused": getattr(config, "SECTION_BOOST_FOCUSED", 1.100),
                "section_umbrella": getattr(config, "SECTION_BOOST_UMBRELLA", 1.020),
                "subsection_focused": getattr(config, "SUBSECTION_BOOST_FOCUSED", 1.050),
                "subsection_umbrella": getattr(config, "SUBSECTION_BOOST_UMBRELLA", 1.020),
                "popularity": config.POPULARITY_BOOST_MULT,
                "recency_max": getattr(config, "RECENCY_MAX_BOOST_MULT", 1.040),
            },
            "agreement_bounds": {"min": -0.03, "max": 0.03, "additive": True},
            "pipeline_version": config.get_pipeline_version(),
        },
    }



@router.post("/clear")
def clear_database(db: Session = Depends(get_db)):
    """
    Clears all stored articles, precomputed recommendations, and ingestion jobs from PostgreSQL.
    Allows editor to reset and start completely fresh from scratch.
    """
    try:
        stats = ArticleRepository.clear_all_data(db)
        try:
            _reload_live_engine()  # the in-memory corpus is now empty
        except Exception:
            pass

        return {
            "status": "success",
            "message": "Database cleared successfully. All tables reset.",
            "deleted": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


@router.post("/reload")
def reload_engine():
    """Trigger reload of in-memory engine from PostgreSQL articles."""
    try:
        _reload_live_engine()
        return {"status": "success", "message": "In-memory engine synchronized with PostgreSQL."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


class ManualArticlePayload(BaseModel):
    title: str = Field(..., min_length=3, description="Article headline")
    description: Optional[str] = Field("", description="Summary or lead")
    body: str = Field(..., min_length=10, description="Full story content")
    section: Optional[str] = Field("india", description="Primary category")
    subsection: Optional[str] = Field("", description="Subcategory or topic tag")
    date: Optional[str] = Field(None, description="Publish date ISO string")
    url: Optional[str] = Field("", description="Canonical article URL")
    image: Optional[str] = Field("", description="Article lead image URL")
    is_live: Optional[bool] = Field(False, description="Whether article is a live blog")
    page_views: Optional[int] = Field(0, ge=0, description="Initial page views")
    unique_users: Optional[int] = Field(0, ge=0, description="Initial unique users")
    premium: Optional[bool] = Field(False, description="Paywalled / premium article")
    linked_ids: Optional[Any] = Field([], description="Editorially linked article IDs")
    dataset: Optional[str] = Field("default", description="Target dataset")
    precompute_recommendations: Optional[bool] = Field(True, description="Precompute top-10 recs")
    reverse_push: Optional[bool] = Field(True, description="Update matching older articles")


@router.post("/articles/manual")
def add_manual_article(payload: ManualArticlePayload, db: Session = Depends(get_db)):
    """
    Ingests a single article published manually through the Editor Studio:
    1. Validates and runs through clean_dataframe() ensuring all token budgets & normalizations.
    2. Encodes embedding vector (384d).
    3. Upserts article into PostgreSQL database in the targeted dataset.
    4. Computes top-10 precomputed recommendations against the corpus.
    5. Optionally runs reverse-push for existing articles that match the new article.
    6. Triggers live in-memory search engine reload.
    7. Emits an execution record for centralised performance monitoring.
    """
    target_ds = (payload.dataset or "default").strip()
    _require_idle(db, dataset_id=target_ds)
    schema = resolve_schema(db, target_ds)
    set_search_path(db, schema)

    t_start = time.perf_counter()
    snap_start = take_snapshot()

    title = payload.title.strip()
    body = payload.body.strip()
    if not title or not body:
        raise HTTPException(status_code=400, detail="Title and body cannot be empty.")

    date_val = payload.date
    if not date_val:
        date_val = pd.Timestamp.now(tz="UTC").isoformat()

    raw_row = {
        "title": title,
        "description": (payload.description or "").strip(),
        "body": body,
        "section": (payload.section or "india").strip().lower(),
        "subsection": (payload.subsection or "").strip().lower(),
        "date": date_val,
        "url": (payload.url or "").strip(),
        "image": (payload.image or "").strip(),
        "is_live": bool(payload.is_live),
        "page_views": int(payload.page_views or 0),
        "unique_users": int(payload.unique_users or 0),
        "premium": bool(payload.premium or False),
        "linked_ids": payload.linked_ids or [],
    }

    try:
        clean_df = clean_dataframe(pd.DataFrame([raw_row]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to clean article: {e}")

    r = clean_df.iloc[0]
    art_id = str(r["id"])

    # 1. Upsert article record into PostgreSQL (targeted schema)
    row_dict = {
        "id": art_id,
        "title": str(r["title"]),
        "description": str(r["description"]),
        "body": str(r["body"]),
        "section": str(r["section"]),
        "subsection": str(r["subsection"]),
        "date": r["date"].to_pydatetime() if pd.notna(r["date"]) else None,
        "url": str(r["url"]),
        "image": str(r["image"]),
        "is_live": bool(r["is_live"]),
        "premium": bool(r.get("premium", False)),
        "page_views": int(r["page_views"]),
        "unique_users": int(r["unique_users"]),
        "linked_ids": json_ids(r["linked_ids"]),
        "text_embed": str(r["text_embed"]),
        "text_bm25": str(r["text_bm25"]),
        "text_rerank": str(r["text_rerank"]),
        "embed_token_count": int(r.get("embed_token_count", 0)),
        "rerank_token_count": int(r.get("rerank_token_count", 0)),
        "is_truncated": bool(r.get("is_truncated", False)),
        "recipe_version": str(r.get("recipe_version", config.TEXT_RECIPE_VERSION)),
    }
    ArticleRepository.bulk_upsert_articles_fast(db, [row_dict])

    # 2. Compute 384d dense vector embedding
    try:
        embedder = get_embedder()
        vector = embedder.encode([str(r["text_embed"])], normalize_embeddings=True)[0]
        ArticleRepository.bulk_set_embeddings(db, [art_id], [vector])
    except Exception as e:
        print(f"[Manual Article] Embedding generation failed: {e}")

    # 3. Precompute recommendations & reverse-push
    recs = []
    refreshed_infos = []
    if payload.precompute_recommendations:
        try:
            records, emb, id_to_pos = _load_full_corpus(db)
            if art_id in id_to_pos and len(records) > 1:
                idx = id_to_pos[art_id]
                bm25 = BM25Index(records)
                reranker = Reranker()
                max_views = max((rec.get("page_views") or 0 for rec in records), default=1)
                max_date = max((rec.get("date") for rec in records if rec.get("date") is not None), default=None)

                recs, cands = _precompute_for_article(
                    idx, records, bm25, reranker, emb, max_views, max_date,
                    config.N_RETRIEVE, config.PRECOMPUTE_CANDIDATES
                )
                if recs:
                    ArticleRepository.save_recommendations(db, art_id, recs)

                if payload.reverse_push and cands:
                    refreshed_infos = _reverse_push_for_old(
                        db, [idx], {idx: cands}, records, emb, bm25, reranker,
                        max_views, max_date, {art_id}, id_to_pos
                    )
        except Exception as e:
            print(f"[Manual Article] Precomputation failed: {e}")

    # 4. Trigger live search engine reload for this dataset
    try:
        _reload_live_engine(dataset_id=target_ds)
    except Exception as e:
        print(f"[Manual Article] Live engine reload failed: {e}")

    # 5. Record monitoring telemetry
    dur_ms = round((time.perf_counter() - t_start) * 1000, 2)
    snap_end = take_snapshot()
    try:
        exec_rec = MonitoringRepository.create_record(
            db,
            job_type="manual_ingestion",
            pipeline_version=config.get_pipeline_version(),
            dataset_id=target_ds,
            articles_total=1,
            metadata={"article_id": art_id, "title": title, "schema": schema},
        )
        MonitoringRepository.close_record(
            db,
            exec_rec.id,
            status="completed",
            duration_ms=dur_ms,
            articles_success=1,
            articles_failed=0,
            resource_summary={
                "cpu_start_pct": snap_start.cpu_percent,
                "cpu_peak_pct": snap_end.cpu_percent,
                "ram_start_mb": snap_start.ram_rss_mb,
                "ram_peak_mb": snap_end.ram_rss_mb,
                "ram_end_mb": snap_end.ram_rss_mb,
                "gpu_available": snap_end.gpu_available,
                "gpu_peak_mem_mb": snap_end.gpu_memory_used_mb,
            },
            stage_timings={"total_ms": dur_ms, "recs_count": len(recs)},
        )
    except Exception as mon_err:
        print(f"[Manual Article Monitoring] {mon_err}")

    return {
        "status": "success",
        "message": f"Article '{art_id}' successfully published and processed.",
        "article_id": art_id,
        "article": {
            "id": art_id,
            "title": str(r["title"]),
            "section": str(r["section"]),
            "subsection": str(r["subsection"]),
            "date": r["date"].isoformat() if pd.notna(r["date"]) else None,
            "page_views": int(r["page_views"]),
            "is_live": bool(r["is_live"]),
            "url": str(r["url"]),
        },
        "recommendations_count": len(recs),
        "refreshed_old_articles": len(refreshed_infos),
        "refreshed_articles_summary": [
            f"{info['id']}: {info['title'][:40]}..." for info in refreshed_infos[:5]
        ],
    }


class ManualLiveArticlePayload(BaseModel):
    title: str = Field(..., min_length=3, description="Article headline")
    description: Optional[str] = Field("", description="Summary or lead")
    body: str = Field(..., min_length=10, description="Full story content")
    section: Optional[str] = Field("india", description="Primary category")
    subsection: Optional[str] = Field("", description="Subcategory or topic tag")
    date: Optional[str] = Field(None, description="Publish date ISO string")
    url: Optional[str] = Field("", description="Canonical article URL")
    image: Optional[str] = Field("", description="Article lead image URL")
    is_live: Optional[bool] = Field(False, description="Whether article is a live blog")
    page_views: Optional[int] = Field(0, ge=0, description="Initial page views")
    unique_users: Optional[int] = Field(0, ge=0, description="Initial unique users")
    premium: Optional[bool] = Field(False, description="Paywalled / premium article")
    linked_ids: Optional[Any] = Field([], description="Editorially linked article IDs")
    dataset: Optional[str] = Field("default", description="Target dataset")


@router.post("/articles/manual-live")
def add_manual_live_article(payload: ManualLiveArticlePayload, db: Session = Depends(get_db)):
    """
    Pipeline B: Fast Single Article Live Ingestion.
    Cleans text representations, generates 384d BGE embedding, and stores into PostgreSQL.
    Completely skips Cross-Encoder reranking and precomputed recommendation tables.
    Uses ``clean_dataframe_live`` so no reranker text is tokenised either.
    """
    target_ds = (payload.dataset or "default").strip()
    _require_idle(db, dataset_id=target_ds)
    schema = resolve_schema(db, target_ds)
    set_search_path(db, schema)

    t_start = time.perf_counter()
    snap_start = take_snapshot()

    title = payload.title.strip()
    body = payload.body.strip()
    if not title or not body:
        raise HTTPException(status_code=400, detail="Title and body cannot be empty.")

    date_val = payload.date
    if not date_val:
        date_val = pd.Timestamp.now(tz="UTC").isoformat()

    raw_row = {
        "title": title,
        "description": (payload.description or "").strip(),
        "body": body,
        "section": (payload.section or "india").strip().lower(),
        "subsection": (payload.subsection or "").strip().lower(),
        "date": date_val,
        "url": (payload.url or "").strip(),
        "image": (payload.image or "").strip(),
        "is_live": bool(payload.is_live),
        "page_views": int(payload.page_views or 0),
        "unique_users": int(payload.unique_users or 0),
        "premium": bool(payload.premium or False),
        "linked_ids": payload.linked_ids or [],
    }

    try:
        clean_df = clean_dataframe_live(pd.DataFrame([raw_row]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to clean article: {e}")

    r = clean_df.iloc[0]
    art_id = str(r["id"])

    row_dict = {
        "id": art_id,
        "title": str(r["title"]),
        "description": str(r["description"]),
        "body": str(r["body"]),
        "section": str(r["section"]),
        "subsection": str(r["subsection"]),
        "date": r["date"].to_pydatetime() if pd.notna(r["date"]) else None,
        "url": str(r["url"]),
        "image": str(r["image"]),
        "is_live": bool(r["is_live"]),
        "premium": bool(r.get("premium", False)),
        "page_views": int(r["page_views"]),
        "unique_users": int(r["unique_users"]),
        "linked_ids": json_ids(r["linked_ids"]),
        "text_embed": str(r["text_embed"]),
        "text_bm25": str(r["text_bm25"]),
        "text_rerank": str(r["text_rerank"]),
        "embed_token_count": int(r.get("embed_token_count", 0)),
        "rerank_token_count": int(r.get("rerank_token_count", 0)),
        "is_truncated": bool(r.get("is_truncated", False)),
        "recipe_version": str(r.get("recipe_version", config.TEXT_RECIPE_VERSION)),
    }

    # Generate 384d dense vector embedding before inserting for atomic storage
    try:
        embedder = get_embedder()
        vector = embedder.encode([str(r["text_embed"])], normalize_embeddings=True)[0]
        row_dict["embedding"] = vector.tolist()
    except Exception as e:
        print(f"[Manual Live Article] Embedding generation failed: {e}")

    ArticleRepository.bulk_upsert_articles_fast(db, [row_dict])

    try:
        _reload_live_engine(dataset_id=target_ds)
    except Exception as e:
        print(f"[Manual Live Article] Live engine reload failed: {e}")

    dur_ms = round((time.perf_counter() - t_start) * 1000, 2)
    snap_end = take_snapshot()
    try:
        exec_rec = MonitoringRepository.create_record(
            db,
            job_type="live_manual_ingestion",
            pipeline_version=config.get_pipeline_version(),
            dataset_id=target_ds,
            articles_total=1,
            metadata={"article_id": art_id, "title": title, "schema": schema},
        )
        MonitoringRepository.close_record(
            db,
            exec_rec.id,
            status="completed",
            duration_ms=dur_ms,
            articles_success=1,
            articles_failed=0,
            resource_summary={
                "cpu_start_pct": snap_start.cpu_percent,
                "cpu_peak_pct": snap_end.cpu_percent,
                "ram_start_mb": snap_start.ram_rss_mb,
                "ram_peak_mb": snap_end.ram_rss_mb,
                "ram_end_mb": snap_end.ram_rss_mb,
                "gpu_available": snap_end.gpu_available,
                "gpu_peak_mem_mb": snap_end.gpu_memory_used_mb,
            },
            stage_timings={"total_ms": dur_ms, "embed_ms": dur_ms},
        )
    except Exception as mon_err:
        print(f"[Manual Live Monitoring] {mon_err}")

    return {
        "status": "success",
        "pipeline_type": "live",
        "message": f"Article '{art_id}' successfully ingested and embedded for live retrieval.",
        "article_id": art_id,
        "article": {
            "id": art_id,
            "title": str(r["title"]),
            "section": str(r["section"]),
            "subsection": str(r["subsection"]),
            "date": r["date"].isoformat() if pd.notna(r["date"]) else None,
            "page_views": int(r["page_views"]),
            "is_live": bool(r["is_live"]),
            "url": str(r["url"]),
        },
        "duration_ms": dur_ms,
    }
