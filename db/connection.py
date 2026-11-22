"""
Database connection, session management, and extension initialization.
"""
import re
import os
import threading
from typing import Dict, Any, Generator, Optional, Union
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Read Postgres connection details from environment variables
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "news_recommender")

# Construct URL (or use direct DATABASE_URL if provided)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy Base and Session Setup
Base = declarative_base()

try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # checks liveness before giving connections
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    # Try fallback to standard postgresql://
    try:
        alt_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(alt_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e2:
        engine = None
        SessionLocal = None
        print(f"[DB Warning] Could not initialize engine: {e2}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset isolation across pooled connections
# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL's `SET search_path` is CONNECTION-scoped, but SQLAlchemy returns a
# connection to the pool after every commit. The next session can therefore
# inherit a different dataset's schema — which is how a live upload ended up
# writing its embeddings into `public`. Remember the intended schema per thread
# and re-apply it whenever the pool hands us a connection.
_schema_ctx = threading.local()


def _schema_for_thread() -> str:
    return getattr(_schema_ctx, "schema", "public")


if engine is not None:
    @event.listens_for(engine, "checkout")
    def _apply_search_path_on_checkout(dbapi_conn, connection_record, connection_proxy=None):
        schema = _schema_for_thread()
        cur = dbapi_conn.cursor()
        try:
            cur.execute(f'SET search_path TO "{schema}", public')
        finally:
            cur.close()


def set_search_path(session_or_conn, schema: str = "public") -> str:
    """Set the target schema for this thread and for the current connection.

    The thread-local part is what survives commits and pool checkouts; the
    explicit SET makes the change immediate on a connection already checked out.
    Always call this before the first query on a new session.
    """
    safe_schema = re.sub(r"[^a-zA-Z0-9_]", "", str(schema or "public").strip()) or "public"
    _schema_ctx.schema = safe_schema
    session_or_conn.execute(text(f'SET search_path TO {safe_schema}, public'))
    return safe_schema


def reset_search_path() -> None:
    """Return this thread to the protected default schema."""
    _schema_ctx.schema = "public"


def resolve_schema(session_or_conn, dataset_id_or_schema: Optional[str] = None) -> str:
    """
    Resolves a dataset ID or schema name to a concrete PostgreSQL schema name.
    If None or 'default' or 'public' -> returns 'public'.
    Otherwise looks up public.datasets by id or schema_name.
    """
    if not dataset_id_or_schema or dataset_id_or_schema in ("default", "public"):
        return "public"
    target = dataset_id_or_schema.strip()
    try:
        row = session_or_conn.execute(
            text("SELECT schema_name FROM public.datasets WHERE id = :id OR schema_name = :id LIMIT 1"),
            {"id": target}
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        pass
    # Sanitize and fallback to clean identifier
    return re.sub(r"[^a-zA-Z0-9_]", "", target) or "public"


def get_db() -> Generator:
    """FastAPI dependency for obtaining a database session."""
    if SessionLocal is None:
        raise RuntimeError("Database engine is not configured.")
    db = SessionLocal()
    try:
        set_search_path(db, "public")  # never inherit a dataset from the pooled connection
        yield db
    finally:
        db.close()
        reset_search_path()


def check_db_health() -> Dict[str, Any]:
    """
    Check if PostgreSQL is reachable and whether the pgvector extension is installed.
    """
    health = {
        "connected": False,
        "database": DB_NAME,
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "pgvector_ready": False,
        "tables_ready": False,
        "error": None,
    }
    if not engine:
        health["error"] = "SQLAlchemy engine not initialized"
        return health

    try:
        with engine.connect() as conn:
            # 1. Test basic connectivity
            conn.execute(text("SELECT 1"))
            health["connected"] = True

            # 2. Check if pgvector extension is available or installed
            try:
                result = conn.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                ).fetchone()
                if result:
                    health["pgvector_ready"] = True
                else:
                    # Try to create extension if user has superuser privileges
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                    health["pgvector_ready"] = True
            except Exception as ext_err:
                conn.rollback()
                health["pgvector_ready"] = False
                health["vector_warning"] = (
                    f"pgvector extension is not installed in PostgreSQL: {str(ext_err)}"
                )

            # 3. Check if articles table exists
            check_table = conn.execute(
                text(
                    "SELECT to_regclass('public.articles')"
                )
            ).scalar()
            health["tables_ready"] = bool(check_table)

    except OperationalError as op_err:
        health["error"] = f"Cannot connect to PostgreSQL at {DB_HOST}:{DB_PORT}. Is PostgreSQL running? Error: {op_err.orig if hasattr(op_err, 'orig') else str(op_err)}"
    except Exception as err:
        health["error"] = str(err)

    return health


# Columns added after the first release. `Base.metadata.create_all` only creates
# missing TABLES, never alters an existing one, so every new column is declared
# here once and applied idempotently at startup / ingestion.
_ADDED_COLUMNS = [
    # (table, column, SQL type, why it exists)
    ("article_recommendations", "rerank_score", "DOUBLE PRECISION",
     "raw cross-encoder score, so the reverse push can reuse it without re-scoring"),
    ("article_recommendations", "rank_raw", "INTEGER",
     "rank before boosts and MMR reordered the list"),
    ("article_recommendations", "boosted_score", "DOUBLE PRECISION",
     "relevance after boosts, before calibration - the value the ordering used"),
    ("article_recommendations", "pipeline_version", "VARCHAR(32)",
     "config fingerprint this row was scored under"),
    ("ingestion_jobs", "refreshed_articles", "INTEGER DEFAULT 0",
     "existing articles the reverse push refreshed in this upload"),
    ("ingestion_jobs", "pipeline_version", "VARCHAR(32)",
     "config fingerprint this job ran under"),
    ("articles", "embed_token_count", "INTEGER DEFAULT 0",
     "tokens actually sent to the embedder"),
    ("articles", "rerank_token_count", "INTEGER DEFAULT 0",
     "tokens actually sent to the cross-encoder"),
    ("articles", "is_truncated", "BOOLEAN DEFAULT FALSE",
     "the text was cut at the token budget"),
    ("articles", "recipe_version", "VARCHAR(32)",
     "text-composition recipe used for this row"),
    ("articles", "premium", "BOOLEAN DEFAULT FALSE",
     "paywalled/premium flag carried from the source CSV"),
    ("execution_records", "dataset_id", "VARCHAR(64) DEFAULT 'default'",
     "dataset this execution ran against"),
    ("datasets", "pipeline_type", "VARCHAR(32) DEFAULT 'precomputed'",
     "specifies whether dataset is precomputed (full) or live (embedding only)"),
    ("ingestion_jobs", "pipeline_type", "VARCHAR(32) DEFAULT 'precomputed'",
     "specifies whether ingestion was precomputed or live embedding"),
]

_ADDED_INDEXES = [
    # "popular in <section>, within N days" filters on both columns, so a composite
    # index beats the section-only one that came with the first release.
    ("ix_articles_section_date", "ON articles (section, date)"),
    # Monitoring queries filter heavily on job_type, status, and created_at.
    ("ix_exec_records_job_type", "ON execution_records (job_type)"),
    ("ix_exec_records_status", "ON execution_records (status)"),
    ("ix_exec_records_created_at", "ON execution_records (created_at DESC)"),
    ("ix_exec_stages_record_id", "ON execution_stages (record_id)"),
    ("ix_exec_records_dataset_id", "ON execution_records (dataset_id)"),
    ("ix_datasets_pipeline_type", "ON datasets (pipeline_type)"),
    # Industrial Scale: HNSW graph index for sub-25ms vector search across 600k+ rows
    ("ix_articles_embedding_hnsw", "ON articles USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"),
    # Industrial Scale: GIN index on text expressions for sub-15ms lexical search
    ("ix_articles_fts", "ON articles USING gin (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, '')))"),
]


def ensure_schema() -> bool:
    """Idempotent, lightweight migrations (safe on every startup and ingestion)."""
    if not engine:
        return False
    try:
        from . import models  # noqa: register models with Base
        with engine.begin() as conn:
            conn.execute(text('SET LOCAL search_path TO public'))
            Base.metadata.create_all(bind=conn)
    except Exception as e:
        print(f"[DB Schema] Table verification notice: {e}")

    statements = [f"ALTER TABLE public.{table} ADD COLUMN IF NOT EXISTS {column} {sql_type}"
                  for table, column, sql_type, _why in _ADDED_COLUMNS]
    statements += [f"CREATE INDEX IF NOT EXISTS {name} {target.replace('ON ', 'ON public.', 1)}"
                   for name, target in _ADDED_INDEXES]
    try:
        with engine.connect() as conn:
            for statement in statements:
                conn.execute(text(statement))
            # Seed default dataset record if not already present
            conn.execute(text("""
                INSERT INTO datasets (id, name, schema_name, pipeline_type, description, is_protected)
                VALUES ('default', 'Default (Production)', 'public', 'precomputed', 'Original production dataset in public schema', TRUE)
                ON CONFLICT (id) DO UPDATE SET is_protected = TRUE, pipeline_type = COALESCE(datasets.pipeline_type, 'precomputed');
            """))
            conn.commit()
        ensure_dataset_schemas()
        reset_search_path()
        return True
    except Exception as e:
        print(f"[DB Schema] Could not ensure schema: {e}")
        return False


def ensure_dataset_schemas() -> int:
    """Migrate every registered isolated schema without creating shared tables there."""
    if not engine:
        return 0
    migrated = 0
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT schema_name, pipeline_type FROM public.datasets WHERE schema_name <> 'public'"
        )).fetchall()
        for schema, pipeline_type in rows:
            safe = re.sub(r"[^a-zA-Z0-9_]", "", str(schema or ""))
            if not safe:
                continue
            tables = {r[0] for r in conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema=:s"
            ), {"s": safe})}
            for table_name, column, sql_type, _why in _ADDED_COLUMNS:
                if table_name in tables and table_name in {"articles", "article_recommendations", "ingestion_jobs"}:
                    conn.execute(text(f"ALTER TABLE {safe}.{table_name} ADD COLUMN IF NOT EXISTS {column} {sql_type}"))

            common_indexes = (
                ("articles_section_idx", "articles (section)"),
                ("articles_date_idx", "articles (date)"),
                ("articles_section_date_idx", "articles (section, date)"),
            )
            if "articles" in tables:
                for name, target in common_indexes:
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON {safe}.{target}"))
                if pipeline_type == "live":
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS articles_embedding_idx ON {safe}.articles USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)"))
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS articles_to_tsvector_idx ON {safe}.articles USING gin (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(description, '')))"))

            if pipeline_type == "precomputed" and "article_recommendations" in tables:
                seq = f"{safe}.article_recommendations_id_seq"
                conn.execute(text(f"CREATE SEQUENCE IF NOT EXISTS {seq}"))
                conn.execute(text(f"ALTER SEQUENCE {seq} OWNED BY {safe}.article_recommendations.id"))
                conn.execute(text(f"ALTER TABLE {safe}.article_recommendations ALTER COLUMN id SET DEFAULT nextval('{seq}'::regclass)"))
                max_id = conn.execute(text(f"SELECT COALESCE(MAX(id), 0) FROM {safe}.article_recommendations")).scalar() or 0
                if max_id:
                    conn.execute(text("SELECT setval(CAST(:seq AS regclass), :value, true)"), {"seq": seq, "value": max_id})
            migrated += 1
    return migrated


def init_db():
    """
    Ensures the pgvector extension is created and all declared tables exist.
    """
    health = check_db_health()
    if not health["connected"]:
        print(f"[DB Init Error] Database connection failed: {health['error']}")
        return False

    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("[DB Init] 'vector' extension verified/created.")
        except Exception as e:
            print(f"[DB Init Notice] Note on pgvector: {e}")

    # Import models so Base has them registered before create_all
    from . import models  # noqa
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    print("[DB Init] All database tables created successfully.")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing PostgreSQL Connection & pgvector Setup...")
    print("=" * 60)
    res = check_db_health()
    for k, v in res.items():
        print(f"{k.ljust(18)}: {v}")
    print("=" * 60)
    if res["connected"]:
        print("Running table initialization...")
        init_db()
        print("Done!")
    else:
        print("Please check your .env settings or make sure PostgreSQL is running.")
