"""Regression test for pooled-connection search_path leakage.

A live upload once wrote its embeddings into ``public`` instead of the selected
live dataset: ``SET search_path`` is connection-scoped and SQLAlchemy returns the
connection to the pool after every commit, so a later session could inherit a
different dataset's schema.  This test proves the thread-local + checkout hook
in :mod:`db.connection` keeps schemas isolated across threads and commits.
"""
import threading

from sqlalchemy import text

from db.connection import SessionLocal, reset_search_path, set_search_path


def _show(db):
    return db.execute(text("SHOW search_path")).scalar()


def test_schema_does_not_leak_between_threads():
    result = {}

    def worker(schema):
        db = SessionLocal()
        try:
            set_search_path(db, schema)
            db.execute(text("SELECT 1")).scalar()
            db.commit()  # releases the pooled connection
            result["worker"] = _show(db)
        finally:
            db.close()

    t = threading.Thread(target=worker, args=("ds_schema_isolation_probe",))
    t.start()
    t.join()

    other = SessionLocal()
    try:
        result["other"] = _show(other)
    finally:
        other.close()
        reset_search_path()

    assert "ds_schema_isolation_probe" in result["worker"]
    assert "ds_schema_isolation_probe" not in result["other"], (
        "a dataset schema leaked into an unrelated thread via the connection pool"
    )


def test_schema_survives_commit_and_recheckout():
    db = SessionLocal()
    try:
        set_search_path(db, "ds_schema_isolation_probe")
        db.execute(text("SELECT 1")).scalar()
        db.commit()
        assert "ds_schema_isolation_probe" in _show(db)

        # Force the session to check the connection back out of the pool.
        db.close()
        db = SessionLocal()
        assert "ds_schema_isolation_probe" in _show(db)
    finally:
        db.close()
        reset_search_path()
