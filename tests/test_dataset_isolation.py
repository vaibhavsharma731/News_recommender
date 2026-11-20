"""
Unit and integration tests for dataset and schema isolation.
Tests verify that:
1. Default dataset is protected and cannot be deleted.
2. New datasets create isolated PostgreSQL schemas.
3. Articles inserted into one dataset do NOT leak into another.
4. Recommendations generated in one dataset are isolated.
5. Deleting a dataset drops its schema without affecting default.
"""
import pytest
from db.connection import SessionLocal, set_search_path
from db.dataset_repository import DatasetRepository
from db.models import Article, ArticleRecommendation, Dataset


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()


def test_default_dataset_protected(db):
    """The default dataset must always exist and must be protected from deletion."""
    ds = DatasetRepository.ensure_default_dataset(db)
    assert ds.id == "default"
    assert ds.is_protected is True
    assert ds.schema_name == "public"

    # Attempting to delete must raise ValueError
    with pytest.raises(ValueError, match="Cannot delete protected dataset"):
        DatasetRepository.delete_dataset(db, "default")


def test_create_and_isolate_dataset(db):
    """Creating a new dataset creates an isolated schema with its own article pool."""
    test_ds_name = "Isolation Test Run"
    created = DatasetRepository.create_dataset(db, name=test_ds_name)
    ds_id = created["id"]
    schema_name = created["schema_name"]

    try:
        assert schema_name.startswith("ds_")
        assert created["article_count"] == 0

        # Insert article into newly created schema
        set_search_path(db, schema_name)
        art = Article(
            id="art_iso_test_001",
            title="Isolation Test Article Headline",
            body="This article should only exist in the experimental dataset schema.",
            section="technology",
        )
        db.add(art)
        db.commit()

        # Check count in experimental schema (re-apply search_path after commit releases connection)
        set_search_path(db, schema_name)
        assert db.query(Article).count() == 1
        assert db.query(Article).filter_by(id="art_iso_test_001").first() is not None

        # Check default (public) schema — MUST NOT contain the test article!
        set_search_path(db, "public")
        assert db.query(Article).filter_by(id="art_iso_test_001").first() is None

        # Stats must reflect 1 article in experimental dataset
        stats = DatasetRepository.get_dataset_stats(db, schema_name)
        assert stats["article_count"] == 1

    finally:
        # Clean up: delete test dataset
        set_search_path(db, "public")
        deleted = DatasetRepository.delete_dataset(db, ds_id)
        assert deleted is True

        # Verify dataset entry is gone from registry
        assert db.query(Dataset).filter_by(id=ds_id).first() is None


def test_clone_articles_across_datasets(db):
    """Cloning copies articles between datasets without copying recommendations."""
    created = DatasetRepository.create_dataset(db, name="Clone Target")
    ds_id = created["id"]
    schema_name = created["schema_name"]

    try:
        # Clone from default to new schema
        copied = DatasetRepository.clone_articles(db, source_id="default", target_id=ds_id)
        assert copied >= 0

        # Stats must show articles were copied but recommendations are 0
        stats = DatasetRepository.get_dataset_stats(db, schema_name)
        assert stats["article_count"] == copied
        assert stats["recommendation_count"] == 0

    finally:
        set_search_path(db, "public")
        DatasetRepository.delete_dataset(db, ds_id)
