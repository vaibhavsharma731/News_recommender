"""
Dataset management REST API routes.
Provides complete CRUD and cloning for isolated experimentation datasets.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.connection import get_db
from db.dataset_repository import DatasetRepository

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])


class CreateDatasetPayload(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Display name for the dataset")
    description: Optional[str] = Field("", description="Optional purpose or notes for this dataset")
    dataset_id: Optional[str] = Field(None, max_length=64, description="Optional custom unique dataset ID")
    pipeline_type: Optional[str] = Field("precomputed", description="Pipeline type: 'precomputed' or 'live'")


class CloneDatasetPayload(BaseModel):
    source_dataset_id: str = Field("default", description="Source dataset ID to copy articles from")
    with_embeddings: bool = Field(True, description="Whether to copy 384d embedding vectors")


@router.get("")
def list_datasets(pipeline_type: Optional[str] = None, db: Session = Depends(get_db)):
    """List all available datasets with live article, embedding, and recommendation counts."""
    datasets = DatasetRepository.list_datasets(db, pipeline_type=pipeline_type)
    return {"datasets": datasets, "total": len(datasets)}


@router.post("")
def create_dataset(payload: CreateDatasetPayload, db: Session = Depends(get_db)):
    """Creates a new isolated PostgreSQL schema with independent articles and recommendations tables."""
    try:
        ds = DatasetRepository.create_dataset(
            db,
            name=payload.name,
            description=payload.description,
            dataset_id=payload.dataset_id,
            pipeline_type=payload.pipeline_type or "precomputed",
        )
        return {
            "status": "success",
            "message": f"Isolated dataset '{ds['name']}' created successfully.",
            "dataset": ds,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {e}")


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Fetch complete metadata and live statistics for a specific dataset."""
    ds = DatasetRepository.get_dataset(db, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return ds


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """
    Deletes an isolated experimental dataset and drops its PostgreSQL schema.
    Protected datasets (like 'default'/public) cannot be deleted.
    """
    try:
        DatasetRepository.delete_dataset(db, dataset_id)
        return {
            "status": "success",
            "message": f"Dataset '{dataset_id}' and all its isolated tables deleted.",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e}")


@router.post("/{dataset_id}/clone")
def clone_articles(
    dataset_id: str,
    payload: CloneDatasetPayload,
    db: Session = Depends(get_db),
):
    """
    Copies articles from a source dataset to this dataset without copying recommendations.
    Enables testing new recommendation logic on an identical article corpus.
    """
    try:
        copied = DatasetRepository.clone_articles(
            db,
            source_id=payload.source_dataset_id,
            target_id=dataset_id,
            with_embeddings=payload.with_embeddings,
        )
        return {
            "status": "success",
            "message": f"Cloned {copied} articles into dataset '{dataset_id}'.",
            "copied_articles": copied,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clone failed: {e}")
