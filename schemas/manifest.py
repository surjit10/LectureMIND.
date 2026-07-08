# schemas/manifest.py
# Manifest schema for the knowledge_package.zip boundary artifact.
# Validated before D1 (Neo4j load) and D2 (Qdrant load).
# embedding_dimension MUST equal 1024.
#
# V7: Optional metadata fields added for richer package description.
# All new fields default to None/empty for backward compatibility with
# existing packages that do not include them.

from typing import Optional
from pydantic import BaseModel, ConfigDict, field_validator


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Core fields (required) ---
    lecture_id: str
    created_at: str
    embedding_model: str
    embedding_dimension: int
    chunk_count: int
    segment_count: int
    entity_count: int
    relation_count: int
    reranker_version: str

    # --- V7 optional metadata fields ---
    # All default to None so old packages continue loading without reprocessing.
    display_name: Optional[str] = None
    course_name: Optional[str] = None
    speaker: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None        # lecture duration in seconds
    pipeline_version: Optional[str] = None  # cloud pipeline version tag
    package_version: Optional[str] = None   # package format version

    @field_validator("embedding_dimension")
    @classmethod
    def must_be_1024(cls, v: int) -> int:
        if v != 1024:
            raise ValueError(
                f"embedding_dimension must be 1024 (bge-large-en-v1.5). Got {v}."
            )
        return v

    @field_validator("lecture_id", "embedding_model", "reranker_version", "created_at")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty.")
        return v

    @field_validator("chunk_count", "segment_count", "entity_count", "relation_count")
    @classmethod
    def must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Count fields must be >= 0.")
        return v
