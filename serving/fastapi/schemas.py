# serving/fastapi/schemas.py
# API-level request/response schemas for FastAPI endpoints.
#
# Uses Chunk 1 schemas (QueryRequest, QueryResponse) as the source of truth.
# Additional endpoint-specific schemas defined here without modifying Chunk 1.

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict


# Re-export Chunk 1 API schemas.
from schemas.query import QueryRequest, QueryResponse


class LectureInfo(BaseModel):
    """Knowledge package metadata for GET /lectures and GET /lecture/{id}."""
    model_config = ConfigDict(extra="ignore")

    lecture_id: str
    title: str = ""
    display_name: Optional[str] = None
    status: str = "UNKNOWN"
    chunk_count: int = 0
    segment_count: int = 0
    # V7 optional metadata
    duration: Optional[float] = None
    course_name: Optional[str] = None
    speaker: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    created_at: Optional[str] = None
    pipeline_version: Optional[str] = None
    package_version: Optional[str] = None


class UploadResponse(BaseModel):
    """Response for POST /upload."""
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    status: str
    message: str = ""


class StatusResponse(BaseModel):
    """Response for GET /status/{lecture_id}."""
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    status: str
    progress: float = 0.0


class NotesRequest(BaseModel):
    """Request for POST /notes."""
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    format: str = "markdown"


class NotesResponse(BaseModel):
    """Response for POST /notes."""
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    notes: str = ""


class FlashcardsRequest(BaseModel):
    """Request for POST /flashcards."""
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    count: int = 10


class FlashcardsResponse(BaseModel):
    """Response for POST /flashcards."""
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    flashcards: List[Dict[str, str]] = []


class QuizRequest(BaseModel):
    """Request for POST /quiz."""
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    count: int = 5


class QuizResponse(BaseModel):
    """Response for POST /quiz."""
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    questions: List[Dict[str, Any]] = []


class LearningPathRequest(BaseModel):
    """Request for POST /learning_path."""
    model_config = ConfigDict(extra="forbid")

    lecture_id: str


class LearningPathResponse(BaseModel):
    """Response for POST /learning_path."""
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    path: List[Dict[str, str]] = []


class SettingsUpdate(BaseModel):
    """Request for PUT /settings."""

    ollama_model: Optional[str] = None
    top_k: Optional[int] = None


class SettingsResponse(BaseModel):
    """Response for PUT /settings."""

    status: str = "updated"
    settings: Dict[str, Any] = {}
