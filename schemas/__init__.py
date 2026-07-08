# schemas/__init__.py
# All Pydantic v2 models for LECTUREMIND V2.
# Chunk 1 — master data contracts.
from .enums import EntityType, RelationType, RetrievalRoute, PipelineStatus
from .manifest import Manifest
from .metadata import VideoMetadata
from .transcript import TranscriptSegment
from .frames import Frame
from .vlm import VLMCaption
from .ocr import OCRResult
from .chunk import MultimodalChunk
from .segment import LectureSegment
from .chunk_map import ChunkSegmentMap
from .entity import Entity
from .relation import Relation
from .triplet import RerankerTriplet
from .query import QueryRequest, QueryResponse
from .pipeline_state import QueryPipelineState

__all__ = [
    "EntityType",
    "RelationType",
    "RetrievalRoute",
    "PipelineStatus",
    "Manifest",
    "VideoMetadata",
    "TranscriptSegment",
    "Frame",
    "VLMCaption",
    "OCRResult",
    "MultimodalChunk",
    "LectureSegment",
    "ChunkSegmentMap",
    "Entity",
    "Relation",
    "RerankerTriplet",
    "QueryRequest",
    "QueryResponse",
    "QueryPipelineState",
]
