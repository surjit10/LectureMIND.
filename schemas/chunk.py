# schemas/chunk.py
# MultimodalChunk — MASTER CONTRACT for LECTUREMIND V2.
#
# CRITICAL: This is the single most important schema in the project.
# Every downstream stage (A7, B0, B1, A8, A9) reads this exact schema.
# If a field is renamed here, ALL consumers must be regenerated.
#
# Schema per spec:
# {
#   "lecture_id": "lec_001",
#   "chunk_id": "lec_001_chunk_000001",
#   "timestamp": 20.5,
#   "transcript": "BFS uses queue",
#   "visual_context": "BFS traversal diagram",
#   "ocr_text": "Breadth First Search O(V+E)"
# }
#
# segment_id is intentionally absent — assigned by Stage A7 via segments.json.
#
# V2 semantic chunking adds OPTIONAL provenance fields with defaults:
#   start_time   — first Whisper segment start (seconds)
#   end_time     — last Whisper segment end (seconds)
#   segment_ids  — Whisper segment ids merged into this chunk
# All three are set by A6 fusion when merge_segments=True; they default to
# 0.0 / 0.0 / [] for legacy 1:1 chunks and old packages, so every existing
# consumer (A7, A8, A9, B0, B1, C1, local loaders) validates unchanged.

from pydantic import BaseModel, ConfigDict, field_validator


class MultimodalChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    chunk_id: str
    timestamp: float
    transcript: str
    visual_context: str
    ocr_text: str

    # Provenance fields (V2 semantic chunking).
    # Set by A6 fusion when merge_segments=True; for 1:1 chunks these
    # equal the single segment's timing and segment_id. Defaults preserve
    # backward compatibility with pre-V2 packages.
    start_time: float = 0.0
    end_time: float = 0.0
    segment_ids: list[int] = []

    @field_validator("lecture_id", "chunk_id")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("lecture_id and chunk_id must not be empty.")
        return v

    @field_validator("timestamp")
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be >= 0.")
        return v

    @field_validator("end_time")
    @classmethod
    def end_time_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("end_time must be >= 0.")
        return v

    @field_validator("start_time")
    @classmethod
    def start_time_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("start_time must be >= 0.")
        return v
