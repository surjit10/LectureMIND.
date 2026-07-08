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

from pydantic import BaseModel, ConfigDict, field_validator


class MultimodalChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    chunk_id: str
    timestamp: float
    transcript: str
    visual_context: str
    ocr_text: str

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
