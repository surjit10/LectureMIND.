# schemas/segment.py
# LectureSegment produced by Stage A7.
# segment_id is assigned here — this is the authoritative source for chunk→segment mapping.

from typing import List
from pydantic import BaseModel, ConfigDict, field_validator


class LectureSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    title: str
    start: float
    end: float
    chunks: List[str]

    @field_validator("segment_id", "title")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("segment_id and title must not be empty.")
        return v

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be greater than start.")
        return v

    @field_validator("chunks")
    @classmethod
    def chunks_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("A segment must reference at least one chunk.")
        return v
