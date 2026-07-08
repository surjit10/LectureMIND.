# schemas/transcript.py
# TranscriptSegment produced by Stage A2 (Whisper large-v3).
# Field names are FIXED — Stage A6 reads start/end/text/segment_id exactly.

from pydantic import BaseModel, ConfigDict, field_validator


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: int
    start: float
    end: float
    text: str

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be greater than start.")
        return v

    @field_validator("text")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty.")
        return v
