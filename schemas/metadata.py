# schemas/metadata.py
# Video metadata produced by Stage A1.

from pydantic import BaseModel, ConfigDict, field_validator


class VideoMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lecture_id: str
    duration: float  # seconds
    fps: float

    @field_validator("lecture_id")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("lecture_id must not be empty.")
        return v

    @field_validator("duration", "fps")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("duration and fps must be positive.")
        return v
