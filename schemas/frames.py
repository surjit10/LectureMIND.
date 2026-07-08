# schemas/frames.py
# Frame produced by Stage A3 (opencv slide-change detection).

from pydantic import BaseModel, ConfigDict, field_validator


class Frame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: int
    timestamp: float
    image_path: str

    @field_validator("timestamp")
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be >= 0.")
        return v

    @field_validator("image_path")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("image_path must not be empty.")
        return v
