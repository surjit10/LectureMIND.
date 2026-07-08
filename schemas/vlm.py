# schemas/vlm.py
# VLMCaption produced by Stage A4 (Qwen2-VL-7B-Instruct).
# A6 reads the `caption` field as `visual_context`.

from typing import List
from pydantic import BaseModel, ConfigDict, field_validator


class VLMCaption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: int
    caption: str
    objects: List[str]

    @field_validator("caption")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("caption must not be empty.")
        return v
