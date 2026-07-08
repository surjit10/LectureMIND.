# schemas/entity.py
# Entity produced by Stage A8 (Qwen2.5-7B-Instruct structured extraction).
#
# entity_id format: {lecture_id}_entity_{number}
# Example: lec_001_entity_000123
#
# Rules (from spec):
# * entity_id is lecture-scoped.
# * entity_id must be unique within a lecture.
# * entity names may repeat across lectures.
# * relation records must reference entity_id values, never entity names.

from pydantic import BaseModel, ConfigDict, field_validator
from .enums import EntityType


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    name: str
    type: EntityType

    @field_validator("entity_id", "name")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("entity_id and name must not be empty.")
        return v
