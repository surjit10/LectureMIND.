# schemas/relation.py
# Relation produced by Stage A9 (Qwen2.5-7B-Instruct structured extraction).
#
# CRITICAL: source_entity_id and target_entity_id reference entity_id values only.
# Entity names are NEVER valid graph identifiers.
#
# Schema per spec:
# {
#   "relation_id": "rel_001",
#   "source_entity_id": "lec_001_entity_000012",
#   "relation": "PREREQUISITE_OF",
#   "target_entity_id": "lec_001_entity_000037"
# }

from pydantic import BaseModel, ConfigDict, field_validator
from .enums import RelationType


class Relation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation_id: str
    source_entity_id: str
    relation: RelationType
    target_entity_id: str

    @field_validator("relation_id", "source_entity_id", "target_entity_id")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("relation_id, source_entity_id, target_entity_id must not be empty.")
        return v
