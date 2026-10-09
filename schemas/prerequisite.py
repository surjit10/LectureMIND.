"""Schema and graph-integrity checks for the optional prerequisite artifact."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PrerequisiteEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_id: str
    target_id: str
    confidence: float

    @field_validator("source_id", "target_id")
    @classmethod
    def non_empty_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entity IDs must not be empty")
        return value

    @field_validator("confidence")
    @classmethod
    def valid_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value


class PrerequisiteArtifact(BaseModel):
    model_config = ConfigDict(extra="allow")

    prerequisites: List[PrerequisiteEdge]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    threshold: float | None = None


def validate_prerequisite_artifact(
    data: Dict[str, Any], entity_ids: Iterable[str]
) -> List[Dict[str, Any]]:
    """Validate edge schema, references, uniqueness, and DAG acyclicity."""
    artifact = PrerequisiteArtifact.model_validate(data)
    known_ids = set(entity_ids)
    pairs: set[tuple[str, str]] = set()
    adjacency: Dict[str, List[str]] = {}

    for edge in artifact.prerequisites:
        pair = (edge.source_id, edge.target_id)
        if edge.source_id not in known_ids or edge.target_id not in known_ids:
            raise ValueError(f"prerequisite references an unknown entity: {pair}")
        if edge.source_id == edge.target_id:
            raise ValueError(f"prerequisite self-loop is not allowed: {pair}")
        if pair in pairs:
            raise ValueError(f"duplicate prerequisite edge: {pair}")
        pairs.add(pair)
        adjacency.setdefault(edge.source_id, []).append(edge.target_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("prerequisite graph contains a directed cycle")
        if node in visited:
            return
        visiting.add(node)
        for neighbor in adjacency.get(node, []):
            visit(neighbor)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(adjacency):
        visit(node)

    return [edge.model_dump() for edge in artifact.prerequisites]
