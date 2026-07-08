# cloud/packaging/validator.py
# Stage C1 — Knowledge Package Validation.
#
# Validates all pipeline outputs before export.
# Checks: file existence, schema validation, embedding dimension,
# referential integrity, manifest consistency.
#
# Any failure raises an exception — hard fail, no warnings.
# Environment: Kaggle only.

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from config import CloudSettings, SharedSettings
from schemas.manifest import Manifest
from schemas.chunk import MultimodalChunk
from schemas.segment import LectureSegment
from schemas.entity import Entity
from schemas.relation import Relation

logger = logging.getLogger(__name__)

# Required files for a valid knowledge package.
REQUIRED_FILES = [
    "manifest.json",
    "segments.json",
    "entities.json",
    "relations.json",
    "multimodal_chunks.json",
    "embeddings.npy",
    "embedding_ids.json",
]

# NOTE: reranker_model/ is no longer required in the package.
# The global reranker is trained separately and loaded at local application startup.
REQUIRED_DIRS: list = []


class ValidationError(Exception):
    """Raised when package validation fails."""
    pass


def validate_package(
    lecture_id: str,
    cloud_settings: CloudSettings | None = None,
    shared_settings: SharedSettings | None = None,
) -> Dict[str, any]:
    """
    Validate all pipeline outputs for a lecture before export.

    Performs:
    1. File existence checks
    2. Schema validation
    3. Embedding dimension validation
    4. Referential integrity
    5. Manifest consistency

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings.
        shared_settings: Injected SharedSettings.

    Returns:
        Validation summary dict.

    Raises:
        ValidationError: On any validation failure (hard fail).
    """
    cs = cloud_settings or CloudSettings()
    ss = shared_settings or SharedSettings()
    lecture_dir = Path(cs.lecture_dir(lecture_id))

    logger.info("C1: Validating package for lecture '%s'...", lecture_id)

    # 1. FILE EXISTENCE
    _check_file_existence(lecture_dir)

    # 2. LOAD AND VALIDATE SCHEMAS
    manifest = _validate_manifest(lecture_dir)
    chunks = _validate_chunks(lecture_dir)
    segments = _validate_segments(lecture_dir)
    entities = _validate_entities(lecture_dir)
    relations = _validate_relations(lecture_dir)
    embedding_ids = _validate_embedding_ids(lecture_dir)

    # 3. EMBEDDING DIMENSION
    _validate_embedding_dimension(lecture_dir, ss.EMBEDDING_DIMENSION)

    # 4. REFERENTIAL INTEGRITY
    _validate_referential_integrity(chunks, segments, entities, relations, embedding_ids)

    # 5. MANIFEST CONSISTENCY
    _validate_manifest_counts(manifest, chunks, segments, entities, relations)

    summary = {
        "lecture_id": lecture_id,
        "status": "VALID",
        "chunk_count": len(chunks),
        "segment_count": len(segments),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "embedding_count": len(embedding_ids),
        "embedding_dimension": ss.EMBEDDING_DIMENSION,
    }
    logger.info("C1: Validation PASSED for '%s'.", lecture_id)
    return summary


def _check_file_existence(lecture_dir: Path) -> None:
    """Check all required files and directories exist."""
    for filename in REQUIRED_FILES:
        path = lecture_dir / filename
        if not path.exists():
            raise ValidationError(f"C1: Required file missing: {path}")

    for dirname in REQUIRED_DIRS:
        path = lecture_dir / dirname
        if not path.is_dir():
            raise ValidationError(f"C1: Required directory missing: {path}")


def _validate_manifest(lecture_dir: Path) -> Manifest:
    """Load and validate manifest.json against Chunk 1 schema."""
    data = json.loads((lecture_dir / "manifest.json").read_text(encoding="utf-8"))
    try:
        return Manifest(**data)
    except Exception as exc:
        raise ValidationError(f"C1: manifest.json schema validation failed: {exc}")


def _validate_chunks(lecture_dir: Path) -> List[Dict]:
    """Load and validate multimodal_chunks.json."""
    data = json.loads((lecture_dir / "multimodal_chunks.json").read_text(encoding="utf-8"))
    for i, chunk in enumerate(data):
        try:
            MultimodalChunk(**chunk)
        except Exception as exc:
            raise ValidationError(f"C1: Chunk {i} validation failed: {exc}")
    return data


def _validate_segments(lecture_dir: Path) -> List[Dict]:
    """Load and validate segments.json."""
    data = json.loads((lecture_dir / "segments.json").read_text(encoding="utf-8"))
    for i, seg in enumerate(data):
        try:
            LectureSegment(**seg)
        except Exception as exc:
            raise ValidationError(f"C1: Segment {i} validation failed: {exc}")
    return data


def _validate_entities(lecture_dir: Path) -> List[Dict]:
    """Load and validate entities.json."""
    data = json.loads((lecture_dir / "entities.json").read_text(encoding="utf-8"))
    for i, ent in enumerate(data):
        try:
            Entity(**ent)
        except Exception as exc:
            raise ValidationError(f"C1: Entity {i} validation failed: {exc}")
    return data


def _validate_relations(lecture_dir: Path) -> List[Dict]:
    """Load and validate relations.json."""
    data = json.loads((lecture_dir / "relations.json").read_text(encoding="utf-8"))
    for i, rel in enumerate(data):
        try:
            Relation(**rel)
        except Exception as exc:
            raise ValidationError(f"C1: Relation {i} validation failed: {exc}")
    return data


def _validate_embedding_ids(lecture_dir: Path) -> List[Dict]:
    """Load embedding_ids.json."""
    data = json.loads((lecture_dir / "embedding_ids.json").read_text(encoding="utf-8"))
    for entry in data:
        if "row_index" not in entry or "chunk_id" not in entry:
            raise ValidationError(f"C1: embedding_ids entry missing required keys: {entry}")
    return data


def _validate_embedding_dimension(lecture_dir: Path, expected_dim: int) -> None:
    """Verify embeddings.npy has the correct dimension."""
    embeddings = np.load(str(lecture_dir / "embeddings.npy"))
    if embeddings.ndim != 2:
        raise ValidationError(f"C1: embeddings.npy must be 2D, got {embeddings.ndim}D.")
    if embeddings.shape[1] != expected_dim:
        raise ValidationError(
            f"C1: Embedding dimension mismatch. Expected {expected_dim}, got {embeddings.shape[1]}. STOP EXPORT."
        )


def _validate_referential_integrity(
    chunks: List[Dict],
    segments: List[Dict],
    entities: List[Dict],
    relations: List[Dict],
    embedding_ids: List[Dict],
) -> None:
    """Check all foreign key references are valid."""
    chunk_ids: Set[str] = {c["chunk_id"] for c in chunks}
    segment_ids: Set[str] = {s["segment_id"] for s in segments}
    entity_ids: Set[str] = {e["entity_id"] for e in entities}

    # Every embedding_ids.chunk_id must exist in chunks.
    for entry in embedding_ids:
        if entry["chunk_id"] not in chunk_ids:
            raise ValidationError(
                f"C1: embedding_ids references non-existent chunk_id: {entry['chunk_id']}"
            )

    # Every segment's chunks must exist.
    for seg in segments:
        for cid in seg.get("chunks", []):
            if cid not in chunk_ids:
                raise ValidationError(
                    f"C1: Segment '{seg['segment_id']}' references non-existent chunk_id: {cid}"
                )

    # Every relation's source/target entity must exist.
    for rel in relations:
        if rel["source_entity_id"] not in entity_ids:
            raise ValidationError(
                f"C1: Relation '{rel['relation_id']}' references non-existent source_entity_id: "
                f"{rel['source_entity_id']}"
            )
        if rel["target_entity_id"] not in entity_ids:
            raise ValidationError(
                f"C1: Relation '{rel['relation_id']}' references non-existent target_entity_id: "
                f"{rel['target_entity_id']}"
            )

    # Optional: chunk_segment_map.json referential check.
    # Not required since segments.json is source of truth.


def _validate_manifest_counts(
    manifest: Manifest,
    chunks: List[Dict],
    segments: List[Dict],
    entities: List[Dict],
    relations: List[Dict],
) -> None:
    """Verify manifest counts match actual artifact counts."""
    if manifest.chunk_count != len(chunks):
        raise ValidationError(
            f"C1: Manifest chunk_count ({manifest.chunk_count}) != actual ({len(chunks)})"
        )
    if manifest.segment_count != len(segments):
        raise ValidationError(
            f"C1: Manifest segment_count ({manifest.segment_count}) != actual ({len(segments)})"
        )
    if manifest.entity_count != len(entities):
        raise ValidationError(
            f"C1: Manifest entity_count ({manifest.entity_count}) != actual ({len(entities)})"
        )
    if manifest.relation_count != len(relations):
        raise ValidationError(
            f"C1: Manifest relation_count ({manifest.relation_count}) != actual ({len(relations)})"
        )
    if manifest.embedding_dimension != 1024:
        raise ValidationError(
            f"C1: Manifest embedding_dimension ({manifest.embedding_dimension}) != 1024"
        )
